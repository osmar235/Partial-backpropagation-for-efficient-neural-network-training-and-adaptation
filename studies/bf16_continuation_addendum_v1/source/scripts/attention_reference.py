"""Prospective same-policy comparison with pinned upstream timm attention."""
import argparse
import collections
import copy
import gc
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import torch.nn.functional as F
import qkv_layout as q
from attention_reference_adapter import install, UPSTREAM

ROOT=q.ROOT; HERE=ROOT/'protocol/attention_reference'
SPEC=HERE/'reference_spec_v1.json'; FREEZE=HERE/'FROZEN_REFERENCE_v1.json'
OUT=ROOT/'results/attention_reference_v1'

def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def save(path, value): q.v.atomic_json(path,value)
def sources():
    files=[ROOT/p for p in ('scripts/attention_reference_adapter.py','scripts/attention_reference.py',
        'scripts/review_attention_reference.py','scripts/test_attention_reference.py','scripts/run_attention_reference.cmd',
        'protocol/attention_reference/reference_spec_v1.json','protocol/attention_reference/REFERENCE_PROTOCOL_v1.md')]
    files+=list(UPSTREAM.iterdir())
    return {**q.sources(),**{str(p):q.d.sha(p) for p in files if p.is_file()}}

def configuration(smoke, device):
    spec=read(SPEC)
    if smoke: spec.update(seeds=[99173],batch_sizes=[3],sessions=2,discarded_warmup_updates=1,steady_updates=2)
    cfg=q.freeze(spec,smoke,device);cfg['sources']=sources()
    cfg['upstream']=read(UPSTREAM/'PROVENANCE.json')
    cfg.pop('fingerprint');cfg['fingerprint']=q.v.digest(cfg)
    return cfg

def make(seed,spec,smoke,arm,device):
    model=q.make(seed,spec,smoke,'qkv_unbind' if arm=='qkv_unbind' else 'math').to(device)
    return install(model) if arm=='timm_reference' else model

def order(spec,session,ci,ri):
    permutations=list(itertools.permutations(spec['arms']))
    return list(permutations[(ri+session+ci)%len(permutations)])

def inputs(seed,batch,spec,device):
    gen=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch)
    return torch.randn(batch,3,32,32,generator=gen).to(device),torch.randint(0,10,(batch,),generator=gen).to(device)

def snapshots(seed,arm,cfg,x,y,device):
    spec=cfg['spec'];model=make(seed,spec,cfg['smoke'],arm,device);model.train()
    opt=q.optimizer(model,spec);result=[]
    initial=q.old.state_hash(model.state_dict())
    with q.sdpa_kernel(q.SDPBackend.MATH):
        q.policy()
        for step in range(spec['correctness_updates']):
            torch.manual_seed(seed+spec['dropout_seed_offset']+step);opt.zero_grad(set_to_none=True)
            logits=model(x);loss=F.cross_entropy(logits,y,label_smoothing=spec['label_smoothing']);loss.backward()
            gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            assert len(gradients)==len(list(model.parameters()))
            opt.step();assert all(int(opt.state[p]['step'])==step+1 for p in model.parameters())
            result.append(dict(logits=logits.detach().cpu(),loss=loss.detach().cpu(),gradient=gradients,
                parameter={n:p.detach().cpu().clone() for n,p in model.named_parameters()},
                exp_avg={n:opt.state[p]['exp_avg'].detach().cpu().clone() for n,p in model.named_parameters()},
                exp_avg_sq={n:opt.state[p]['exp_avg_sq'].detach().cpu().clone() for n,p in model.named_parameters()},
                rng=q.rng(device),model_hash=q.old.state_hash(model.state_dict()),optimizer_hash=q.opt_hash(opt)))
    return initial,result

def probe(seed,arm,cfg,x,y,device):
    spec=cfg['spec'];model=make(seed,spec,cfg['smoke'],arm,device);model.train();opt=q.optimizer(model,spec)
    with q.sdpa_kernel(q.SDPBackend.MATH):
        policy=q.policy()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:
            q.update(model,opt,x,y,seed,0,spec);q.v.sync(device)
    counts=collections.Counter(event.name for event in prof.events())
    layers=len(model.body.encoder.layers)
    assert counts[q.MATH_OP]==layers
    assert all(counts[op]==0 for op in q.frozen_backend.OPS.values() if op!=q.MATH_OP)
    assert counts['UnbindBackward0']==(0 if arm=='math' else layers)
    assert all(p.requires_grad and p.dtype==torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    copies=collections.Counter((e.name,str(e.input_shapes)) for e in prof.events() if e.name in ('aten::contiguous','aten::clone','aten::copy_'))
    return dict(policy=policy,attention_operators={k:n for k,n in counts.items() if 'scaled_dot_product' in k},
        select_backward_nodes=counts['SelectBackward0'],unbind_backward_nodes=counts['UnbindBackward0'],
        copy_operations=[dict(operator=k[0],shapes=k[1],count=n) for k,n in sorted(copies.items())],full_finite_gradients=True)

def trial(seed,batch,arm,ri,position,cfg,x,y,device):
    spec=cfg['spec'];gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    with q.sdpa_kernel(q.SDPBackend.MATH):
        q.policy();q.v.sync(device);start=time.perf_counter()
        model=make(seed,spec,cfg['smoke'],arm,device);model.train();opt=q.optimizer(model,spec)
        q.v.sync(device);setup=time.perf_counter()-start
        initial=copy.deepcopy(model.state_dict());initial_hash=q.old.state_hash(initial)
        q.v.sync(device);start=time.perf_counter()
        for step in range(spec['discarded_warmup_updates']):q.update(model,opt,x,y,seed,-100+step,spec)
        del opt;model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True);opt=q.optimizer(model,spec)
        del initial
        q.v.sync(device);warmup=time.perf_counter()-start
        assert q.old.state_hash(model.state_dict())==initial_hash and not opt.state
        base=q.prior.memory_start(device);q.v.sync(device);start=time.perf_counter()
        first_loss=q.update(model,opt,x,y,seed,0,spec);q.v.sync(device);first=time.perf_counter()-start
        first_memory=q.prior.memory_end(device,base)
        base=q.prior.memory_start(device);q.v.sync(device);start=time.perf_counter()
        losses=[q.update(model,opt,x,y,seed,step,spec) for step in range(1,1+spec['steady_updates'])]
        q.v.sync(device);steady=time.perf_counter()-start;memory=q.prior.memory_end(device,base)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.isfinite(p).all() for p in model.parameters())
        age=1+spec['steady_updates'];assert all(int(opt.state[p]['step'])==age for p in model.parameters())
        return dict(seed=seed,batch=batch,arm=arm,round=ri,position=position,initial_sha256=initial_hash,
            input_sha256=q.tensor_hash(x),labels_sha256=q.tensor_hash(y),setup_seconds=setup,warmup_restore_seconds=warmup,
            first_seconds=first,steady_seconds=steady,steady_update_seconds=steady/spec['steady_updates'],
            first_loss=first_loss,steady_losses=losses,first_memory=first_memory,steady_memory=memory,
            optimizer_age=age,final_rng=q.rng(device),final_model_sha256=q.old.state_hash(model.state_dict()),
            final_optimizer_sha256=q.opt_hash(opt),all_gradients_finite=True)

def complete(parent,fingerprint,attempt=None):
    scope=attempt if attempt is not None else parent
    files=[p for p in scope.rglob('*') if p.is_file() and p!=parent/'COMPLETE.json']
    rec=dict(fingerprint=fingerprint,sha256={str(p):q.d.sha(p) for p in files})
    if attempt is not None:rec['attempt']=str(attempt)
    save(parent/'COMPLETE.json',rec)

def condition(parent,seed,batch,ci,session,cfg,device):
    spec=cfg['spec'];parent.mkdir(parents=True,exist_ok=True)
    rc=dict(root_fingerprint=cfg['fingerprint'],seed=seed,batch=batch,condition_index=ci,session=session)
    rc['fingerprint']=q.v.digest(rc);q.old.write_freeze(parent/'run_config.json',rc)
    if q.d.validate_complete(parent,rc['fingerprint']):return
    attempt=q.d.next_attempt(parent);save(attempt/'process.json',dict(pid=os.getpid()))
    x,y=inputs(seed,batch,spec,device);initial,reference=snapshots(seed,'math',cfg,x,y,device)
    checks={};probes={};eligible=[]
    for arm in spec['arms']:
        probes[arm]=probe(seed,arm,cfg,x,y,device)
        actual_initial,actual=snapshots(seed,arm,cfg,x,y,device)
        repeat_initial,repeat=snapshots(seed,arm,cfg,x,y,device)
        assert initial==actual_initial==repeat_initial
        check=q.compare(reference,actual,spec['tolerances'])
        check['deterministic_repeat']=all(all(a[k]==b[k] for k in ('model_hash','optimizer_hash','rng')) for a,b in zip(actual,repeat))
        if not check['deterministic_repeat']:check['status']='FAIL_DETERMINISM'
        checks[arm]=check
        if check['status']=='PASS':eligible.append(arm)
        print('CHECK',session,seed,batch,arm,check['status'],flush=True)
    assert 'math' in eligible
    save(attempt/'correctness.json',checks);save(attempt/'mechanism.json',probes)
    del reference,actual,repeat
    trials=[]
    for ri in range(spec['rounds']):
        for position,arm in enumerate(order(spec,session,ci,ri)):
            if arm not in eligible:continue
            row=trial(seed,batch,arm,ri,position,cfg,x,y,device);trials.append(row)
            print('TIMING',session,seed,batch,ri,arm,round(row['steady_update_seconds']*1000,3),'ms',flush=True)
    save(attempt/'trials.json',trials);complete(parent,rc['fingerprint'],attempt)

def run_session(out,index,cfg,device):
    parent=out/f'session_{index:02d}';parent.mkdir(exist_ok=True)
    if q.d.validate_complete(parent,cfg['fingerprint']):return
    launch=q.d.next_attempt(parent/'launches')
    save(launch/'process.json',dict(pid=os.getpid(),session=index))
    save(launch/'preflight.json',{'smoke':True} if cfg['smoke'] else q.v.idle_preflight())
    save(launch/'telemetry_before.json',{'smoke':True} if cfg['smoke'] else q.v.telemetry())
    conditions=list(itertools.product(cfg['spec']['seeds'],cfg['spec']['batch_sizes']))
    indices=list(range(len(conditions)))
    if index%2:indices.reverse()
    for ci in indices:
        seed,batch=conditions[ci]
        condition(parent/f'seed_{seed}_batch_{batch}',seed,batch,ci,index,cfg,device)
    save(launch/'telemetry_after.json',{'smoke':True} if cfg['smoke'] else q.v.telemetry())
    complete(parent,cfg['fingerprint'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--run-all',action='store_true')
    ap.add_argument('--session',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--tag',default='v1')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=ap.parse_args()
    assert args.tag.replace('_','').isalnum();q.m.configure();device=torch.device(args.device)
    assert args.smoke or device.type=='cuda';assert device.type=='cpu' or torch.cuda.is_available()
    cfg=configuration(args.smoke,device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();q.old.write_freeze(FREEZE,cfg);print('FROZEN',cfg['fingerprint']);return
    out=ROOT/f'results/attention_reference_smoke_{args.tag}_{device.type}' if args.smoke else OUT
    if not args.smoke:assert cfg==read(FREEZE),'Source/spec/environment changed after freeze'
    out.mkdir(exist_ok=True);q.old.write_freeze(out/'configuration.json',cfg)
    if args.session is not None:
        assert 0<=args.session<cfg['spec']['sessions'];run_session(out,args.session,cfg,device);return
    assert args.run_all,'Use --run-all or --session'
    from review_attention_reference import audit
    if q.d.validate_complete(out,cfg['fingerprint']):print(json.dumps(audit(out),indent=2));return
    for index in range(cfg['spec']['sessions']):
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--session',str(index),'--device',args.device,'--tag',args.tag]
        if args.smoke:cmd.append('--smoke')
        subprocess.run(cmd,check=True)
    result=audit(out,write=True);complete(out,cfg['fingerprint'])
    print(json.dumps(result,indent=2));print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
