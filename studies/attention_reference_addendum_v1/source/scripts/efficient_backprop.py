"""Forced FP32 attention-backend feasibility with full gradients; artificial only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import collections
import copy
import gc
import hashlib
import json
import math
import platform
import time
import warnings
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
import sampled_weight_cost as prior

m,v,d,old=prior.m,prior.v,prior.d,prior.old
ROOT=v.ROOT;HERE=ROOT/'protocol/efficient_backprop'
SPEC=HERE/'backend_spec_v1.json';FREEZE=HERE/'FROZEN_BACKENDS_v1.json';OUT=ROOT/'results/efficient_backprop_v1'
BACKENDS={'math':SDPBackend.MATH,'efficient':SDPBackend.EFFICIENT_ATTENTION,
    'flash':SDPBackend.FLASH_ATTENTION,'cudnn':SDPBackend.CUDNN_ATTENTION}
OPS={'math':'aten::_scaled_dot_product_attention_math','efficient':'aten::_scaled_dot_product_efficient_attention',
    'flash':'aten::_scaled_dot_product_flash_attention','cudnn':'aten::_scaled_dot_product_cudnn_attention'}

def sources():
    import torch.nn.functional as functional
    import torch.nn.attention as attention
    paths=[ROOT/p for p in ('scripts/efficient_backprop.py','scripts/review_efficient_backprop.py',
        'scripts/test_efficient_backprop.py','scripts/run_efficient_backprop.cmd',
        'scripts/review_stage14C.py','scripts/side_adaptation_prototype.py',
        'protocol/efficient_backprop/backend_spec_v1.json','protocol/efficient_backprop/BACKEND_PROTOCOL_v1.md')]
    paths += [Path(functional.__file__),Path(attention.__file__)]
    return {**prior.sources(),**{str(p):d.sha(p) for p in paths}}

def freeze(spec,smoke,device):
    rec=dict(spec=spec,smoke=smoke,synthetic=True,device=str(device),sources=sources(),
        python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
        hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',execution=m.execution_settings())
    rec['fingerprint']=v.digest(rec);return rec

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def rng(device):return dict(cpu=tensor_hash(torch.get_rng_state()),cuda=tensor_hash(torch.cuda.get_rng_state(device)) if device.type=='cuda' else None)
def opt_hash(opt):
    sd=opt.state_dict()
    return v.digest(dict(groups=sd['param_groups'],state={str(i):{k:tensor_hash(z) if torch.is_tensor(z) else z for k,z in state.items()} for i,state in sd['state'].items()}))
def optimizer(model,spec):return torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
def make(seed,spec,fake):return prior.make(seed,'native',spec,fake)
def update(model,opt,x,y,seed,step,spec):return old.update(model,opt,x,y,seed+spec['dropout_seed_offset']+step,spec['label_smoothing'])
def policy():
    settings=m.execution_settings()
    assert settings['deterministic'] and not settings['warn_only'] and not settings['tf32_matmul'] and not settings['tf32_cudnn']
    assert not torch.is_autocast_enabled('cuda')
    return settings
def unsupported(error):
    msg=str(error).lower()
    return any(z in msg for z in ('no available kernel','no viable backend','no execution plans support',
        'no suitable kernel','does not have a deterministic implementation','not supported with deterministic',
        'not compiled with','requires sm','no kernel image is available'))

def probe(seed,batch,arm,spec,fake,x,y,device):
    if device.type=='cpu' and arm!='math':return dict(status='UNSUPPORTED',reason='CUDA backend screen; CPU smoke checks math only.',warnings=[])
    model=make(seed,spec,fake).to(device);model.train();opt=optimizer(model,spec)
    caught=[]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with sdpa_kernel(BACKENDS[arm]):
                settings=policy()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    loss=update(model,opt,x,y,seed,0,spec);v.sync(device)
        counts=collections.Counter(event.name for event in prof.events())
        layers=len(model.body.encoder.layers)
        assert counts[OPS[arm]]==layers,(arm,counts[OPS[arm]],layers)
        for other in OPS:
            if other!=arm:assert counts[OPS[other]]==0,(arm,'unexpected fallback',other)
        if arm!='math':assert counts[OPS[arm]+'_backward']==layers,(arm,'missing fused backward')
        assert all(p.dtype==torch.float32 and p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        return dict(status='SUPPORTED',loss=loss,inside_context=settings,
            attention_operators={k:n for k,n in counts.items() if 'scaled_dot_product' in k},
            all_parameters_have_finite_full_gradients=True,warnings=[str(w.message) for w in caught])
    except RuntimeError as error:
        if arm=='math' or not unsupported(error):raise
        return dict(status='UNSUPPORTED',reason=str(error),warnings=[str(w.message) for w in caught])
    finally:
        del model,opt
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()

def snapshots(seed,arm,spec,fake,x,y,device):
    model=make(seed,spec,fake).to(device);model.train();opt=optimizer(model,spec);result=[]
    with sdpa_kernel(BACKENDS[arm]):
        policy()
        for step in range(spec['correctness_updates']):
            torch.manual_seed(seed+spec['dropout_seed_offset']+step);opt.zero_grad(set_to_none=True)
            logits=model(x);loss=F.cross_entropy(logits,y,label_smoothing=spec['label_smoothing']);loss.backward()
            gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            assert len(gradients)==len(list(model.parameters()))
            opt.step()
            assert all(int(opt.state[p]['step'])==step+1 for p in model.parameters())
            result.append(dict(logits=logits.detach().cpu(),loss=loss.detach().cpu(),gradient=gradients,
                parameter={n:p.detach().cpu().clone() for n,p in model.named_parameters()},
                exp_avg={n:opt.state[p]['exp_avg'].detach().cpu().clone() for n,p in model.named_parameters()},
                exp_avg_sq={n:opt.state[p]['exp_avg_sq'].detach().cpu().clone() for n,p in model.named_parameters()},
                rng=rng(device),model_hash=old.state_hash(model.state_dict()),optimizer_hash=opt_hash(opt)))
    return result

def compare(reference,actual,tolerances):
    metrics=[];passed=True
    for step,(expected,got) in enumerate(zip(reference,actual)):
        if expected['rng']!=got['rng']:passed=False
        for kind in tolerances:
            a,b=expected[kind],got[kind]
            if not isinstance(a,dict):a={'value':a};b={'value':b}
            assert a.keys()==b.keys()
            for name in a:
                assert a[name].shape==b[name].shape and a[name].dtype==b[name].dtype==torch.float32
                delta=(a[name]-b[name]).abs();tol=tolerances[kind]
                finite=bool(torch.isfinite(a[name]).all() and torch.isfinite(b[name]).all())
                ratio=float((delta/(tol['atol']+tol['rtol']*a[name].abs())).max()) if finite else None
                ok=finite and ratio<=1;passed &= ok
                metrics.append(dict(step=step,kind=kind,name=name,max_abs=float(delta.max()) if finite else None,
                    max_scaled_error=ratio,pass_tolerance=ok))
    return dict(status='PASS' if passed else 'FAIL_NUMERICAL',rng_match=all(a['rng']==b['rng'] for a,b in zip(reference,actual)),metrics=metrics)

def trial(seed,arm,batch,ri,position,spec,fake,x,y,device):
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    with sdpa_kernel(BACKENDS[arm]):
        policy();v.sync(device);base=prior.memory_start(device);tick=time.perf_counter()
        model=make(seed,spec,fake);initial=copy.deepcopy(model.state_dict());model.to(device);model.train();opt=optimizer(model,spec)
        v.sync(device);setup=time.perf_counter()-tick;setup_memory=prior.memory_end(device,base)
        initial_hash=old.state_hash(model.state_dict());v.sync(device);tick=time.perf_counter()
        for step in range(spec['discarded_warmup_updates']):update(model,opt,x,y,seed,-100+step,spec)
        del opt;model.load_state_dict(initial);model.zero_grad(set_to_none=True);opt=optimizer(model,spec)
        assert old.state_hash(model.state_dict())==initial_hash and not opt.state
        v.sync(device);warmup=time.perf_counter()-tick
        base=prior.memory_start(device);tick=time.perf_counter();first_loss=update(model,opt,x,y,seed,0,spec);v.sync(device)
        first=time.perf_counter()-tick;first_memory=prior.memory_end(device,base)
        base=prior.memory_start(device);tick=time.perf_counter();losses=[]
        for step in range(1,1+spec['steady_updates']):losses.append(update(model,opt,x,y,seed,step,spec))
        v.sync(device);steady=time.perf_counter()-tick;steady_memory=prior.memory_end(device,base)
        assert all(p.requires_grad and p.dtype==torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        age=1+spec['steady_updates'];assert len(opt.state)==len(list(model.parameters()))
        assert all(int(opt.state[p]['step'])==age for p in model.parameters())
        return dict(seed=seed,arm=arm,batch=batch,round=ri,position=position,initial_sha256=initial_hash,
            input_sha256=tensor_hash(x),labels_sha256=tensor_hash(y),setup_seconds=setup,warmup_restore_seconds=warmup,
            first_seconds=first,steady_seconds=steady,steady_update_seconds=steady/spec['steady_updates'],
            first_loss=first_loss,steady_losses=losses,setup_memory=setup_memory,first_memory=first_memory,
            steady_memory=steady_memory,optimizer_age=age,final_rng=rng(device),all_gradients_finite=True,
            final_model_sha256=old.state_hash(model.state_dict()),final_optimizer_sha256=opt_hash(opt))

def order(arms,ci,ri):
    base=arms if ci%2==0 else list(reversed(arms));shift=ri%len(base)
    return base[shift:]+base[:shift]

def condition(parent,seed,batch,ci,cfg,device):
    spec=cfg['spec'];fake=cfg['smoke'];parent.mkdir(parents=True,exist_ok=True)
    rc=dict(root_fingerprint=cfg['fingerprint'],seed=seed,batch=batch,condition_index=ci);rc['fingerprint']=v.digest(rc)
    old.write_freeze(parent/'run_config.json',rc)
    if d.validate_complete(parent,rc['fingerprint']):print('SKIP',seed,batch,flush=True);return
    attempt=d.next_attempt(parent)
    gen=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch)
    x=torch.randn(batch,3,32,32,generator=gen).to(device);y=torch.randint(0,10,(batch,),generator=gen).to(device)
    probes={};checks={};eligible=[]
    reference=snapshots(seed,'math',spec,fake,x,y,device)
    for arm in spec['arms']:
        probes[arm]=probe(seed,batch,arm,spec,fake,x,y,device)
        if probes[arm]['status']=='SUPPORTED':
            actual=snapshots(seed,arm,spec,fake,x,y,device);repeat=snapshots(seed,arm,spec,fake,x,y,device)
            check=compare(reference,actual,spec['tolerances'])
            check['deterministic_repeat']=all(all(a[k]==b[k] for k in ('model_hash','optimizer_hash','rng')) for a,b in zip(actual,repeat))
            if not check['deterministic_repeat']:check['status']='FAIL_DETERMINISM'
            checks[arm]=check
            if check['status']=='PASS':eligible.append(arm)
        print('BACKEND',seed,batch,arm,probes[arm]['status'],checks.get(arm,{}).get('status',''),flush=True)
    assert 'math' in eligible
    v.atomic_json(attempt/'capability.json',probes);v.atomic_json(attempt/'correctness.json',checks)
    del reference,actual,repeat
    trials=[];position=0
    for ri in range(spec['rounds']):
        for arm in order(eligible,ci,ri):
            row=trial(seed,arm,batch,ri,position,spec,fake,x,y,device);position+=1;trials.append(row)
            print('TIMING',seed,batch,ri,arm,round(1000*row['steady_update_seconds'],3),'ms',flush=True)
    v.atomic_json(attempt/'trials.json',trials)
    v.atomic_json(parent/'COMPLETE.json',dict(fingerprint=rc['fingerprint'],attempt=str(attempt),
        sha256={str(p):d.sha(p) for p in attempt.rglob('*') if p.is_file()}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--tag',default='v1');args=ap.parse_args()
    assert args.tag.replace('_','').isalnum();m.configure();device=torch.device(args.device)
    assert device.type=='cuda' or args.smoke;assert device.type=='cpu' or torch.cuda.is_available()
    spec=json.loads(SPEC.read_text())
    if args.smoke:spec.update(seeds=[99073],batch_sizes=[3],rounds=2,discarded_warmup_updates=1,steady_updates=2)
    cfg=freeze(spec,args.smoke,device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();old.write_freeze(FREEZE,cfg);print('FROZEN',cfg['fingerprint']);return
    out=ROOT/f'results/efficient_backprop_smoke_{args.tag}_{device.type}' if args.smoke else OUT
    if not args.smoke:assert cfg==json.loads(FREEZE.read_text()),'Frozen configuration changed'
    out.mkdir(exist_ok=True);old.write_freeze(out/'configuration.json',cfg)
    from review_efficient_backprop import audit
    if d.validate_complete(out,cfg['fingerprint']):print(json.dumps(audit(out,False),indent=2));return
    launch=d.next_attempt(out/'launches');v.atomic_json(launch/'preflight.json',{'smoke':True} if args.smoke else v.idle_preflight())
    v.atomic_json(launch/'telemetry_before.json',{'smoke':True} if args.smoke else v.telemetry())
    ci=0
    for seed in spec['seeds']:
        for batch in spec['batch_sizes']:
            condition(out/f'seed_{seed}_batch_{batch}',seed,batch,ci,cfg,device);ci+=1
    v.atomic_json(launch/'telemetry_after.json',{'smoke':True} if args.smoke else v.telemetry())
    result=audit(out,True)
    paths=[out/p for p in ('configuration.json','audit.json','summary.csv','REPORT.md')]
    paths+=list(out.glob('seed_*/COMPLETE.json'))+list(out.glob('seed_*/run_config.json'))+list((out/'launches').rglob('*.json'))
    v.atomic_json(out/'COMPLETE.json',dict(fingerprint=cfg['fingerprint'],sha256={str(p):d.sha(p) for p in paths}))
    print(json.dumps(result,indent=2));print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
