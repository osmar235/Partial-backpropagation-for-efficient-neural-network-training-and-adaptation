"""Frozen artificial operation-cost gate for exact-input/sampled-weight gradients."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
import torch
import stage10_models as m
import stage10_representation as old
import stage9_boundary_diagnostic as d
from side_cost_benchmark import memory_start,memory_end
from sampled_weight_gradient import install,set_sampling_seed

v=m.v;ROOT=v.ROOT;HERE=ROOT/'protocol/sampled_weight_cost'
SPEC=HERE/'cost_spec_v1.json';FREEZE=HERE/'FROZEN_COST_v1.json';OUT=ROOT/'results/sampled_weight_cost_v1'

def sources():
    files=['scripts/sampled_weight_gradient.py','scripts/test_sampled_weight_gradient.py','scripts/sampled_weight_cost.py',
        'scripts/review_sampled_weight_cost.py','scripts/side_cost_benchmark.py','scripts/run_sampled_weight_cost.cmd',
        'protocol/sampled_weight_cost/cost_spec_v1.json','protocol/sampled_weight_cost/COST_PROTOCOL_v1.md']
    return {**old.sources(),**{str(ROOT/p):d.sha(ROOT/p) for p in files}}

def freeze(spec,smoke):
    rec=dict(date='2026-10-25',spec=spec,smoke=smoke,synthetic=True,sources=sources(),python=platform.python_version(),
        torch=str(torch.__version__),execution=m.execution_settings())
    rec['fingerprint']=v.digest(rec);return rec

def make(seed,arm,spec,smoke):
    core=m.make_model('full_rgb32',seed,smoke)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+320001)
        with torch.no_grad():core.head.weight.normal_(std=.02)
    if arm!='native':install(core,spec['patches_total'] if arm=='exact_custom' else spec['patches_kept'])
    return core

def order(spec,si,bi,ri):
    arms=spec['arms'] if (si+bi)%2==0 else list(reversed(spec['arms']))
    shift=ri%len(arms);return arms[shift:]+arms[:shift]

def update(model,opt,x,y,seed,step,spec):
    set_sampling_seed(model,seed+spec['sampling_seed_offset']+step)
    return old.update(model,opt,x,y,seed+spec['dropout_seed_offset']+step,spec['label_smoothing'])

def trial(seed,arm,batch,ri,position,spec,smoke,x,y,device):
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    v.sync(device);base=memory_start(device);tick=time.perf_counter()
    model=make(seed,arm,spec,smoke);initial=copy.deepcopy(model.state_dict());model.to(device);model.train();opt=optimizer(model)
    v.sync(device);setup_seconds=time.perf_counter()-tick;setup_memory=memory_end(device,base)
    initial_hash=old.state_hash(model.state_dict());assert not opt.state
    v.sync(device);base=memory_start(device);tick=time.perf_counter()
    for i in range(spec['discarded_warmup_updates']):update(model,opt,x,y,seed,-100+i,spec)
    del opt;model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True);opt=optimizer(model)
    v.sync(device);warmup_seconds=time.perf_counter()-tick;warmup_memory=memory_end(device,base)
    assert not opt.state and old.state_hash(model.state_dict())==initial_hash
    v.sync(device);base=memory_start(device);tick=time.perf_counter()
    first_loss=update(model,opt,x,y,seed,0,spec);v.sync(device)
    first_seconds=time.perf_counter()-tick;first_memory=memory_end(device,base)
    v.sync(device);base=memory_start(device);tick=time.perf_counter();losses=[]
    for step in range(1,spec['steady_updates_per_trial']+1):losses.append(update(model,opt,x,y,seed,step,spec))
    v.sync(device);steady_seconds=time.perf_counter()-tick;steady_memory=memory_end(device,base)
    assert all(p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert len(opt.state)==len(list(model.parameters()))
    assert all(int(opt.state[p]['step'])==1+spec['steady_updates_per_trial'] for p in model.parameters())
    assert old.state_hash(model.state_dict())!=initial_hash
    return dict(seed=seed,arm=arm,batch=batch,round=ri,position=position,initial_sha256=initial_hash,
        input_sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
        labels_sha256=hashlib.sha256(y.detach().cpu().numpy().tobytes()).hexdigest(),
        trainable_parameters=sum(p.numel() for p in model.parameters()),frozen_parameters=0,
        setup_seconds=setup_seconds,warmup_restore_seconds=warmup_seconds,first_seconds=first_seconds,
        steady_seconds=steady_seconds,steady_update_seconds=steady_seconds/spec['steady_updates_per_trial'],
        first_loss=first_loss,steady_losses=losses,setup_memory=setup_memory,warmup_memory=warmup_memory,
        first_memory=first_memory,steady_memory=steady_memory,optimizer_age=1+spec['steady_updates_per_trial'],
        initial_state_restored=True,all_trainable_gradients_finite=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--freeze-only',action='store_true')
    args=ap.parse_args();m.configure();spec=json.loads(SPEC.read_text())
    if args.smoke:spec.update(seeds=[99041],batch_sizes=[3,4],rounds=3,discarded_warmup_updates=1,steady_updates_per_trial=2)
    record=freeze(spec,args.smoke)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();old.write_freeze(FREEZE,record);print('FROZEN',record['fingerprint']);return
    out=ROOT/'results/sampled_weight_cost_smoke_v1' if args.smoke else OUT
    if not args.smoke:assert json.loads(FREEZE.read_text())==record
    out.mkdir(exist_ok=True);old.write_freeze(out/'configuration.json',record)
    from review_sampled_weight_cost import audit
    if d.validate_complete(out,record['fingerprint']):print(json.dumps(audit(out,False),indent=2));return
    device=torch.device('cpu' if args.smoke else 'cuda');assert args.smoke or torch.cuda.is_available()
    attempt=d.next_attempt(out);v.atomic_json(attempt/'preflight.json',{'smoke':True} if args.smoke else v.idle_preflight())
    v.atomic_json(attempt/'device.json',dict(device=str(device),name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'))
    rows=[];telemetry=[]
    for si,seed in enumerate(spec['seeds']):
        for bi,batch in enumerate(spec['batch_sizes']):
            gen=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch)
            x=torch.randn(batch,3,32,32,generator=gen).to(device);y=torch.randint(0,10,(batch,),generator=gen).to(device)
            for ri in range(spec['rounds']):
                if not args.smoke:telemetry.append(dict(seed=seed,batch=batch,round=ri,sample=v.telemetry()))
                for position,arm in enumerate(order(spec,si,bi,ri)):
                    rows.append(trial(seed,arm,batch,ri,position,spec,args.smoke,x,y,device));v.atomic_json(attempt/'trials.json',rows)
                print(f'WEIGHT COST seed={seed} batch={batch} round={ri+1}/{spec["rounds"]}',flush=True)
            del x,y
    v.atomic_json(attempt/'telemetry.json',telemetry)
    v.atomic_json(out/'RAW_COMPLETE.json',dict(fingerprint=record['fingerprint'],attempt=str(attempt),sha256={str(p):d.sha(p) for p in attempt.iterdir() if p.is_file()}))
    result=audit(out,True)
    paths=[out/p for p in ('configuration.json','RAW_COMPLETE.json','summary.csv','paired.csv','audit.json','REPORT.md')]
    v.atomic_json(out/'COMPLETE.json',dict(fingerprint=record['fingerprint'],sha256={str(p):d.sha(p) for p in paths}));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
