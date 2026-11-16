"""Bounded artificial GPU operation benchmark; no real checkpoint/data access."""
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
from side_adaptation_prototype import ResidualSideNet

v=m.v;ROOT=v.ROOT
HERE=ROOT/'protocol/side_cost'
SPEC=HERE/'cost_spec_v1.json';FREEZE=HERE/'FROZEN_COST_v1.json'
OUT=ROOT/'results/side_cost_v1'


def sources():
    files=['scripts/side_cost_benchmark.py','scripts/review_side_cost.py',
        'scripts/side_adaptation_prototype.py','protocol/side_cost/cost_spec_v1.json',
        'protocol/side_cost/COST_PROTOCOL_v1.md']
    return {**old.sources(),**{str(ROOT/p):d.sha(ROOT/p) for p in files}}


def freeze(spec,smoke):
    record=dict(date='2026-10-25',spec=spec,smoke=smoke,synthetic=True,sources=sources(),
        python=platform.python_version(),torch=str(torch.__version__),execution=m.execution_settings())
    record['fingerprint']=v.digest(record)
    return record


def memory_start(device):
    if device.type=='cpu':return 0
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def memory_end(device,baseline):
    peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
    return dict(baseline=baseline,peak=peak,incremental=peak-baseline,
        reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)


def make(seed,arm,spec,smoke):
    core=m.make_model('full_rgb32',seed,smoke)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+310001)
        with torch.no_grad():core.head.weight.normal_(std=.02)
    if arm=='head':core.body.requires_grad_(False)
    if arm in ('full','head'):return core
    return ResidualSideNet(core,side_width=spec['side_width'],taps=tuple(spec['taps']),
        final_features_only=arm=='final_side',seed=seed+311001)


def optimizer(model,spec):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=spec['learning_rate'],weight_decay=spec['weight_decay'])


def order(spec,seed_index,batch_index,round_index):
    base=spec['arms'] if (seed_index+batch_index)%2==0 else list(reversed(spec['arms']))
    rotation=round_index%len(base)
    return base[rotation:]+base[:rotation]


def trial(seed,arm,batch,round_index,position,spec,smoke,x,y,device):
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    v.sync(device);base=memory_start(device);tick=time.perf_counter()
    model=make(seed,arm,spec,smoke)
    initial=copy.deepcopy(model.state_dict())
    model.to(device);model.train();opt=optimizer(model,spec)
    v.sync(device);setup_seconds=time.perf_counter()-tick;setup_memory=memory_end(device,base)
    initial_hash=old.state_hash(model.state_dict());body_before=old.state_hash(model.body.state_dict())
    assert not opt.state
    base=memory_start(device);v.sync(device);tick=time.perf_counter()
    for step in range(spec['discarded_warmup_updates']):
        old.update(model,opt,x,y,seed+312001+step,spec['label_smoothing'])
    del opt
    model.load_state_dict(initial,strict=True);model.zero_grad(set_to_none=True)
    opt=optimizer(model,spec)
    v.sync(device);warmup_restore_seconds=time.perf_counter()-tick;warmup_memory=memory_end(device,base)
    assert not opt.state and old.state_hash(model.state_dict())==initial_hash
    base=memory_start(device);v.sync(device);tick=time.perf_counter()
    first_loss=old.update(model,opt,x,y,seed+313001,spec['label_smoothing'])
    v.sync(device);first_seconds=time.perf_counter()-tick;first_memory=memory_end(device,base)
    base=memory_start(device);v.sync(device);tick=time.perf_counter()
    losses=[]
    for step in range(spec['steady_updates_per_trial']):
        losses.append(old.update(model,opt,x,y,seed+313002+step,spec['label_smoothing']))
    v.sync(device);steady_seconds=time.perf_counter()-tick;steady_memory=memory_end(device,base)
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert len(opt.state)==len(trainable)
    assert all(int(opt.state[p]['step'])==1+spec['steady_updates_per_trial'] for p in trainable)
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable)
    assert old.state_hash(model.state_dict())!=initial_hash
    if arm!='full':
        assert all(p.grad is None for p in model.body.parameters())
        assert old.state_hash(model.body.state_dict())==body_before
    row=dict(seed=seed,batch=batch,round=round_index,position=position,arm=arm,
        input_sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
        initial_sha256=initial_hash,trainable_parameters=sum(p.numel() for p in trainable),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        setup_seconds=setup_seconds,warmup_restore_seconds=warmup_restore_seconds,
        first_seconds=first_seconds,steady_seconds=steady_seconds,
        steady_update_seconds=steady_seconds/spec['steady_updates_per_trial'],
        setup_memory=setup_memory,warmup_memory=warmup_memory,first_memory=first_memory,steady_memory=steady_memory,
        first_loss=first_loss,steady_losses=losses,optimizer_age=1+spec['steady_updates_per_trial'],
        initial_state_restored=True,all_trainable_gradients_finite=True,frozen_body_verified=arm!='full')
    del model,opt,initial,trainable
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();m.configure();spec=json.loads(SPEC.read_text())
    if args.smoke:
        spec.update(seeds=[99011],batch_sizes=[3,4],rounds=4,side_width=6,taps=[1,2],
            discarded_warmup_updates=1,steady_updates_per_trial=2)
    record=freeze(spec,args.smoke)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();old.write_freeze(FREEZE,record)
        print('FROZEN',record['fingerprint']);return
    out=ROOT/'results/side_cost_smoke_v1' if args.smoke else OUT
    if not args.smoke:assert json.loads(FREEZE.read_text())==record
    out.mkdir(exist_ok=True);old.write_freeze(out/'configuration.json',record)
    from review_side_cost import audit
    if d.validate_complete(out,record['fingerprint']):
        print(json.dumps(audit(out,False),indent=2));return
    device=torch.device('cpu' if args.smoke else 'cuda')
    assert args.smoke or torch.cuda.is_available()
    attempt=d.next_attempt(out);v.atomic_json(attempt/'preflight.json',{'smoke':True} if args.smoke else v.idle_preflight())
    v.atomic_json(attempt/'device.json',dict(device=str(device),name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'))
    rows=[];telemetry=[]
    for si,seed in enumerate(spec['seeds']):
        for bi,batch in enumerate(spec['batch_sizes']):
            gen=torch.Generator().manual_seed(seed+314001+batch)
            x=torch.randn(batch,3,32,32,generator=gen).to(device)
            y=torch.randint(0,10,(batch,),generator=gen).to(device)
            for ri in range(spec['rounds']):
                if not args.smoke:telemetry.append(dict(seed=seed,batch=batch,round=ri,sample=v.telemetry()))
                for position,arm in enumerate(order(spec,si,bi,ri)):
                    row=trial(seed,arm,batch,ri,position,spec,args.smoke,x,y,device)
                    rows.append(row);v.atomic_json(attempt/'trials.json',rows)
                print(f'COST seed={seed} batch={batch} round={ri+1}/{spec["rounds"]}',flush=True)
            del x,y
    v.atomic_json(attempt/'telemetry.json',telemetry)
    v.atomic_json(out/'RAW_COMPLETE.json',dict(fingerprint=record['fingerprint'],attempt=str(attempt),
        sha256={str(p):d.sha(p) for p in attempt.iterdir() if p.is_file()}))
    result=audit(out,True)
    paths=[out/p for p in ('configuration.json','RAW_COMPLETE.json','summary.csv','paired.csv','audit.json','REPORT.md')]
    v.atomic_json(out/'COMPLETE.json',dict(fingerprint=record['fingerprint'],sha256={str(p):d.sha(p) for p in paths}))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
