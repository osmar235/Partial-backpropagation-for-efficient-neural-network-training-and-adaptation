"""Bounded eager execution-policy feasibility; separate numerical sanity gates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import collections
from contextlib import contextmanager
import copy
import gc
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import attention_reference as prior

q=prior.q;ROOT=prior.ROOT;HERE=ROOT/'protocol/optimized_execution'
SPEC=HERE/'execution_spec_v1.json';FREEZE=HERE/'FROZEN_EXECUTION_v1.json'
CAPABILITY=ROOT/'results/optimized_execution_preparation_v1/environment_probe.json'
OUT=ROOT/'results/optimized_execution_v1'
read=prior.read;save=prior.save

class NonfiniteError(RuntimeError):pass

def properties(arm):
    implementation,precision,backend=arm.split('_')
    assert implementation in ('native','reference') and precision in ('fp32','bf16') and backend in ('math','auto')
    return implementation,precision,backend

def sources():
    names=['scripts/optimized_execution.py','scripts/review_optimized_execution.py','scripts/test_optimized_execution.py',
        'scripts/run_optimized_execution.cmd','protocol/optimized_execution/execution_spec_v1.json',
        'protocol/optimized_execution/EXECUTION_PROTOCOL_v1.md','protocol/optimized_execution/environment_probe.py']
    return {**prior.sources(),**{str(ROOT/p):q.d.sha(ROOT/p) for p in names}}

def configuration(smoke,device):
    q.m.configure();spec=read(SPEC)
    if smoke:spec.update(seeds=[99203],batch_sizes=[3],sessions=2,discarded_warmup_updates=1,steady_updates=2)
    cfg=q.freeze(spec,smoke,device);cfg['sources']=sources()
    cfg['compiler_capability']=read(CAPABILITY);cfg['capability_sha256']=q.d.sha(CAPABILITY)
    cfg['triton_installed_now']=importlib.util.find_spec('triton') is not None
    assert cfg['triton_installed_now']==cfg['compiler_capability']['triton_installed']
    cfg.pop('fingerprint');cfg['fingerprint']=q.v.digest(cfg);return cfg

@contextmanager
def policy(arm):
    _,_,backend=properties(arm);q.m.configure()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    if backend=='auto':
        torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False
    backends=[q.SDPBackend.MATH] if backend=='math' else [q.SDPBackend.FLASH_ATTENTION,q.SDPBackend.EFFICIENT_ATTENTION,q.SDPBackend.CUDNN_ATTENTION,q.SDPBackend.MATH]
    try:
        with q.sdpa_kernel(backends):yield
    finally:q.m.configure()

def settings():
    return dict(**q.m.execution_settings(),fp16_reduced_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        math_reduced_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())

def make(seed,arm,cfg,device,qualification=False):
    implementation,_,_=properties(arm)
    model=prior.make(seed,cfg['spec'],cfg['smoke'],'math' if implementation=='native' else 'timm_reference',device)
    model.train()
    if qualification:
        for module in model.modules():
            if isinstance(module,nn.Dropout):module.p=0.
    return model

def forward_loss(model,x,y,precision,spec):
    with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
        logits=model(x);loss=F.cross_entropy(logits,y,label_smoothing=spec['label_smoothing'])
    assert logits.dtype==(torch.bfloat16 if precision=='bf16' else torch.float32)
    return logits,loss

def update(model,opt,x,y,seed,step,arm,spec):
    torch.manual_seed(seed+spec['dropout_seed_offset']+step);opt.zero_grad(set_to_none=True)
    _,loss=forward_loss(model,x,y,properties(arm)[1],spec)
    value=loss.item();loss.backward();opt.step();return value

def snapshots(seed,arm,cfg,x,y,device):
    spec=cfg['spec'];model=make(seed,arm,cfg,device,qualification=True);opt=q.optimizer(model,spec)
    initial=q.old.state_hash(model.state_dict());result=[]
    with policy(arm):
        for step in range(spec['correctness_updates']):
            torch.manual_seed(seed+spec['dropout_seed_offset']+step);opt.zero_grad(set_to_none=True)
            logits,loss=forward_loss(model,x,y,properties(arm)[1],spec);loss.backward()
            assert all(p.dtype==torch.float32 and p.grad is not None and p.grad.dtype==torch.float32 for p in model.parameters())
            gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
            opt.step();assert all(int(opt.state[p]['step'])==step+1 for p in model.parameters())
            state=dict(logits=logits.detach().float().cpu(),loss=loss.detach().float().cpu(),gradient=gradients,
                parameter={n:p.detach().cpu().clone() for n,p in model.named_parameters()},
                exp_avg={n:opt.state[p]['exp_avg'].detach().cpu().clone() for n,p in model.named_parameters()},
                exp_avg_sq={n:opt.state[p]['exp_avg_sq'].detach().cpu().clone() for n,p in model.named_parameters()},
                rng=q.rng(device),model_hash=q.old.state_hash(model.state_dict()),optimizer_hash=q.opt_hash(opt))
            for kind in spec['fp32_tolerances']:
                values=state[kind].values() if isinstance(state[kind],dict) else [state[kind]]
                if not all(bool(torch.isfinite(v).all()) for v in values):raise NonfiniteError(f'Nonfinite {kind}, step {step}')
            result.append(state)
    return initial,result

def aggregate(a,b):
    if not isinstance(a,dict):a={'value':a};b={'value':b}
    assert a.keys()==b.keys();rows=[];an=bn=dn=dot=0.;maximum=0.
    for name in a:
        assert a[name].shape==b[name].shape
        x=a[name].double().reshape(-1);y=b[name].double().reshape(-1);delta=x-y
        aa=float(x@x);bb=float(y@y);dd=float(delta@delta);ab=float(x@y);mx=float(delta.abs().max())
        an+=aa;bn+=bb;dn+=dd;dot+=ab;maximum=max(maximum,mx)
        rows.append(dict(name=name,max_abs=mx,reference_l2=aa**.5,relative_l2=dd**.5/max(aa**.5,1e-12)))
    cosine=dot/(an*bn)**.5 if an and bn else (1. if not an and not bn else 0.)
    return dict(max_abs=maximum,relative_l2=dn**.5/max(an**.5,1e-12),cosine=cosine,tensors=rows)

def compare(reference,actual,arm,spec):
    if properties(arm)[1]=='fp32':return q.compare(reference,actual,spec['fp32_tolerances'])
    metrics=[];checks=[];budget=spec['bf16_sanity_budgets']
    for step,(a,b) in enumerate(zip(reference,actual)):
        bykind={kind:aggregate(a[kind],b[kind]) for kind in spec['fp32_tolerances']}
        for kind,value in bykind.items():metrics.append(dict(step=step,kind=kind,**value))
        for key,limit in budget.items():
            if key.endswith('_cosine_min'):kind=key[:-11];value=bykind[kind]['cosine'];ok=value>=limit
            elif key.endswith('_relative_l2'):kind=key[:-12];value=bykind[kind]['relative_l2'];ok=value<=limit
            else:kind=key[:-8];value=bykind[kind]['max_abs'];ok=value<=limit
            checks.append(dict(step=step,budget=key,value=value,limit=limit,pass_budget=bool(ok)))
    return dict(status='PASS_SANITY_ONLY' if all(c['pass_budget'] for c in checks) else 'FAIL_NUMERICAL_SANITY',
        checks=checks,metrics=metrics,quality_equivalence_established=False,
        rng_match=all(a['rng']==b['rng'] for a,b in zip(reference,actual)))

def supported(arm,device):
    if properties(arm)[1]=='bf16':
        if device.type!='cuda':return 'CUDA-only mixed-precision screen; CPU smoke skips BF16 arms'
        if not torch.cuda.is_bf16_supported():return 'Installed GPU/runtime does not report BF16 support'
    return None

def probe(seed,arm,cfg,x,y,device):
    reason=supported(arm,device)
    if reason:return dict(status='UNSUPPORTED',reason=reason)
    spec=cfg['spec'];model=make(seed,arm,cfg,device);opt=q.optimizer(model,spec)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with policy(arm):
                flags=settings()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                    loss=update(model,opt,x,y,seed,0,arm,spec);q.v.sync(device)
                if not math.isfinite(loss) or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(torch.isfinite(p).all()) for p in model.parameters()):
                    return dict(status='FAIL_NONFINITE',reason='Nonfinite profiled update')
        counts=collections.Counter(event.name for event in profiler.events());layers=2 if cfg['smoke'] else 6
        dispatch={name:counts[op] for name,op in q.frozen_backend.OPS.items()}
        assert sum(dispatch.values())==layers,dispatch
        if properties(arm)[2]=='math':assert dispatch['math']==layers
        return dict(status='SUPPORTED',policy=flags,dispatch=dispatch,
            attention_operators={k:n for k,n in counts.items() if 'scaled_dot_product' in k},
            loss=loss,output_dtype=properties(arm)[1],parameter_dtype='float32',all_gradients_finite=True,
            warnings=[str(w.message) for w in caught])
    except RuntimeError as error:
        if not (q.frozen_backend.unsupported(error) or 'cudnn_status_not_supported' in str(error).lower()):raise
        return dict(status='UNSUPPORTED',reason=str(error))

def order(spec,session,ci,ri):
    base=list(spec['arms'])
    if (session+ci)%2:base.reverse()
    shift=ri%len(base);return base[shift:]+base[:shift]

def trial(seed,batch,arm,ri,position,cfg,x,y,device):
    spec=cfg['spec'];gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    with policy(arm):
        q.v.sync(device);start=time.perf_counter()
        model=make(seed,arm,cfg,device);opt=q.optimizer(model,spec);q.v.sync(device);setup=time.perf_counter()-start
        initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()};initial_hash=q.old.state_hash(initial)
        q.v.sync(device);start=time.perf_counter();startup_loss=update(model,opt,x,y,seed,-101,arm,spec)
        q.v.sync(device);startup=time.perf_counter()-start
        q.v.sync(device);start=time.perf_counter()
        for step in range(spec['discarded_warmup_updates']):update(model,opt,x,y,seed,-100+step,arm,spec)
        del opt;model.load_state_dict(initial);model.zero_grad(set_to_none=True);opt=q.optimizer(model,spec)
        q.v.sync(device);warmup=time.perf_counter()-start;del initial
        assert q.old.state_hash(model.state_dict())==initial_hash and not opt.state
        q.v.sync(device);start=time.perf_counter();first_loss=update(model,opt,x,y,seed,0,arm,spec)
        q.v.sync(device);first=time.perf_counter()-start
        base=q.prior.memory_start(device);q.v.sync(device);start=time.perf_counter()
        losses=[update(model,opt,x,y,seed,i,arm,spec) for i in range(1,1+spec['steady_updates'])]
        q.v.sync(device);steady=time.perf_counter()-start;mem=q.prior.memory_end(device,base)
        assert all(p.dtype==torch.float32 and p.grad is not None and p.grad.dtype==torch.float32 for p in model.parameters())
        if not all(torch.isfinite(p).all() and torch.isfinite(p.grad).all() for p in model.parameters()):raise NonfiniteError('Nonfinite timed parameters/gradients')
        if not all(math.isfinite(z) for z in [startup_loss,first_loss,*losses]):raise NonfiniteError('Nonfinite timed loss')
        age=1+spec['steady_updates'];assert all(int(opt.state[p]['step'])==age for p in model.parameters())
        return dict(seed=seed,batch=batch,arm=arm,round=ri,position=position,initial_sha256=initial_hash,
            input_sha256=q.tensor_hash(x),labels_sha256=q.tensor_hash(y),setup_seconds=setup,
            startup_update_seconds=startup,startup_loss=startup_loss,warmup_restore_seconds=warmup,
            first_seconds=first,steady_seconds=steady,steady_update_seconds=steady/spec['steady_updates'],
            first_loss=first_loss,steady_losses=losses,steady_memory=mem,optimizer_age=age,
            final_rng=q.rng(device),final_model_sha256=q.old.state_hash(model.state_dict()),
            final_optimizer_sha256=q.opt_hash(opt),all_gradients_finite=True)

def condition(parent,seed,batch,ci,session,cfg,device):
    spec=cfg['spec'];parent.mkdir(parents=True,exist_ok=True)
    rc=dict(root_fingerprint=cfg['fingerprint'],session=session,seed=seed,batch=batch,condition_index=ci)
    rc['fingerprint']=q.v.digest(rc);q.old.write_freeze(parent/'run_config.json',rc)
    if q.d.validate_complete(parent,rc['fingerprint']):return
    attempt=q.d.next_attempt(parent);save(attempt/'process.json',dict(pid=os.getpid()))
    x,y=prior.inputs(seed,batch,spec,device);initial,reference=snapshots(seed,spec['required_baseline'],cfg,x,y,device)
    probes={};checks={};eligible=[]
    for arm in spec['arms']:
        probes[arm]=probe(seed,arm,cfg,x,y,device)
        if probes[arm]['status']!='SUPPORTED':checks[arm]=dict(status=probes[arm]['status'],reason=probes[arm]['reason']);continue
        try:
            start,actual=snapshots(seed,arm,cfg,x,y,device);again,repeat=snapshots(seed,arm,cfg,x,y,device)
            assert initial==start==again
            check=compare(reference,actual,arm,spec)
            check['deterministic_repeat']=all(all(a[k]==b[k] for k in ('rng','model_hash','optimizer_hash')) for a,b in zip(actual,repeat))
            if properties(arm)[2]=='math' and not check['deterministic_repeat']:check['status']='FAIL_DETERMINISM'
            checks[arm]=check
            if check['status'] in ('PASS','PASS_SANITY_ONLY'):eligible.append(arm)
            del actual,repeat
        except NonfiniteError as error:checks[arm]=dict(status='FAIL_NONFINITE',reason=str(error))
        print('CHECK',session,seed,batch,arm,checks[arm]['status'],flush=True)
    del reference
    save(attempt/'capability.json',probes);save(attempt/'correctness.json',checks)
    blocked=spec['required_baseline'] not in eligible
    rows=[];failures=[]
    if not blocked:
        failed=set()
        for ri in range(spec['rounds']):
            for pos,arm in enumerate(order(spec,session,ci,ri)):
                if arm not in eligible or arm in failed:continue
                try:row=trial(seed,batch,arm,ri,pos,cfg,x,y,device)
                except NonfiniteError as error:
                    failed.add(arm);failures.append(dict(arm=arm,round=ri,position=pos,reason=str(error)));continue
                rows.append(row);print('TIMING',session,seed,batch,ri,arm,round(1000*row['steady_update_seconds'],3),'ms',flush=True)
    save(attempt/'trials.json',rows);save(attempt/'timing_status.json',dict(required_baseline_blocked=blocked,failures=failures))
    prior.complete(parent,rc['fingerprint'],attempt)

def run_session(out,index,cfg,device):
    parent=out/f'session_{index:02d}';parent.mkdir(exist_ok=True)
    if q.d.validate_complete(parent,cfg['fingerprint']):return
    launch=q.d.next_attempt(parent/'launches');save(launch/'process.json',dict(pid=os.getpid(),session=index))
    save(launch/'preflight.json',{'smoke':True} if cfg['smoke'] else q.v.idle_preflight())
    save(launch/'telemetry_before.json',{'smoke':True} if cfg['smoke'] else q.v.telemetry())
    conditions=list(itertools.product(cfg['spec']['seeds'],cfg['spec']['batch_sizes']))
    indices=list(range(len(conditions)))
    if index%2:indices.reverse()
    for ci in indices:
        seed,batch=conditions[ci];condition(parent/f'seed_{seed}_batch_{batch}',seed,batch,ci,index,cfg,device)
    save(launch/'telemetry_after.json',{'smoke':True} if cfg['smoke'] else q.v.telemetry());prior.complete(parent,cfg['fingerprint'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--run-all',action='store_true')
    ap.add_argument('--session',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--tag',default='v1')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=ap.parse_args()
    assert args.tag.replace('_','').isalnum();device=torch.device(args.device);assert args.smoke or device.type=='cuda'
    cfg=configuration(args.smoke,device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();q.old.write_freeze(FREEZE,cfg);print('FROZEN',cfg['fingerprint']);return
    out=ROOT/f'results/optimized_execution_smoke_{args.tag}_{device.type}' if args.smoke else OUT
    if not args.smoke:assert cfg==read(FREEZE),'Frozen source/spec/environment changed'
    out.mkdir(exist_ok=True);q.old.write_freeze(out/'configuration.json',cfg)
    if args.session is not None:
        assert 0<=args.session<cfg['spec']['sessions'];run_session(out,args.session,cfg,device);return
    assert args.run_all
    from review_optimized_execution import audit
    if q.d.validate_complete(out,cfg['fingerprint']):print(json.dumps(audit(out),indent=2));return
    for index in range(cfg['spec']['sessions']):
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--session',str(index),'--device',args.device,'--tag',args.tag]
        if args.smoke:cmd.append('--smoke')
        subprocess.run(cmd,check=True)
    result=audit(out,write=True);prior.complete(out,cfg['fingerprint'])
    print(json.dumps(result,indent=2));print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
