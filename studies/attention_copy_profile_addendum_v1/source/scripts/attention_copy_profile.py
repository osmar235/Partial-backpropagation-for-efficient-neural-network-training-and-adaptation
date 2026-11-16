"""Bounded matched-state profiling of qualified attention and compatibility overhead."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
from contextlib import nullcontext
import gc
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid
import torch
import bf16_continuation as c
import runtime_profile as attribution
from attention_copy_adapter import install,enable_ranges

ROOT=c.ROOT;HERE=ROOT/'protocol/attention_copy_profile'
SPEC=HERE/'profile_spec_v1.json';FREEZE=HERE/'FROZEN_PROFILE_v1.json';OUT=ROOT/'results/attention_copy_profile_v1'
read=c.read;save=c.r.v.atomic_json;sha=c.r.d.sha


def sources():
    names=['scripts/attention_copy_adapter.py','scripts/attention_copy_profile.py','scripts/review_attention_copy_profile.py',
           'scripts/test_attention_copy_profile.py','scripts/run_attention_copy_profile.cmd',
           'protocol/attention_copy_profile/profile_spec_v1.json','protocol/attention_copy_profile/PROFILE_PROTOCOL_v1.md']
    return {**c.sources(),**attribution.sources(),**{str(ROOT/p):sha(ROOT/p) for p in names}}


def configuration(smoke,device):
    cfg=c.configuration(smoke,device);spec=read(SPEC)
    if smoke:spec.update(seeds=[9971],batch_sizes=[3],sessions=2,anchor_updates=1,hardware_warmup_updates=1,
                         plain_updates=3,profile_active_updates=1,execution_policy='reference_bf16_auto' if device.type=='cuda' else 'reference_fp32_math')
    cfg['spec'].update(spec);cfg['sources']=sources()
    assert not c.e.supported(spec['execution_policy'],device)
    if not smoke:
        assert c.configuration(False,device)==read(c.FREEZE)
        for p in (c.FREEZE,ROOT/'results/prefix_adaptation_review_v1/REPORT.md',ROOT/'results/prefix_adaptation_review_v1/independent_audit.json'):
            cfg['evidence'][str(p)]=sha(p)
    cfg.pop('fingerprint');cfg['fingerprint']=c.r.v.digest(cfg);return cfg


def cases(spec,session):
    result=list(itertools.product(spec['seeds'],spec['batch_sizes'],spec['dropout_modes']))
    if session//2%2:
        half=len(result)//2;result=result[half:]+result[:half]
    if session%2:result=list(reversed(result))
    return result


def arm_order(spec,session,index):
    canonical=cases(spec,0).index(cases(spec,session)[index])
    return spec['arms'] if (session+canonical)%2==0 else list(reversed(spec['arms']))


def make(seed,arm,dropout,cfg,device):
    core,cp=c.load_source(seed,cfg)
    model,opt=c.r.make_condition(core,cp,seed,'full',dict(cfg['spec'],modes=['full']),device)
    install(model,arm);model.train()
    if dropout=='disabled':
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
    else:assert dropout=='original'
    for group in opt.param_groups:group['lr']=cfg['spec']['learning_rate']
    assert all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())
    return model,opt


def fixed_batch(seed,batch,spec,x,y):
    gen=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch)
    indices=torch.randperm(len(y),generator=gen)[:batch];assert len(indices)==batch
    aug=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch+1)
    mix=c.np.random.default_rng(seed+spec['input_seed_offset']+batch+2)
    perm=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch+3)
    xx=c.r.v.frozen6.augment_and_normalize(x[indices.to(x.device)],aug)
    xx,targets,lam=c.r.v.mix_batch(xx,y[indices.to(y.device)],mix,perm,spec['mixup_alpha'],10)
    return xx,targets,dict(indices=indices.tolist(),mixup_lambda=lam,input_sha256=c.q.tensor_hash(xx),targets_sha256=c.q.tensor_hash(targets))


def restore(model,opt,anchor):
    model.load_state_dict(anchor['weights'],strict=True);opt.load_state_dict(copy.deepcopy(anchor['optimizer']))
    model.zero_grad(set_to_none=True);model.train();enable_ranges(model,False)
    assert c.r.old.state_hash(model.state_dict())==anchor['model_sha256']
    assert c.r.optimizer_hash(opt)==anchor['optimizer_sha256']


def state(model,opt):
    return dict(model_sha256=c.r.old.state_hash(model.state_dict()),optimizer_sha256=c.r.optimizer_hash(opt))


def plain_update(model,opt,x,y,seed,index,spec):
    return c.update(model,opt,x,y,seed+spec['dropout_seed_offset']+index,spec['execution_policy'],spec)


def instrumented_update(model,opt,x,y,seed,index,spec,rec):
    rec.step=index
    with rec.phase('seed_zero'):
        torch.manual_seed(seed+spec['dropout_seed_offset']+index);opt.zero_grad(set_to_none=True)
    with rec.phase('forward_loss'):_,loss=c.forward_loss(model,x,y,spec['execution_policy'],spec)
    with rec.phase('loss_readback'):
        value=loss.item()
        if not math.isfinite(value):raise RuntimeError('Nonfinite loss')
    with rec.phase('backward'):loss.backward()
    with rec.phase('optimizer'):opt.step()
    return value


def run_regime(folder,model,opt,anchor,x,y,seed,spec,mode,updates,device):
    folder.mkdir();restore(model,opt,anchor)
    rec=attribution.PhaseRecorder('plain' if mode in ('plain','short_plain') else mode,device)
    prof=None;requested=False;operators=[];cuda_events=0
    if mode=='profile':
        activities=[torch.profiler.ProfilerActivity.CPU]
        requested=device.type=='cuda' and torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
        if requested:activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof=torch.profiler.profile(activities=activities,record_shapes=True,with_stack=False,profile_memory=False,
            schedule=torch.profiler.schedule(wait=0,warmup=spec['profile_warmup_updates'],active=spec['profile_active_updates'],repeat=1))
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    losses=[]
    with c.e.policy(spec['execution_policy']):
        flags=c.e.settings();c.r.v.sync(device);start=time.perf_counter()
        with prof if prof is not None else nullcontext():
            with attribution.module_ranges(model) if prof is not None else nullcontext():
                enable_ranges(model,prof is not None)
                for i in range(updates):
                    losses.append(plain_update(model,opt,x,y,seed,i,spec) if mode in ('plain','short_plain') else
                                  instrumented_update(model,opt,x,y,seed,i,spec,rec))
                    if prof is not None:prof.step()
                c.r.v.sync(device)
        c.r.v.sync(device);seconds=time.perf_counter()-start
    enable_ranges(model,False)
    phase_rows=rec.rows()
    if prof is not None:
        operators,cuda_events=attribution.collect_operators(prof)
        save(folder/'operators.json',operators);prof.export_chrome_trace(str(folder/'trace.json'))
    c.r.validate_optimizer(model,opt,'full',spec['anchor_updates']+updates,spec)
    assert all(p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(torch.isfinite(p).all()) for p in model.parameters())
    dispatch={k:sum(row['calls'] for row in operators if row['operator']==op) for k,op in c.q.frozen_backend.OPS.items()}
    if prof is not None:assert sum(dispatch.values())==spec['profile_active_updates']*len(model.body.encoder.layers)
    result=dict(mode=mode,updates=updates,wall_seconds=seconds,losses=losses,**state(model,opt),
                rng=c.q.rng(device),optimizer_age=spec['anchor_updates']+updates,all_gradients_finite=True,
                peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                execution_policy=flags,cuda_activity_requested=requested,cuda_events=cuda_events,
                gpu_operator_attribution_available=bool(requested and cuda_events and sum(r['self_device_us'] for r in operators)>0),dispatch=dispatch)
    save(folder/'measurement.json',result)
    if phase_rows:save(folder/'phases.json',phase_rows)
    return result


def identity(result):
    return {k:result[k] for k in ('updates','losses','model_sha256','optimizer_sha256','rng','optimizer_age')}


def run_case(folder,seed,batch,dropout,index,session,cfg,device,data):
    folder.mkdir();spec=cfg['spec'];x,y,inputs=fixed_batch(seed,batch,spec,*data)
    model,opt=make(seed,'compatible',dropout,cfg,device)
    with c.e.policy(spec['execution_policy']):
        for i in range(spec['anchor_updates']):plain_update(model,opt,x,y,seed,-100+i,spec)
    anchor=dict(weights={n:p.detach().cpu().clone() for n,p in model.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),**state(model,opt))
    save(folder/'anchor.json',dict(seed=seed,batch=batch,dropout=dropout,optimizer_age=spec['anchor_updates'],**inputs,**state(model,opt)))
    del model,opt
    records={};order=arm_order(spec,session,index)
    # Both primary arms run before any events or profiler in this case.
    for arm in order:
        parent=folder/arm;parent.mkdir()
        model,opt=make(seed,arm,dropout,cfg,device);restore(model,opt,anchor)
        with c.e.policy(spec['execution_policy']):
            for i in range(spec['hardware_warmup_updates']):plain_update(model,opt,x,y,seed,-200+i,spec)
        records[arm]=run_regime(parent/'plain',model,opt,anchor,x,y,seed,spec,'plain',spec['plain_updates'],device)
        del model,opt;gc.collect()
    for arm in order if session==spec['instrument_session'] else []:
        parent=folder/arm;model,opt=make(seed,arm,dropout,cfg,device)
        event=run_regime(parent/'events',model,opt,anchor,x,y,seed,spec,'events',spec['plain_updates'],device)
        short_n=spec['profile_warmup_updates']+spec['profile_active_updates']
        short=run_regime(parent/'short_plain',model,opt,anchor,x,y,seed,spec,'short_plain',short_n,device)
        profile=run_regime(parent/'profile',model,opt,anchor,x,y,seed,spec,'profile',short_n,device)
        assert identity(event)==identity(records[arm]),'Events changed computation'
        assert identity(short)==identity(profile),'Profiler changed computation'
        del model,opt;gc.collect()
    save(folder/'comparison.json',dict(seed=seed,batch=batch,dropout=dropout,session=session,arm_order=order,
        anchor_model_sha256=anchor['model_sha256'],anchor_optimizer_sha256=anchor['optimizer_sha256'],
        learning_identity_across_layouts=identity(records['compatible'])==identity(records['native_layout_control']),
        diagnostic_control_only=True,quality_tested=False))
    print(f'PROFILE CASE session={session} seed={seed} batch={batch} dropout={dropout}',flush=True)


def session_run(out,session,cfg,device):
    folder=out/f'session_{session:02d}';folder.mkdir(exist_ok=True)
    if c.r.d.validate_complete(folder,cfg['fingerprint']):print('SKIP completed session',session);return
    attempt=c.r.d.next_attempt(folder)
    save(attempt/'process.json',dict(session=session,uuid=str(uuid.uuid4()),pid=os.getpid()))
    try:
        save(attempt/'preflight.json',{'smoke':True} if cfg['smoke'] else c.r.v.idle_preflight())
        x,y,ev,ey,split=c.r.v.load_data('development',device,fake=cfg['smoke']);assert split==cfg['split'];del ev,ey
        for index,(seed,batch,dropout) in enumerate(cases(cfg['spec'],session)):
            run_case(attempt/f'{index:02d}_{seed}_{batch}_{dropout}',seed,batch,dropout,index,session,cfg,device,(x,y))
        save(attempt/'telemetry_after.json',{'smoke':True} if cfg['smoke'] else c.r.v.telemetry())
        save(folder/'COMPLETE.json',dict(fingerprint=cfg['fingerprint'],attempt=str(attempt),sha256={str(p):sha(p) for p in attempt.rglob('*') if p.is_file()}))
    except Exception as error:
        save(attempt/'FAILED.json',dict(type=type(error).__name__,message=str(error)));raise


def main():
    ap=argparse.ArgumentParser();actions=ap.add_mutually_exclusive_group(required=True)
    actions.add_argument('--freeze-only',action='store_true');actions.add_argument('--run-all',action='store_true');actions.add_argument('--session',type=int)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--tag',default='v1')
    args=ap.parse_args();assert __debug__ and args.tag.replace('_','').isalnum()
    c.r.m.configure();device=torch.device(args.device);assert args.smoke or device.type=='cuda'
    cfg=configuration(args.smoke,device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();c.r.old.write_freeze(FREEZE,cfg);print('FROZEN',cfg['fingerprint']);return
    if not args.smoke:assert cfg==read(FREEZE),'Frozen environment/source/data changed'
    out=ROOT/f'results/attention_copy_profile_smoke_{args.tag}_{device.type}' if args.smoke else OUT
    out.mkdir(exist_ok=True);c.r.old.write_freeze(out/'configuration.json',cfg)
    from review_attention_copy_profile import audit
    if c.r.d.validate_complete(out,cfg['fingerprint']):print(audit(out));return
    if args.session is not None:
        assert 0<=args.session<cfg['spec']['sessions'];session_run(out,args.session,cfg,device);return
    for session in range(cfg['spec']['sessions']):
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--session',str(session),'--device',args.device,'--tag',args.tag]
        if args.smoke:command.append('--smoke')
        subprocess.run(command,cwd=ROOT,check=True)
    result=audit(out,write=True)
    c.seal(out,cfg['fingerprint'],[out/p for p in ('configuration.json','audit.json','pairings.json','REPORT.md')]+list(out.glob('session_*/COMPLETE.json')))
    print(result);print('COMPLETE. Send',out,'for review.')


if __name__=='__main__':main()
