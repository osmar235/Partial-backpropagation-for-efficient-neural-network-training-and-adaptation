"""Artificial matched-state runtime attribution; no model efficacy experiment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
from contextlib import contextmanager,nullcontext
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import sampled_weight_cost as cost
from sampled_weight_gradient import set_sampling_seed

m,v,d,old=cost.m,cost.v,cost.d,cost.old
ROOT=v.ROOT;HERE=ROOT/'protocol/runtime_profile'
SPEC=HERE/'profile_spec_v1.json';FREEZE=HERE/'FROZEN_PROFILE_v1.json';OUT=ROOT/'results/runtime_profile_v1'
PHASES=('seed_zero','forward','loss_readback','backward','optimizer')

def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def optimizer_hash(opt):
    sd=opt.state_dict()
    return v.digest(dict(groups=sd['param_groups'],state={str(i):{k:tensor_hash(z) if torch.is_tensor(z) else z for k,z in state.items()} for i,state in sd['state'].items()}))

def sources():
    names=['scripts/runtime_profile.py','scripts/review_runtime_profile.py','scripts/test_runtime_profile.py',
        'scripts/run_runtime_profile.cmd','protocol/runtime_profile/profile_spec_v1.json','protocol/runtime_profile/PROFILE_PROTOCOL_v1.md',
        'scripts/side_adaptation_prototype.py','scripts/review_stage14C.py']
    return {**cost.sources(),**{str(ROOT/p):d.sha(ROOT/p) for p in names}}

def freeze(spec,smoke,device):
    rec=dict(date='2026-10-25',spec=spec,synthetic=True,smoke=smoke,device=str(device),
        hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',sources=sources(),
        python=platform.python_version(),torch=str(torch.__version__),execution=m.execution_settings())
    rec['fingerprint']=v.digest(rec);return rec

class PhaseRecorder:
    def __init__(self,mode,device):self.mode=mode;self.device=device;self.items=[];self.step=0

    @contextmanager
    def phase(self,name):
        if self.mode=='plain':yield;return
        if self.mode=='profile':
            with torch.profiler.record_function('phase::'+name):yield
            return
        start=end=None
        if self.device.type=='cuda':
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        tick=time.perf_counter()
        try:yield
        finally:
            host=time.perf_counter()-tick
            if end is not None:end.record()
            self.items.append(dict(step=self.step,phase=name,host_seconds=host,start=start,end=end))

    def rows(self):
        v.sync(self.device)
        return [dict(step=row['step'],phase=row['phase'],host_seconds=row['host_seconds'],
            cuda_stream_ms=row['start'].elapsed_time(row['end']) if row['start'] is not None else None) for row in self.items]

@contextmanager
def module_ranges(model):
    handles=[];stacks={}
    def add(module,label):
        stack=[];stacks[label]=stack
        def enter(module,inputs):
            ctx=torch.profiler.record_function('module::'+label);ctx.__enter__();stack.append(ctx)
        def leave(module,inputs,output):
            if stack:stack.pop().__exit__(None,None,None)
        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(leave,always_call=True))
    add(model.body.conv_proj,'patch_embedding');add(model.head,'classifier')
    for i,block in enumerate(model.body.encoder.layers):
        add(block.self_attention,f'attention_{i}');add(block.mlp,f'mlp_{i}')
        add(block.ln_1,f'ln1_{i}');add(block.ln_2,f'ln2_{i}')
    try:yield
    finally:
        for handle in handles:handle.remove()
        for stack in stacks.values():
            while stack:stack.pop().__exit__(None,None,None)

def instrumented_update(model,opt,x,y,seed,step,spec,rec):
    rec.step=step
    with rec.phase('seed_zero'):
        set_sampling_seed(model,seed+spec['sampling_seed_offset']+step)
        torch.manual_seed(seed+spec['dropout_seed_offset']+step);opt.zero_grad(set_to_none=True)
    with rec.phase('forward'):logits=model(x)
    with rec.phase('loss_readback'):
        loss=F.cross_entropy(logits,y,label_smoothing=spec['label_smoothing']);value=loss.item()
        if not math.isfinite(value):raise FloatingPointError('Nonfinite profiling loss')
    with rec.phase('backward'):loss.backward()
    with rec.phase('optimizer'):opt.step()
    return value

def collect_operators(prof):
    # Non-overlapping self times for CPU events only. Do not add raw CUDA events
    # again: linked kernel durations are already attributed to CPU operators.
    grouped={};cuda_events=0
    for event in prof.events():
        device_type=str(event.device_type)
        if 'CUDA' in device_type:cuda_events+=1
        if 'CPU' not in device_type:continue
        current=event;phase='outside';module=''
        while current is not None:
            if current.name.startswith('module::') and not module:module=current.name[8:]
            if current.name.startswith('phase::'):
                phase=current.name[7:];break
            current=current.cpu_parent
        shapes=json.dumps(event.input_shapes,separators=(',',':'))
        key=(phase,module,event.name,shapes)
        row=grouped.setdefault(key,dict(phase=phase,forward_module=module,operator=event.name,input_shapes=shapes,
            calls=0,self_cpu_us=0.,self_device_us=0.))
        row['calls']+=1;row['self_cpu_us']+=float(event.self_cpu_time_total)
        row['self_device_us']+=float(event.self_device_time_total)
    return sorted(grouped.values(),key=lambda row:(row['phase'],-row['self_cpu_us'],row['operator'])),cuda_events

def run_regime(model,opt,x,y,seed,spec,mode,device,folder):
    rec=PhaseRecorder(mode,device);losses=[];steps=spec['profile_warmup_updates']+spec['profile_active_updates']
    prof=None;requested=False
    if mode=='profile':
        activities=[torch.profiler.ProfilerActivity.CPU]
        requested=device.type=='cuda' and torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
        if requested:activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof=torch.profiler.profile(activities=activities,record_shapes=True,with_stack=False,profile_memory=False,
            schedule=torch.profiler.schedule(wait=0,warmup=spec['profile_warmup_updates'],active=spec['profile_active_updates'],repeat=1))
    model.train();v.sync(device);started=time.perf_counter()
    with prof if prof is not None else nullcontext():
        with module_ranges(model) if mode=='profile' else nullcontext():
            for step in range(steps):
                losses.append(cost.update(model,opt,x,y,seed,step,spec) if mode=='plain' else
                    instrumented_update(model,opt,x,y,seed,step,spec,rec))
                if prof is not None:prof.step()
            v.sync(device)
    v.sync(device);seconds=time.perf_counter()-started
    phase_rows=rec.rows();operators=[];cuda_events=0
    if prof is not None:
        operators,cuda_events=collect_operators(prof)
        v.write_csv(folder/'operators.csv',operators);prof.export_chrome_trace(str(folder/'trace.json'))
    for p in model.parameters():
        assert p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all())
        assert int(opt.state[p]['step'])==spec['burnin_updates']+steps
    result=dict(mode=mode,wall_seconds=seconds,updates=steps,losses=losses,
        final_model_sha256=old.state_hash(model.state_dict()),final_optimizer_sha256=optimizer_hash(opt),
        optimizer_age=spec['burnin_updates']+steps,all_gradients_finite=True,
        cuda_activity_requested=requested,cuda_events=cuda_events,
        gpu_operator_attribution_available=bool(requested and cuda_events>0 and sum(row['self_device_us'] for row in operators)>0),
        cpu_rng_sha256=tensor_hash(torch.get_rng_state()),
        cuda_rng_sha256=tensor_hash(torch.cuda.get_rng_state(device)) if device.type=='cuda' else None)
    v.atomic_json(folder/'measurement.json',result)
    if phase_rows:v.write_csv(folder/'phases.csv',phase_rows)
    return result

def condition(parent,seed,batch,arm,position,cfg,device):
    spec=cfg['spec'];fake=cfg['smoke'];parent.mkdir(parents=True,exist_ok=True)
    rc=dict(root_fingerprint=cfg['fingerprint'],seed=seed,batch=batch,arm=arm,position=position)
    rc['fingerprint']=v.digest(rc);old.write_freeze(parent/'run_config.json',rc)
    if d.validate_complete(parent,rc['fingerprint']):print('SKIP',seed,batch,arm,flush=True);return
    attempt=d.next_attempt(parent)
    gen=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch)
    x=torch.randn(batch,3,32,32,generator=gen).to(device);y=torch.randint(0,10,(batch,),generator=gen).to(device)
    model=cost.make(seed,arm,spec,fake).to(device);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    for step in range(spec['burnin_updates']):cost.update(model,opt,x,y,seed,-100+step,spec)
    model.zero_grad(set_to_none=True)
    initial=copy.deepcopy(model.state_dict());initial_opt=copy.deepcopy(opt.state_dict())
    anchor_model=old.state_hash(initial);anchor_opt=optimizer_hash(opt)
    v.atomic_json(attempt/'anchor.json',dict(model_sha256=anchor_model,optimizer_sha256=anchor_opt,
        optimizer_age=spec['burnin_updates'],input_sha256=tensor_hash(x),labels_sha256=tensor_hash(y)))
    regimes=[];baseline=None
    sequence=[]
    for ri in range(spec['rounds']):
        modes=['plain','events'] if (position+ri)%2==0 else ['events','plain']
        sequence.extend((ri,mode) for mode in modes)
    sequence.append((spec['rounds'],'profile'))
    for ri,mode in sequence:
        model.load_state_dict(initial,strict=True);opt.load_state_dict(copy.deepcopy(initial_opt));model.zero_grad(set_to_none=True)
        assert old.state_hash(model.state_dict())==anchor_model and optimizer_hash(opt)==anchor_opt
        folder=attempt/f'{ri:02d}_{mode}';folder.mkdir()
        rec=run_regime(model,opt,x,y,seed,spec,mode,device,folder)
        identity={k:rec[k] for k in ('losses','final_model_sha256','final_optimizer_sha256','cpu_rng_sha256','cuda_rng_sha256')}
        if baseline is None:baseline=identity
        assert identity==baseline,'Instrumentation changed computation'
        regimes.append(dict(round=ri,mode=mode,folder=str(folder),wall_seconds=rec['wall_seconds']))
    v.atomic_json(attempt/'regimes.json',regimes)
    v.atomic_json(parent/'COMPLETE.json',dict(fingerprint=rc['fingerprint'],attempt=str(attempt),
        sha256={str(p):d.sha(p) for p in attempt.rglob('*') if p.is_file()}))
    print('PROFILE COMPLETE',seed,batch,arm,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args();m.configure()
    spec=json.loads(SPEC.read_text());device=torch.device(args.smoke_device if args.smoke else 'cuda')
    if args.smoke:spec.update(seeds=[99061],batch_sizes=[3],rounds=1,burnin_updates=1,profile_active_updates=1)
    assert device.type=='cpu' or torch.cuda.is_available();cfg=freeze(spec,args.smoke,device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();old.write_freeze(FREEZE,cfg);print('FROZEN',cfg['fingerprint']);return
    out=ROOT/f'results/runtime_profile_smoke_v1_{device.type}' if args.smoke else OUT
    if not args.smoke:assert cfg==json.loads(FREEZE.read_text()),'Frozen configuration changed'
    out.mkdir(exist_ok=True);old.write_freeze(out/'configuration.json',cfg)
    from review_runtime_profile import audit
    if d.validate_complete(out,cfg['fingerprint']):print(json.dumps(audit(out,False),indent=2));return
    launches=d.next_attempt(out/'launches')
    v.atomic_json(launches/'preflight.json',{'smoke':True} if args.smoke else v.idle_preflight())
    v.atomic_json(launches/'telemetry_before.json',{'smoke':True} if args.smoke else v.telemetry())
    position=0
    for si,seed in enumerate(spec['seeds']):
        for bi,batch in enumerate(spec['batch_sizes']):
            arms=spec['arms'] if (si+bi)%2==0 else list(reversed(spec['arms']))
            for arm in arms:
                condition(out/f'seed_{seed}_batch_{batch}'/arm,seed,batch,arm,position,cfg,device);position+=1
    v.atomic_json(launches/'telemetry_after.json',{'smoke':True} if args.smoke else v.telemetry())
    result=audit(out,True)
    paths=[out/p for p in ('configuration.json','audit.json','summary.csv','phase_summary.csv','REPORT.md')]
    paths+=list(out.glob('seed_*/*/COMPLETE.json'))+list(out.glob('seed_*/*/run_config.json'))+list((out/'launches').rglob('*.json'))
    v.atomic_json(out/'COMPLETE.json',dict(fingerprint=cfg['fingerprint'],sha256={str(p):d.sha(p) for p in paths}))
    print(json.dumps(result,indent=2));print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
