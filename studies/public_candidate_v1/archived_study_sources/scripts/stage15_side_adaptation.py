"""Fixed five-arm side adaptation screen; real execution is explicitly user-run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
import math
import re
import time
from pathlib import Path
import numpy as np
import torch
import stage14_optimizer_history as previous
from side_adaptation_prototype import ResidualSideNet

t,v,d,m,old=previous.t,previous.v,previous.d,previous.m,previous.old
ROOT=v.ROOT
SPEC=ROOT/'protocol/stage15a/side_adaptation_spec_v1.json'
FREEZE=ROOT/'protocol/stage15a/FROZEN_SIDE_ADAPTATION_v1.json'
REAL_OUT=ROOT/'results/stage15A_side_adaptation_v1'
tensor_hash=previous.tensor_hash
optimizer_hash=previous.optimizer_hash
memory_start=previous.memory_start
memory_end=previous.memory_end
score=previous.score
source=previous.source

def sources():
    names=['scripts/stage15_side_adaptation.py','scripts/review_stage15A.py',
        'scripts/test_stage15_side_adaptation.py','scripts/run_stage15A_side_adaptation.cmd',
        'scripts/side_adaptation_prototype.py','protocol/stage15a/side_adaptation_spec_v1.json',
        'protocol/stage15a/SIDE_ADAPTATION_v1.md']
    return {**previous.sources(),**{str(ROOT/p):d.sha(ROOT/p) for p in names}}

def freeze_record(spec,fake):
    rec=previous.freeze_record(spec,fake)
    rec.update(prepared_date='2026-10-25',source_sha256=sources())
    # The prior full_fresh trajectory is a compulsory reproduction control.
    evidence=[previous.FREEZE,ROOT/'protocol/side_cost/cost_artifacts_sha256_v1.json',
        ROOT/'protocol/side_cost/POST_COST_DECISION_v1.md',ROOT/'protocol/strategy/SIDE_ADAPTATION_DESIGN_v1.md']
    mutable={str(ROOT/p) for p in ('CURRENT_RELEASE.md','README_STAGE9.md')}
    t.verify({p:h for p,h in json.loads(evidence[1].read_text()).items() if p not in mutable})
    rec['full_reproduction']={}
    for seed in ([] if fake else spec['seeds']):
        parent=previous.REAL_OUT/f'seed_{seed}'/'full_fresh'
        rc=json.loads((parent/'run_config.json').read_text())
        marker=d.validate_complete(parent,rc['fingerprint']);assert marker
        rec['full_reproduction'][str(seed)]=marker['attempt']
        evidence += [parent/'COMPLETE.json',parent/'run_config.json']+[Path(p) for p in marker['sha256']]
    rec['evidence_sha256'].update({str(p):d.sha(p) for p in evidence})
    rec.pop('freeze_hash');rec['freeze_hash']=v.digest(rec)
    return rec

def make_condition(core,cp,seed,mode,spec,device):
    assert mode in spec['modes']
    core.load_state_dict(cp['model_state'],strict=True)
    core.requires_grad_(True)
    if mode in ('multi_side','final_side'):
        model=ResidualSideNet(core,spec['side_width'],tuple(spec['taps']),
            final_features_only=mode=='final_side',seed=seed+spec['side_seed_offset'])
    else:
        model=core
        if mode in ('head','lp_full'):model.body.requires_grad_(False)
    model.to(device);model.zero_grad(set_to_none=True)
    # One canonical group, including inactive parameters. AdamW skips grad=None,
    # so inactive parameters have no moments, step counters or weight decay.
    opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    groups=copy.deepcopy(cp['optimizer_state']['param_groups'])
    assert len(groups)==1 and cp['updates']==spec['source_optimizer_age']
    assert groups[0]['weight_decay']==spec['weight_decay'] and tuple(groups[0]['betas'])==(.9,.999)
    groups[0]['params']=list(range(len(list(model.parameters()))))
    opt.load_state_dict(dict(state={},param_groups=groups));assert not opt.state
    return model,opt

def activate_epoch(model,opt,mode,epoch,spec):
    if mode=='lp_full' and epoch==spec['head_epochs']+1:
        before=optimizer_hash(opt)
        assert not any(p in opt.state for p in model.body.parameters())
        model.body.requires_grad_(True)
        assert optimizer_hash(opt)==before

def ages(model,mode,updates,spec):
    result={}
    for name,p in model.named_parameters():
        body=name.startswith('body.')
        if body and mode in ('head','multi_side','final_side'):continue
        age=max(0,updates-spec['head_epochs']*spec['expected_batches_per_epoch']) if body and mode=='lp_full' else updates
        if age:result[name]=age
    return result

def validate_optimizer(model,opt,mode,updates,spec):
    expected=ages(model,mode,updates,spec)
    names=dict(model.named_parameters())
    assert {name for name,p in names.items() if p in opt.state}==set(expected)
    for name,age in expected.items():
        state=opt.state[names[name]]
        assert int(state['step'])==age
        for key in ('exp_avg','exp_avg_sq'):
            assert state[key].shape==names[name].shape and bool(torch.isfinite(state[key]).all())
        assert bool((state['exp_avg_sq']>=0).all())

def gradient_record(model,step,mode):
    active=[p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in active)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    norm=lambda ps:float(torch.cat([p.grad.detach().cpu().double().flatten() for p in ps if p.grad is not None]).norm())
    row=dict(step=step,gradient_l2=norm(active),head_gradient_l2=norm(list(model.head.parameters())),
        backbone_gradient_present=any(p.grad is not None for p in model.body.parameters()))
    if mode in ('multi_side','final_side'):
        row['side_output_gradient_l2']=norm(list(model.output.parameters()))
        row['upstream_side_gradient_l2']=norm(list(model.projections.parameters())+list(model.blocks.parameters())+list(model.side_norm.parameters()))
        assert row['side_output_gradient_l2']>0
        assert row['upstream_side_gradient_l2']==0 if step==1 else row['upstream_side_gradient_l2']>0
    else:
        row.update(side_output_gradient_l2=0.,upstream_side_gradient_l2=0.)
    assert row['head_gradient_l2']>0
    return row

def run_one(out,seed,mode,position,cfg,freeze,data,device,shared_data_seconds=0.,launch_record=None):
    spec,fake=cfg['spec'],cfg['synthetic']
    parent=out/f'seed_{seed}'/mode;parent.mkdir(parents=True,exist_ok=True)
    rc=dict(root_fingerprint=cfg['fingerprint'],seed=seed,mode=mode,position=position)
    rc['fingerprint']=v.digest(rc);old.write_freeze(parent/'run_config.json',rc)
    if d.validate_complete(parent,rc['fingerprint']):print('SKIP completed',seed,mode,flush=True);return
    attempt=d.next_attempt(parent);v.sync(device);started=time.perf_counter()
    x,y,ev,ey,_=data;labels=ey.cpu().numpy()
    base=memory_start(device);tick=time.perf_counter()
    core,cp=source(seed,fake,freeze)
    model,opt=make_condition(core,cp,seed,mode,spec,device);del core
    initial=copy.deepcopy(model.state_dict());initial_opt=copy.deepcopy(opt.state_dict())
    initial_hash=old.state_hash(initial);initial_opt_hash=optimizer_hash(opt)
    source_body_hash=old.state_hash(model.body.state_dict());source_head_hash=old.state_hash(model.head.state_dict())
    initial_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    v.sync(device);setup_seconds=time.perf_counter()-tick;setup_memory=memory_end(device,base)
    curves,logits,steps,diagnostics,telemetry,transitions=[],{},[],[],[],[]
    validation_seconds=0.
    def evaluate(epoch):
        nonlocal validation_seconds
        v.sync(device);tick=time.perf_counter();base=memory_start(device)
        z=t.predictions(model,ev,spec['eval_batch_size']);loss,accuracy=score(z,labels);v.sync(device)
        seconds=time.perf_counter()-tick;validation_seconds+=seconds;logits[f'epoch_{epoch}']=z
        curves.append(dict(seed=seed,mode=mode,epoch=epoch,updates=epoch*spec['expected_batches_per_epoch'],
            loss=loss,accuracy=accuracy,validation_seconds=seconds,**memory_end(device,base)))
        return z
    initial_logits=evaluate(0)
    if not fake:
        with np.load(freeze['checkpoints'][str(seed)]['predictions_path'],allow_pickle=False) as prior:
            assert np.array_equal(labels,prior['labels'])
            assert np.max(np.abs(initial_logits-prior['full']))<=spec['source_logit_tolerance']
            assert np.array_equal(initial_logits.argmax(1),prior['full'].argmax(1)),'Starting predictions changed'
            loss,acc=score(prior['full'],labels)
            assert abs(curves[0]['loss']-loss)<=spec['source_loss_tolerance'] and curves[0]['accuracy']==acc
    base=memory_start(device);tick=time.perf_counter();model.train()
    xx=v.frozen6.normalize_cifar(x[:spec['batch_size']]);yy=y[:len(xx)]
    for i in range(spec['hardware_warmup_steps']):old.update(model,opt,xx,yy,seed+249001+i,spec['label_smoothing'])
    model.load_state_dict(initial,strict=True);opt.load_state_dict(initial_opt);model.zero_grad(set_to_none=True)
    del xx,yy,initial,initial_opt,cp
    v.sync(device);warmup_restore_seconds=time.perf_counter()-tick;warmup_memory=memory_end(device,base)
    assert old.state_hash(model.state_dict())==initial_hash and optimizer_hash(opt)==initial_opt_hash
    v.atomic_json(attempt/'restoration.json',dict(model_sha256=initial_hash,optimizer_sha256=initial_opt_hash,
        source_body_sha256=source_body_hash,source_head_sha256=source_head_hash,initial_trainable_parameters=initial_trainable,
        parameter_names=list(dict(model.named_parameters())),initial_state_entries=len(opt.state),
        source_optimizer_age=spec['source_optimizer_age'],initial_optimizer_age=0,source_restored_after_warmup=True,
        setup_seconds=setup_seconds,warmup_restore_seconds=warmup_restore_seconds,setup_memory=setup_memory,warmup_memory=warmup_memory))
    bg,ag,mg,pg=previous.generators(seed,spec)
    batches=math.ceil(len(y)/spec['batch_size']);assert batches==spec['expected_batches_per_epoch']
    total=spec['epochs']*batches;idx=0
    check_steps={1,2,total,spec['head_epochs']*batches,spec['head_epochs']*batches+1}
    for epoch in range(1,spec['epochs']+1):
        model.train();v.sync(device);tick=time.perf_counter()
        before=optimizer_hash(opt);body_before=old.state_hash(model.body.state_dict())
        activate_epoch(model,opt,mode,epoch,spec)
        validate_optimizer(model,opt,mode,idx,spec)
        if mode=='lp_full' and epoch==spec['head_epochs']+1:
            assert body_before==source_body_hash
            transitions.append(dict(epoch=epoch,updates=idx,optimizer_before=before,optimizer_after=optimizer_hash(opt),
                body_sha256=body_before,body_state_entries=sum(p in opt.state for p in model.body.parameters()),
                head_age=int(opt.state[model.head.weight]['step']),seconds=time.perf_counter()-tick))
        if not fake:telemetry.append(dict(epoch=epoch,sample=v.telemetry()))
        for indices in v.frozen6.epoch_batches(len(y),spec['batch_size'],bg):
            n=len(indices);indices_sha=tensor_hash(indices);augmentation_before=tensor_hash(ag.get_state())
            opt.zero_grad(set_to_none=True);v.sync(device);tick=time.perf_counter()
            ix=indices.to(device);xx=v.frozen6.augment_and_normalize(x[ix],ag)
            xx,targets,lam=v.mix_batch(xx,y[ix],mg,pg,spec['mixup_alpha'],10)
            v.sync(device);transform_seconds=time.perf_counter()-tick
            lr=v.learning_rate(idx,total,batches,spec['learning_rate'],spec)
            for group in opt.param_groups:group['lr']=lr
            dropout=seed+spec['dropout_seed_offset']+idx
            base=memory_start(device);tick=time.perf_counter()
            loss=old.update(model,opt,xx,targets,dropout,spec['label_smoothing']);v.sync(device)
            seconds=time.perf_counter()-tick;mem=memory_end(device,base);idx+=1
            if idx in check_steps:diagnostics.append(gradient_record(model,idx,mode))
            steps.append(dict(epoch=epoch,step=idx,batch_examples=n,indices_sha256=indices_sha,
                augmentation_before=augmentation_before,augmentation_after=tensor_hash(ag.get_state()),
                permutation_after=tensor_hash(pg.get_state()),mixup_state=v.digest(mg.bit_generator.state),
                mixup_lambda=lam,dropout_seed=dropout,lr=lr,loss=loss,head_age=idx,
                body_age=ages(model,mode,idx,spec).get('body.conv_proj.weight',0),
                gradient_checked=idx in check_steps,transform_seconds=transform_seconds,update_seconds=seconds,**mem))
            del xx,targets,ix
        validate_optimizer(model,opt,mode,idx,spec)
        if mode in ('head','multi_side','final_side') or (mode=='lp_full' and epoch<=spec['head_epochs']):
            assert old.state_hash(model.body.state_dict())==source_body_hash
        model.zero_grad(set_to_none=True);evaluate(epoch)
        v.write_csv(attempt/'steps.csv',steps);v.write_csv(attempt/'curves.csv',curves)
        print(f'SIDE seed={seed} {mode} epoch={epoch}/{spec["epochs"]} accuracy={100*curves[-1]["accuracy"]:.2f}%',flush=True)
    assert idx==total and all(bool(torch.isfinite(p).all()) for p in model.parameters())
    model.zero_grad(set_to_none=True);model_hash=old.state_hash(model.state_dict());opt_hash=optimizer_hash(opt)
    v.write_csv(attempt/'diagnostics.csv',diagnostics);v.atomic_json(attempt/'transitions.json',transitions)
    v.atomic_json(attempt/'telemetry.json',telemetry);np.savez(attempt/'predictions.npz',labels=labels,**logits)
    tick=time.perf_counter()
    old.checkpoint(attempt/'final_checkpoint.pt',dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),
        seed=seed,mode=mode,updates=idx,root_fingerprint=cfg['fingerprint'],run_fingerprint=rc['fingerprint'],
        batch_rng=bg.get_state(),augmentation_rng=ag.get_state(),permutation_rng=pg.get_state(),mixup_rng=mg.bit_generator.state))
    v.sync(device);checkpoint_seconds=time.perf_counter()-tick
    v.atomic_json(attempt/'adaptation_artifacts.json',{str(p):d.sha(p) for p in attempt.iterdir() if p.is_file()})
    v.sync(device);adaptation_seconds=time.perf_counter()-started
    gen=torch.Generator().manual_seed(seed+spec['inference_seed_offset'])
    ii=torch.randperm(len(y),generator=gen)[:spec['inference_trials']*spec['batch_size']]
    assert len(ii)==spec['inference_trials']*spec['batch_size']
    inference=[];v.sync(device);tick=time.perf_counter()
    for trial,indices in enumerate(ii.reshape(spec['inference_trials'],spec['batch_size'])):
        xx=v.frozen6.normalize_cifar(x[indices.to(device)]);yy=y[indices.to(device)]
        if trial==0:
            for j in range(spec['inference_warmups']):t.operation(model,xx,yy,'inference',seed+250001+j,device)
        row=t.operation(model,xx,yy,'inference',seed+251001+trial,device)
        inference.append(dict(trial=trial,indices_sha256=tensor_hash(indices),**row));del xx,yy
    v.sync(device);inference_seconds=time.perf_counter()-tick
    assert old.state_hash(model.state_dict())==model_hash and optimizer_hash(opt)==opt_hash
    v.write_csv(attempt/'inference.csv',inference)
    src=freeze['checkpoints'].get(str(seed),{})
    summary=dict(seed=seed,mode=mode,updates=idx,initial_accuracy=curves[0]['accuracy'],final_accuracy=curves[-1]['accuracy'],
        initial_loss=curves[0]['loss'],final_loss=curves[-1]['loss'],initial_model_sha256=initial_hash,
        initial_optimizer_sha256=initial_opt_hash,final_model_sha256=model_hash,final_optimizer_sha256=opt_hash,
        initial_trainable_parameters=initial_trainable,final_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),adaptation_seconds=adaptation_seconds,
        shared_data_seconds=shared_data_seconds,launch_record=str(launch_record) if launch_record else None,
        complete_adaptation_seconds=adaptation_seconds+shared_data_seconds,
        training_seconds=sum(row['transform_seconds']+row['update_seconds'] for row in steps),
        setup_seconds=setup_seconds,warmup_restore_seconds=warmup_restore_seconds,validation_seconds=validation_seconds,
        checkpoint_seconds=checkpoint_seconds,inference_benchmark_seconds=inference_seconds,observed_arm_seconds=time.perf_counter()-started,
        source_training_seconds=src.get('source_training_seconds',0),source_total_run_seconds=src.get('source_total_run_seconds',0),
        max_training_incremental_allocated=max(row['incremental_allocated'] for row in steps),
        max_training_peak_allocated=max(row['peak_allocated'] for row in steps),
        max_measured_arm_allocated=max([setup_memory['peak_allocated'],warmup_memory['peak_allocated']]+
            [row['peak_allocated'] for row in steps+curves+inference]))
    v.atomic_json(attempt/'summary.json',summary)
    v.atomic_json(parent/'COMPLETE.json',dict(fingerprint=rc['fingerprint'],attempt=str(attempt),
        sha256={str(p):d.sha(p) for p in attempt.iterdir() if p.is_file()}))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--smoke-tag',default='final_cpu');ap.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu')
    args=ap.parse_args();assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag)
    m.configure();spec=json.loads(SPEC.read_text());fake=args.smoke
    if fake:spec.update(seeds=[9971,9972],taps=[1,2],side_width=6,epochs=2,batch_size=3,eval_batch_size=4,
        expected_batches_per_epoch=6,source_optimizer_age=2,hardware_warmup_steps=1,inference_trials=3,inference_warmups=1)
    freeze=freeze_record(spec,fake)
    if args.freeze_only:
        assert not fake and not REAL_OUT.exists();old.write_freeze(FREEZE,freeze);print('FROZEN',freeze['freeze_hash']);return
    device=torch.device(args.smoke_device if fake else 'cuda');assert device.type=='cpu' or torch.cuda.is_available()
    out=ROOT/f'results/stage15A_sanity_side_v1_{args.smoke_tag}' if fake else REAL_OUT
    if not fake:assert FREEZE.exists() and json.loads(FREEZE.read_text())==freeze,'Frozen inputs/environment changed'
    out.mkdir(exist_ok=True)
    if fake:old.write_freeze(out/'SYNTHETIC_FREEZE.json',freeze)
    cfg=dict(spec=spec,synthetic=fake,freeze_hash=freeze['freeze_hash'],split=freeze['expected_split'],
        device=str(device),hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU')
    cfg['fingerprint']=v.digest(cfg);old.write_freeze(out/'configuration.json',cfg)
    from review_stage15A import audit
    if d.validate_complete(out,cfg['fingerprint']):
        print(json.dumps(audit(out,False),indent=2));print('SKIP completed; audit passed');return
    launch=d.next_attempt(out/'launches')
    v.atomic_json(launch/'preflight.json',{'synthetic':True} if fake else v.idle_preflight())
    tick=time.perf_counter();data=v.load_data('development',device,fake=fake);v.sync(device)
    assert data[-1]==cfg['split']
    v.atomic_json(launch/'data.json',dict(seconds=time.perf_counter()-tick,split=data[-1],training_examples=len(data[1]),validation_examples=len(data[3])))
    shared=json.loads((launch/'data.json').read_text())['seconds']/(len(spec['seeds'])*len(spec['modes']))
    for seed,order in zip(spec['seeds'],spec['orders']):
        for position,mode in enumerate(order):run_one(out,seed,mode,position,cfg,freeze,data,device,shared,launch/'data.json')
    del data;audit(out,True)
    paths=[out/name for name in ('configuration.json','audit.json','summary.csv','screen.csv','REPORT.md')]
    paths+=list(out.glob('seed_*/*/COMPLETE.json'))+list(out.glob('seed_*/*/run_config.json'))+list((out/'launches').rglob('*.json'))
    v.atomic_json(out/'COMPLETE.json',dict(fingerprint=cfg['fingerprint'],synthetic=fake,sha256={str(p):d.sha(p) for p in paths}))
    print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
