"""Matched real-data workflow timing, unchanged QKV unbind versus native."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import gc
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import torch
import stage15_side_adaptation as r
import qkv_layout as q
from qkv_unbind import install

ROOT=r.ROOT;HERE=ROOT/'protocol/qkv_training'
SPEC=HERE/'training_spec_v1.json';FREEZE=HERE/'FROZEN_TRAINING_v1.json';OUT=ROOT/'results/qkv_training_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sources():
    names=['scripts/qkv_training.py','scripts/review_qkv_training.py','scripts/test_qkv_training.py',
           'scripts/run_qkv_training.cmd','protocol/qkv_training/training_spec_v1.json',
           'protocol/qkv_training/TRAINING_PROTOCOL_v1.md','scripts/review_qkv_confirmation.py']
    return {**r.sources(),**q.sources(),**{str(ROOT/p):r.d.sha(ROOT/p) for p in names}}

def configuration(smoke,device):
    spec=read(SPEC);base=read(r.FREEZE)
    if smoke:spec.update(sessions=2,seeds=[9971,9972],epochs=2,batch_size=3,eval_batch_size=4,
        expected_batches_per_epoch=6,source_optimizer_age=2,hardware_warmup_steps=1)
    evidence={};checkpoints={} if smoke else base['checkpoints'];references={}
    if not smoke:
        evidence[str(r.FREEZE)]=r.d.sha(r.FREEZE)
        for seed in spec['seeds']:
            src=checkpoints[str(seed)];assert r.d.sha(src['path'])==src['sha256']
            evidence[src['path']]=src['sha256'];evidence[src['predictions_path']]=r.d.sha(src['predictions_path'])
            parent=r.REAL_OUT/f'seed_{seed}'/'full';rc=read(parent/'run_config.json')
            marker=r.d.validate_complete(parent,rc['fingerprint']);assert marker
            references[str(seed)]=marker['attempt']
            evidence[str(parent/'COMPLETE.json')]=r.d.sha(parent/'COMPLETE.json')
            evidence.update(marker['sha256'])
        for name in ['batches.meta']+[f'data_batch_{i}' for i in range(1,6)]:
            p=ROOT/'data/cifar-10-batches-py'/name;evidence[str(p)]=r.d.sha(p)
    cfg=dict(spec=spec,smoke=smoke,device=str(device),hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        sources=sources(),evidence=evidence,checkpoints=checkpoints,references=references,
        split={'kind':'artificial'} if smoke else base['expected_split'],execution=r.m.execution_settings(),
        python=platform.python_version(),torch=str(torch.__version__),torchvision=str(r.t.torchvision.__version__),numpy=np.__version__)
    cfg['fingerprint']=r.v.digest(cfg);return cfg

def schedule(spec,session):
    seeds=spec['seeds'] if session%2==0 else list(reversed(spec['seeds']))
    result=[]
    for seed in seeds:
        index=spec['seeds'].index(seed)
        arms=spec['arms'] if (session//2+index)%2==0 else list(reversed(spec['arms']))
        result.extend((seed,arm) for arm in arms)
    return result

def load_source(seed,cfg):
    if cfg['smoke']:return r.source(seed,True,cfg)
    src=cfg['checkpoints'][str(seed)]
    cp=torch.load(src['path'],map_location='cpu',weights_only=False)
    assert cp['seed']==seed and cp['variant']=='full'
    for k in ('root_fingerprint','run_fingerprint','updates'):assert cp[k]==src[k]
    core=r.m.make_model('full_rgb32',seed,False);core.load_state_dict(cp['model_state'],strict=True)
    return core,cp

def make(seed,arm,cfg,device):
    core,cp=load_source(seed,cfg)
    # Reuse the archived full-arm optimizer setup, with its explicit mode contract.
    model,opt=r.make_condition(core,cp,seed,'full',dict(cfg['spec'],modes=['full']),device)
    if arm=='qkv_unbind':install(model)
    else:assert arm=='native'
    return model,opt,cp

def stream_states(bg,ag,mg,pg):
    return dict(batch_rng=bg.get_state(),augmentation_rng=ag.get_state(),
                permutation_rng=pg.get_state(),mixup_rng=copy.deepcopy(mg.bit_generator.state))

def seal(parent,fingerprint,paths):
    r.v.atomic_json(parent/'COMPLETE.json',dict(fingerprint=fingerprint,sha256={str(p):r.d.sha(p) for p in paths}))

def run_one(out,session,seed,arm,position,cfg,device):
    spec=cfg['spec'];parent=out/f'session_{session:02d}'/f'seed_{seed}'/arm
    parent.mkdir(parents=True,exist_ok=True)
    rc=dict(root_fingerprint=cfg['fingerprint'],session=session,seed=seed,arm=arm,position=position)
    rc['fingerprint']=r.v.digest(rc);r.old.write_freeze(parent/'run_config.json',rc)
    if r.d.validate_complete(parent,rc['fingerprint']):print('SKIP',session,seed,arm,flush=True);return
    attempt=r.d.next_attempt(parent)
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    r.v.sync(device);baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else 0
    started=time.perf_counter();tick=started
    data=r.v.load_data('development',device,fake=cfg['smoke'])
    x,y,ev,ey,split=data;assert split==cfg['split'];labels=ey.cpu().numpy()
    r.v.sync(device);data_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    model,opt,cp=make(seed,arm,cfg,device)
    initial=copy.deepcopy(cp['model_state']);initial_opt=copy.deepcopy(opt.state_dict())
    r.v.sync(device);setup_seconds=time.perf_counter()-tick
    curves=[];predictions={};steps=[];orders=[];training_seconds=0.;validation_seconds=0.
    def evaluate(epoch):
        nonlocal validation_seconds
        r.v.sync(device);tick=time.perf_counter()
        logits=r.t.predictions(model,ev,spec['eval_batch_size']);loss,accuracy=r.score(logits,labels)
        r.v.sync(device);seconds=time.perf_counter()-tick;validation_seconds+=seconds
        predictions[f'epoch_{epoch}']=logits
        curves.append(dict(epoch=epoch,loss=loss,accuracy=accuracy,seconds=seconds))
    with q.sdpa_kernel(q.SDPBackend.MATH):
        q.policy();evaluate(0);model.train();r.v.sync(device);tick=time.perf_counter()
        xx=r.v.frozen6.normalize_cifar(x[:spec['batch_size']]);yy=y[:len(xx)]
        for i in range(spec['hardware_warmup_steps']):r.old.update(model,opt,xx,yy,seed+249001+i,spec['label_smoothing'])
        model.load_state_dict(initial,strict=True);opt.load_state_dict(initial_opt);model.zero_grad(set_to_none=True)
        del xx,yy
        r.v.sync(device);warmup_seconds=time.perf_counter()-tick
        # Small restoration hashes are included in the workflow clock and enforce reset.
        initial_hash=r.old.state_hash(model.state_dict());initial_optimizer_hash=r.optimizer_hash(opt)
        assert initial_hash==r.old.state_hash(initial) and not opt.state
        bg,ag,mg,pg=r.previous.generators(seed,spec)
        batches=spec['expected_batches_per_epoch'];assert (len(y)+spec['batch_size']-1)//spec['batch_size']==batches
        total=batches*spec['epochs'];index=0
        for epoch in range(1,spec['epochs']+1):
            model.train();r.v.sync(device);tick=time.perf_counter()
            permutation=torch.randperm(len(y),generator=bg);orders.append(permutation)
            for start in range(0,len(y),spec['batch_size']):
                ix=permutation[start:start+spec['batch_size']].to(device)
                xx=r.v.frozen6.augment_and_normalize(x[ix],ag)
                xx,targets,lam=r.v.mix_batch(xx,y[ix],mg,pg,spec['mixup_alpha'],10)
                lr=r.v.learning_rate(index,total,batches,spec['learning_rate'],spec)
                for group in opt.param_groups:group['lr']=lr
                dropout=seed+spec['dropout_seed_offset']+index
                loss=r.old.update(model,opt,xx,targets,dropout,spec['label_smoothing']);index+=1
                steps.append(dict(epoch=epoch,step=index,batch_examples=len(ix),loss=loss,lr=lr,mixup_lambda=lam,dropout_seed=dropout))
            del xx,targets,ix
            r.v.sync(device);training_seconds+=time.perf_counter()-tick
            evaluate(epoch)
            print(f'TRAIN session={session+1} seed={seed} {arm} epoch={epoch}/{spec["epochs"]} accuracy={100*curves[-1]["accuracy"]:.2f}%',flush=True)
    r.v.sync(device);tick=time.perf_counter()
    # Capture states before any post-run audit/reconstruction can touch RNG.
    rng=q.rng(device);streams=stream_states(bg,ag,mg,pg)
    final=dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),seed=seed,arm=arm,
        updates=index,root_fingerprint=cfg['fingerprint'],run_fingerprint=rc['fingerprint'],**streams)
    r.old.checkpoint(attempt/'final_checkpoint.pt',final)
    np.savez(attempt/'predictions.npz',labels=labels,**predictions)
    r.v.write_csv(attempt/'steps.csv',steps);r.v.write_csv(attempt/'curves.csv',curves)
    r.old.checkpoint(attempt/'streams.pt',dict(epoch_orders=orders,**streams,final_global_rng=rng))
    r.v.sync(device);artifacts_seconds=time.perf_counter()-tick
    complete_seconds=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
    reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0
    # Everything below is an excluded, untimed scientific integrity check.
    assert index==total
    r.validate_optimizer(model,opt,'full',index,spec)
    assert all(p.requires_grad and p.dtype==torch.float32 and p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    summary=dict(session=session,seed=seed,arm=arm,position=position,updates=index,
        complete_workflow_seconds=complete_seconds,data_seconds=data_seconds,setup_seconds=setup_seconds,
        warmup_restore_seconds=warmup_seconds,training_seconds=training_seconds,validation_seconds=validation_seconds,
        artifacts_seconds=artifacts_seconds,residual_seconds=complete_seconds-sum((data_seconds,setup_seconds,warmup_seconds,training_seconds,validation_seconds,artifacts_seconds)),
        initial_model_sha256=initial_hash,initial_optimizer_sha256=initial_optimizer_hash,
        final_model_sha256=r.old.state_hash(model.state_dict()),final_optimizer_sha256=r.optimizer_hash(opt),
        final_global_rng=rng,initial_accuracy=curves[0]['accuracy'],final_accuracy=curves[-1]['accuracy'],
        initial_loss=curves[0]['loss'],final_loss=curves[-1]['loss'],all_full_gradients_finite=True,
        trainable_parameters=sum(p.numel() for p in model.parameters()),baseline_allocated=baseline,
        peak_allocated=peak,incremental_allocated=peak-baseline,peak_reserved=reserved,
        source_training_seconds=cfg['checkpoints'].get(str(seed),{}).get('source_training_seconds',0))
    r.v.atomic_json(attempt/'summary.json',summary)
    r.v.atomic_json(parent/'COMPLETE.json',dict(fingerprint=rc['fingerprint'],attempt=str(attempt),
        sha256={str(p):r.d.sha(p) for p in attempt.iterdir() if p.is_file()}))

def session_run(out,session,cfg,device):
    parent=out/f'session_{session:02d}';parent.mkdir(exist_ok=True)
    if r.d.validate_complete(parent,cfg['fingerprint']):print('SKIP session',session,flush=True);return
    launch=r.d.next_attempt(parent/'launches')
    r.v.atomic_json(launch/'process.json',dict(pid=os.getpid(),uuid=str(uuid.uuid4()),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    r.v.atomic_json(launch/'preflight.json',{'smoke':True} if cfg['smoke'] else r.v.idle_preflight())
    for pos,(seed,arm) in enumerate(schedule(cfg['spec'],session)):
        run_one(out,session,seed,arm,pos,cfg,device)
        r.v.atomic_json(launch/f'trajectory_{pos:02d}.json',dict(seed=seed,arm=arm,telemetry={'smoke':True} if cfg['smoke'] else r.v.telemetry()))
    paths=list(parent.glob('seed_*/*/COMPLETE.json'))+list(parent.glob('seed_*/*/run_config.json'))+list((parent/'launches').rglob('*.json'))
    seal(parent,cfg['fingerprint'],paths)

def main():
    ap=argparse.ArgumentParser();actions=ap.add_mutually_exclusive_group(required=True)
    actions.add_argument('--freeze-only',action='store_true');actions.add_argument('--run-all',action='store_true');actions.add_argument('--session',type=int)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--tag',default='v1')
    args=ap.parse_args();assert args.tag.replace('_','').isalnum()
    r.m.configure();device=torch.device(args.device);assert args.smoke or device.type=='cuda'
    assert device.type=='cpu' or torch.cuda.is_available()
    cfg=configuration(args.smoke,device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();r.old.write_freeze(FREEZE,cfg);print('FROZEN',cfg['fingerprint']);return
    if not args.smoke:assert cfg==read(FREEZE),'Frozen source/data/checkpoint/environment changed'
    out=ROOT/f'results/qkv_training_smoke_{args.tag}_{device.type}' if args.smoke else OUT
    out.mkdir(exist_ok=True);r.old.write_freeze(out/'configuration.json',cfg)
    from review_qkv_training import audit
    if r.d.validate_complete(out,cfg['fingerprint']):print(json.dumps(audit(out),indent=2));return
    if args.session is not None:
        assert 0<=args.session<cfg['spec']['sessions'];session_run(out,args.session,cfg,device);return
    for session in range(cfg['spec']['sessions']):
        argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--session',str(session),'--device',args.device,'--tag',args.tag]
        if args.smoke:argv.append('--smoke')
        subprocess.run(argv,cwd=ROOT,check=True)
        if not args.smoke and session+1<cfg['spec']['sessions']:time.sleep(5)
    result=audit(out,write=True)
    paths=[out/p for p in ('configuration.json','audit.json','pairings.json','REPORT.md')]+list(out.glob('session_*/COMPLETE.json'))
    seal(out,cfg['fingerprint'],paths)
    print(json.dumps(result,indent=2));print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
