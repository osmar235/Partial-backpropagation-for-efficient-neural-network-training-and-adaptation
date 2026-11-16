"""Frozen coarse/detail representation screen; every arm uses full backprop."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, copy, csv, hashlib, json, math, platform, re, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import stage8_vit_v2 as v
import stage9_boundary_diagnostic as d
import stage10_models as m

ROOT=v.ROOT
SPEC_PATH=ROOT/'protocol/stage10/representation_spec_v1.json'
FREEZE_PATH=ROOT/'protocol/stage10/FROZEN_REPRESENTATION_v1.json'
REAL_OUT=ROOT/'results/stage10A_representation_v1'

def state_hash(state):
    h=hashlib.sha256()
    for name,tensor in state.items():h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def update(model,opt,x,targets,seed,smoothing=.1):
    torch.manual_seed(seed);opt.zero_grad(set_to_none=True)
    loss=F.cross_entropy(model(x),targets,label_smoothing=smoothing);value=loss.item()
    if not math.isfinite(value):raise FloatingPointError('Nonfinite representation-screen loss')
    loss.backward();opt.step();return value

def sources():
    names=['scripts/stage10_models.py','scripts/stage10_representation.py','scripts/review_stage10A.py',
        'scripts/test_stage10_representation.py','scripts/run_stage10A_representation.cmd',
        'protocol/stage10/representation_spec_v1.json','protocol/stage10/REPRESENTATION_SCREEN_v1.md']
    return {**d.source_hashes(),**{str((ROOT/n).resolve()):d.sha(ROOT/n) for n in names}}

def freeze_record(spec,fake):
    evidence=[ROOT/n for n in ['results/stage9E_review_v1/REPORT.md','results/stage9E_review_v1/independent_audit.json',
        'protocol/stage9/POST_PILOT_DECISION_v1.md','results/stage9E_quality_pilot_v1/configuration.json']]
    prior=json.loads(evidence[-1].read_text())
    record=dict(version=spec['version'],prepared_date='2026-10-21',synthetic=fake,spec=spec,
        source_sha256=sources(),evidence_sha256={str(p):d.sha(p) for p in evidence},
        expected_split={'kind':'artificial'} if fake else prior['split'],required_execution=m.execution_settings(),
        role='representation/architecture development screen; no local optimizer/evolution/new-method claim')
    record['freeze_hash']=v.digest(record);return record

def write_freeze(path,record):
    if path.exists():assert json.loads(path.read_text())==record,'Frozen source/spec changed; use a prospectively amended study, never overwrite'
    else:v.atomic_json(path,record)

def checkpoint(path,value):
    temp=path.with_suffix('.pt.tmp');torch.save(value,temp);temp.replace(path)

def run_one(out,seed,variant,position,config,data,spec,device,fake):
    parent=out/f'seed_{seed}'/variant;parent.mkdir(parents=True,exist_ok=True)
    rc=dict(root_fingerprint=config['fingerprint'],seed=seed,variant=variant,order_position=position)
    fp=v.digest(rc);rc['fingerprint']=fp;path=parent/'run_config.json'
    if path.exists():assert json.loads(path.read_text())==rc
    else:v.atomic_json(path,rc)
    if d.validate_complete(parent,fp):print('SKIP complete',seed,variant,flush=True);return
    attempt=d.next_attempt(parent);process_start=time.perf_counter()
    x,y,ev,ey,_=data;v.seed_all(seed);model=m.make_model(variant,seed,fake).to(device)
    initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
    components={name:state_hash(module.state_dict()) for name,module in model.named_children()}
    opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    ag=torch.Generator().manual_seed(seed+191001);mg=np.random.default_rng(seed+192001);pg=torch.Generator().manual_seed(seed+193001)
    xx=v.frozen6.augment_and_normalize(x[:spec['batch_size']],ag)
    xx,targets,_=v.mix_batch(xx,y[:spec['batch_size']],mg,pg,spec['mixup_alpha'],10)
    v.sync(device);t=time.perf_counter()
    for i in range(spec['hardware_warmup_steps']):update(model,opt,xx,targets,seed+194001+i,spec['label_smoothing'])
    v.sync(device);hardware_seconds=time.perf_counter()-t
    model.load_state_dict(initial);model.zero_grad(set_to_none=True)
    opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    bg=torch.Generator().manual_seed(seed+91001);ag=torch.Generator().manual_seed(seed+92001)
    mg=np.random.default_rng(seed+98001);pg=torch.Generator().manual_seed(seed+99001)
    batch=spec['batch_size'];batches=math.ceil(len(y)/batch);assert batches==spec['expected_batches_per_epoch']
    step_fields=['epoch','step','batch_examples','loss','lr','mixup_lambda','dropout_seed']
    epoch_fields=['epoch','updates','training_loss','validation_loss','validation_accuracy','epoch_training_seconds',
        'cumulative_training_seconds','validation_seconds','cumulative_validation_seconds','cumulative_observed_seconds']
    sf=(attempt/'steps.csv').open('w',encoding='utf8',newline='');ef=(attempt/'epochs.csv').open('w',encoding='utf8',newline='')
    sw=csv.DictWriter(sf,fieldnames=step_fields);sw.writeheader();ew=csv.DictWriter(ef,fieldnames=epoch_fields);ew.writeheader()
    epochs=[];telemetry=[];idx=0;train_total=val_total=0.
    v.sync(device);started=time.perf_counter()
    try:
        for epoch in range(1,spec['epochs']+1):
            if not fake and (epoch==1 or epoch%spec['telemetry_every_epochs']==0):
                telemetry.append(dict(epoch=epoch,sample=json.dumps(v.telemetry())));v.write_csv(attempt/'telemetry.csv',telemetry)
            rows=[];loss_sum=seen=0;v.sync(device);t=time.perf_counter()
            for indices in v.frozen6.epoch_batches(len(y),batch,bg):
                indices=indices.to(device);n=len(indices)
                xx=v.frozen6.augment_and_normalize(x[indices],ag)
                xx,targets,lam=v.mix_batch(xx,y[indices],mg,pg,spec['mixup_alpha'],10)
                lr=v.learning_rate(idx,spec['schedule_horizon_epochs']*batches,batches,spec['learning_rate'],spec)
                for group in opt.param_groups:group['lr']=lr
                dropout_seed=seed+spec['dropout_seed_offset']+idx
                loss=update(model,opt,xx,targets,dropout_seed,spec['label_smoothing']);idx+=1
                loss_sum+=loss*n;seen+=n
                rows.append(dict(epoch=epoch,step=idx,batch_examples=n,loss=loss,lr=lr,mixup_lambda=lam,dropout_seed=dropout_seed))
            v.sync(device);seconds=time.perf_counter()-t;train_total+=seconds
            for param in model.parameters():assert int(opt.state[param]['step'])==idx
            sw.writerows(rows);sf.flush();v.sync(device);t=time.perf_counter()
            ev_loss,accuracy=v.evaluate(model,ev,ey,spec['eval_batch_size']);v.sync(device)
            ev_seconds=time.perf_counter()-t;val_total+=ev_seconds;observed=time.perf_counter()-started
            row=dict(epoch=epoch,updates=idx,training_loss=loss_sum/seen,validation_loss=ev_loss,validation_accuracy=accuracy,
                epoch_training_seconds=seconds,cumulative_training_seconds=train_total,validation_seconds=ev_seconds,
                cumulative_validation_seconds=val_total,cumulative_observed_seconds=observed)
            ew.writerow(row);ef.flush();epochs.append(row)
            print(f'REPRESENTATION seed={seed} {variant} epoch={epoch}/{spec["epochs"]} val={100*accuracy:.2f}% observed_s={observed:.1f}',flush=True)
    finally:sf.close();ef.close()
    assert idx==spec['expected_updates_per_run'];v.sync(device);t=time.perf_counter()
    checkpoint(attempt/'final_checkpoint.pt',dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),seed=seed,variant=variant,
        updates=idx,root_fingerprint=config['fingerprint'],run_fingerprint=fp,batch_rng=bg.get_state(),augmentation_rng=ag.get_state(),
        permutation_rng=pg.get_state(),mixup_rng=mg.bit_generator.state,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
    cp_seconds=time.perf_counter()-t
    summary=dict(status='complete',seed=seed,variant=variant,order_position=position,root_fingerprint=config['fingerprint'],fingerprint=fp,
        epochs=spec['epochs'],updates=idx,parameter_count=sum(p.numel() for p in model.parameters()),initial_state_sha256=state_hash(initial),
        initial_component_sha256=components,final_validation_accuracy=epochs[-1]['validation_accuracy'],final_validation_loss=epochs[-1]['validation_loss'],
        training_seconds=train_total,validation_seconds=val_total,observed_seconds=epochs[-1]['cumulative_observed_seconds'],
        hardware_warmup_seconds=hardware_seconds,checkpoint_seconds=cp_seconds,experiment_elapsed_seconds=time.perf_counter()-process_start)
    v.atomic_json(attempt/'summary.json',summary)
    v.atomic_json(parent/'COMPLETE.json',dict(status='complete',fingerprint=fp,attempt=str(attempt),sha256={str(p):d.sha(p) for p in attempt.iterdir() if p.is_file()}))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-tag',default='final_cpu')
    parser.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag);m.configure();fake=args.smoke;spec=json.loads(SPEC_PATH.read_text())
    if fake:spec.update(development_seeds=[9951,9952],epochs=3,schedule_horizon_epochs=10,batch_size=3,eval_batch_size=4,
        hardware_warmup_steps=2,primary_target=0.,secondary_target=1.,sustained_epochs=2,expected_batches_per_epoch=6,expected_updates_per_run=18)
    record=freeze_record(spec,fake)
    if args.freeze_only:
        assert not fake and not REAL_OUT.exists();write_freeze(FREEZE_PATH,record);print('FROZEN:',FREEZE_PATH,record['freeze_hash']);return
    out=ROOT/'results'/('stage10_sanity_representation_v1_'+args.smoke_tag) if fake else REAL_OUT
    if fake:out.mkdir(exist_ok=True);write_freeze(out/'SYNTHETIC_FREEZE.json',record)
    else:assert FREEZE_PATH.exists() and json.loads(FREEZE_PATH.read_text())==record,'Prospective freeze missing or changed'
    device=torch.device(args.smoke_device if fake else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    config=dict(spec=spec,synthetic=fake,freeze_hash=record['freeze_hash'],source_sha256=record['source_sha256'],split=record['expected_split'],
        execution=m.execution_settings(),device=str(device),gpu='cpu' if device.type=='cpu' else torch.cuda.get_device_name(device),
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,torchvision=__import__('torchvision').__version__,cuda=torch.version.cuda)
    fp=v.digest(config);config['fingerprint']=fp;out.mkdir(exist_ok=True);path=out/'configuration.json'
    if path.exists():assert json.loads(path.read_text())==config,'Code/settings/environment changed; do not overwrite study'
    else:v.atomic_json(path,config)
    from review_stage10A import audit
    if (out/'COMPLETE.json').exists():d.validate_complete(out,fp);audit(out,write=False);print('SKIP completed, re-audited representation screen');return
    launch=d.next_attempt(out/'launches');v.atomic_json(launch/'preflight.json',v.idle_preflight() if device.type=='cuda' else [])
    t=time.perf_counter();data=v.load_data('development',device,fake=fake);assert data[-1]==config['split']
    v.atomic_json(launch/'data_loading.json',dict(seconds=time.perf_counter()-t))
    for seed,order in zip(spec['development_seeds'],spec['orders']):
        for position,variant in enumerate(order):run_one(out,seed,variant,position,config,data,spec,device,fake)
    audit(out,write=True)
    paths=[out/n for n in ['configuration.json','audit.json','all_runs.csv','targets.csv','screen.csv','REPORT.md']]
    paths+=list(out.glob('seed_*/*/COMPLETE.json'))+list(out.glob('seed_*/*/run_config.json'))+list((out/'launches').rglob('*.json'))
    v.atomic_json(out/'COMPLETE.json',dict(status='complete',synthetic=fake,fingerprint=fp,runs=10,sha256={str(p):d.sha(p) for p in paths}))
    print('REPRESENTATION SCREEN COMPLETE:',out)

if __name__=='__main__':main()
