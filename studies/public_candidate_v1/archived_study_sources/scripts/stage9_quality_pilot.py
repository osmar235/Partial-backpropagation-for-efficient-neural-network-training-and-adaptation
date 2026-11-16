"""Prospectively frozen, resource-matched five-arm development quality pilot."""
from __future__ import annotations
import argparse, collections, copy, csv, hashlib, json, math, platform, random, re, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import stage8_vit_v2 as v
import stage9_boundary_diagnostic as d
from stage9_split_backward_proof import split_forward
from stage9_deployment_cost import predecided_forward

ROOT=v.ROOT
SPEC_PATH=ROOT/'protocol/stage9/quality_pilot_spec_v1.json'
FREEZE_PATH=ROOT/'protocol/stage9/FROZEN_QUALITY_PILOT_v1.json'
REAL_OUT=ROOT/'results/stage9E_quality_pilot_v1'

def ways(n,k,previous):
    if n<0 or k<0:return 0
    if k==0:return 1
    top=n-k+1-int(previous)
    return math.comb(top,k) if top>=k else 0

def spaced(n,k):return {int((i+.5)*n/k) for i in range(k)} if k else set()

class Budget:
    def __init__(self,spec):self.spec=spec;self.previous=False;self.n=self.k=self.position=0
    def begin_epoch(self,regular):self.left=regular;self.n=0
    def choose(self,strategy,request=False,draw=0.):
        if not self.n:
            self.n=min(self.left,self.spec['quota_window']);assert self.n>0
            self.k=round(self.spec['omission_fraction']*self.n);self.position=0
            self.plan=spaced(self.n,self.k)
        n,k=self.n,self.k
        skip_ways=ways(n-1,k-1,True) if not self.previous else 0
        full_ways=ways(n-1,k,False)
        assert skip_ways+full_ways>0
        requested=(self.position in self.plan) if strategy=='periodic' else (
            draw<skip_ways/(skip_ways+full_ways) if strategy=='random' else bool(request))
        if not skip_ways:
            partial=False;reason='refresh' if self.previous else 'quota_exhausted'
        elif not full_ways:partial=True;reason='quota_deadline'
        else:partial=requested;reason=strategy
        assert not (partial and self.previous)
        if strategy=='periodic':assert partial==(self.position in self.plan)
        self.n-=1;self.k-=int(partial);self.left-=1;self.position+=1;self.previous=partial
        assert ways(self.n,self.k,self.previous)>0
        return partial,reason,int(requested),n,k
    def mandatory_full(self):self.previous=False
    def end_epoch(self):assert self.left==self.n==self.k==0

class Controller:
    def __init__(self,variant,seed,spec):
        self.variant=variant;self.spec=spec;self.history=collections.deque(maxlen=spec['history_size'])
        self.budget=Budget(spec);self.rng=random.Random(seed+171001)
    def decide(self,signal,active,regular):
        threshold='';before=len(self.history)
        if not active or not regular or self.variant=='full_adamw':
            self.budget.mandatory_full()
            return False,dict(reason='full' if self.variant=='full_adamw' else ('warmup' if not active else 'short_batch'),
                requested=0,threshold=threshold,history_count=before,remaining_n=-1,remaining_k=-1,random_draw='')
        if self.variant in ['mixup_only','boundary_rms']:
            assert before==self.spec['history_size'] and signal is not None
            threshold=float(np.quantile(list(self.history),self.spec['calibration_quantile']))
            request=signal<=threshold
        else:request=False
        draw=self.rng.random() if self.variant=='random' else 0.
        partial,reason,requested,n,k=self.budget.choose(self.variant,request,draw)
        return partial,dict(reason=reason,requested=requested,threshold=threshold,history_count=before,
            remaining_n=n,remaining_k=k,random_draw=draw if self.variant=='random' else '')
    def observe(self,signal):
        if signal is not None:self.history.append(float(signal))

def kernel(model,opt,x,targets,cut,decide,boundary=False):
    opt.zero_grad(set_to_none=True)
    if boundary:
        logits,h,bridge=split_forward(model,x,cut)
        loss=F.cross_entropy(logits,targets,label_smoothing=.1);lv=loss.item();loss.backward()
        signal=float(bridge.grad.detach().square().mean().sqrt().item())
        partial,metadata=decide(signal)
        if not partial:h.backward(bridge.grad)
    else:
        signal=None;partial,metadata=decide(None)
        logits=predecided_forward(model,x,cut) if partial else model(x)
        loss=F.cross_entropy(logits,targets,label_smoothing=.1);lv=loss.item();loss.backward()
    if not math.isfinite(lv):raise FloatingPointError('Nonfinite pilot loss')
    if signal is not None and not math.isfinite(signal):raise FloatingPointError('Nonfinite boundary signal')
    opt.step()
    return lv,partial,signal,metadata

def source_hashes():
    files=[Path(__file__),SPEC_PATH,ROOT/'protocol/stage9/QUALITY_PILOT_v1.md',
        ROOT/'scripts/stage9_deployment_cost.py',ROOT/'scripts/review_stage9E.py',
        ROOT/'scripts/test_stage9_quality_pilot.py',ROOT/'scripts/run_stage9E_quality_pilot.cmd']
    return {**d.source_hashes(),**{str(p.resolve()):d.sha(p) for p in files}}

def freeze_record(spec,fake):
    evidence=[ROOT/f'results/{name}/{file}' for name in ['stage9C_review_v1','stage9D_review_v1'] for file in ['REPORT.md','audit.json']]
    prior=json.loads((ROOT/'results/stage9C_boundary_diagnostic_v1/configuration.json').read_text())
    record=dict(version='stage9E-quality-pilot-v1',prepared_date='2026-10-21',synthetic=fake,spec=spec,
        source_sha256=source_hashes(),evidence_sha256={str(p):d.sha(p) for p in evidence},
        expected_split={'kind':'artificial'} if fake else prior['split'],
        role='development only; reused validation; no confirmatory or publication inference')
    record['freeze_hash']=v.digest(record);return record

def write_freeze(path,record):
    if path.exists():assert json.loads(path.read_text())==record,'Frozen study differs; do not overwrite or repair it retrospectively'
    else:v.atomic_json(path,record)

def checkpoint(path,value):
    temp=path.with_suffix('.pt.tmp');torch.save(value,temp);temp.replace(path)

def initial_hash(state):
    h=hashlib.sha256()
    for name,t in state.items():h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def run_one(out,seed,variant,order,root_config,data,spec,device,fake):
    parent=out/f'seed_{seed}'/variant;parent.mkdir(parents=True,exist_ok=True)
    config=dict(root_fingerprint=root_config['fingerprint'],seed=seed,variant=variant,order_position=order)
    fp=v.digest(config);config['fingerprint']=fp
    cfg=parent/'run_config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:v.atomic_json(cfg,config)
    complete=d.validate_complete(parent,fp)
    if complete:print('SKIP completed',seed,variant,flush=True);return
    attempt=d.next_attempt(parent);process_start=time.perf_counter()
    x,y,ev,ey,_=data;v.seed_all(seed)
    model=v.make_model(tiny=fake).to(device);prefix,suffix,prefix_names=v.partition(model,spec['cut'])
    initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()};ih=initial_hash(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    warm_ag=torch.Generator().manual_seed(seed+172001);warm_mg=np.random.default_rng(seed+173001)
    warm_pg=torch.Generator().manual_seed(seed+174001)
    xx=v.frozen6.augment_and_normalize(x[:spec['batch_size']],warm_ag)
    xx,targets,_=v.mix_batch(xx,y[:spec['batch_size']],warm_mg,warm_pg,spec['mixup_alpha'],10)
    v.sync(device);warm_start=time.perf_counter()
    for i in range(spec['hardware_warmup_steps']):
        mode=i%4;kernel(model,opt,xx,targets,spec['cut'],lambda signal:(mode in [1,3],{}),boundary=mode in [2,3])
    v.sync(device);warm_seconds=time.perf_counter()-warm_start
    model.load_state_dict(initial);model.zero_grad(set_to_none=True)
    opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    v.seed_all(seed+95001)
    bg=torch.Generator().manual_seed(seed+91001);ag=torch.Generator().manual_seed(seed+92001)
    mg=np.random.default_rng(seed+98001);pg=torch.Generator().manual_seed(seed+99001)
    controller=Controller(variant,seed,spec)
    batch=spec['batch_size'];batches=math.ceil(len(y)/batch);regular=len(y)//batch
    assert batches==spec['expected_batches_per_epoch'] and regular==spec['expected_regular_batches_per_epoch']
    assert regular>=spec['history_size']
    steps_fields=['epoch','step','batch_examples','partial','loss','lr','mixup_lambda','signal','threshold',
        'history_count','reason','requested','remaining_n','remaining_k','random_draw']
    epoch_fields=['epoch','updates','training_loss','validation_loss','validation_accuracy','partial_updates',
        'omitted_prefix_examples','epoch_training_seconds','cumulative_training_seconds','validation_seconds',
        'cumulative_validation_seconds','cumulative_observed_seconds']
    step_file=(attempt/'steps.csv').open('w',newline='',encoding='utf8')
    epoch_file=(attempt/'epochs.csv').open('w',newline='',encoding='utf8')
    sw=csv.DictWriter(step_file,fieldnames=steps_fields);sw.writeheader()
    ew=csv.DictWriter(epoch_file,fieldnames=epoch_fields);ew.writeheader()
    telemetry=[];epochs=[];idx=partial_total=omitted_total=0;train_total=validation_total=0.
    v.sync(device);training_started=time.perf_counter()
    try:
        for epoch in range(1,spec['epochs']+1):
            if not fake and (epoch==1 or epoch%spec['telemetry_every_epochs']==0):
                telemetry.append(dict(epoch=epoch,sample=json.dumps(v.telemetry())))
                v.write_csv(attempt/'telemetry.csv',telemetry)
            active=epoch>spec['full_warmup_epochs']
            if active and variant!='full_adamw':controller.budget.begin_epoch(regular)
            step_rows=[];loss_sum=seen=ep_partial=ep_omitted=0
            v.sync(device);started=time.perf_counter()
            for indices in v.frozen6.epoch_batches(len(y),batch,bg):
                indices=indices.to(device);n=len(indices);is_regular=n==batch
                xx=v.frozen6.augment_and_normalize(x[indices],ag)
                xx,targets,lam=v.mix_batch(xx,y[indices],mg,pg,spec['mixup_alpha'],10)
                lr=v.learning_rate(idx,spec['schedule_horizon_epochs']*batches,batches,spec['learning_rate'],v.SPEC)
                for group in opt.param_groups:group['lr']=lr
                collect=is_regular and epoch>=spec['full_warmup_epochs']
                cheap_signal=-min(lam,1-lam) if collect and variant=='mixup_only' else None
                boundary=collect and variant=='boundary_rms'
                def decide(signal):
                    return controller.decide(signal if boundary else cheap_signal,active,is_regular)
                lv,partial,signal,metadata=kernel(model,opt,xx,targets,spec['cut'],decide,boundary=boundary)
                actual_signal=signal if boundary else cheap_signal
                controller.observe(actual_signal)
                idx+=1;ep_partial+=int(partial);ep_omitted+=n*int(partial)
                loss_sum+=lv*n;seen+=n
                step_rows.append(dict(epoch=epoch,step=idx,batch_examples=n,partial=int(partial),loss=lv,lr=lr,
                    mixup_lambda=lam,signal='' if actual_signal is None else actual_signal,**metadata))
            v.sync(device);epoch_seconds=time.perf_counter()-started;train_total+=epoch_seconds
            if active and variant!='full_adamw':controller.budget.end_epoch()
            partial_total+=ep_partial;omitted_total+=ep_omitted
            expected=spec['expected_partial_updates_per_active_epoch'] if active and variant!='full_adamw' else 0
            assert ep_partial==expected and ep_omitted==expected*batch
            for param in prefix:assert int(opt.state[param]['step'])==idx-partial_total
            for param in suffix:assert int(opt.state[param]['step'])==idx
            sw.writerows(step_rows);step_file.flush()
            v.sync(device);ev_start=time.perf_counter()
            ev_loss,accuracy=v.evaluate(model,ev,ey,spec['eval_batch_size']);v.sync(device)
            ev_seconds=time.perf_counter()-ev_start;validation_total+=ev_seconds
            observed=time.perf_counter()-training_started
            row=dict(epoch=epoch,updates=idx,training_loss=loss_sum/seen,validation_loss=ev_loss,validation_accuracy=accuracy,
                partial_updates=ep_partial,omitted_prefix_examples=ep_omitted,epoch_training_seconds=epoch_seconds,
                cumulative_training_seconds=train_total,validation_seconds=ev_seconds,cumulative_validation_seconds=validation_total,
                cumulative_observed_seconds=observed)
            epochs.append(row);ew.writerow(row);epoch_file.flush()
            print(f'PILOT seed={seed} {variant} epoch={epoch}/{spec["epochs"]} val={100*accuracy:.2f}% omit={ep_partial}/{batches} observed_s={observed:.1f}',flush=True)
    finally:step_file.close();epoch_file.close()
    assert idx==spec['expected_total_updates_per_arm']
    assert partial_total==(0 if variant=='full_adamw' else spec['expected_total_partial_updates_per_arm'])
    v.sync(device);cp_start=time.perf_counter()
    checkpoint(attempt/'final_checkpoint.pt',dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),
        seed=seed,variant=variant,updates=idx,partial_updates=partial_total,root_fingerprint=root_config['fingerprint'],
        run_fingerprint=fp,controller_history=list(controller.history),allocation_rng=controller.rng.getstate(),
        batch_rng=bg.get_state(),augmentation_rng=ag.get_state(),mixup_rng=mg.bit_generator.state,
        permutation_rng=pg.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
    cp_seconds=time.perf_counter()-cp_start
    summary=dict(status='complete',seed=seed,variant=variant,order_position=order,root_fingerprint=root_config['fingerprint'],
        fingerprint=fp,initial_state_sha256=ih,epochs=spec['epochs'],updates=idx,partial_updates=partial_total,
        omitted_prefix_examples=omitted_total,final_validation_accuracy=epochs[-1]['validation_accuracy'],
        final_validation_loss=epochs[-1]['validation_loss'],training_seconds=train_total,validation_seconds=validation_total,
        observed_seconds=epochs[-1]['cumulative_observed_seconds'],hardware_warmup_seconds=warm_seconds,
        checkpoint_seconds=cp_seconds,experiment_elapsed_seconds=time.perf_counter()-process_start,
        prefix_parameters=sum(p.numel() for p in prefix),suffix_parameters=sum(p.numel() for p in suffix))
    v.atomic_json(attempt/'summary.json',summary)
    files=[p for p in attempt.iterdir() if p.is_file()]
    v.atomic_json(parent/'COMPLETE.json',dict(status='complete',fingerprint=fp,attempt=str(attempt),sha256={str(p):d.sha(p) for p in files}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-only',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-tag',default='final');p.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag)
    spec=json.loads(SPEC_PATH.read_text());fake=args.smoke
    if fake:
        torch.set_num_threads(1)
        spec.update(development_seeds=[9901,9902],epochs=3,schedule_horizon_epochs=10,full_warmup_epochs=1,
            cut=1,batch_size=3,eval_batch_size=4,quota_window=5,history_size=4,hardware_warmup_steps=4,
            primary_target=0.,secondary_target=1.,sustained_epochs=2,expected_batches_per_epoch=6,
            expected_regular_batches_per_epoch=5,expected_partial_updates_per_active_epoch=2,
            expected_total_partial_updates_per_arm=4,expected_total_updates_per_arm=18,expected_omitted_examples_per_arm=12)
    record=freeze_record(spec,fake)
    if args.freeze_only:
        assert not fake and not REAL_OUT.exists(),'Freeze must precede scientific output'
        write_freeze(FREEZE_PATH,record);print('FROZEN:',FREEZE_PATH,record['freeze_hash']);return
    out=ROOT/'results'/('stage9_sanity_quality_pilot_v1_'+args.smoke_tag) if fake else REAL_OUT
    if fake:out.mkdir(exist_ok=True);write_freeze(out/'SYNTHETIC_FREEZE.json',record)
    else:assert FREEZE_PATH.exists() and json.loads(FREEZE_PATH.read_text())==record,'Prospective freeze missing or modified'
    device=torch.device(args.smoke_device if fake else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    v.configure()
    config=dict(spec=spec,synthetic=fake,freeze_hash=record['freeze_hash'],source_sha256=record['source_sha256'],
        split=record['expected_split'],device=str(device),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        torchvision=__import__('torchvision').__version__,cuda=torch.version.cuda,gpu='cpu' if device.type=='cpu' else torch.cuda.get_device_name(device))
    fp=v.digest(config);config['fingerprint']=fp;out.mkdir(exist_ok=True)
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'Code/settings/environment changed; do not overwrite a study'
    else:v.atomic_json(cp,config)
    from review_stage9E import audit
    if (out/'COMPLETE.json').exists():
        d.validate_complete(out,fp);audit(out,write=False);print('SKIP complete, re-audited pilot');return
    preflight=v.idle_preflight() if device.type=='cuda' else []
    launch=d.next_attempt(out/'launches');v.atomic_json(launch/'preflight.json',preflight)
    loaded=time.perf_counter();data=v.load_data('development',device,fake=fake)
    assert data[-1]==config['split']
    v.atomic_json(launch/'data_loading.json',dict(seconds=time.perf_counter()-loaded))
    for seed,order in zip(spec['development_seeds'],spec['orders']):
        for position,variant in enumerate(order):run_one(out,seed,variant,position,config,data,spec,device,fake)
    audit(out,write=True)
    paths=[out/n for n in ['configuration.json','audit.json','all_runs.csv','targets.csv','candidate_screen.csv','REPORT.md']]
    paths+=list(out.glob('seed_*/*/COMPLETE.json'))+list(out.glob('seed_*/*/run_config.json'))+list((out/'launches').rglob('*.json'))
    v.atomic_json(out/'COMPLETE.json',dict(status='complete',fingerprint=fp,synthetic=fake,runs=10,sha256={str(p):d.sha(p) for p in paths}))
    print('PILOT COMPLETE:',out)

if __name__=='__main__':main()
