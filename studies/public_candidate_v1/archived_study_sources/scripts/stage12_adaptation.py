"""Fixed pretrained selective-adaptation opportunity screen."""
import argparse,csv,json,math,platform,re,time
import torch
from pathlib import Path
import stage12_pretrained as p
import stage8_vit_v2 as v
import stage9_boundary_diagnostic as d
import stage10_representation as old

ROOT=v.ROOT;SPEC=ROOT/'protocol/stage12/adaptation_spec_v1.json'
FREEZE=ROOT/'protocol/stage12/FROZEN_ADAPTATION_v1.json';REAL=ROOT/'results/stage12A_adaptation_v1'
CONTROLS=['full','head','lp_full','lp_suffix']
def sources():
    import torchvision.models.resnet as resnet_source
    files=[ROOT/n for n in ['scripts/stage12_pretrained.py','scripts/stage12_adaptation.py','scripts/review_stage12A.py',
        'scripts/test_stage12_adaptation.py','scripts/run_stage12A_adaptation.cmd','protocol/stage12/adaptation_spec_v1.json',
        'protocol/stage12/ADAPTATION_SCREEN_v1.md']]+[Path(resnet_source.__file__)]
    return {**old.sources(),**{str(f):p.sha(f) for f in files}}

def freeze_record(spec,fake):
    evidence=[ROOT/n for n in ['results/stage11B_review_v1/REPORT.md','results/stage11B_review_v1/independent_audit.json',
        'results/stage11B_growth_pilot_v1/configuration.json']]
    split=json.loads(evidence[-1].read_text())['split']
    record=dict(date='2026-10-22',version=spec['version'],synthetic=fake,spec=spec,source_sha256=sources(),
        evidence_sha256={str(f):p.sha(f) for f in evidence},split={'kind':'artificial'} if fake else split,
        pretrained=None if fake else dict(path=str(p.WEIGHTS),sha256=p.sha(p.WEIGHTS),metadata_sha256=p.sha(p.WEIGHTS.with_suffix('.json'))),
        required_execution=p.execution.execution_settings())
    record['freeze_hash']=v.digest(record);return record

def screen(records,spec):
    best=max(records[c]['final_accuracy'] for c in CONTROLS)
    qualified=[c for c in CONTROLS if 100*(best-records[c]['final_accuracy'])<=spec['final_accuracy_margin_pp']+1e-12]
    fastest=min(qualified,key=lambda c:(records[c]['total_run_seconds'],c))
    candidate=records['lp_periodic'];delta=100*(candidate['final_accuracy']-best)
    ratio=candidate['total_run_seconds']/records[fastest]['total_run_seconds']
    full_ratio=candidate['total_run_seconds']/records['full']['total_run_seconds']
    ready=records['full']['ready'] or records['lp_full']['ready']
    quality=delta>=-spec['final_accuracy_margin_pp']-1e-12
    speed=max(ratio,full_ratio)<=1-spec['minimum_time_gain']
    return dict(best_control_accuracy=best,qualified_controls=';'.join(qualified),fastest_qualified_control=fastest,
        delta_vs_best_control_pp=delta,delta_vs_fixed_suffix_pp=100*(candidate['final_accuracy']-records['lp_suffix']['final_accuracy']),
        time_ratio_to_fastest_qualified=ratio,time_ratio_to_full=full_ratio,baseline_ready=ready,candidate_ready=candidate['ready'],
        quality_pass=quality,speed_pass=speed,opportunity_pass=quality and speed and ready and candidate['ready'])

def run_one(out,seed,arm,position,cfg,data,spec,device,fake):
    parent=out/f'seed_{seed}'/arm;parent.mkdir(parents=True,exist_ok=True)
    rc=dict(seed=seed,arm=arm,position=position,root_fingerprint=cfg['fingerprint']);rc['fingerprint']=v.digest(rc)
    rcp=parent/'run_config.json'
    if rcp.exists():assert json.loads(rcp.read_text())==rc
    else:v.atomic_json(rcp,rc)
    if d.validate_complete(parent,rc['fingerprint']):print('SKIP',seed,arm,flush=True);return
    attempt=d.next_attempt(parent);v.sync(device);start=time.perf_counter()
    x,y,ev,ey,_=data;model=p.make_model(seed,fake).to(device)
    initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()};initial_hash=old.state_hash(initial)
    opt=p.optimizer(model,spec);xx=p.transform(x[:spec['batch_size']],spec['image_size'])
    v.sync(device);t=time.perf_counter()
    # Every path warms full kernels, restores weights/buffers, and discards moments.
    for _ in range(spec['hardware_warmup_steps']):p.update(model,opt,xx,y[:len(xx)],'full',spec['label_smoothing'])
    v.sync(device);hardware=time.perf_counter()-t
    model.load_state_dict(initial);opt=p.optimizer(model,spec);model.zero_grad(set_to_none=True)
    bg=torch.Generator().manual_seed(seed+281001);ag=torch.Generator().manual_seed(seed+282001)
    counts=dict(prefix=0,suffix=0,head=0);idx=0;train_total=val_total=0.;epochs=[];telemetry=[]
    def snapshot():
        return dict(seed=seed,arm=arm,root_fingerprint=cfg['fingerprint'],fingerprint=rc['fingerprint'],updates=idx,
            group_updates=counts.copy(),model_state=model.state_dict(),optimizer_state=opt.state_dict(),batch_rng=bg.get_state(),augmentation_rng=ag.get_state())
    sf=(attempt/'steps.csv').open('w',encoding='utf8',newline='');ef=(attempt/'epochs.csv').open('w',encoding='utf8',newline='')
    sw=csv.DictWriter(sf,fieldnames=['epoch','step','batch_examples','mode','loss','backbone_lr','head_lr','prefix_updates','suffix_updates','head_updates'])
    ew=csv.DictWriter(ef,fieldnames=['epoch','updates','training_loss','validation_loss','validation_accuracy','epoch_training_seconds',
        'training_seconds','epoch_validation_seconds','validation_seconds','observed_seconds'])
    sw.writeheader();ew.writeheader();v.sync(device);observed_start=time.perf_counter()
    try:
        for epoch in range(1,spec['epochs']+1):
            if not fake:
                telemetry.append(dict(epoch=epoch,sample=json.dumps(v.telemetry())));v.write_csv(attempt/'telemetry.csv',telemetry)
            rows=[];loss_sum=seen=0;v.sync(device);t=time.perf_counter()
            for ix in v.frozen6.epoch_batches(len(y),spec['batch_size'],bg):
                ix=ix.to(device);xx=p.transform(x[ix],spec['image_size'],ag);mode=p.mode_at(arm,idx,spec);factor=p.rate_factor(idx,spec)
                for group in opt.param_groups:group['lr']=factor*(spec['head_lr'] if group['role']=='head' else spec['backbone_lr'])
                loss=p.update(model,opt,xx,y[ix],mode,spec['label_smoothing']);idx+=1
                counts['prefix']+=int(mode=='full');counts['suffix']+=int(mode!='head');counts['head']+=1
                rows.append(dict(epoch=epoch,step=idx,batch_examples=len(ix),mode=mode,loss=loss,
                    backbone_lr=factor*spec['backbone_lr'],head_lr=factor*spec['head_lr'],**{k+'_updates':n for k,n in counts.items()}))
                loss_sum+=loss*len(ix);seen+=len(ix)
            v.sync(device);seconds=time.perf_counter()-t;train_total+=seconds;sw.writerows(rows);sf.flush()
            v.sync(device);t=time.perf_counter();loss,acc=p.evaluate(model,ev,ey,spec);v.sync(device);val_seconds=time.perf_counter()-t;val_total+=val_seconds
            if epoch==spec['head_warmup_epochs']:old.checkpoint(attempt/'warmup_checkpoint.pt',snapshot())
            row=dict(epoch=epoch,updates=idx,training_loss=loss_sum/seen,validation_loss=loss,validation_accuracy=acc,
                epoch_training_seconds=seconds,training_seconds=train_total,epoch_validation_seconds=val_seconds,
                validation_seconds=val_total,observed_seconds=time.perf_counter()-observed_start)
            ew.writerow(row);ef.flush();epochs.append(row)
            print(f'ADAPTATION seed={seed} {arm} epoch={epoch}/{spec["epochs"]} val={100*acc:.2f}% train_s={train_total:.1f}',flush=True)
    finally:sf.close();ef.close()
    assert idx==spec['updates_per_run'];v.sync(device);t=time.perf_counter();old.checkpoint(attempt/'final_checkpoint.pt',snapshot());v.sync(device)
    cp_seconds=time.perf_counter()-t;total=time.perf_counter()-start
    summary=dict(status='complete',seed=seed,arm=arm,position=position,root_fingerprint=cfg['fingerprint'],fingerprint=rc['fingerprint'],
        updates=idx,epochs=spec['epochs'],initial_hash=initial_hash,group_updates=counts,final_accuracy=epochs[-1]['validation_accuracy'],
        final_loss=epochs[-1]['validation_loss'],training_seconds=train_total,validation_seconds=val_total,hardware_warmup_seconds=hardware,
        final_checkpoint_seconds=cp_seconds,observed_seconds=epochs[-1]['observed_seconds'],total_run_seconds=total)
    v.atomic_json(attempt/'summary.json',summary)
    v.atomic_json(parent/'COMPLETE.json',dict(status='complete',fingerprint=rc['fingerprint'],attempt=str(attempt),sha256={str(f):p.sha(f) for f in attempt.iterdir() if f.is_file()}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--smoke-tag',default='final_cpu');ap.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag);p.execution.configure();spec=json.loads(SPEC.read_text());fake=args.smoke
    if fake:spec.update(seeds=[9961,9962],epochs=4,head_warmup_epochs=1,image_size=32,batch_size=3,eval_batch_size=4,
        batches_per_epoch=6,updates_per_run=24,hardware_warmup_steps=2,readiness_target=0.,secondary_target=1.)
    record=freeze_record(spec,fake)
    if args.freeze_only:
        assert not fake and not REAL.exists();old.write_freeze(FREEZE,record);print('FROZEN',record['freeze_hash']);return
    out=ROOT/'results'/('stage12_sanity_adaptation_v1_'+args.smoke_tag) if fake else REAL
    if not fake:assert FREEZE.exists() and json.loads(FREEZE.read_text())==record,'Prospective source/evidence freeze changed'
    out.mkdir(exist_ok=True)
    if fake:old.write_freeze(out/'SYNTHETIC_FREEZE.json',record)
    device=torch.device(args.smoke_device if fake else 'cuda')
    if device.type=='cuda':assert torch.cuda.is_available()
    cfg=dict(spec=spec,synthetic=fake,source_sha256=record['source_sha256'],freeze_hash=record['freeze_hash'],split=record['split'],
        pretrained=record['pretrained'],execution=p.execution.execution_settings(),python=platform.python_version(),torch=torch.__version__,
        torchvision=__import__('torchvision').__version__,numpy=__import__('numpy').__version__,cuda=torch.version.cuda,
        device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu')
    cfg['fingerprint']=v.digest(cfg);cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg,'Environment/source/config changed'
    else:v.atomic_json(cp,cfg)
    from review_stage12A import audit
    if (out/'COMPLETE.json').exists():audit(out,False);print('Completed screen verified read-only');return
    launch=d.next_attempt(out/'launches');v.atomic_json(launch/'preflight.json',v.idle_preflight() if device.type=='cuda' else [])
    t=time.perf_counter();data=p.load_data(device,fake);assert data[-1]==cfg['split']
    assert math.ceil(len(data[1])/spec['batch_size'])==spec['batches_per_epoch']
    v.atomic_json(launch/'data_loading.json',dict(seconds=time.perf_counter()-t))
    for seed,order in zip(spec['seeds'],spec['orders']):
        for position,arm in enumerate(order):run_one(out,seed,arm,position,cfg,data,spec,device,fake)
    audit(out,True)
    files=[out/n for n in ['configuration.json','audit.json','all_runs.csv','targets.csv','screen.csv','REPORT.md']]
    files+=list(out.glob('seed_*/*/COMPLETE.json'))+list(out.glob('seed_*/*/run_config.json'))+list((out/'launches').rglob('*.json'))
    v.atomic_json(out/'COMPLETE.json',dict(status='complete',fingerprint=cfg['fingerprint'],synthetic=fake,runs=10,sha256={str(f):p.sha(f) for f in files}))
    print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
