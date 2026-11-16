"""Frozen midpoint-growth pilot with uninterrupted and Adam-reset controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,json,math,platform,re,time
from pathlib import Path
import numpy as np
import torch
import stage8_vit_v2 as v
import stage9_boundary_diagnostic as d
import stage10_models as m
import stage10_representation as old
import stage11_growth as g
import stage11_growth_feasibility as feasibility

ROOT=v.ROOT
SPEC=ROOT/'protocol/stage11/growth_pilot_spec_v1.json'
FREEZE=ROOT/'protocol/stage11/FROZEN_GROWTH_PILOT_v1.json'
REAL_OUT=ROOT/'results/stage11B_growth_pilot_v1'
CONTROLS=['full','small','full_reset','small_reset']

def initial_variant(arm):return 'full_rgb32' if arm.startswith('full') else 'small_rgb32'
def resets(arm):return arm.endswith('_reset') or arm=='grow'

def sources():
    names=['scripts/stage11_growth_pilot.py','scripts/review_stage11B.py','scripts/test_stage11_growth_pilot.py',
           'scripts/run_stage11B_growth_pilot.cmd','protocol/stage11/growth_pilot_spec_v1.json','protocol/stage11/GROWTH_PILOT_v1.md']
    return {**feasibility.sources(),**{str(ROOT/n):d.sha(ROOT/n) for n in names}}

def freeze_record(spec,fake):
    paths=[ROOT/n for n in ['results/stage11A_review_v1/REPORT.md','results/stage11A_review_v1/independent_audit.json',
                            'results/stage11A_growth_feasibility_v1/configuration.json','results/stage11A_growth_feasibility_v1/COMPLETE.json']]
    previous=json.loads(paths[2].read_text())
    record=dict(version=spec['version'],prepared_date='2026-10-22',synthetic=fake,spec=spec,source_sha256=sources(),
                evidence_sha256={str(p):d.sha(p) for p in paths},expected_split={'kind':'artificial'} if fake else previous['split'],
                required_execution=m.execution_settings())
    record['freeze_hash']=v.digest(record);return record

def transition(model,opt,arm,seed,fake,spec,probe):
    device=probe.device;v.sync(device);started=time.perf_counter()
    before_hash=old.state_hash(model.state_dict());before_count=sum(p.numel() for p in model.parameters())
    model.eval()
    with torch.no_grad():before=model(probe)
    v.sync(device);t=time.perf_counter()
    if arm=='grow':model=g.expand(model,seed,fake)
    if resets(arm):opt=g.reset_adam(model,opt.param_groups[0]['lr'],spec['weight_decay'])
    v.sync(device);mapping=time.perf_counter()-t
    model.eval()
    with torch.no_grad():after=model(probe)
    scaled=(after-before).abs()/(spec['logit_atol']+spec['logit_rtol']*before.abs())
    assert torch.isfinite(after).all() and float(scaled.max())<=1.
    assert torch.equal(before.argmax(1),after.argmax(1))
    model.train();v.sync(device)
    info=dict(action='grow_and_reset' if arm=='grow' else 'reset' if resets(arm) else 'unchanged',
              model_sha_before=before_hash,model_sha_after=old.state_hash(model.state_dict()),
              parameter_count_before=before_count,parameter_count_after=sum(p.numel() for p in model.parameters()),
              optimizer_entries_after=len(opt.state),max_abs_logit_error=float((after-before).abs().max()),
              max_scaled_logit_error=float(scaled.max()),prediction_changes=0,mapping_reset_seconds=mapping,
              operation_seconds=time.perf_counter()-started)
    return model,opt,info

def practical_screen(records,spec):
    best=max(records[c]['final_validation_accuracy'] for c in CONTROLS)
    qualified=[c for c in CONTROLS if 100*(best-records[c]['final_validation_accuracy'])<=spec['final_accuracy_margin_pp']+1e-12]
    fastest=min(qualified,key=lambda c:(records[c]['total_run_seconds'],c))
    candidate=records['grow'];full=records['full']
    delta=100*(candidate['final_validation_accuracy']-best)
    quality=delta>=-spec['final_accuracy_margin_pp']-1e-12
    rf=candidate['total_run_seconds']/full['total_run_seconds']
    rb=candidate['total_run_seconds']/records[fastest]['total_run_seconds']
    speed=max(rf,rb)<=1-spec['minimum_time_gain']
    ready=full['readiness_reached'] and candidate['readiness_reached']
    return dict(best_control_accuracy=best,qualified_controls=';'.join(qualified),fastest_qualified_control=fastest,
                final_delta_vs_best_control_pp=delta,total_time_ratio_to_full=rf,total_time_ratio_to_best_qualified=rb,
                baseline_ready=full['readiness_reached'],candidate_ready=candidate['readiness_reached'],
                quality_pass=quality,speed_pass=speed,seed_screen_pass=quality and speed and ready)

def run_one(out,seed,arm,position,cfg,data,spec,device,fake):
    parent=out/f'seed_{seed}'/arm;parent.mkdir(parents=True,exist_ok=True)
    rc=dict(root_fingerprint=cfg['fingerprint'],seed=seed,variant=arm,order_position=position)
    rc['fingerprint']=v.digest(rc);fp=rc['fingerprint'];p=parent/'run_config.json'
    if p.exists():assert json.loads(p.read_text())==rc
    else:v.atomic_json(p,rc)
    if d.validate_complete(parent,fp):print('SKIP complete',seed,arm,flush=True);return
    attempt=d.next_attempt(parent);v.sync(device);process_start=time.perf_counter()
    x,y,ev,ey,_=data;v.seed_all(seed);model=m.make_model(initial_variant(arm),seed,fake).to(device)
    initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()};initial_hash=old.state_hash(initial)
    opt=g.reset_adam(model,spec['learning_rate'],spec['weight_decay'])
    ag=torch.Generator().manual_seed(seed+191001);mg=np.random.default_rng(seed+192001);pg=torch.Generator().manual_seed(seed+193001)
    xx=v.frozen6.augment_and_normalize(x[:spec['batch_size']],ag)
    xx,targets,_=v.mix_batch(xx,y[:spec['batch_size']],mg,pg,spec['mixup_alpha'],10)
    v.sync(device);t=time.perf_counter()
    for i in range(spec['hardware_warmup_steps']):old.update(model,opt,xx,targets,seed+194001+i,spec['label_smoothing'])
    v.sync(device);hardware=time.perf_counter()-t
    model.load_state_dict(initial);model.zero_grad(set_to_none=True);opt=g.reset_adam(model,spec['learning_rate'],spec['weight_decay'])
    bg=torch.Generator().manual_seed(seed+91001);ag=torch.Generator().manual_seed(seed+92001)
    mg=np.random.default_rng(seed+98001);pg=torch.Generator().manual_seed(seed+99001)
    batches=math.ceil(len(y)/spec['batch_size']);assert batches==spec['expected_batches_per_epoch']
    idx=0;train_total=val_total=0.;transition_seconds=0.;epochs=[];telemetry=[]
    def snapshot():
        return dict(seed=seed,variant=arm,updates=idx,root_fingerprint=cfg['fingerprint'],run_fingerprint=fp,
                    model_state=model.state_dict(),optimizer_state=opt.state_dict(),batch_rng=bg.get_state(),augmentation_rng=ag.get_state(),
                    permutation_rng=pg.get_state(),mixup_rng=mg.bit_generator.state)
    sf=(attempt/'steps.csv').open('w',encoding='utf8',newline='');ef=(attempt/'epochs.csv').open('w',encoding='utf8',newline='')
    sw=csv.DictWriter(sf,fieldnames=['epoch','step','batch_examples','loss','lr','mixup_lambda','dropout_seed','width','optimizer_step'])
    ew=csv.DictWriter(ef,fieldnames=['epoch','updates','training_loss','validation_loss','validation_accuracy','epoch_training_seconds',
                                  'cumulative_training_seconds','validation_seconds','cumulative_validation_seconds','cumulative_observed_seconds'])
    sw.writeheader();ew.writeheader();v.sync(device);started=time.perf_counter()
    try:
        for epoch in range(1,spec['epochs']+1):
            if epoch==spec['transition_epoch']+1:
                v.sync(device);t=time.perf_counter();old.checkpoint(attempt/'transition_before.pt',snapshot())
                probe=v.frozen6.normalize_cifar(x[:spec['transition_probe_examples']])
                model,opt,info=transition(model,opt,arm,seed,fake,spec,probe)
                old.checkpoint(attempt/'transition_after.pt',snapshot());v.sync(device)
                transition_seconds=time.perf_counter()-t
                info.update(after_epoch=spec['transition_epoch'],updates=idx,total_transition_seconds=transition_seconds)
                v.atomic_json(attempt/'transition.json',info)
            if not fake and (epoch==1 or epoch%spec['telemetry_every_epochs']==0):
                telemetry.append(dict(epoch=epoch,sample=json.dumps(v.telemetry())));v.write_csv(attempt/'telemetry.csv',telemetry)
            rows=[];loss_sum=seen=0;v.sync(device);t=time.perf_counter()
            for indices in v.frozen6.epoch_batches(len(y),spec['batch_size'],bg):
                indices=indices.to(device);n=len(indices)
                xx=v.frozen6.augment_and_normalize(x[indices],ag)
                xx,targets,lam=v.mix_batch(xx,y[indices],mg,pg,spec['mixup_alpha'],10)
                lr=v.learning_rate(idx,spec['schedule_horizon_epochs']*batches,batches,spec['learning_rate'],spec)
                for group in opt.param_groups:group['lr']=lr
                dropout_seed=seed+spec['dropout_seed_offset']+idx
                loss=old.update(model,opt,xx,targets,dropout_seed,spec['label_smoothing']);idx+=1
                age=idx-(spec['transition_epoch']*batches if resets(arm) and epoch>spec['transition_epoch'] else 0)
                rows.append(dict(epoch=epoch,step=idx,batch_examples=n,loss=loss,lr=lr,mixup_lambda=lam,dropout_seed=dropout_seed,
                                 width=model.body.hidden_dim,optimizer_step=age));loss_sum+=loss*n;seen+=n
            v.sync(device);seconds=time.perf_counter()-t;train_total+=seconds
            assert all(int(opt.state[p]['step'])==age for p in model.parameters())
            sw.writerows(rows);sf.flush();v.sync(device);t=time.perf_counter()
            loss,accuracy=v.evaluate(model,ev,ey,spec['eval_batch_size']);v.sync(device);val_seconds=time.perf_counter()-t;val_total+=val_seconds
            row=dict(epoch=epoch,updates=idx,training_loss=loss_sum/seen,validation_loss=loss,validation_accuracy=accuracy,
                     epoch_training_seconds=seconds,cumulative_training_seconds=train_total,validation_seconds=val_seconds,
                     cumulative_validation_seconds=val_total,cumulative_observed_seconds=time.perf_counter()-started)
            ew.writerow(row);ef.flush();epochs.append(row)
            print(f'GROWTH PILOT seed={seed} {arm} epoch={epoch}/{spec["epochs"]} val={100*accuracy:.2f}% train_s={train_total:.1f}',flush=True)
    finally:sf.close();ef.close()
    assert idx==spec['expected_updates_per_run'];v.sync(device);t=time.perf_counter()
    old.checkpoint(attempt/'final_checkpoint.pt',snapshot());v.sync(device);cp_seconds=time.perf_counter()-t
    total_seconds=time.perf_counter()-process_start
    summary=dict(status='complete',seed=seed,variant=arm,order_position=position,root_fingerprint=cfg['fingerprint'],fingerprint=fp,
                 updates=idx,epochs=spec['epochs'],initial_state_sha256=initial_hash,initial_parameter_count=sum(t.numel() for t in initial.values()),
                 final_parameter_count=sum(p.numel() for p in model.parameters()),final_optimizer_steps=age,
                 final_validation_accuracy=epochs[-1]['validation_accuracy'],final_validation_loss=epochs[-1]['validation_loss'],
                 training_seconds=train_total,validation_seconds=val_total,observed_seconds=epochs[-1]['cumulative_observed_seconds'],
                 hardware_warmup_seconds=hardware,transition_seconds=transition_seconds,checkpoint_seconds=cp_seconds,total_run_seconds=total_seconds)
    v.atomic_json(attempt/'summary.json',summary)
    v.atomic_json(parent/'COMPLETE.json',dict(status='complete',fingerprint=fp,attempt=str(attempt),sha256={str(p):d.sha(p) for p in attempt.iterdir() if p.is_file()}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze-only',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-tag',default='final_cpu');p.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu');args=p.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag);m.configure();spec=json.loads(SPEC.read_text());fake=args.smoke
    if fake:spec.update(development_seeds=[9971,9972],epochs=4,transition_epoch=2,schedule_horizon_epochs=10,batch_size=3,eval_batch_size=4,
                        hardware_warmup_steps=2,readiness_target=0.,secondary_target=1.,sustained_epochs=2,expected_batches_per_epoch=6,expected_updates_per_run=24)
    record=freeze_record(spec,fake)
    if args.freeze_only:
        assert not fake and not REAL_OUT.exists();old.write_freeze(FREEZE,record);print('FROZEN',record['freeze_hash']);return
    out=ROOT/'results'/('stage11_sanity_growth_pilot_v1_'+args.smoke_tag) if fake else REAL_OUT
    if fake:out.mkdir(exist_ok=True);old.write_freeze(out/'SYNTHETIC_FREEZE.json',record)
    else:assert FREEZE.exists() and json.loads(FREEZE.read_text())==record,'Prospective freeze changed or missing'
    device=torch.device(args.smoke_device if fake else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    cfg=dict(spec=spec,synthetic=fake,freeze_hash=record['freeze_hash'],source_sha256=record['source_sha256'],split=record['expected_split'],
             execution=m.execution_settings(),device=str(device),gpu='cpu' if device.type=='cpu' else torch.cuda.get_device_name(device),
             python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,torchvision=__import__('torchvision').__version__,cuda=torch.version.cuda)
    cfg['fingerprint']=v.digest(cfg);out.mkdir(exist_ok=True);cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg,'Environment/source/config changed; do not overwrite'
    else:v.atomic_json(cp,cfg)
    from review_stage11B import audit
    if (out/'COMPLETE.json').exists():audit(out,write=False);print('SKIP completed, re-audited growth pilot');return
    launch=d.next_attempt(out/'launches');v.atomic_json(launch/'preflight.json',v.idle_preflight() if device.type=='cuda' else [])
    t=time.perf_counter();data=v.load_data('development',device,fake=fake);assert data[-1]==cfg['split']
    v.atomic_json(launch/'data_loading.json',dict(seconds=time.perf_counter()-t))
    for seed,order in zip(spec['development_seeds'],spec['orders']):
        for position,arm in enumerate(order):run_one(out,seed,arm,position,cfg,data,spec,device,fake)
    audit(out,write=True)
    paths=[out/n for n in ['configuration.json','audit.json','all_runs.csv','targets.csv','screen.csv','REPORT.md']]
    paths+=list(out.glob('seed_*/*/COMPLETE.json'))+list(out.glob('seed_*/*/run_config.json'))+list((out/'launches').rglob('*.json'))
    v.atomic_json(out/'COMPLETE.json',dict(status='complete',fingerprint=cfg['fingerprint'],synthetic=fake,runs=10,sha256={str(p):d.sha(p) for p in paths}))
    print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
