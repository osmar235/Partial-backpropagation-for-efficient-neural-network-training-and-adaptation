"""Post-run Stage9E evidence preservation, independent outcomes and optional rescore."""
from pathlib import Path
import argparse, collections, csv, hashlib, json, math, statistics as st

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/stage9E_quality_pilot_v1'
OUT=ROOT/'results/stage9E_review_v1'
MUTABLE={str(ROOT/'README_STAGE9.md'),str(ROOT/'CURRENT_RELEASE.md')}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,rows):
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10),(a,b)
def sustained(rows,threshold,n):
    streak=0
    for i,row in enumerate(rows):
        streak=streak+1 if float(row['validation_accuracy'])>=threshold else 0
        if streak==n:return i,row
    return None,None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rescore-cuda',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(1)
    from review_stage9E import audit
    audit_result=audit(RAW,write=False)
    assert not audit_result['synthetic'] and audit_result['audited_steps']==176000
    cfg=json.loads((RAW/'configuration.json').read_text());spec=cfg['spec'];protected={}
    def verify(path,h):
        assert sha(path)==h,path
        if str(path) not in MUTABLE:protected[str(path)]=h
    for path in RAW.rglob('*'):
        if path.is_file():verify(path,sha(path))
    for row in read(ROOT/'protocol/stage9/quality_pilot_preparation_sha256_v1.csv'):
        if row['path'] not in MUTABLE:verify(row['path'],row['sha256'])
    prior=json.loads((ROOT/'protocol/stage9/quality_pilot_preservation_v1.json').read_text())
    for path,h in prior['prior_immutable_sha256'].items():verify(path,h)
    for path,h in cfg['source_sha256'].items():verify(path,h)

    records={};epoch_records={};target_records={};cp_paths={};features=[];thermal=[];warm_losses={};warm_checks=[]
    for seed in spec['development_seeds']:
        for variant in spec['variants']:
            marker=json.loads((RAW/f'seed_{seed}/{variant}/COMPLETE.json').read_text());attempt=Path(marker['attempt'])
            summary=json.loads((attempt/'summary.json').read_text());epochs=read(attempt/'epochs.csv');steps=read(attempt/'steps.csv')
            _,reached=sustained(epochs,spec['primary_target'],spec['sustained_epochs'])
            observed=float(reached['cumulative_observed_seconds']) if reached else None
            training=float(reached['cumulative_training_seconds']) if reached else None
            target_records[seed,variant]=dict(seed=seed,variant=variant,reached=reached is not None,
                confirming_epoch=int(reached['epoch']) if reached else '',observed_seconds=observed,training_seconds=training)
            records[seed,variant]=summary;epoch_records[seed,variant]=epochs;cp_paths[seed,variant]=attempt/'final_checkpoint.pt'
            primary=next(r for r in read(RAW/'targets.csv') if int(r['seed'])==seed and r['variant']==variant and r['target_role']=='primary')
            assert (primary['reached']=='True')==(reached is not None)
            if reached:close(observed,primary['observed_seconds']);assert int(primary['confirming_epoch'])==int(reached['epoch'])
            else:assert primary['observed_seconds']=='' and int(primary['censoring_epoch'])==100
            warm=[float(r['loss']) for r in steps[:spec['full_warmup_epochs']*spec['expected_batches_per_epoch']]]
            warm_losses.setdefault(seed,warm)
            differences=[abs(a-b) for a,b in zip(warm_losses[seed],warm)]
            reference_epochs=epoch_records.get((seed,'full_adamw'),epochs)
            for first in range(spec['full_warmup_epochs']):
                local=differences[first*spec['expected_batches_per_epoch']:(first+1)*spec['expected_batches_per_epoch']]
                warm_checks.append(dict(seed=seed,variant=variant,epoch=first+1,max_loss_difference_vs_full=max(local),
                    validation_accuracy=float(epochs[first]['validation_accuracy']),reference_validation_accuracy=float(reference_epochs[first]['validation_accuracy']),
                    validation_delta_vs_full_pp=100*(float(epochs[first]['validation_accuracy'])-float(reference_epochs[first]['validation_accuracy']))))
            active=[r for r in steps if int(r['epoch'])>spec['full_warmup_epochs'] and int(r['batch_examples'])==spec['batch_size']]
            if variant!='full_adamw':
                for label,test in [('all',lambda r:True),('weak_mix_le_0p05',lambda r:min(float(r['mixup_lambda']),1-float(r['mixup_lambda']))<=.05),
                    ('strong_mix_ge_0p20',lambda r:min(float(r['mixup_lambda']),1-float(r['mixup_lambda']))>=.20)]:
                    subset=[r for r in active if test(r)];partial=[r for r in subset if r['partial']=='1']
                    deadlines=sum(r['reason']=='quota_deadline' for r in partial)
                    features.append(dict(seed=seed,variant=variant,group=label,n=len(subset),partial_updates=len(partial),
                        omission_fraction=len(partial)/len(subset),deadline_partials=deadlines,deadline_fraction_of_partials=deadlines/len(partial) if partial else '',
                        mean_selected_mix_strength=st.mean(min(float(r['mixup_lambda']),1-float(r['mixup_lambda'])) for r in partial) if partial else ''))
            telemetry=read(attempt/'telemetry.csv')
            for r in telemetry:
                sample=json.loads(r['sample']);assert sample['available']
                for gpu in sample['gpus']:
                    thermal.append(dict(seed=seed,variant=variant,epoch=int(r['epoch']),temperature=float(gpu['temperature.gpu']),clock=float(gpu['clocks.sm']),utilization=float(gpu['utilization.gpu'])))
    comparisons=[]
    for seed in spec['development_seeds']:
        full=records[seed,'full_adamw'];ft=target_records[seed,'full_adamw']
        assert ft['reached']
        for variant in spec['variants']:
            a=records[seed,variant];at=target_records[seed,variant]
            controls=['periodic','random']+(['mixup_only'] if variant=='boundary_rms' else [])
            valid=[target_records[seed,c] for c in controls if target_records[seed,c]['reached']]
            best=min(valid,key=lambda r:(r['observed_seconds'],r['variant']))
            best_accuracy=records[seed,best['variant']]['final_validation_accuracy']
            delta=100*(a['final_validation_accuracy']-full['final_validation_accuracy'])
            best_delta=100*(a['final_validation_accuracy']-best_accuracy)
            ratio=at['observed_seconds']/ft['observed_seconds'] if at['reached'] else None
            blind_ratio=at['observed_seconds']/best['observed_seconds'] if at['reached'] else None
            speed=ratio is not None and blind_ratio is not None and max(ratio,blind_ratio)<=.95
            quality=min(delta,best_delta)>=-.2
            comparisons.append(dict(seed=seed,variant=variant,final_accuracy=a['final_validation_accuracy'],final_delta_vs_full_pp=delta,
                training_wall_ratio=a['training_seconds']/full['training_seconds'],primary_reached=at['reached'],confirming_epoch=at['confirming_epoch'],
                observed_target_seconds=at['observed_seconds'] if at['reached'] else '',time_ratio_to_full=ratio if ratio is not None else '',
                best_predecided_control=best['variant'],time_ratio_to_best_control=blind_ratio if blind_ratio is not None else '',
                final_delta_vs_best_control_pp=best_delta,speed_screen_pass=speed,quality_screen_pass=quality,seed_screen_pass=speed and quality))
    saved=read(RAW/'candidate_screen.csv')
    for r in saved:
        a=next(a for a in comparisons if a['seed']==int(r['seed']) and a['variant']==r['candidate'])
        for field in ['time_ratio_to_full','final_delta_vs_full_pp']:
            if r[field]=='':assert a[field]==''
            else:close(r[field],a[field])
        if r['time_ratio_to_fastest_control']:close(r['time_ratio_to_fastest_control'],a['time_ratio_to_best_control'])
        close(r['final_delta_vs_control_pp'],a['final_delta_vs_best_control_pp'])
        assert (r['seed_screen_pass']=='True')==a['seed_screen_pass']
    aggregate=[]
    for variant in spec['variants']:
        a=[r for r in comparisons if r['variant']==variant]
        aggregate.append(dict(variant=variant,n=2,mean_final_accuracy=st.mean(r['final_accuracy'] for r in a),
            mean_accuracy_delta_vs_full_pp=st.mean(r['final_delta_vs_full_pp'] for r in a),mean_paired_training_wall_ratio=st.mean(r['training_wall_ratio'] for r in a),
            primary_reached=sum(r['primary_reached'] for r in a),primary_time_ratio_seed1=a[0]['time_ratio_to_full'],primary_time_ratio_seed2=a[1]['time_ratio_to_full']))
    save('independent_seed_results.csv',comparisons);save('independent_mean_results.csv',aggregate);save('allocation_diagnostics_posthoc.csv',features);save('telemetry_samples.csv',thermal);save('initial_full_warmup_differences.csv',warm_checks)
    if args.rescore_cuda:
        import torch
        import stage8_vit_v2 as v
        assert torch.cuda.is_available();v.configure();preflight=v.idle_preflight()
        x,y,ev,ey,split=v.load_data('development',torch.device('cuda'));assert split==cfg['split']
        rescored=[]
        for (seed,variant),path in cp_paths.items():
            cp=torch.load(path,map_location='cpu',weights_only=True);model=v.make_model().to('cuda');model.load_state_dict(cp['model_state'])
            loss,accuracy=v.evaluate(model,ev,ey,spec['eval_batch_size']);expected=records[seed,variant]
            assert abs(accuracy-expected['final_validation_accuracy'])<1e-12
            assert abs(loss-expected['final_validation_loss'])<1e-6
            rescored.append(dict(seed=seed,variant=variant,checkpoint_sha256=sha(path),validation_accuracy=accuracy,validation_loss=loss,
                recorded_accuracy=expected['final_validation_accuracy'],recorded_loss=expected['final_validation_loss']))
            del model,cp
        save('checkpoint_validation_rescore.csv',rescored)
        (OUT/'rescore_metadata.json').write_text(json.dumps(dict(status='PASS',device=torch.cuda.get_device_name(0),test_dataset_loaded=False,training_performed=False,preflight=preflight),indent=2)+'\n')
    baseline=OUT/'preserved_inputs_sha256.csv'
    if baseline.exists():
        for r in read(baseline):verify(r['path'],r['sha256'])
    else:save('preserved_inputs_sha256.csv',[dict(path=p,sha256=h) for p,h in sorted(protected.items())])
    result=dict(status='PASS',runs=10,updates=176000,initial_full_warmup_losses_bitwise_identical=all(r['max_loss_difference_vs_full']==0 for r in warm_checks),
        initial_full_warmup_max_loss_difference=max(r['max_loss_difference_vs_full'] for r in warm_checks),
        warmup_note='Matched initialization and RNG streams do not guarantee bitwise CUDA trajectories; native full-update arms already drift during warmup. No deterministic-algorithm setting was required by frozen protocol.',
        independent_targets_and_practical_screens_match=True,preserved_immutable_files=len(protected),
        final_checkpoint_validation_rescored=(OUT/'checkpoint_validation_rescore.csv').exists(),
        temperature_range=[min(r['temperature'] for r in thermal),max(r['temperature'] for r in thermal)],
        clock_range=[min(r['clock'] for r in thermal),max(r['clock'] for r in thermal)],
        decision='Both adaptive candidates failed frozen development screen; no automatic extension or parameter search')
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
