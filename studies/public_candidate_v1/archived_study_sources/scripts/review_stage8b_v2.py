"""Review completed frozen Stage8 evaluation without training or data loading."""
from pathlib import Path
import csv, hashlib, json, math, statistics as st
import torch
import stage8_vit_v2 as v
from stage8_audit_v2 import audit, close, records


def interval(values,log=False):
    assert len(values)==6
    x=[math.log(z) for z in values] if log else values
    mean=st.mean(x);half=2.570581835636314*st.stdev(x)/math.sqrt(6)
    return [math.exp(z) for z in (mean,mean-half,mean+half)] if log else [mean,mean-half,mean+half]


def main():
    torch.set_num_threads(1)
    root=v.ROOT/'results/stage8B_vit_evaluation_v2'
    out=v.ROOT/'results/stage8B_review_v2';out.mkdir(parents=True,exist_ok=True)
    freeze=v.ROOT/'protocol/stage8/FROZEN_EVALUATION_v2.json'
    frozen=json.loads(freeze.read_text());spec=frozen['spec']
    assert not frozen['fake'] and not frozen['tiny']
    for path,h in frozen['development_input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    integrity=audit(root,freeze)
    lookup={};all_runs=[];curves=[];mixing=[];telemetry=[];state_checks=[]
    for seed in spec['evaluation_seeds']:
        for variant in spec['variants']:
            d=root/f'seed_{seed}'/variant
            r=json.loads((d/'summary.json').read_text());lookup[seed,variant]=r
            assert r['phase']=='evaluation' and r['epochs']==300 and r['updates']==58800
            config=json.loads((d/'run_config.json').read_text())
            assert config['phase']=='evaluation' and config['device']=='cuda' and not config['fake']
            assert config['split']=={'kind':'full_50000_train_10000_test'}
            assert config['profile']==frozen['periodic_epoch_q']
            all_runs.append(dict(seed=seed,variant=variant,accuracy_pct=100*r['final_eval_accuracy'],
                training_seconds=r['training_wall_seconds'],elapsed_seconds=r['total_experiment_elapsed_seconds'],
                partial_fraction=r['partial_fraction'],full_updates=r['full_updates'],partial_updates=r['partial_updates']))
            epochs=records(d/'epoch_trace.csv')
            curves += [dict(seed=seed,variant=variant,**row) for row in epochs]
            # Stream the large trace rather than retaining a second 1.4M-row copy.
            stats={0:[0,0.,0.],1:[0,0.,0.]}
            with (d/'step_trace.csv').open(newline='') as f:
                for row in csv.DictReader(f):
                    p=int(row['partial']);lam=float(row['mixup_lambda']);acc=stats[p]
                    acc[0]+=1;acc[1]+=min(lam,1-lam);acc[2]+=float(row['loss'])
            for p,(n,strength,loss) in stats.items():
                if n:mixing.append(dict(seed=seed,variant=variant,partial=p,updates=n,
                    mean_mixing_strength=strength/n,mean_training_loss=loss/n))
            gpu=[g for row in records(d/'gpu_telemetry.csv') for g in json.loads(row['sample']).get('gpus',[])]
            assert len(gpu)==601
            telemetry.append(dict(seed=seed,variant=variant,samples=len(gpu),
                min_sm_clock=min(float(g['clocks.sm']) for g in gpu),max_sm_clock=max(float(g['clocks.sm']) for g in gpu),
                min_temperature=min(float(g['temperature.gpu']) for g in gpu),max_temperature=max(float(g['temperature.gpu']) for g in gpu),
                first50_mean_epoch_seconds=st.mean(float(e['epoch_training_seconds']) for e in epochs[:50]),
                last50_mean_epoch_seconds=st.mean(float(e['epoch_training_seconds']) for e in epochs[-50:])))
            # Independent finite checkpoint-state check in addition to the native counter audit.
            ck=torch.load(d/'final_checkpoint.pt',weights_only=True,map_location='cpu')
            assert all(bool(torch.isfinite(t).all()) for t in ck['model_state'].values())
            assert all(bool(torch.isfinite(t).all()) for state in ck['optimizer_state']['state'].values() for t in state.values())
            state_checks.append(dict(seed=seed,variant=variant,finite=True,checkpoint_sha256=r['checkpoint_sha256']))
            print('REVIEWED',seed,variant,flush=True)
    effects=[];targets=[]
    planned=[('lgpb','full_adamw'),('periodic_matched','full_adamw'),('replay_prefix','full_adamw'),
             ('lgpb','periodic_matched'),('lgpb','replay_prefix')]
    for name,ref in planned:
        pairs=[(lookup[seed,name],lookup[seed,ref]) for seed in spec['evaluation_seeds']]
        accuracy=interval([100*(a['final_eval_accuracy']-b['final_eval_accuracy']) for a,b in pairs])
        wall=interval([a['training_wall_seconds']/b['training_wall_seconds'] for a,b in pairs],True)
        elapsed=interval([a['total_experiment_elapsed_seconds']/b['total_experiment_elapsed_seconds'] for a,b in pairs],True)
        effects.append(dict(variant=name,reference=ref,n=6,accuracy_delta_pp_ci95=accuracy,
            wall_ratio_ci95=wall,elapsed_ratio_ci95=elapsed,
            faster_seeds=sum(a['training_wall_seconds']<b['training_wall_seconds'] for a,b in pairs),
            more_accurate_seeds=sum(a['final_eval_accuracy']>b['final_eval_accuracy'] for a,b in pairs)))
        for target in spec['targets']:
            key=f'training_only_time_to_{round(100*target)}pct'
            ratios=[a[key]/b[key] for a,b in pairs if a[key] is not None and b[key] is not None]
            ci=interval(ratios,True) if len(ratios)==6 else None
            targets.append(dict(variant=name,reference=ref,target=target,
                variant_reached=sum(a[key] is not None for a,b in pairs),reference_reached=sum(b[key] is not None for a,b in pairs),
                paired_n=len(ratios),time_ratio_ci95=ci))
    existing=records(root/'stage8_paired_estimates.csv')
    for actual,expected in zip(existing,effects):
        assert actual['variant']==expected['variant'] and actual['reference']==expected['reference']
        for keys,values in [(('accuracy_delta_pp','accuracy_ci95_low','accuracy_ci95_high'),expected['accuracy_delta_pp_ci95']),
                            (('wall_ratio','wall_ci95_low','wall_ci95_high'),expected['wall_ratio_ci95'])]:
            for k,value in zip(keys,values):close(actual[k],value)
    existing_targets=records(root/'stage8_all_targets.csv');assert len(existing_targets)==len(targets)==15
    for actual,expected in zip(existing_targets,targets):
        assert actual['variant']==expected['variant'] and actual['reference']==expected['reference']
        close(actual['target'],expected['target'])
        for key in ('variant_reached','reference_reached','paired_n'):assert int(actual[key])==expected[key]
        if expected['time_ratio_ci95']:
            for k,value in zip(('training_only_time_ratio','ci95_low','ci95_high'),expected['time_ratio_ci95']):close(actual[k],value)
        else:assert all(actual[k]=='' for k in ('training_only_time_ratio','ci95_low','ci95_high'))
    criteria=json.loads((root/'stage8_primary_criteria.json').read_text())
    assert criteria['accuracy_pass']==(effects[0]['accuracy_delta_pp_ci95'][1]>-1)==False
    assert criteria['speed_pass']==(effects[0]['wall_ratio_ci95'][2]<1)==True
    summary=[]
    for variant in spec['variants']:
        runs=[lookup[seed,variant] for seed in spec['evaluation_seeds']]
        summary.append(dict(variant=variant,mean_accuracy_pct=st.mean(100*r['final_eval_accuracy'] for r in runs),
            mean_training_seconds=st.mean(r['training_wall_seconds'] for r in runs),
            mean_partial_fraction=st.mean(r['partial_fraction'] for r in runs)))
    result=dict(status='PASS',date='2026-10-20',integrity=integrity,means=summary,contrasts=effects,targets=targets,
        primary_accuracy_pass=False,primary_speed_pass=True,joint_primary_success=False,
        interpretation='Architecture/partition/recipe transfer failed the new accuracy criterion; all 24 runs retained.')
    v.atomic_json(out/'independent_results.json',result)
    for name,rows in [('all_24_runs.csv',all_runs),('mean_results.csv',summary),('learning_curves.csv',curves),
        ('mixup_gate_descriptive.csv',mixing),('telemetry_summary.csv',telemetry),('checkpoint_finiteness.csv',state_checks)]:
        v.write_csv(out/name,rows)
    files=[p for p in root.rglob('*') if p.is_file()]+[Path(p) for p in v.source_hashes()]+[freeze]
    v.write_csv(out/'preserved_evaluation_sha256.csv',[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)])
    lines=['# Stage8 frozen ViT evaluation: independent review','',
        'All24 runs /1,411,200 updates passed trace, provenance, checkpoint/counter and finite-state checks.',
        'All five planned effects and all15 target contrasts were independently recomputed.',
        'Frozen development inputs match their hashes. No training or dataset loading performed.','',
        '| Arm | Mean final test accuracy | Mean training wall | Mean partial use |',
        '|---|---:|---:|---:|']
    lines += [f"| {r['variant']} | {r['mean_accuracy_pct']:.3f}% | {r['mean_training_seconds']:.2f}s | {100*r['mean_partial_fraction']:.3f}% |" for r in summary]
    lines += ['','| Contrast | Accuracy delta pp [95% CI] | Training wall ratio [95% CI] |',
        '|---|---:|---:|']
    for e in effects:
        a=e['accuracy_delta_pp_ci95'];t=e['wall_ratio_ci95']
        lines.append(f"| {e['variant']} / {e['reference']} | {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}] | {t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}] |")
    lines += ['', 'The new joint primary criterion FAILED: accuracy failed, speed passed. Original Stage6 inference is unchanged.',
        'LGPB is less accurate in all six seeds. End-of-training wall savings are not useful time-to-quality savings:',
        'LGPB reached70% in all seeds but required7.61x baseline training time (95% CI6.45--8.99);',
        'it reached neither80% nor85% in any seed. Replay also reached neither; periodic reached80% in6/6',
        'and85% at some epoch in5/6. Final-epoch and ever-reached accuracy are distinct.','',
        'Periodic is much more accurate at nearly identical partial use and training time. This whole-algorithm',
        'contrast is adverse evidence for the loss-informed policy in this ViT/MixUp/partition recipe.',
        'Replay does not remedy the large loss. These controls do not isolate a unique cause, prove all',
        'transformers fail, or identify MixUp as causal. Mixed-batch gate associations are descriptive.',
        'Mechanism/target intervals are unadjusted. Different recipes/datasets/hardware remain untested.',
        'Retain all runs and telemetry; inspect timing drift rather than deleting outliers.','',
        'Decision: all planned scientific training is complete. No replacement seeds, third recipe or hidden',
        'tuning. Revise the manuscript around workload-specific benefits and this explicit failure boundary.',
        'Disclose the failed v1 screen, reused development validation and adverse v2 calibration. Complete',
        'public code/data archiving and editorial packaging before submission; no public release is implied.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',means=summary,primary_success=False),indent=2),flush=True)


if __name__=='__main__':main()
