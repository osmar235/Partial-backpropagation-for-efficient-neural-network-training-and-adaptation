"""Independent stdlib reconstruction of saved evidence; no training or timing."""
from collections import Counter,defaultdict
import itertools
import math
from pathlib import Path
from statistics import median
from record_preparation import ROOT,sha,read,save,verify,MUTABLE
import hashlib
import json

RAW=ROOT/'results/optimized_execution_v1'
OUT=ROOT/'results/optimized_execution_review_v1'
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def main():
    assert not OUT.exists()
    prep=ROOT/'results/optimized_execution_preparation_v1'
    seal=read(prep/'review_artifacts_sha256.json');verify(seal)
    protected={p:h for p,h in {**read(prep/'preserved_inputs_sha256.json'),**seal}.items() if Path(p) not in MUTABLE}
    protected[str(prep/'review_artifacts_sha256.json')]=sha(prep/'review_artifacts_sha256.json')
    protected.update({str(p):sha(p) for p in RAW.rglob('*') if p.is_file()})
    verify(protected);OUT.mkdir()
    save(OUT/'preserved_inputs_sha256.json',protected)
    save(OUT/'mutable_pointers_before.json',{str(p):sha(p) for p in MUTABLE})
    cfg=read(RAW/'configuration.json');spec=cfg['spec']
    assert cfg==read(ROOT/'protocol/optimized_execution/FROZEN_EXECUTION_v1.json')
    assert cfg['fingerprint']==digest({k:v for k,v in cfg.items() if k!='fingerprint'})
    verify(cfg['sources']);assert cfg['capability_sha256']==sha(prep/'environment_probe.json')
    for marker in RAW.rglob('COMPLETE.json'):verify(read(marker)['sha256'])
    rows_all=[];pairs=[];numerics=[];profiles=[];pids=[];telemetry=[];state_groups=defaultdict(list);status_counts=Counter()
    baseline='native_fp32_math';candidate='reference_fp32_math'
    for session in range(spec['sessions']):
        parent=RAW/f'session_{session:02d}';launches=list((parent/'launches').glob('attempt_*'));assert len(launches)==1
        process=read(launches[0]/'process.json');assert process['session']==session;pids.append(process['pid'])
        assert read(parent/'COMPLETE.json')['fingerprint']==cfg['fingerprint']
        for name in ('preflight','telemetry_before','telemetry_after'):
            telemetry.append(dict(session=session,kind=name,record=read(launches[0]/(name+'.json'))))
        for ci,(seed,batch) in enumerate(itertools.product(spec['seeds'],spec['batch_sizes'])):
            folder=parent/f'seed_{seed}_batch_{batch}';attempts=list(folder.glob('attempt_*'));assert len(attempts)==1
            attempt=attempts[0];rc=read(folder/'run_config.json');marker=read(folder/'COMPLETE.json')
            assert (rc['root_fingerprint'],rc['session'],rc['seed'],rc['batch'],rc['condition_index'])==(cfg['fingerprint'],session,seed,batch,ci)
            assert rc['fingerprint']==digest({k:v for k,v in rc.items() if k!='fingerprint'})==marker['fingerprint']
            assert Path(marker['attempt']).resolve()==attempt.resolve()
            assert set(marker['sha256'])=={str(p.resolve()) for p in attempt.rglob('*') if p.is_file()}
            assert read(attempt/'process.json')['pid']==process['pid']
            checks=read(attempt/'correctness.json');probes=read(attempt/'capability.json')
            assert list(checks)==list(probes)==spec['arms']
            for arm,check in checks.items():
                probe=probes[arm];assert probe['status']=='SUPPORTED' and probe['all_gradients_finite']
                flags=probe['policy'];is_math=arm.endswith('_math');bf16='_bf16_' in arm
                assert flags['deterministic']==flags['cudnn_deterministic']==is_math
                assert not any(flags[k] for k in ('tf32_matmul','tf32_cudnn','cudnn_benchmark','warn_only','fp16_reduced_reduction','bf16_reduced_reduction','math_reduced_reduction'))
                assert probe['parameter_dtype']=='float32' and probe['output_dtype']==('bf16' if bf16 else 'fp32')
                assert sum(probe['dispatch'].values())==6
                assert probe['dispatch']['math' if is_math else 'efficient']==6
                profiles.append(dict(session=session,seed=seed,batch=batch,arm=arm,**probe))
                metrics=check['metrics'];assert {m['step'] for m in metrics}==set(range(3))
                assert {m['kind'] for m in metrics}==set(spec['fp32_tolerances'])
                if is_math:assert check['deterministic_repeat']
                if bf16:
                    assert check['quality_equivalence_established'] is False
                    indexed={(m['step'],m['kind']):m for m in metrics}
                    assert len(indexed)==18 and len(check['checks'])==24
                    assert {(c['step'],c['budget']) for c in check['checks']}==set(itertools.product(range(3),spec['bf16_sanity_budgets']))
                    for m in metrics:
                        assert m['max_abs']==max(t['max_abs'] for t in m['tensors'])
                        assert len({t['name'] for t in m['tensors']})==len(m['tensors'])
                        # Reconstruct global relative L2 from serialized tensor norms.
                        numerator=sum((t['relative_l2']*max(t['reference_l2'],1e-12))**2 for t in m['tensors'])**.5
                        denominator=max(sum(t['reference_l2']**2 for t in m['tensors'])**.5,1e-12)
                        assert math.isclose(m['relative_l2'],numerator/denominator,rel_tol=1e-10,abs_tol=1e-12)
                    failed=[]
                    for c in check['checks']:
                        key=c['budget'];limit=spec['bf16_sanity_budgets'][key]
                        if key.endswith('_cosine_min'):metric=indexed[c['step'],key[:-11]];value=metric['cosine'];ok=value>=limit
                        elif key.endswith('_relative_l2'):metric=indexed[c['step'],key[:-12]];value=metric['relative_l2'];ok=value<=limit
                        else:metric=indexed[c['step'],key[:-8]];value=metric['max_abs'];ok=value<=limit
                        assert math.isfinite(value) and c['value']==value and c['limit']==limit and c['pass_budget']==ok
                        if not ok:failed.append(c)
                        numerics.append(dict(session=session,seed=seed,batch=batch,arm=arm,**c,
                            worst_tensor=max(metric['tensors'],key=lambda t:t['max_abs'])['name'] if key=='parameter_max_abs' else None))
                    assert check['status']=='FAIL_NUMERICAL_SANITY' and len(failed)==2
                    assert {(c['step'],c['budget']) for c in failed}=={(1,'parameter_max_abs'),(2,'parameter_max_abs')}
                else:
                    assert check['rng_match'] and check['status']=='PASS'
                    for m in metrics:
                        assert m['pass_tolerance'] and math.isfinite(m['max_scaled_error']) and m['max_scaled_error']<=1
                status_counts[check['status']]+=1
            assert read(attempt/'timing_status.json')=={'required_baseline_blocked':False,'failures':[]}
            rows=read(attempt/'trials.json');assert len(rows)==12
            base=list(spec['arms'])
            if (session+ci)%2:base.reverse()
            expected=[(ri,pos,arm) for ri in range(6) for pos,arm in enumerate(base[ri:]+base[:ri]) if arm in (baseline,candidate)]
            assert [(r['round'],r['position'],r['arm']) for r in rows]==expected
            for key in ('initial_sha256','input_sha256','labels_sha256'):assert len({r[key] for r in rows})==1
            for r in rows:
                assert (r['seed'],r['batch'],r['optimizer_age'])==(seed,batch,33)
                assert len(r['steady_losses'])==32 and r['all_gradients_finite']
                assert all(math.isfinite(v) for v in [r['startup_loss'],r['first_loss'],*r['steady_losses']])
                assert all(math.isfinite(r[k]) and r[k]>0 for k in ('setup_seconds','startup_update_seconds','warmup_restore_seconds','first_seconds','steady_seconds'))
                assert math.isclose(r['steady_update_seconds'],r['steady_seconds']/32,abs_tol=1e-12)
                mem=r['steady_memory'];assert mem['peak']-mem['baseline']==mem['incremental']
                state_groups[seed,batch,r['arm']].append([r[k] for k in ('final_model_sha256','final_optimizer_sha256','final_rng','first_loss','steady_losses')])
                rows_all.append(dict(session=session,**r))
            for ri in range(6):
                a=next(r for r in rows if r['round']==ri and r['arm']==baseline);b=next(r for r in rows if r['round']==ri and r['arm']==candidate)
                pairs.append(dict(session=session,seed=seed,batch=batch,round=ri,saving_percent=100*(a['steady_seconds']-b['steady_seconds'])/a['steady_seconds'],
                    native_ms=a['steady_update_seconds']*1000,reference_ms=b['steady_update_seconds']*1000,native_first=a['position']<b['position'],
                    peak_memory_reduction_bytes=a['steady_memory']['peak']-b['steady_memory']['peak'],
                    max_loss_difference=max(abs(x-y) for x,y in zip([a['first_loss'],*a['steady_losses']],[b['first_loss'],*b['steady_losses']])),
                    exact_learning_record=all(a[k]==b[k] for k in ('final_model_sha256','final_optimizer_sha256','final_rng','first_loss','steady_losses'))))
    assert len(set(pids))==4 and len(rows_all)==192 and len(pairs)==96
    for group in state_groups.values():assert len(group)==24 and all(row==group[0] for row in group)
    assert rows_all==read(RAW/'trials_export.json')
    summaries=[];order_summaries=[]
    for seed,batch in sorted({(r['seed'],r['batch']) for r in pairs}):
        subset=[r for r in pairs if (r['seed'],r['batch'])==(seed,batch)]
        sessions=[median(r['saving_percent'] for r in subset if r['session']==s) for s in range(4)]
        summary=dict(seed=seed,batch=batch,session_medians=sessions,median_saving_percent=median(sessions),
            positive_sessions=sum(x>0 for x in sessions),faster_pairs=sum(r['saving_percent']>0 for r in subset),pairs=len(subset),
            peak_memory_reduction_bytes_min=min(r['peak_memory_reduction_bytes'] for r in subset),peak_memory_reduction_bytes_max=max(r['peak_memory_reduction_bytes'] for r in subset))
        summaries.append(summary)
        original=next(r for r in read(RAW/'audit.json')['summary'] if (r['seed'],r['batch'],r['baseline'])==(seed,batch,baseline))
        assert math.isclose(original['median_saving_percent'],summary['median_saving_percent'],abs_tol=1e-12)
        for native_first in (False,True):
            group=[r for r in subset if r['native_first']==native_first];assert len(group)==12
            order_summaries.append(dict(seed=seed,batch=batch,native_first=native_first,pairs=len(group),median_saving_percent=median(r['saving_percent'] for r in group)))
    budgets=[]
    for arm,budget in sorted({(r['arm'],r['budget']) for r in numerics}):
        group=[r for r in numerics if (r['arm'],r['budget'])==(arm,budget)]
        budgets.append(dict(arm=arm,budget=budget,min_value=min(r['value'] for r in group),max_value=max(r['value'] for r in group),failed_checks=sum(not r['pass_budget'] for r in group),checks=len(group)))
    failed=[r for r in numerics if not r['pass_budget']]
    result=dict(status='PASS',trials=192,measured_postrestore_updates=6336,discarded_startup_updates=192,discarded_warmup_updates=960,
        qualification_updates=16*(3+6*2*3),profiled_updates=96,eligibility_counts=dict(status_counts),summary=summaries,
        fp32_reference_faster_pairs=sum(r['saving_percent']>0 for r in pairs),total_pairs=96,
        reference_positive_session_conditions=sum(r['positive_sessions'] for r in summaries),max_timed_fp32_loss_difference=max(r['max_loss_difference'] for r in pairs),
        exact_crossarm_learning_records=sum(r['exact_learning_record'] for r in pairs),bf16_failed_checks=len(failed),
        worst_bf16_parameter_difference=max(r['value'] for r in failed),bf16_parameter_limit=spec['bf16_sanity_budgets']['parameter_max_abs'],
        lowest_bf16_gradient_cosine=min(r['value'] for r in numerics if r['budget']=='gradient_cosine_min'),
        max_bf16_gradient_relative_l2=max(r['value'] for r in numerics if r['budget']=='gradient_relative_l2'),
        max_bf16_loss_difference=max(r['value'] for r in numerics if r['budget']=='loss_max_abs'),
        failed_parameter_tensor_counts=dict(Counter(r['worst_tensor'] for r in failed)),process_ids=pids,fragmented_sessions=0,
        numerical_tensors_recomputed=False,real_data_quality_established=False,bf16_performance_measured=False,experiment_rerun=False)
    save(OUT/'independent_audit.json',result);save(OUT/'independent_pairs.json',pairs);save(OUT/'trials_export.json',rows_all)
    save(OUT/'numerical_summary.json',budgets);save(OUT/'numerical_checks.json',numerics);save(OUT/'dispatch.json',profiles)
    save(OUT/'order_summary.json',order_summaries);save(OUT/'telemetry.json',telemetry)
    save(OUT/'frozen_audit_recheck_receipt.json',dict(status='PASS',exit_code=0,experiment_rerun=False,
        command='python -B scripts/review_optimized_execution.py results/optimized_execution_v1',record_type='Observed successful tool execution; separate independent arithmetic review saved here'))
    verify(protected)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
