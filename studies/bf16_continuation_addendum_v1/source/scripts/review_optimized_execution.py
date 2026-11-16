"""Read-only audit; sanity-qualified mixed precision is never quality-qualified."""
import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import median

def audit(out,write=False):
    import optimized_execution as e
    out=Path(out).resolve(strict=True);cfg=e.read(out/'configuration.json');spec=cfg['spec']
    assert cfg['fingerprint']==e.q.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})
    for p,h in cfg['sources'].items():assert e.q.d.sha(p)==h,p
    assert e.q.d.sha(e.CAPABILITY)==cfg['capability_sha256']
    if not cfg['smoke']:assert cfg==e.read(e.FREEZE)
    if (out/'COMPLETE.json').exists():assert e.q.d.validate_complete(out,cfg['fingerprint'])
    rows_all=[];eligibility=[];pairs=[];fragmented=[];pids=[];failed_trials=[];blocked=[];repeats={}
    for session in range(spec['sessions']):
        parent=out/f'session_{session:02d}';assert e.q.d.validate_complete(parent,cfg['fingerprint'])
        launches=sorted((parent/'launches').glob('attempt_*'));processes=[e.read(p/'process.json')['pid'] for p in launches]
        pids.append(processes[-1])
        if len(launches)!=1:fragmented.append(session)
        for ci,(seed,batch) in enumerate(itertools.product(spec['seeds'],spec['batch_sizes'])):
            folder=parent/f'seed_{seed}_batch_{batch}';rc=e.read(folder/'run_config.json')
            assert (rc['root_fingerprint'],rc['session'],rc['seed'],rc['batch'],rc['condition_index'])==(cfg['fingerprint'],session,seed,batch,ci)
            assert rc['fingerprint']==e.q.v.digest({k:v for k,v in rc.items() if k!='fingerprint'})
            marker=e.q.d.validate_complete(folder,rc['fingerprint']);assert marker
            attempt=Path(marker['attempt']).resolve();assert attempt.parent==folder
            assert set(marker['sha256'])=={str(p) for p in attempt.rglob('*') if p.is_file()}
            assert e.read(attempt/'process.json')['pid'] in processes
            if len(list(folder.glob('attempt_*')))!=1 and session not in fragmented:fragmented.append(session)
            probes=e.read(attempt/'capability.json');checks=e.read(attempt/'correctness.json')
            assert list(probes)==spec['arms'] and list(checks)==spec['arms'];eligible=[]
            for arm in spec['arms']:
                probe=probes[arm];check=checks[arm];precision=e.properties(arm)[1];backend=e.properties(arm)[2]
                if probe['status']!='SUPPORTED':
                    assert probe['status'] in ('UNSUPPORTED','FAIL_NONFINITE') and check['status']==probe['status']
                else:
                    flags=probe['policy'];assert not flags['tf32_matmul'] and not flags['tf32_cudnn'] and not flags['cudnn_benchmark']
                    assert not any(flags[k] for k in ('fp16_reduced_reduction','bf16_reduced_reduction','math_reduced_reduction','warn_only'))
                    assert flags['deterministic']==(backend=='math') and flags['cudnn_deterministic']==(backend=='math')
                    assert sum(probe['dispatch'].values())==(2 if cfg['smoke'] else 6)
                    if backend=='math':assert probe['dispatch']['math']==sum(probe['dispatch'].values())
                    assert probe['output_dtype']==precision and probe['parameter_dtype']=='float32' and probe['all_gradients_finite']
                    if check['status']!='FAIL_NONFINITE':
                        if precision=='fp32':
                            metrics=check['metrics'];assert metrics and {m['step'] for m in metrics}==set(range(spec['correctness_updates']))
                            assert {m['kind'] for m in metrics}==set(spec['fp32_tolerances'])
                            for m in metrics:
                                ratio=m['max_scaled_error'];ok=ratio is not None and math.isfinite(ratio) and ratio<=1
                                assert m['pass_tolerance']==ok
                            status='PASS' if check['rng_match'] and all(m['pass_tolerance'] for m in metrics) else 'FAIL_NUMERICAL'
                        else:
                            assert check['quality_equivalence_established'] is False
                            assert len(check['checks'])==len(spec['bf16_sanity_budgets'])*spec['correctness_updates']
                            assert {(c['step'],c['budget']) for c in check['checks']}==set(itertools.product(range(spec['correctness_updates']),spec['bf16_sanity_budgets']))
                            metrics={(m['step'],m['kind']):m for m in check['metrics']}
                            for c in check['checks']:
                                assert c['limit']==spec['bf16_sanity_budgets'][c['budget']]
                                key=c['budget']
                                if key.endswith('_cosine_min'):value=metrics[c['step'],key[:-11]]['cosine'];ok=value>=c['limit']
                                elif key.endswith('_relative_l2'):value=metrics[c['step'],key[:-12]]['relative_l2'];ok=value<=c['limit']
                                else:value=metrics[c['step'],key[:-8]]['max_abs'];ok=value<=c['limit']
                                assert math.isfinite(value) and c['value']==value and c['pass_budget']==ok
                            status='PASS_SANITY_ONLY' if all(c['pass_budget'] for c in check['checks']) else 'FAIL_NUMERICAL_SANITY'
                        if backend=='math' and not check['deterministic_repeat']:status='FAIL_DETERMINISM'
                        assert check['status']==status
                if check['status'] in ('PASS','PASS_SANITY_ONLY'):eligible.append(arm)
                eligibility.append(dict(session=session,seed=seed,batch=batch,arm=arm,status=check['status'],dispatch=probe.get('dispatch'),repeat_exact=check.get('deterministic_repeat')))
            timing=e.read(attempt/'timing_status.json');is_blocked=spec['required_baseline'] not in eligible
            assert timing['required_baseline_blocked']==is_blocked
            if is_blocked:blocked.append(dict(session=session,seed=seed,batch=batch))
            failures=timing['failures'];assert len({f['arm'] for f in failures})==len(failures)
            failed_trials.extend(dict(session=session,seed=seed,batch=batch,**f) for f in failures)
            failure_map={f['arm']:f for f in failures};rows=e.read(attempt/'trials.json')
            planned=[]
            if not is_blocked:
                for ri in range(spec['rounds']):
                    for pos,arm in enumerate(e.order(spec,session,ci,ri)):
                        if arm in eligible and (arm not in failure_map or ri<failure_map[arm]['round']):planned.append((ri,pos,arm))
                for arm,f in failure_map.items():
                    assert arm in eligible and e.order(spec,session,ci,f['round'])[f['position']]==arm
            else:assert not failures
            assert [(r['round'],r['position'],r['arm']) for r in rows]==planned
            if rows:assert len({r['initial_sha256'] for r in rows})==1
            x,y=e.prior.inputs(seed,batch,spec,e.torch.device('cpu'))
            for r in rows:
                assert (r['seed'],r['batch'],r['optimizer_age'])==(seed,batch,1+spec['steady_updates'])
                assert r['input_sha256']==e.q.tensor_hash(x) and r['labels_sha256']==e.q.tensor_hash(y)
                assert len(r['steady_losses'])==spec['steady_updates'] and r['all_gradients_finite']
                assert all(math.isfinite(v) for v in [r['startup_loss'],r['first_loss'],*r['steady_losses']])
                assert all(math.isfinite(r[k]) and r[k]>0 for k in ('setup_seconds','startup_update_seconds','warmup_restore_seconds','first_seconds','steady_seconds'))
                assert math.isclose(r['steady_update_seconds'],r['steady_seconds']/spec['steady_updates'],abs_tol=1e-12)
                state=[r[k] for k in ('final_model_sha256','final_optimizer_sha256','final_rng','first_loss','steady_losses')]
                key=(seed,batch,r['arm'])
                if e.properties(r['arm'])[2]=='math' and key in repeats:assert repeats[key]==state
                repeats[key]=state;rows_all.append(dict(session=session,**r))
            comparisons=[(spec['required_baseline'],arm) for arm in spec['arms'] if arm!=spec['required_baseline']]
            comparisons += [('native_'+suffix,'reference_'+suffix) for suffix in ('fp32_math','bf16_math','bf16_auto')]
            for base,cand in comparisons:
                if is_blocked or base not in eligible or cand not in eligible or base in failure_map or cand in failure_map:continue
                for ri in range(spec['rounds']):
                    a=next(r for r in rows if r['round']==ri and r['arm']==base);b=next(r for r in rows if r['round']==ri and r['arm']==cand)
                    pairs.append(dict(session=session,seed=seed,batch=batch,round=ri,baseline=base,candidate=cand,
                        saving_percent=100*(a['steady_seconds']-b['steady_seconds'])/a['steady_seconds'],
                        baseline_ms=1000*a['steady_update_seconds'],candidate_ms=1000*b['steady_update_seconds']))
    assert len(set(pids))==spec['sessions']
    summary=[]
    for seed,batch,base,cand in sorted({(r['seed'],r['batch'],r['baseline'],r['candidate']) for r in pairs}):
        group=[r for r in pairs if (r['seed'],r['batch'],r['baseline'],r['candidate'])==(seed,batch,base,cand)]
        values=[]
        for session in range(spec['sessions']):
            samples=[r['saving_percent'] for r in group if r['session']==session]
            if samples:assert len(samples)==spec['rounds'];values.append(dict(session=session,median_saving_percent=median(samples)))
        nums=[r['median_saving_percent'] for r in values]
        summary.append(dict(seed=seed,batch=batch,baseline=base,candidate=cand,session_values=values,
            median_saving_percent=median(nums),min_session_percent=min(nums),max_session_percent=max(nums),
            positive_sessions=sum(v>0 for v in nums),all_sessions_eligible=len(nums)==spec['sessions']))
    result=dict(status='PASS',smoke=cfg['smoke'],successful_trials=len(rows_all),failed_timed_trials=len(failed_trials),
        measured_postrestore_updates=(len(rows_all)+len(failed_trials))*(1+spec['steady_updates']),
        discarded_startup_updates=len(rows_all)+len(failed_trials),discarded_warmup_updates=(len(rows_all)+len(failed_trials))*spec['discarded_warmup_updates'],
        blocked_conditions=blocked,fragmented_sessions=sorted(fragmented),process_ids=pids,
        compiler_capability=cfg['compiler_capability']['compile_status'],compiled_performance_measured=False,
        eligibility_counts={status:sum(r['status']==status for r in eligibility) for status in sorted({r['status'] for r in eligibility})},
        summary=summary,real_data_quality_established=False,complete_workflow_timing=False)
    if write:
        e.save(out/'audit.json',result);e.save(out/'eligibility.json',eligibility);e.save(out/'paired_results.json',pairs)
        e.save(out/'trials_export.json',rows_all);e.save(out/'failed_trials.json',failed_trials)
        lines=['# Optimized execution feasibility','',f'Audit PASS; {len(rows_all)} successful trials, {len(failed_trials)} failed timed trials.',
            'Numerical sanity is not real-data quality equivalence. Compiled performance was not measured.',
            f'Compiler capability: {result["compiler_capability"]}. Fragmented sessions: {sorted(fragmented)}.','',
            '| Seed | Batch | Baseline | Candidate | Median saving | Eligible sessions |','|---|---|---|---|---:|---:|']
        for r in summary:lines.append(f'| {r["seed"]} | {r["batch"]} | {r["baseline"]} | {r["candidate"]} | {r["median_saving_percent"]:.3f}% | {len(r["session_values"])}/{spec["sessions"]} |')
        lines+=['','No new method, automatic follow-up or real-data performance claim. Review every failed/unsupported condition and dispatch record before selecting another study.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    import optimized_execution as e
    e.q.m.configure();print(json.dumps(audit(args.output),indent=2))
