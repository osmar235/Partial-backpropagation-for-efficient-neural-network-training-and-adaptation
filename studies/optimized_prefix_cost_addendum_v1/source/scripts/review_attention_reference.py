"""Read-only reconstruction of eligibility, planned order and paired timings."""
import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import median

def audit(out,write=False):
    import attention_reference as e
    out=Path(out).resolve(strict=True);cfg=e.read(out/'configuration.json');spec=cfg['spec']
    if (out/'COMPLETE.json').exists():assert e.q.d.validate_complete(out,cfg['fingerprint'])
    assert e.q.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    for name,digest in cfg['sources'].items():assert e.q.d.sha(name)==digest,name
    if not cfg['smoke']:assert cfg==e.read(e.FREEZE)
    pairs=[];eligibility=[];total=0;fragmented=[];pids=[];states={}
    for session in range(spec['sessions']):
        parent=out/f'session_{session:02d}'
        assert e.q.d.validate_complete(parent,cfg['fingerprint'])
        launches=list((parent/'launches').glob('attempt_*'))
        processes=[e.read(p/'process.json')['pid'] for p in launches]
        pids.append(processes[-1])
        if len(launches)!=1:fragmented.append(session)
        for ci,(seed,batch) in enumerate(itertools.product(spec['seeds'],spec['batch_sizes'])):
            condition=parent/f'seed_{seed}_batch_{batch}';rc=e.read(condition/'run_config.json')
            assert (rc['seed'],rc['batch'],rc['condition_index'],rc['session'],rc['root_fingerprint'])==(seed,batch,ci,session,cfg['fingerprint'])
            assert rc['fingerprint']==e.q.v.digest({k:v for k,v in rc.items() if k!='fingerprint'})
            marker=e.q.d.validate_complete(condition,rc['fingerprint']);assert marker
            attempt=Path(marker['attempt']).resolve();assert attempt.parent==condition
            assert set(marker['sha256'])=={str(p) for p in attempt.rglob('*') if p.is_file()}
            assert e.read(attempt/'process.json')['pid'] in processes
            if len(list(condition.glob('attempt_*')))!=1 and session not in fragmented:fragmented.append(session)
            checks=e.read(attempt/'correctness.json');probes=e.read(attempt/'mechanism.json')
            assert list(checks)==spec['arms'] and list(probes)==spec['arms']
            eligible=[];layers=2 if cfg['smoke'] else 6
            expected_metrics={(m['step'],m['kind'],m['name']) for m in checks['math']['metrics']}
            for arm in spec['arms']:
                check=checks[arm];metrics=check['metrics'];assert metrics
                assert {(m['step'],m['kind'],m['name']) for m in metrics}==expected_metrics
                assert {m['step'] for m in metrics}==set(range(spec['correctness_updates']))
                assert {m['kind'] for m in metrics}==set(spec['tolerances'])
                for m in metrics:
                    ratio=m['max_scaled_error'];ok=ratio is not None and math.isfinite(ratio) and ratio<=1
                    assert m['pass_tolerance']==ok
                status='PASS' if check['rng_match'] and all(m['pass_tolerance'] for m in metrics) else 'FAIL_NUMERICAL'
                if not check['deterministic_repeat']:status='FAIL_DETERMINISM'
                assert check['status']==status
                if status=='PASS':eligible.append(arm)
                eligibility.append(dict(session=session,seed=seed,batch=batch,arm=arm,status=status,
                    failures=[m for m in metrics if not m['pass_tolerance']]))
                probe=probes[arm];policy=probe['policy']
                assert policy['deterministic'] and not policy['warn_only'] and policy['math_sdp']
                assert not any(policy[k] for k in ('tf32_matmul','tf32_cudnn','flash_sdp','mem_efficient_sdp','cudnn_sdp'))
                assert probe['attention_operators'][e.q.MATH_OP]==layers
                assert all(probe['attention_operators'].get(op,0)==0 for op in e.q.frozen_backend.OPS.values() if op!=e.q.MATH_OP)
                assert probe['unbind_backward_nodes']==(0 if arm=='math' else layers)
                assert probe['full_finite_gradients']
            assert 'math' in eligible
            rows=e.read(attempt/'trials.json');total+=len(rows)
            planned=[(ri,pos,arm) for ri in range(spec['rounds']) for pos,arm in enumerate(e.order(spec,session,ci,ri)) if arm in eligible]
            assert [(r['round'],r['position'],r['arm']) for r in rows]==planned
            x,y=e.inputs(seed,batch,spec,e.torch.device('cpu'))
            assert len({r['initial_sha256'] for r in rows})==1
            for row in rows:
                assert (row['seed'],row['batch'])==(seed,batch)
                assert row['input_sha256']==e.q.tensor_hash(x) and row['labels_sha256']==e.q.tensor_hash(y)
                assert row['optimizer_age']==1+spec['steady_updates'] and row['all_gradients_finite']
                assert len(row['steady_losses'])==spec['steady_updates']
                assert all(math.isfinite(v) for v in [row['first_loss'],*row['steady_losses']])
                assert all(math.isfinite(row[k]) and row[k]>0 for k in ('steady_seconds','first_seconds','setup_seconds','warmup_restore_seconds'))
                assert math.isclose(row['steady_update_seconds'],row['steady_seconds']/spec['steady_updates'],abs_tol=1e-12)
                key=(seed,batch,row['arm'])
                state=[row[k] for k in ('final_model_sha256','final_optimizer_sha256','final_rng','first_loss','steady_losses')]
                if key in states:assert states[key]==state,'Within-arm repeated trajectory changed'
                else:states[key]=state
            for baseline,candidate in [('math','qkv_unbind'),('math','timm_reference'),('timm_reference','qkv_unbind')]:
                if baseline not in eligible or candidate not in eligible:continue
                for ri in range(spec['rounds']):
                    a=next(r for r in rows if r['arm']==baseline and r['round']==ri)
                    b=next(r for r in rows if r['arm']==candidate and r['round']==ri)
                    pairs.append(dict(session=session,seed=seed,batch=batch,round=ri,baseline=baseline,candidate=candidate,
                        baseline_ms=1000*a['steady_update_seconds'],candidate_ms=1000*b['steady_update_seconds'],
                        saving_percent=100*(1-b['steady_update_seconds']/a['steady_update_seconds']),
                        exact_timed_learning_records=all(a[k]==b[k] for k in ('first_loss','steady_losses','final_model_sha256','final_optimizer_sha256','final_rng'))))
    assert len(set(pids))==spec['sessions'],'Sessions must have distinct processes'
    grouped=[]
    keys=sorted({(r['seed'],r['batch'],r['baseline'],r['candidate']) for r in pairs})
    for seed,batch,baseline,candidate in keys:
        session_values=[]
        for session in range(spec['sessions']):
            values=[r['saving_percent'] for r in pairs if (r['seed'],r['batch'],r['baseline'],r['candidate'],r['session'])==(seed,batch,baseline,candidate,session)]
            if values:assert len(values)==spec['rounds'];session_values.append(dict(session=session,median_saving_percent=median(values)))
        vals=[r['median_saving_percent'] for r in session_values]
        grouped.append(dict(seed=seed,batch=batch,baseline=baseline,candidate=candidate,
            session_values=session_values,all_sessions_eligible=len(vals)==spec['sessions'],median_saving_percent=median(vals),
            min_session_percent=min(vals),max_session_percent=max(vals),positive_sessions=sum(v>0 for v in vals)))
    result=dict(status='PASS',smoke=cfg['smoke'],trials=total,measured_updates=total*(1+spec['steady_updates']),
        discarded_warmup_updates=total*spec['discarded_warmup_updates'],fragmented_sessions=sorted(fragmented),
        process_ids=pids,numerically_ineligible_conditions=sum(r['status']!='PASS' for r in eligibility),
        summary=grouped,meaning='Audit pass is not a scientific superiority or quality claim.')
    if write:
        e.save(out/'audit.json',result);e.save(out/'paired_results.json',pairs);e.save(out/'eligibility.json',eligibility)
        lines=['# Competitive attention screen','',f'Audit PASS; {total} eligible trials. Numerical failures: {result["numerically_ineligible_conditions"]}.',
               f'Fragmented sessions: {sorted(fragmented)}. Results are descriptive and require review.','',
               '| Seed | Batch | Baseline | Candidate | Median session saving | Eligible sessions |','|---|---|---|---|---:|---:|']
        for r in grouped:lines.append(f'| {r["seed"]} | {r["batch"]} | {r["baseline"]} | {r["candidate"]} | {r["median_saving_percent"]:.3f}% | {len(r["session_values"])}/{spec["sessions"]} |')
        lines+=['','Positive saving favors the candidate. Only numerically eligible arms are timed.',
                'Artificial full-model update cost; not real-data accuracy, full-workflow cost, novel unbind, or latest-timm/optimized-execution superiority.',
                'Do not treat incomplete eligibility or fragmented sessions as a completed independent confirmation.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    import attention_reference as e
    e.q.m.configure();print(json.dumps(audit(args.output),indent=2))
