"""Read-only reconstruction of session-level QKV confirmation evidence."""
import argparse
import json
import math
from pathlib import Path
import statistics as st

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def statistics(values,family):
    assert values and all(math.isfinite(x) for x in values)
    n=len(values);wins=sum(x>0 for x in values)
    p=sum(math.comb(n,k) for k in range(wins,n+1))/2**n
    return dict(sessions=n,positive_sessions=wins,session_savings_percent=values,
        median_saving_percent=st.median(values),median_interval_percent=[min(values),max(values)],
        marginal_interval_coverage=1-2/2**n,
        simultaneous_interval_coverage_lower_bound=max(0,1-family*2/2**n),
        one_sided_sign_p=p,bonferroni_p=min(1,p*family),directional_pass=min(1,p*family)<=.05)

def same_trajectory(a,b):
    return all(a[k]==b[k] for k in ('first_loss','steady_losses','final_model_sha256',
                                  'final_optimizer_sha256','final_rng'))

def audit_session(out,session,cfg,require_complete=True):
    import qkv_confirmation as c
    e=c.e;spec=cfg['spec'];parent=out/f'session_{session:02d}'
    if require_complete:
        completed=e.d.validate_complete(parent,cfg['fingerprint']);assert completed
        paths=list(parent.glob('seed_*/COMPLETE.json'))+list(parent.glob('seed_*/run_config.json'))
        paths+=list((parent/'launches').rglob('*.json'))
        assert set(completed['sha256'])=={str(p) for p in paths}
    launches=sorted((parent/'launches').glob('attempt_*'))
    assert launches
    for launch in launches:assert read(launch/'process.json')['uuid']
    # Even an aborted idle preflight is retained and conservatively marks fragmentation.
    fragmented=len(launches)!=1
    last=launches[-1]
    for name in ('preflight.json','telemetry_before.json','telemetry_after.json'):
        assert (last/name).is_file()
    if not cfg['smoke']:
        samples=read(last/'preflight.json');assert len(samples)==3
        assert all(s['available'] for s in samples)
        for i in range(len(samples[0]['gpus'])):
            assert st.median(float(s['gpus'][i]['utilization.gpu']) for s in samples)<=20
    result=[]
    for position,(seed,batch) in enumerate(c.schedule(spec,session)):
        observed=read(last/f'condition_{position:02d}.json')
        assert (observed['seed'],observed['batch'])==(seed,batch)
        target=parent/f'seed_{seed}_batch_{batch}';rc=read(target/'run_config.json')
        ci=position+session
        assert (rc['seed'],rc['batch'],rc['condition_index'],rc['root_fingerprint'])==(seed,batch,ci,cfg['fingerprint'])
        assert rc['fingerprint']==e.v.digest({k:v for k,v in rc.items() if k!='fingerprint'})
        marker=e.d.validate_complete(target,rc['fingerprint']);assert marker
        attempt=Path(marker['attempt']).resolve(strict=True);assert attempt.parent==target
        assert set(marker['sha256'])=={str(p) for p in attempt.rglob('*') if p.is_file()}
        probes=read(attempt/'capability.json');checks=read(attempt/'correctness.json')
        layers=2 if cfg['smoke'] else 6;eligible=[]
        assert list(probes)==list(checks)==spec['arms']
        for arm in spec['arms']:
            probe=probes[arm];assert probe['status']=='SUPPORTED'
            settings=probe['inside_context']
            assert settings==cfg['execution']
            assert settings['math_sdp'] and settings['deterministic'] and not settings['warn_only']
            assert not any(settings[k] for k in ('tf32_matmul','tf32_cudnn','flash_sdp','mem_efficient_sdp','cudnn_sdp'))
            assert probe['attention_operators'][e.MATH_OP]==layers
            assert all(probe['attention_operators'].get(op,0)==0 for op in e.frozen_backend.OPS.values() if op!=e.MATH_OP)
            assert probe['packed_shape']==[3,65,batch,24 if cfg['smoke'] else 192]
            assert probe['projection_split_counts']['aten::unbind']==(layers if arm=='qkv_unbind' else 0)
            assert probe['unbind_backward_nodes']==(layers if arm=='qkv_unbind' else 0)
            if arm=='math':assert probe['projection_split_counts']['aten::select']==3*layers
            assert probe['all_parameters_have_finite_full_gradients']
            check=checks[arm];metrics=check['metrics'];assert metrics
            assert {z['step'] for z in metrics}==set(range(spec['correctness_updates']))
            assert {z['kind'] for z in metrics}==set(spec['tolerances'])
            for z in metrics:
                value=z['max_scaled_error']
                ok=value is not None and math.isfinite(value) and value<=1
                assert z['pass_tolerance']==ok
            status='PASS' if check['rng_match'] and all(z['pass_tolerance'] for z in metrics) else 'FAIL_NUMERICAL'
            if not check['deterministic_repeat']:status='FAIL_DETERMINISM'
            assert check['status']==status
            if status=='PASS':eligible.append(arm)
        assert 'math' in eligible
        assert probes['math']['select_backward_nodes']-probes['qkv_unbind']['select_backward_nodes']==3*layers
        trials=read(attempt/'trials.json');assert len(trials)==len(eligible)*spec['rounds']
        gen=e.torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch)
        x=e.torch.randn(batch,3,32,32,generator=gen);y=e.torch.randint(0,10,(batch,),generator=gen)
        xhash=e.tensor_hash(x);yhash=e.tensor_hash(y);by={};index=0;initial=None
        for ri in range(spec['rounds']):
            for arm in e.order(eligible,ci,ri):
                row=trials[index]
                assert (row['seed'],row['batch'],row['round'],row['arm'],row['position'])==(seed,batch,ri,arm,index)
                assert row['input_sha256']==xhash and row['labels_sha256']==yhash
                if initial is None:initial=row['initial_sha256']
                assert row['initial_sha256']==initial and row['final_model_sha256']!=initial
                assert row['optimizer_age']==1+spec['steady_updates'] and row['all_gradients_finite']
                assert len(row['steady_losses'])==spec['steady_updates']
                assert all(math.isfinite(z) and z>0 for z in [row['first_loss'],*row['steady_losses']])
                assert row['steady_update_seconds']==row['steady_seconds']/spec['steady_updates']
                for k in ('first_seconds','steady_seconds','setup_seconds','warmup_restore_seconds'):
                    assert math.isfinite(row[k]) and row[k]>0
                for k in ('setup_memory','first_memory','steady_memory'):
                    mem=row[k];assert mem['peak']>=mem['baseline']>=0 and mem['incremental']==mem['peak']-mem['baseline']
                by[ri,arm]=row;index+=1
        deterministic=all(same_trajectory(by[0,arm],by[ri,arm]) for arm in eligible for ri in range(spec['rounds']))
        paired_exact='qkv_unbind' in eligible and all(same_trajectory(by[ri,'math'],by[ri,'qkv_unbind']) for ri in range(spec['rounds']))
        ratios=[];orders={'native_first':[],'candidate_first':[]}
        if 'qkv_unbind' in eligible:
            for ri in range(spec['rounds']):
                a,b=by[ri,'math'],by[ri,'qkv_unbind'];ratio=b['steady_seconds']/a['steady_seconds'];ratios.append(ratio)
                orders['native_first' if a['position']<b['position'] else 'candidate_first'].append(100*(1-ratio))
        result.append(dict(session=session,seed=seed,batch=batch,position=position,fragmented=fragmented,
            numerical_pass=len(eligible)==2,deterministic=deterministic,paired_exact=paired_exact,
            paired_ratios=ratios,saving_percent=100*(1-st.median(ratios)) if ratios else None,
            order_savings_percent=orders,
            timing_by_arm={arm:dict(steady_median_ms=1000*st.median(by[ri,arm]['steady_update_seconds'] for ri in range(spec['rounds'])),
                first_median_ms=1000*st.median(by[ri,arm]['first_seconds'] for ri in range(spec['rounds'])),
                setup_median_ms=1000*st.median(by[ri,arm]['setup_seconds'] for ri in range(spec['rounds'])),
                steady_peak_bytes=st.median(by[ri,arm]['steady_memory']['peak'] for ri in range(spec['rounds'])),
                steady_incremental_bytes=st.median(by[ri,arm]['steady_memory']['incremental'] for ri in range(spec['rounds']))) for arm in eligible},
            final_identity_by_arm={arm:{k:by[0,arm][k] for k in ('final_model_sha256','final_optimizer_sha256','final_rng')} for arm in eligible},
            trials=len(trials)))
    return result

def audit(out,write=False):
    import qkv_confirmation as c
    e=c.e;out=Path(out).resolve(strict=True);cfg=read(out/'configuration.json');spec=cfg['spec']
    assert cfg['synthetic'] and cfg['fingerprint']==e.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})
    assert cfg['sources']==c.sources()
    if not cfg['smoke']:assert cfg==read(c.FREEZE)
    if not write:
        completed=e.d.validate_complete(out,cfg['fingerprint']);assert completed
        paths=[out/p for p in ('configuration.json','audit.json','session_summary.json','REPORT.md')]
        paths+=list(out.glob('session_*/COMPLETE.json'))
        assert set(completed['sha256'])=={str(p) for p in paths}
    rows=[]
    for session in range(spec['sessions']):rows+=audit_session(out,session,cfg)
    family=len(spec['seeds'])*len(spec['batch_sizes']);conditions=[]
    for seed in spec['seeds']:
        for batch in spec['batch_sizes']:
            selected=[r for r in rows if (r['seed'],r['batch'])==(seed,batch)]
            assert len(selected)==spec['sessions']
            identities=[r['final_identity_by_arm'] for r in selected]
            across=all(z==identities[0] for z in identities)
            valid=across and all(r['numerical_pass'] and r['deterministic'] and r['paired_exact'] and not r['fragmented'] for r in selected)
            values=[r['saving_percent'] for r in selected]
            stats=statistics(values,family) if all(z is not None for z in values) else None
            conditions.append(dict(seed=seed,batch=batch,identity_across_sessions=across,valid_for_confirmation=valid,
                statistics=stats,confirmed=bool(not cfg['smoke'] and valid and stats and stats['directional_pass'])))
    confirmed=all(z['confirmed'] for z in conditions)
    trials=sum(r['trials'] for r in rows)
    result=dict(status='PASS',smoke=cfg['smoke'],sessions=spec['sessions'],conditions=conditions,
        trials=trials,timed_updates=trials*(1+spec['steady_updates']),discarded_warmup_updates=trials*spec['discarded_warmup_updates'],
        correctness_updates=len(rows)*5*spec['correctness_updates'],mechanism_updates=len(rows)*2,
        directional_saving_confirmed=confirmed,
        decision='smoke_only' if cfg['smoke'] else 'review_confirmed_operation_saving' if confirmed else 'confirmation_not_established',
        original_ten_percent_screen='FAILED_UNCHANGED',real_training_run=False,automatic_followup=False)
    if write:
        for name in ('audit.json','session_summary.json','REPORT.md'):assert not (out/name).exists()
        e.v.atomic_json(out/'session_summary.json',rows);e.v.atomic_json(out/'audit.json',result)
        lines=['# QKV confirmation results','',f'Decision: {result["decision"]}. Integrity: PASS.','',
            '| Seed | Batch | Session median saving | Median interval | Positive sessions | Adjusted p | Confirmed |',
            '|---|---:|---:|---|---:|---:|---|']
        for z in conditions:
            s=z['statistics']
            if s:
                lo,hi=s['median_interval_percent']
                lines.append(f'| {z["seed"]} | {z["batch"]} | {s["median_saving_percent"]:.3f}% | [{lo:.3f}, {hi:.3f}]% | {s["positive_sessions"]}/{s["sessions"]} | {s["bonferroni_p"]:.6f} | {z["confirmed"]} |')
            else:lines.append(f'| {z["seed"]} | {z["batch"]} | unavailable | unavailable | — | — | False |')
        lines+=['','Intervals use session extrema: 99.21875% marginal median coverage and >=96.875% simultaneous coverage for the full eight-session/four-condition study, assuming independent identically distributed continuous session effects. Smoke intervals do not have those coverages.',
            'Fresh processes share hardware; correlated drift limits inferential generalization. Review individual session values, order effects and telemetry.',
            'All trajectories and eligibility checks must pass; fragmented sessions withhold confirmation. Integrity PASS alone is not a speed or correctness conclusion.',
            'Original >=10% screen remains failed. This tests positive direction, not replication of the exact 5–7% magnitude or a new practical threshold.',
            'Artificial data, one model/hardware/backend policy. No complete training, accuracy or publication claim. No automatic follow-up.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert read(out/'audit.json')==result
        assert read(out/'session_summary.json')==rows
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    import qkv_confirmation as c
    c.e.m.configure();print(json.dumps(audit(args.output),indent=2))
