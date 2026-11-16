"""Reconstruct QKV layout numerical eligibility, mechanism checks and paired costs."""
import argparse
import json
import math
from pathlib import Path
import statistics as st
import torch
from review_stage14C import compare_csv

def audit(out,write=False):
    import qkv_layout as e
    out=Path(out).resolve(strict=True);cfg=json.loads((out/'configuration.json').read_text());spec=cfg['spec']
    assert cfg['synthetic'] and e.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    for p,h in cfg['sources'].items():assert e.d.sha(p)==h,p
    if not cfg['smoke']:assert cfg==json.loads(e.FREEZE.read_text())
    summary=[];ci=0;trials_count=0;correctness_steps=0;probe_attempts=0
    for seed in spec['seeds']:
        for batch in spec['batch_sizes']:
            parent=out/f'seed_{seed}_batch_{batch}';rc=json.loads((parent/'run_config.json').read_text())
            assert (rc['seed'],rc['batch'],rc['condition_index'],rc['root_fingerprint'])==(seed,batch,ci,cfg['fingerprint'])
            assert rc['fingerprint']==e.v.digest({k:v for k,v in rc.items() if k!='fingerprint'})
            marker=e.d.validate_complete(parent,rc['fingerprint']);assert marker
            attempt=Path(marker['attempt']);assert attempt.parent==parent
            assert set(marker['sha256'])=={str(p) for p in attempt.rglob('*') if p.is_file()}
            probes=json.loads((attempt/'capability.json').read_text());checks=json.loads((attempt/'correctness.json').read_text())
            assert list(probes)==spec['arms'];eligible=[];correctness_steps+=spec['correctness_updates']
            layers=2 if cfg['smoke'] else 6
            for arm in spec['arms']:
                probe=probes[arm];assert probe['status']=='SUPPORTED';probe_attempts+=1
                policy=probe['inside_context'];assert policy['deterministic'] and not policy['warn_only'] and not policy['tf32_matmul'] and not policy['tf32_cudnn']
                assert policy['math_sdp'] and not any(policy[k] for k in ('flash_sdp','mem_efficient_sdp','cudnn_sdp'))
                assert probe['attention_operators'][e.MATH_OP]==layers
                assert all(probe['attention_operators'].get(op,0)==0 for op in e.frozen_backend.OPS.values() if op!=e.MATH_OP)
                assert probe['packed_shape']==[3,65,batch,24 if cfg['smoke'] else 192]
                assert probe['projection_split_counts']['aten::unbind']==(layers if arm=='qkv_unbind' else 0)
                if arm=='math':assert probe['projection_split_counts']['aten::select']==3*layers
                assert probe['unbind_backward_nodes']==(layers if arm=='qkv_unbind' else 0)
                assert probe['all_parameters_have_finite_full_gradients']
                check=checks[arm];metrics=check['metrics'];assert metrics
                assert {z['step'] for z in metrics}==set(range(spec['correctness_updates']))
                assert {z['kind'] for z in metrics}==set(spec['tolerances'])
                for z in metrics:
                    ok=z['max_scaled_error'] is not None and math.isfinite(z['max_scaled_error']) and z['max_scaled_error']<=1
                    assert z['pass_tolerance']==ok
                status='PASS' if check['rng_match'] and all(z['pass_tolerance'] for z in metrics) else 'FAIL_NUMERICAL'
                if not check['deterministic_repeat']:status='FAIL_DETERMINISM'
                assert check['status']==status
                correctness_steps+=2*spec['correctness_updates']
                if status=='PASS':eligible.append(arm)
            assert 'math' in eligible
            assert probes['math']['select_backward_nodes']-probes['qkv_unbind']['select_backward_nodes']==3*layers
            trials=json.loads((attempt/'trials.json').read_text());assert len(trials)==len(eligible)*spec['rounds']
            gen=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch)
            x=torch.randn(batch,3,32,32,generator=gen);y=torch.randint(0,10,(batch,),generator=gen)
            xhash=e.tensor_hash(x);yhash=e.tensor_hash(y);initial=None;position=0;by={}
            for ri in range(spec['rounds']):
                for arm in e.order(eligible,ci,ri):
                    row=trials[position]
                    assert (row['seed'],row['batch'],row['round'],row['arm'],row['position'])==(seed,batch,ri,arm,position)
                    assert row['input_sha256']==xhash and row['labels_sha256']==yhash
                    if initial is None:initial=row['initial_sha256']
                    assert row['initial_sha256']==initial and row['final_model_sha256']!=initial
                    assert row['optimizer_age']==1+spec['steady_updates'] and row['all_gradients_finite']
                    assert len(row['steady_losses'])==spec['steady_updates']
                    assert all(math.isfinite(z) and z>0 for z in [row['first_loss'],*row['steady_losses']])
                    assert row['steady_update_seconds']==row['steady_seconds']/spec['steady_updates']
                    for k in ('first_seconds','steady_seconds','setup_seconds','warmup_restore_seconds'):assert math.isfinite(row[k]) and row[k]>0
                    for k in ('setup_memory','first_memory','steady_memory'):
                        mem=row[k];assert mem['peak']>=mem['baseline']>=0 and mem['incremental']==mem['peak']-mem['baseline']
                    by[ri,arm]=row;position+=1
            # Repeated trajectories must be deterministic; all backend dropout streams agree.
            for arm in eligible:
                first=by[0,arm]
                for ri in range(spec['rounds']):
                    for k in ('first_loss','steady_losses','final_model_sha256','final_optimizer_sha256','final_rng'):assert by[ri,arm][k]==first[k]
                    assert by[ri,arm]['final_rng']==by[ri,'math']['final_rng']
                    tol=spec['tolerances']['loss'];a=by[ri,'math']['first_loss'];b=by[ri,arm]['first_loss']
                    assert abs(a-b)<=tol['atol']+tol['rtol']*abs(a)
            for arm in spec['arms']:
                base=dict(seed=seed,batch=batch,arm=arm,capability=probes[arm]['status'],correctness=checks.get(arm,{}).get('status','NOT_RUN'))
                if arm not in eligible:
                    summary.append(dict(**base,steady_median_ms=None,paired_median_saving_percent=None,passing_rounds=None,
                        median_setup_ms=None,median_first_ms=None,median_peak_bytes=None,median_incremental_bytes=None,cost_pass=False))
                    continue
                values=[by[ri,arm] for ri in range(spec['rounds'])]
                ratios=[by[ri,arm]['steady_update_seconds']/by[ri,'math']['steady_update_seconds'] for ri in range(spec['rounds'])]
                saving=1-st.median(ratios)
                summary.append(dict(**base,steady_median_ms=1000*st.median(z['steady_update_seconds'] for z in values),
                    paired_median_saving_percent=100*saving,passing_rounds=sum(z<=1-spec['required_saving'] for z in ratios),
                    median_setup_ms=1000*st.median(z['setup_seconds'] for z in values),median_first_ms=1000*st.median(z['first_seconds'] for z in values),
                    median_peak_bytes=st.median(z['steady_memory']['peak'] for z in values),median_incremental_bytes=st.median(z['steady_memory']['incremental'] for z in values),
                    cost_pass=arm!='math' and saving>=spec['required_saving']))
            trials_count+=len(trials);ci+=1
    candidates=[arm for arm in spec['arms'] if arm!='math' and all(z['cost_pass'] for z in summary if z['arm']==arm)]
    result=dict(status='PASS',synthetic=True,smoke=cfg['smoke'],conditions=ci,trials=trials_count,
        timed_updates=trials_count*(1+spec['steady_updates']),discarded_warmup_updates=trials_count*spec['discarded_warmup_updates'],
        correctness_updates=correctness_steps,capability_probe_attempts=probe_attempts,
        qualifying_layouts=[] if cfg['smoke'] else candidates,
        decision='smoke_only' if cfg['smoke'] else 'review_qualifying_layout' if candidates else 'layout_does_not_qualify',
        full_gradients=True,precision='FP32',strict_determinism=True,real_data_or_checkpoint_used=False,
        real_quality_measured=False,complete_adaptation_measured=False,automatic_followup=False)
    if write:
        e.v.write_csv(out/'summary.csv',summary);e.v.atomic_json(out/'audit.json',result)
        lines=['# QKV unbind layout screen','','```json',json.dumps(result,indent=2),'```','',
            '| Seed | Batch | Implementation | Capability | Correctness | Update ms | Paired saving | Cost pass |',
            '|---|---:|---|---|---|---:|---:|---|']
        for z in summary:
            ms='—' if z['steady_median_ms'] is None else f"{z['steady_median_ms']:.3f}"
            saving='—' if z['paired_median_saving_percent'] is None else f"{z['paired_median_saving_percent']:.2f}%"
            lines.append(f"| {z['seed']} | {z['batch']} | {z['arm']} | {z['capability']} | {z['correctness']} | {ms} | {saving} | {z['cost_pass']} |")
        lines+=['','Only numerically validated implementations are timed. Both use the same forced math attention backend; the QKV split graph is checked outside timing.',
            'All parameters train; FP32, TF32-off, unchanged dropout and strict deterministic settings. Numerical tolerance is not bitwise equivalence across different implementations.',
            'Profiler used only outside timing to verify dispatch. Timing uses synchronized uninstrumented updates. Setup/first-update costs and allocated memory are reported separately.',
            'Requires >=10% median paired steady-update saving in every seed/batch condition. Failed arms never qualify. Smoke results never qualify an implementation.',
            'Artificial operation screen only: no real-data accuracy, complete adaptation or novelty result. Preserve split/backward mechanism records in capability.json and numerical metrics in correctness.json.',
            'Send the complete result folder for review before any real training or further layout change.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert e.d.validate_complete(out,cfg['fingerprint'])
        assert json.loads((out/'audit.json').read_text())==result
        compare_csv(e.d.records(out/'summary.csv'),summary)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    import stage10_models as m
    m.configure();print(json.dumps(audit(args.output),indent=2))
