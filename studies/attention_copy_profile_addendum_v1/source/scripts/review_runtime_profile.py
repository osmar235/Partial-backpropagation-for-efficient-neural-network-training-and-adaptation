"""Read-only validation and attribution summaries for matched runtime profiles."""
import argparse
import json
import math
import statistics as st
from pathlib import Path
import torch
from review_stage14C import compare_csv

def audit(out,write=False):
    import runtime_profile as r
    out=Path(out);cfg=json.loads((out/'configuration.json').read_text());spec=cfg['spec'];steps=spec['profile_warmup_updates']+spec['profile_active_updates']
    assert cfg['synthetic'] and r.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    for p,h in cfg['sources'].items():assert r.d.sha(Path(p))==h,p
    if not cfg['smoke']:assert cfg==json.loads(r.FREEZE.read_text())
    summaries=[];phase_summary=[];conditions=0;measured=0;gpu_attributions=[];identities=[]
    for si,seed in enumerate(spec['seeds']):
        for bi,batch in enumerate(spec['batch_sizes']):
            arms=spec['arms'] if (si+bi)%2==0 else list(reversed(spec['arms']))
            for arm in arms:
                parent=out/f'seed_{seed}_batch_{batch}'/arm;rc=json.loads((parent/'run_config.json').read_text())
                assert rc['root_fingerprint']==cfg['fingerprint'] and (rc['seed'],rc['batch'],rc['arm'],rc['position'])==(seed,batch,arm,conditions)
                assert rc['fingerprint']==r.v.digest({k:v for k,v in rc.items() if k!='fingerprint'})
                marker=r.d.validate_complete(parent,rc['fingerprint']);assert marker
                attempt=Path(marker['attempt']);assert attempt.parent.resolve()==parent.resolve()
                assert set(marker['sha256'])=={str(p) for p in attempt.rglob('*') if p.is_file()}
                anchor=json.loads((attempt/'anchor.json').read_text());assert anchor['optimizer_age']==spec['burnin_updates']
                gen=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch)
                x=torch.randn(batch,3,32,32,generator=gen);y=torch.randint(0,10,(batch,),generator=gen)
                assert anchor['input_sha256']==r.tensor_hash(x) and anchor['labels_sha256']==r.tensor_hash(y)
                sequence=[]
                for ri in range(spec['rounds']):
                    modes=['plain','events'] if (conditions+ri)%2==0 else ['events','plain'];sequence.extend((ri,mode) for mode in modes)
                sequence.append((spec['rounds'],'profile'))
                regimes=json.loads((attempt/'regimes.json').read_text());assert len(regimes)==len(sequence)
                measurements={};all_phases=[];identity=None
                for record,(ri,mode) in zip(regimes,sequence):
                    assert (record['round'],record['mode'])==(ri,mode)
                    folder=Path(record['folder']);assert folder==attempt/f'{ri:02d}_{mode}'
                    rec=json.loads((folder/'measurement.json').read_text());measurements[ri,mode]=rec
                    assert rec['mode']==mode and rec['updates']==steps and rec['optimizer_age']==steps+spec['burnin_updates'] and rec['all_gradients_finite']
                    assert rec['wall_seconds']==record['wall_seconds'] and math.isfinite(rec['wall_seconds']) and rec['wall_seconds']>0
                    assert len(rec['losses'])==steps and all(math.isfinite(z) and z>0 for z in rec['losses'])
                    this={k:rec[k] for k in ('losses','final_model_sha256','final_optimizer_sha256','cpu_rng_sha256','cuda_rng_sha256')}
                    if identity is None:identity=this
                    assert this==identity and rec['final_model_sha256']!=anchor['model_sha256'] and rec['final_optimizer_sha256']!=anchor['optimizer_sha256']
                    if mode=='events':
                        phases=r.d.records(folder/'phases.csv');assert len(phases)==steps*len(r.PHASES)
                        for index,row in enumerate(phases):
                            assert int(row['step'])==index//len(r.PHASES) and row['phase']==r.PHASES[index%len(r.PHASES)]
                            assert math.isfinite(float(row['host_seconds'])) and float(row['host_seconds'])>=0
                            if cfg['device']=='cuda':assert math.isfinite(float(row['cuda_stream_ms'])) and float(row['cuda_stream_ms'])>=0
                            else:assert row['cuda_stream_ms']==''
                        all_phases.extend(phases)
                    elif mode=='profile':
                        ops=r.d.records(folder/'operators.csv');assert ops
                        assert set(r.PHASES)<=set(row['phase'] for row in ops)
                        assert any(row['forward_module'].startswith('mlp_') for row in ops)
                        assert any(row['forward_module'].startswith('attention_') for row in ops)
                        for row in ops:
                            assert row['phase'] in (*r.PHASES,'outside') and int(row['calls'])>0
                            assert math.isfinite(float(row['self_cpu_us'])) and math.isfinite(float(row['self_device_us']))
                            json.loads(row['input_shapes'])
                        trace=json.loads((folder/'trace.json').read_text(encoding='utf8'));assert trace['traceEvents']
                        assert rec['gpu_operator_attribution_available']==bool(rec['cuda_activity_requested'] and rec['cuda_events']>0 and sum(float(row['self_device_us']) for row in ops)>0)
                        gpu_attributions.append(rec['gpu_operator_attribution_available'])
                    measured+=steps
                identities.append(dict(seed=seed,batch=batch,arm=arm,**identity))
                plain=[measurements[ri,'plain']['wall_seconds']/steps for ri in range(spec['rounds'])]
                event=[measurements[ri,'events']['wall_seconds']/steps for ri in range(spec['rounds'])]
                profile=measurements[spec['rounds'],'profile'];prof_ops=r.d.records(attempt/f'{spec["rounds"]:02d}_profile'/'operators.csv')
                summaries.append(dict(seed=seed,batch=batch,arm=arm,plain_median_update_ms=1000*st.median(plain),
                    events_median_update_ms=1000*st.median(event),median_paired_events_overhead_percent=100*(st.median(b/a for a,b in zip(plain,event))-1),
                    profile_session_ms=1000*profile['wall_seconds'],profile_session_to_plain_ratio=profile['wall_seconds']/(steps*st.median(plain)),
                    profiled_active_updates=spec['profile_active_updates'],cuda_activity_requested=profile['cuda_activity_requested'],
                    cuda_events=profile['cuda_events'],gpu_operator_attribution_available=profile['gpu_operator_attribution_available']))
                for phase in r.PHASES:
                    data=[row for row in all_phases if row['phase']==phase]
                    phase_summary.append(dict(seed=seed,batch=batch,arm=arm,phase=phase,samples=len(data),
                        median_host_ms=1000*st.median(float(row['host_seconds']) for row in data),
                        median_cuda_stream_ms=st.median(float(row['cuda_stream_ms']) for row in data) if cfg['device']=='cuda' else None,
                        profile_self_cpu_us=sum(float(row['self_cpu_us']) for row in prof_ops if row['phase']==phase),
                        profile_self_device_us=sum(float(row['self_device_us']) for row in prof_ops if row['phase']==phase) if profile['gpu_operator_attribution_available'] else None))
                conditions+=1
    result=dict(status='PASS',synthetic=True,smoke=cfg['smoke'],conditions=conditions,measured_artificial_updates=measured,
        burnin_artificial_updates=conditions*spec['burnin_updates'],all_instrumented_losses_states_and_rng_match=True,
        cuda_operator_attribution_conditions=sum(gpu_attributions),total_profile_conditions=conditions,
        decision='diagnostic_only_review_before_design',real_checkpoint_or_data_used=False,efficacy_measured=False,
        complete_adaptation_measured=False,automatic_followup=False)
    if write:
        r.v.write_csv(out/'summary.csv',summaries);r.v.write_csv(out/'phase_summary.csv',phase_summary);r.v.atomic_json(out/'audit.json',result)
        lines=['# Runtime attribution diagnostic','','```json',json.dumps(result,indent=2),'```','',
            '| Seed | Batch | Arm | Plain median update ms | Event overhead | GPU operator attribution |',
            '|---|---:|---|---:|---:|---|']
        for row in summaries:lines.append(f"| {row['seed']} | {row['batch']} | {row['arm']} | {row['plain_median_update_ms']:.3f} | {row['median_paired_events_overhead_percent']:.2f}% | {row['gpu_operator_attribution_available']} |")
        lines+=['','Plain timings use the prior uninstrumented update function. Event timings are perturbed stream intervals including host launch gaps; host phases are not GPU compute durations.',
            'Profiler sessions include initialization/collection/teardown and one warmup update; only active updates appear in operator tables. Never rank implementations by profiler session time.',
            'Operator self times and nested module ranges must not be added together. CPU and GPU totals overlap. Missing GPU attribution is unavailable, never zero GPU cost.',
            'Attention/MLP labels attribute forward work only. Backward operator names/shapes are diagnostic evidence and do not directly isolate module weight-gradient savings.',
            'Artificial workload and fixed historical FP32/math-attention settings only. No speed/quality pass rule, no real training, no automatic next method. Review trace files and phase tables before choosing a design.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text())==result
        compare_csv(r.d.records(out/'summary.csv'),summaries);compare_csv(r.d.records(out/'phase_summary.csv'),phase_summary)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    import stage10_models as m
    m.configure();print(json.dumps(audit(args.output),indent=2))
