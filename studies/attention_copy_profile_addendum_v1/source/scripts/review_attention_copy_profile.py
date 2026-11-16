"""Saved-data audit; profiled device attribution is distinct from uninstrumented time."""
import argparse
import math
from pathlib import Path
from statistics import median
import attention_copy_profile as p

COPY_OPS={'aten::copy_','aten::clone','aten::contiguous','aten::_to_copy'}


def operator_summary(rows,measurement):
    cpu=sum(r['self_cpu_us'] for r in rows);gpu=sum(r['self_device_us'] for r in rows)
    compatible=[r for r in rows if r['forward_module'].startswith('compatibility_copy_')]
    copies=[r for r in rows if r['operator'] in COPY_OPS]
    available=measurement['gpu_operator_attribution_available']
    return dict(cpu_self_us=cpu,linked_gpu_self_us=gpu if available else None,
        compatibility_cpu_self_us=sum(r['self_cpu_us'] for r in compatible),
        compatibility_linked_gpu_self_us=sum(r['self_device_us'] for r in compatible) if available else None,
        compatibility_gpu_share_percent=100*sum(r['self_device_us'] for r in compatible)/gpu if available and gpu>0 else None,
        copy_operator_cpu_self_us=sum(r['self_cpu_us'] for r in copies),
        copy_operator_linked_gpu_self_us=sum(r['self_device_us'] for r in copies) if available else None,
        phase_cpu_self_us={phase:sum(r['self_cpu_us'] for r in rows if r['phase']==phase) for phase in sorted({r['phase'] for r in rows})},
        phase_linked_gpu_self_us={phase:sum(r['self_device_us'] for r in rows if r['phase']==phase) if available else None for phase in sorted({r['phase'] for r in rows})},
        top_cpu=sorted(rows,key=lambda r:r['self_cpu_us'],reverse=True)[:20],
        top_linked_gpu=sorted(rows,key=lambda r:r['self_device_us'],reverse=True)[:20] if available else [])


def audit(out,write=False):
    out=Path(out).resolve(strict=True);cfg=p.read(out/'configuration.json');spec=cfg['spec'];fingerprint=cfg['fingerprint']
    assert fingerprint==p.c.r.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})
    assert cfg['sources']==p.sources();p.c.r.t.verify(cfg['evidence'])
    if not cfg['smoke']:assert cfg==p.read(p.FREEZE)
    if not write:assert p.c.r.d.validate_complete(out,fingerprint)
    pairs=[];profiles=[];retried=[];uuids=[];plain_count=instrument_count=0
    for session in range(spec['sessions']):
        folder=out/f'session_{session:02d}';marker=p.c.r.d.validate_complete(folder,fingerprint);assert marker
        attempt=Path(marker['attempt']).resolve(strict=True);assert attempt.parent==folder
        assert set(marker['sha256'])=={str(q) for q in attempt.rglob('*') if q.is_file()}
        if len(list(folder.glob('attempt_*')))!=1:retried.append(session)
        process=p.read(attempt/'process.json');assert process['session']==session;uuids.append(process['uuid'])
        for index,(seed,batch,dropout) in enumerate(p.cases(spec,session)):
            case=attempt/f'{index:02d}_{seed}_{batch}_{dropout}';anchor=p.read(case/'anchor.json');comparison=p.read(case/'comparison.json')
            assert (anchor['seed'],anchor['batch'],anchor['dropout'])==(seed,batch,dropout)
            assert anchor['optimizer_age']==spec['anchor_updates'] and len(anchor['indices'])==batch and len(set(anchor['indices']))==batch
            assert comparison['arm_order']==p.arm_order(spec,session,index)
            for key in ('model_sha256','optimizer_sha256'):assert comparison['anchor_'+key]==anchor[key]
            measured={}
            for arm in spec['arms']:
                root=case/arm;regimes=['plain']+(['events','short_plain','profile'] if session==spec['instrument_session'] else [])
                assert {q.name for q in root.iterdir()}==set(regimes)
                stored={}
                for mode in regimes:
                    row=p.read(root/mode/'measurement.json');stored[mode]=row
                    updates=spec['plain_updates'] if mode in ('plain','events') else spec['profile_warmup_updates']+spec['profile_active_updates']
                    assert row['mode']==mode and row['updates']==len(row['losses'])==updates
                    assert row['wall_seconds']>0 and math.isfinite(row['wall_seconds'])
                    assert row['optimizer_age']==spec['anchor_updates']+updates and row['all_gradients_finite']
                    assert all(math.isfinite(x) and x>0 for x in row['losses'])
                    if mode in ('events','profile'):
                        if mode=='events':
                            phases=p.read(root/mode/'phases.json');assert len(phases)==updates*5
                            for i in range(updates):
                                selected=[r for r in phases if r['step']==i]
                                assert [r['phase'] for r in selected]==['seed_zero','forward_loss','loss_readback','backward','optimizer']
                                assert all(r['host_seconds']>=0 and math.isfinite(r['host_seconds']) for r in selected)
                                if cfg['device']=='cuda':assert all(r['cuda_stream_ms']>=0 for r in selected)
                                else:assert all(r['cuda_stream_ms'] is None for r in selected)
                        instrument_count+=1
                plain=stored['plain'];measured[arm]=plain;plain_count+=1
                if session==spec['instrument_session']:
                    assert p.identity(stored['events'])==p.identity(plain)
                    assert p.identity(stored['profile'])==p.identity(stored['short_plain'])
                    prof=stored['profile'];rows=p.read(root/'profile/operators.json')
                    assert (root/'profile/trace.json').stat().st_size>0
                    assert all(r['calls']>0 and math.isfinite(r['self_cpu_us']) and math.isfinite(r['self_device_us']) for r in rows)
                    layers=2 if cfg['smoke'] else 6;expected=spec['profile_active_updates']*layers
                    counts={k:sum(r['calls'] for r in rows if r['operator']==op) for k,op in p.c.q.frozen_backend.OPS.items()}
                    assert counts==prof['dispatch'] and sum(counts.values())==expected
                    markers=sum(r['calls'] for r in rows if r['operator'].startswith('module::compatibility_copy_'))
                    assert markers==(expected if arm=='compatible' else 0)
                    summary=operator_summary(rows,prof)
                    profiles.append(dict(seed=seed,batch=batch,dropout=dropout,arm=arm,dispatch=counts,
                        gpu_attribution_available=prof['gpu_operator_attribution_available'],
                        event_wall_overhead_percent=100*(stored['events']['wall_seconds']/plain['wall_seconds']-1),
                        profiler_session_overhead_percent=100*(prof['wall_seconds']/stored['short_plain']['wall_seconds']-1),
                        **summary))
            baseline=measured['compatible'];control=measured['native_layout_control']
            exact=p.identity(baseline)==p.identity(control);assert exact==comparison['learning_identity_across_layouts']
            pairs.append(dict(session=session,seed=seed,batch=batch,dropout=dropout,compatible_first=comparison['arm_order'][0]=='compatible',
                compatible_ms_per_update=1000*baseline['wall_seconds']/spec['plain_updates'],
                control_ms_per_update=1000*control['wall_seconds']/spec['plain_updates'],
                control_saving_percent=100*(1-control['wall_seconds']/baseline['wall_seconds']),
                learning_exact=exact,first_loss_difference=control['losses'][0]-baseline['losses'][0]))
    assert len(set(uuids))==spec['sessions']
    groups=[]
    for seed,batch,dropout in p.cases(spec,0):
        rows=[r for r in pairs if (r['seed'],r['batch'],r['dropout'])==(seed,batch,dropout)]
        vals=[r['control_saving_percent'] for r in rows]
        groups.append(dict(seed=seed,batch=batch,dropout=dropout,median_control_saving_percent=median(vals),range_percent=[min(vals),max(vals)],
                           positive_sessions=sum(v>0 for v in vals),sessions=len(vals),learning_exact_all_sessions=all(r['learning_exact'] for r in rows)))
    result=dict(status='PASS',smoke=cfg['smoke'],plain_blocks=plain_count,primary_updates=plain_count*spec['plain_updates'],
                instrumented_regimes=instrument_count,profiles=profiles,groups=groups,retried_sessions=retried,
                instrumentation_preserves_learning=True,quality_tested=False,method_qualification=False,
                scope='Fixed-batch operation timing; copy removal is a diagnostic control, not a qualified equivalent method')
    if write:
        p.save(out/'audit.json',result);p.save(out/'pairings.json',pairs)
        lines=['# Attention copy / optimized runtime profile','',
               'Integrity PASS. Compare plain timings only; profiler/event overhead and availability are reported separately.',
               'No accuracy, complete adaptation, equivalent-method or novelty claim. Original dropout control can change learning.',
               'Compatibility copy is part of our adapter. Its removal alone is not an improvement over established upstream practice.','',
               '| Source | Batch | Dropout | Median control saving | Range | Positive | Learning exact |',
               '|---|---:|---|---:|---:|---:|---|']
        for g in groups:lines.append(f'| {g["seed"]} | {g["batch"]} | {g["dropout"]} | {g["median_control_saving_percent"]:.2f}% | {g["range_percent"][0]:.2f} to {g["range_percent"][1]:.2f}% | {g["positive_sessions"]}/{g["sessions"]} | {g["learning_exact_all_sessions"]} |')
        lines+=['','See audit.json for phase/operator attribution; raw traces are saved for session0 only.',
                'CPU and GPU times overlap; linked operator self times must not be added to raw kernel events.',
                'Forward compatibility ranges do not identify all backward copy costs. Missing CUDA attribution is unavailable, not zero.',
                'No automatic optimization or learning campaign follows this diagnostic.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    else:assert result==p.read(out/'audit.json') and pairs==p.read(out/'pairings.json')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=p.OUT);args=ap.parse_args()
    assert __debug__;p.c.r.m.configure();print(audit(args.output))
