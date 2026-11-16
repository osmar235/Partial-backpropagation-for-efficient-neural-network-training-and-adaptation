"""Verify fixed artificial timing inputs, counters, pairings and decision arithmetic."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import torch
from review_stage14C import compare_csv

def audit(out,write=False):
    import sampled_weight_cost as r
    out=Path(out);cfg=json.loads((out/'configuration.json').read_text());spec=cfg['spec'];smoke=cfg['smoke']
    assert cfg['synthetic'] and r.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    for p,h in cfg['sources'].items():assert r.d.sha(Path(p))==h,p
    if not smoke:assert cfg==json.loads(r.FREEZE.read_text())
    marker=json.loads((out/'RAW_COMPLETE.json').read_text());assert marker['fingerprint']==cfg['fingerprint']
    attempt=Path(marker['attempt']);assert attempt.parent.resolve()==out.resolve()
    assert set(marker['sha256'])=={str(p) for p in attempt.iterdir() if p.is_file()}
    for p,h in marker['sha256'].items():assert r.d.sha(Path(p))==h,p
    trials=json.loads((attempt/'trials.json').read_text());device=json.loads((attempt/'device.json').read_text())
    assert device['device']==('cpu' if smoke else 'cuda')
    telemetry=json.loads((attempt/'telemetry.json').read_text())
    assert len(telemetry)==(0 if smoke else len(spec['seeds'])*len(spec['batch_sizes'])*spec['rounds'])
    expected=len(spec['seeds'])*len(spec['batch_sizes'])*spec['rounds']*len(spec['arms']);assert len(trials)==expected
    index={};position=0
    for si,seed in enumerate(spec['seeds']):
        initial={arm:r.old.state_hash(r.make(seed,arm,spec,smoke).state_dict()) for arm in spec['arms']}
        assert len(set(initial.values()))==1
        params=sum(p.numel() for p in r.make(seed,'native',spec,smoke).parameters())
        for bi,batch in enumerate(spec['batch_sizes']):
            gen=torch.Generator().manual_seed(seed+spec['input_seed_offset']+batch)
            x=torch.randn(batch,3,32,32,generator=gen);y=torch.randint(0,10,(batch,),generator=gen)
            xhash=hashlib.sha256(x.numpy().tobytes()).hexdigest();yhash=hashlib.sha256(y.numpy().tobytes()).hexdigest()
            for ri in range(spec['rounds']):
                base=spec['arms'] if (si+bi)%2==0 else list(reversed(spec['arms']))
                shift=ri%len(base);order=base[shift:]+base[:shift]
                for pi,arm in enumerate(order):
                    row=trials[position];position+=1
                    assert (row['seed'],row['batch'],row['round'],row['position'],row['arm'])==(seed,batch,ri,pi,arm)
                    index[seed,batch,ri,arm]=row
                    assert row['initial_sha256']==initial[arm] and row['input_sha256']==xhash and row['labels_sha256']==yhash
                    assert row['trainable_parameters']==params and row['frozen_parameters']==0
                    assert row['initial_state_restored'] and row['all_trainable_gradients_finite']
                    assert row['optimizer_age']==1+spec['steady_updates_per_trial']
                    assert len(row['steady_losses'])==spec['steady_updates_per_trial']
                    for key in ('setup_seconds','warmup_restore_seconds','first_seconds','steady_seconds','steady_update_seconds','first_loss'):
                        assert math.isfinite(row[key]) and row[key]>0
                    assert all(math.isfinite(loss) and loss>0 for loss in row['steady_losses'])
                    assert math.isclose(row['steady_update_seconds']*spec['steady_updates_per_trial'],row['steady_seconds'],rel_tol=1e-12)
                    for key in ('setup_memory','warmup_memory','first_memory','steady_memory'):
                        mem=row[key];assert mem['peak']-mem['baseline']==mem['incremental']>=0 and mem['reserved']>=mem['peak']
                        if smoke:assert all(value==0 for value in mem.values())
                        else:assert mem['peak']>0
                first=[index[seed,batch,ri,arm]['first_loss'] for arm in spec['arms']]
                assert max(first)-min(first)<=spec['first_loss_tolerance']
    paired=[];summary=[];conditions=[]
    for seed in spec['seeds']:
        for batch in spec['batch_sizes']:
            for arm in spec['arms']:
                rows=[index[seed,batch,ri,arm] for ri in range(spec['rounds'])]
                ratios=[]
                for ri,row in enumerate(rows):
                    native=index[seed,batch,ri,'native'];ratio=row['steady_update_seconds']/native['steady_update_seconds'];ratios.append(ratio)
                    paired.append(dict(seed=seed,batch=batch,round=ri,arm=arm,ratio_vs_native=ratio,saving_percent=100*(1-ratio),
                        at_least_10_percent_faster=ratio<=1-spec['minimum_time_saving']))
                median=statistics.median(ratios)
                summary.append(dict(seed=seed,batch=batch,arm=arm,median_steady_ms=1000*statistics.median(row['steady_update_seconds'] for row in rows),
                    median_paired_ratio=median,median_paired_saving_percent=100*(1-median),
                    rounds_at_least_10_percent_faster=sum(ratio<=1-spec['minimum_time_saving'] for ratio in ratios),
                    rounds=spec['rounds'],median_first_ms=1000*statistics.median(row['first_seconds'] for row in rows),
                    median_setup_ms=1000*statistics.median(row['setup_seconds'] for row in rows),
                    median_warmup_restore_ms=1000*statistics.median(row['warmup_restore_seconds'] for row in rows),
                    median_steady_incremental_bytes=statistics.median(row['steady_memory']['incremental'] for row in rows)))
                if arm=='sampled_quarter':conditions.append(median<=1-spec['minimum_time_saving'])
    gate=bool(not smoke and all(conditions))
    result=dict(status='PASS',synthetic=True,smoke=smoke,trials=len(trials),
        retained_artificial_updates=len(trials)*(1+spec['steady_updates_per_trial']),discarded_warmup_updates=len(trials)*spec['discarded_warmup_updates'],
        all_inputs_states_first_losses_and_pairings_verified=True,operation_cost_gate=gate,
        passing_seed_batch_conditions=sum(conditions),total_seed_batch_conditions=len(conditions),
        task_quality_measured=False,complete_adaptation_measured=False,real_checkpoint_or_data_used=False,
        decision='synthetic_pipeline_only' if smoke else 'prepare_quality_diagnostic' if gate else 'park_fixed_implementation')
    if write:
        r.v.write_csv(out/'summary.csv',summary);r.v.write_csv(out/'paired.csv',paired);r.v.atomic_json(out/'audit.json',result)
        lines=['# Sampled weight-gradient operation cost','','```json',json.dumps(result,indent=2),'```','',
            '| Artificial seed | Batch | Arm | Median update ms | Median paired saving vs native | Passing rounds |',
            '|---|---:|---|---:|---:|---:|']
        for row in summary:lines.append(f"| {row['seed']} | {row['batch']} | {row['arm']} | {row['median_steady_ms']:.3f} | {row['median_paired_saving_percent']:.2f}% | {row['rounds_at_least_10_percent_faster']} / {row['rounds']} |")
        lines+=['','Operation clock includes sampling, indexing, full forward, loss, backward and AdamW. Setup, fresh first update and memory are separate in summary.csv.',
            'No real task quality or complete adaptation result. Native is the required time comparator; exact-custom measures implementation overhead. No tuning or automatic follow-up.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text())==result
        compare_csv(r.d.records(out/'summary.csv'),summary);compare_csv(r.d.records(out/'paired.csv'),paired)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    import stage10_models as m
    m.configure();print(json.dumps(audit(args.output),indent=2))
