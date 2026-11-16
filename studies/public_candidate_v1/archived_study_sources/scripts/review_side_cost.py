"""Read-only independent arithmetic/integrity audit for the artificial timing check."""
import argparse
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify(entries):
    for path,h in entries.items():assert sha(path)==h,path


def median_ratio(candidate,full):return st.median([a/b for a,b in zip(candidate,full)])


def aggregate(rows,spec,smoke):
    summaries=[];paired=[];passes=[]
    for seed in spec['seeds']:
        for batch in spec['batch_sizes']:
            full=sorted([r for r in rows if r['seed']==seed and r['batch']==batch and r['arm']=='full'],key=lambda r:r['round'])
            for arm in spec['arms']:
                selected=sorted([r for r in rows if r['seed']==seed and r['batch']==batch and r['arm']==arm],key=lambda r:r['round'])
                ratios=[]
                for row,ref in zip(selected,full):
                    ratio=row['steady_seconds']/ref['steady_seconds'];ratios.append(ratio)
                    paired.append(dict(seed=seed,batch=batch,round=row['round'],arm=arm,
                        steady_ratio_to_full=ratio,first_ratio_to_full=row['first_seconds']/ref['first_seconds']))
                ratio=st.median(ratios)
                q1,_,q3=st.quantiles(ratios,n=4,method='inclusive')
                gate=ratio<=1-spec['minimum_operation_time_saving']+1e-12
                if arm=='multi_side':passes.append(gate)
                summaries.append(dict(seed=seed,batch=batch,arm=arm,
                    median_steady_ms=1000*st.median([r['steady_update_seconds'] for r in selected]),
                    median_first_ms=1000*st.median([r['first_seconds'] for r in selected]),
                    median_setup_ms=1000*st.median([r['setup_seconds'] for r in selected]),
                    median_warmup_restore_ms=1000*st.median([r['warmup_restore_seconds'] for r in selected]),
                    median_paired_ratio=ratio,median_paired_saving_percent=100*(1-ratio),
                    min_paired_ratio=min(ratios),q1_paired_ratio=q1,q3_paired_ratio=q3,max_paired_ratio=max(ratios),
                    rounds_at_least_10_percent_faster=sum(z<=.90+1e-12 for z in ratios),rounds=len(ratios),
                    median_steady_peak_bytes=st.median([r['steady_memory']['peak'] for r in selected]),
                    median_first_incremental_bytes=st.median([r['first_memory']['incremental'] for r in selected]),
                    median_steady_incremental_bytes=st.median([r['steady_memory']['incremental'] for r in selected]),
                    trainable_parameters=selected[0]['trainable_parameters'],frozen_parameters=selected[0]['frozen_parameters'],
                    operation_gate=gate if arm=='multi_side' and not smoke else False))
    return summaries,paired,all(passes) and not smoke


def write_csv(path,rows):
    with path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def compare_csv(path,expected):
    with path.open(encoding='utf8',newline='') as f:actual=list(csv.DictReader(f))
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected):
        assert set(a)==set(b)
        for key,val in b.items():
            if isinstance(val,(str,bool)):assert a[key]==str(val)
            else:assert math.isclose(float(a[key]),val,rel_tol=1e-12,abs_tol=1e-12)


def audit(out,write=False):
    out=Path(out);cfg=json.loads((out/'configuration.json').read_text());spec=cfg['spec'];smoke=cfg['smoke']
    canonical=json.dumps({k:v for k,v in cfg.items() if k!='fingerprint'},sort_keys=True,allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==cfg['fingerprint']
    verify(cfg['sources'])
    marker=json.loads((out/'RAW_COMPLETE.json').read_text());assert marker['fingerprint']==cfg['fingerprint']
    verify(marker['sha256']);attempt=Path(marker['attempt']);assert attempt.parent.resolve()==out.resolve()
    assert set(marker['sha256'])=={str(p) for p in attempt.iterdir() if p.is_file()}
    rows=json.loads((attempt/'trials.json').read_text())
    assert len(rows)==len(spec['seeds'])*len(spec['batch_sizes'])*spec['rounds']*len(spec['arms'])
    index=0
    for si,seed in enumerate(spec['seeds']):
        for bi,batch in enumerate(spec['batch_sizes']):
            base=spec['arms'] if (si+bi)%2==0 else list(reversed(spec['arms']))
            for ri in range(spec['rounds']):
                rot=ri%len(base);order=base[rot:]+base[:rot]
                group=rows[index:index+len(order)]
                for position,arm in enumerate(order):
                    row=rows[index];index+=1
                    assert (row['seed'],row['batch'],row['round'],row['position'],row['arm'])==(seed,batch,ri,position,arm)
                    for key in ('setup_seconds','warmup_restore_seconds','first_seconds','steady_seconds','steady_update_seconds','first_loss'):
                        assert math.isfinite(row[key]) and row[key]>0
                    assert math.isclose(row['steady_update_seconds']*spec['steady_updates_per_trial'],row['steady_seconds'],rel_tol=1e-12)
                    assert len(row['steady_losses'])==spec['steady_updates_per_trial'] and all(math.isfinite(z) and z>0 for z in row['steady_losses'])
                    assert row['optimizer_age']==1+spec['steady_updates_per_trial']
                    assert row['initial_state_restored'] and row['all_trainable_gradients_finite']
                    assert row['frozen_body_verified']==(arm!='full')
                    for key in ('setup_memory','warmup_memory','first_memory','steady_memory'):
                        mem=row[key];assert mem['peak']-mem['baseline']==mem['incremental']>=0 and mem['reserved']>=mem['peak']
                        assert all(z==0 for z in mem.values()) if smoke else mem['peak']>0
                    if not smoke:assert row['trainable_parameters']==dict(full=2693578,head=1930,final_side=125002,multi_side=125002)[arm]
                assert len({r['input_sha256'] for r in group})==1
                hashes={r['arm']:r['initial_sha256'] for r in group}
                assert hashes['full']==hashes['head'] and hashes['final_side']==hashes['multi_side']
            for arm in spec['arms']:
                selected=[r for r in rows if r['seed']==seed and r['batch']==batch and r['arm']==arm]
                assert len({r['initial_sha256'] for r in selected})==1
                assert all(sum(r['position']==p for r in selected)==spec['rounds']//4 for p in range(4))
    summaries,paired,passed=aggregate(rows,spec,smoke)
    result=dict(status='PASS',date='2026-10-25',synthetic=True,smoke=smoke,trials=len(rows),
        discarded_warmup_updates=len(rows)*spec['discarded_warmup_updates'],
        retained_artificial_updates=len(rows)*(1+spec['steady_updates_per_trial']),
        paired_condition_count=len(spec['seeds'])*len(spec['batch_sizes']),operation_cost_gate=passed,
        decision='synthetic_smoke_only' if smoke else 'cost_feasible_prepare_quality_screen' if passed else 'park_fixed_side_architecture_for_time_objective',
        real_data_loaded=False,real_checkpoints_loaded=False,task_quality_measured=False,complete_adaptation_time_measured=False)
    if write:
        write_csv(out/'summary.csv',summaries);write_csv(out/'paired.csv',paired)
        (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        lines=['# Side-network operation-cost feasibility','',
            '**Artificial operation timings only. No task quality or complete adaptation result.**','',
            '```json',json.dumps(result,indent=2),'```','',
            '| Seed | Batch | Arm | Median update ms | Paired saving vs full | Paired ratio range |',
            '|---|---:|---|---:|---:|---:|']
        for row in summaries:
            lines.append(f"| {row['seed']} | {row['batch']} | {row['arm']} | {row['median_steady_ms']:.3f} | {row['median_paired_saving_percent']:.2f}% | {row['min_paired_ratio']:.3f}–{row['max_paired_ratio']:.3f} |")
        lines+=['','Operation includes full forward, feature capture, loss/readback, backward and AdamW. Setup, discarded warmup/restoration and first-state allocation are reported separately in summary.csv. Input generation/transfer, augmentation, validation, logging and checkpoint costs are outside operation timing. The gate applies separately to both seeds and both batch sizes. Memory cannot substitute for time.','',
            'Head and final-side quality are unknown. No speed claim against quality-qualified controls follows. No widths/taps were selected by timing search.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text())==result
        compare_csv(out/'summary.csv',summaries);compare_csv(out/'paired.csv',paired)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);args=ap.parse_args()
    print(json.dumps(audit(args.out),indent=2))
