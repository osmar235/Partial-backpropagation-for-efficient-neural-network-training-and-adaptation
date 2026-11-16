"""Read-only post-run verification and project record for sampled-weight cost."""
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/sampled_weight_cost_v1'
OUT=ROOT/'results/sampled_weight_cost_review_v1'
MUTABLE={str(ROOT/name) for name in ('CURRENT_RELEASE.md','README_STAGE9.md')}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify(entries):
    for p,h in entries.items():assert sha(p)==h,p

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')

def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def save(path,records):
    with Path(path).open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)

def main():
    assert not OUT.exists();OUT.mkdir()
    previous=ROOT/'results/sampled_weight_design_v1'
    protected=json.loads((previous/'preserved_inputs_sha256.json').read_text())
    prior_seal=previous/'preparation_artifacts_sha256.json';entries=json.loads(prior_seal.read_text())
    verify(protected);verify(entries)
    protected.update({p:h for p,h in entries.items() if p not in MUTABLE});protected[str(prior_seal)]=sha(prior_seal)
    protected.update({str(p):sha(p) for p in RAW.rglob('*') if p.is_file()})
    before={p:sha(p) for p in sorted(MUTABLE)}
    dump(OUT/'preserved_inputs_sha256.json',protected);dump(OUT/'mutable_pointers_before.json',before)
    import sampled_weight_cost as r
    from review_sampled_weight_cost import audit
    r.m.configure();cfg=json.loads((RAW/'configuration.json').read_text());spec=cfg['spec']
    assert cfg['synthetic'] and not cfg['smoke'] and r.d.validate_complete(RAW,cfg['fingerprint'])
    checked=audit(RAW,False);dump(OUT/'frozen_audit_recheck.json',checked)
    marker=json.loads((RAW/'RAW_COMPLETE.json').read_text());attempt=Path(marker['attempt'])
    trials=json.loads((attempt/'trials.json').read_text());assert len(trials)==72
    summary={(int(row['seed']),int(row['batch']),row['arm']):row for row in rows(RAW/'summary.csv')}
    raw_pairs={(int(row['seed']),int(row['batch']),int(row['round']),row['arm']):row for row in rows(RAW/'paired.csv')}
    comparisons=[];pairings=[]
    for seed in spec['seeds']:
        for batch in spec['batch_sizes']:
            matched={arm:sorted([row for row in trials if row['seed']==seed and row['batch']==batch and row['arm']==arm],
                key=lambda row:row['round']) for arm in spec['arms']}
            assert all(len(group)==6 for group in matched.values())
            for arm,group in matched.items():
                assert all(sum(row['position']==p for row in group)==2 for p in range(3))
                close(summary[seed,batch,arm]['median_steady_ms'],1000*st.median(row['steady_seconds']/8 for row in group))
            ratios=[];custom_ratios=[];overhead=[];memory_ratios=[];peak_ratios=[];first_ratios=[]
            for index in range(6):
                native=matched['native'][index];custom=matched['exact_custom'][index];sample=matched['sampled_quarter'][index]
                assert native['round']==custom['round']==sample['round']==index
                for key in ('input_sha256','labels_sha256','initial_sha256'):
                    assert native[key]==custom[key]==sample[key]
                ratio=sample['steady_seconds']/native['steady_seconds'];cr=sample['steady_seconds']/custom['steady_seconds']
                ratios.append(ratio);custom_ratios.append(cr);overhead.append(custom['steady_seconds']/native['steady_seconds'])
                memory_ratios.append(sample['steady_memory']['incremental']/native['steady_memory']['incremental'])
                peak_ratios.append(sample['steady_memory']['peak']/native['steady_memory']['peak'])
                first_ratios.append(sample['first_seconds']/native['first_seconds'])
                saved=raw_pairs[seed,batch,index,'sampled_quarter'];close(saved['ratio_vs_native'],ratio)
                close(saved['saving_percent'],100*(1-ratio));assert saved['at_least_10_percent_faster']==str(ratio<=.9)
                pairings.append(dict(seed=seed,batch=batch,round=index,ratio_vs_native=ratio,ratio_vs_exact_custom=cr,
                    native_ms=1000*native['steady_seconds']/8,sampled_ms=1000*sample['steady_seconds']/8,
                    at_least_10_percent_faster=ratio<=.9))
            ratio=st.median(ratios);q1,_,q3=st.quantiles(ratios,n=4,method='inclusive')
            close(summary[seed,batch,'sampled_quarter']['median_paired_ratio'],ratio)
            comparisons.append(dict(seed=seed,batch=batch,
                native_median_ms=1000*st.median(row['steady_seconds']/8 for row in matched['native']),
                exact_custom_median_ms=1000*st.median(row['steady_seconds']/8 for row in matched['exact_custom']),
                sampled_median_ms=1000*st.median(row['steady_seconds']/8 for row in matched['sampled_quarter']),
                median_paired_slowdown_percent=100*(ratio-1),min_paired_ratio=min(ratios),q1_paired_ratio=q1,q3_paired_ratio=q3,
                max_paired_ratio=max(ratios),passing_rounds=sum(x<=.9 for x in ratios),any_speedup_rounds=sum(x<1 for x in ratios),
                condition_pass=ratio<=.9,custom_overhead_percent=100*(st.median(overhead)-1),
                saving_vs_exact_custom_percent=100*(1-st.median(custom_ratios)),
                incremental_memory_saving_percent=100*(1-st.median(memory_ratios)),
                absolute_peak_memory_saving_percent=100*(1-st.median(peak_ratios)),
                first_update_slowdown_percent=100*(st.median(first_ratios)-1)))
    save(OUT/'independent_comparisons.csv',comparisons);save(OUT/'independent_pairings.csv',pairings)
    passed=all(row['condition_pass'] for row in comparisons)
    assert passed==checked['operation_cost_gate'] and not passed
    assert sum(row['condition_pass'] for row in comparisons)==checked['passing_seed_batch_conditions']==0
    assert checked['trials']==72 and checked['retained_artificial_updates']==648 and checked['discarded_warmup_updates']==216
    verify(protected)
    result=dict(status='PASS',date='2026-10-25',trials_verified=72,paired_candidate_rounds=24,
        retained_artificial_updates_verified=648,discarded_warmup_updates_verified=216,
        passing_conditions=0,total_conditions=4,passing_candidate_rounds=sum(row['passing_rounds'] for row in comparisons),
        operation_cost_gate=False,decision='park_fixed_implementation',prior_and_raw_files_preserved=len(protected),
        new_benchmark_or_training_performed=False,real_data_or_checkpoint_used=False,task_quality_measured=False,
        complete_adaptation_measured=False)
    dump(OUT/'independent_audit.json',result)
    table=['| Artificial seed | Batch | Native update ms | Sampled update ms | Paired slowdown | Passing rounds / 6 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in comparisons:table.append(f"| {row['seed']} | {row['batch']} | {row['native_median_ms']:.3f} | {row['sampled_median_ms']:.3f} | {row['median_paired_slowdown_percent']:.2f}% | {row['passing_rounds']} / 6 |")
    memory_savings=[row['incremental_memory_saving_percent'] for row in comparisons]
    custom_savings=[row['saving_vs_exact_custom_percent'] for row in comparisons]
    finding=f'''The frozen sampled-weight-gradient implementation FAILS the artificial
operation-cost gate in all four seed/batch conditions. It is slower than native
backpropagation by median paired 4.97% / 13.82% / 5.56% / 11.24%, in the order
shown below. None of 24 paired candidate rounds achieves the required 10% saving.

{chr(10).join(table)}

The exact-custom control is approximately 18.75–20.13% slower than native.
Sampling lowers the custom implementation's measured time by approximately
{min(custom_savings):.2f}–{max(custom_savings):.2f}% (median paired comparisons), but does not overcome its overhead.
This supports an implementation-overhead interpretation, without identifying
which kernels, gathers, RNG operations or dispatch costs dominate. The control
includes all-token index generation/gathering and custom-autograd execution;
it does not isolate Python overhead alone. Fewer arithmetic terms did not
produce faster native-relative updates on this ViT/hardware/configuration.

Incremental steady-update allocated memory is about {min(memory_savings):.2f}–{max(memory_savings):.2f}% lower
than native (median paired). Retain that resource tradeoff; memory cannot replace
the predeclared primary time gate. No real quality, complete adaptation, inference
or larger-model result follows. Initial matching, unbiasedness unit checks and
valid gradients establish implementation properties, not task-level usefulness.

Independent review reran the frozen read-only audit and recomputed every paired
ratio and decision from raw trials. Verified 72 trials, 648 retained artificial
updates and 216 discarded warmups; inputs, initial states, first losses and
counterbalanced order agree. Timing repetitions are not independent training
replications. All prior and raw evidence is unchanged. This review performed
no new benchmark, optimization updates, real checkpoint access or task scoring.

Decision: park this fixed quarter-patch implementation for the time-efficiency
objective. Do not launch the proposed real-data accuracy study, change the
sampling fraction from these timings, weaken the 10% criterion or compare only
against the slower exact-custom control. The result does not refute sampled
backpropagation generally, its prior literature, or a future independently
justified fused implementation. Any such redesign requires a separate explicit
design and new prospective cost check. No replacement experiment is prepared or
launched. Preserve this negative systems result in the paper memory; publication
remains on hold.
'''
    report='# Sampled weight-gradient cost: independent review — 2026-10-25\n\n'+finding
    report+='\n## Verification\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n\n'
    report+='See independent_comparisons.csv for paired ranges, all-token custom overhead, first-update timing and memory; independent_pairings.csv retains each candidate round.\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf8')
    master=ROOT/'adaptive_gradient_refresh_project_master_v7.24.md'
    memory=ROOT/'paper/RESEARCH_DISCUSSION_MEMORY_v13.md'
    decision=ROOT/'protocol/sampled_weight_cost/POST_COST_DECISION_v1.md'
    seal=OUT/'review_artifacts_sha256.json'
    assert not any(p.exists() for p in (master,memory,decision,seal))
    decision.write_text('# Sampled weight-gradient cost decision — 2026-10-25\n\n'+finding,encoding='utf8')
    text=(ROOT/'adaptive_gradient_refresh_project_master_v7.23.md').read_text(encoding='utf8').replace('Project Master v7.23','Project Master v7.24',1)
    text='\n'.join('**Status:** Publication on hold; sampled weight-gradient cost gate fails all four conditions; fixed implementation parked'
        if line.startswith('**Status:**') else line for line in text.splitlines())+'\n'
    anchor='## Current action: sampled weight-gradient feasibility — 2026-10-25';assert anchor in text
    master.write_text(text.replace(anchor,'## Current decision: sampled weight-gradient cost reviewed — 2026-10-25\n\n'+finding+
        '\nReview: results/sampled_weight_cost_review_v1/REPORT.md. Paper memory v13 is current. Earlier statuses below are historical.\n\n## Historical sampled-weight preparation — 2026-10-25',1),encoding='utf8')
    memory.write_text('# Paper discussion memory v13 — sampled-weight cost gate fails — 2026-10-25\n\n'+finding+
        '\n## Historical v12 and earlier notes\n\n'+(ROOT/'paper/RESEARCH_DISCUSSION_MEMORY_v12.md').read_text(encoding='utf8'),encoding='utf8')
    header='''# Current decision: sampled weight-gradient cost reviewed — 2026-10-25

Fixed sampled-weight implementation FAILS all four artificial cost conditions:
4.97–13.82% slower than native; zero of 24 paired rounds reaches 10% saving.
All-token custom control is slower still. Correctness/audit pass; memory savings
do not replace time. Park this implementation; do not launch real-data training.
No replacement experiment prepared or launched. Master v7.24; paper memory v13.
Review: results/sampled_weight_cost_review_v1/REPORT.md. Earlier evidence preserved;
publication on hold. All older statuses and commands below are historical.

'''
    after={}
    for p,h in before.items():
        assert sha(p)==h
        path=Path(p);path.write_text(header+path.read_text(encoding='utf8'),encoding='utf8');after[p]=sha(path)
    dump(OUT/'mutable_pointers_after.json',after);verify(protected)
    dump(OUT/'project_update_verification.json',dict(status='PASS',date='2026-10-25',preserved_immutable_files=len(protected),
        current_master=str(master),paper_memory=str(memory),operation_cost_gate=False,new_real_study_prepared=False,
        new_benchmark_or_training_performed=False,publication_on_hold=True))
    files=list(OUT.iterdir())+[master,memory,decision,Path(__file__)]+[Path(p) for p in after]
    dump(seal,{str(p):sha(p) for p in files if p.is_file() and p!=seal})
    print(json.dumps(result,indent=2));print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
