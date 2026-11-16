"""Paired, descriptive summaries of post-confirmation reviewer experiments.

These are not the Stage6B pre-registered primary tests. Outputs are descriptive
and cannot be used to revise the original confirmatory claims.
"""
from __future__ import annotations
import argparse, csv, json, math, statistics
from pathlib import Path

T_CRIT_95 = {1:12.7062,2:4.3027,3:3.1824,4:2.7764,5:2.5706,6:2.4469,7:2.3646,8:2.3060,9:2.2622,10:2.2281,11:2.2010,12:2.1788,13:2.1604,14:2.1448,15:2.1314,16:2.1199,17:2.1098,18:2.1009,19:2.0930,20:2.0860}

def ci95(values):
    n = len(values)
    if n<2: return [None,None]
    mu = statistics.mean(values)
    try:
        import scipy.stats as st
        tc = st.t.ppf(.975,n-1)
    except ImportError:
        tc = T_CRIT_95.get(n-1,1.96)
    half = tc * statistics.stdev(values) / math.sqrt(n)
    return [mu-half,mu+half]

def write(path,rows):
    if not rows:return
    with open(path,'w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--reference',default='lgpb')
    p.add_argument('--additional-references', nargs='*', default=[])
    p.add_argument('--expected-seeds', nargs='+', type=int)
    p.add_argument('--expected-variants', nargs='+')
    args=p.parse_args()
    root=Path(args.root)
    dirs=sorted(root.glob('seed_*'))
    if not dirs: raise FileNotFoundError(f'No seed_* folders under {root}')
    rows=[]
    for sd in dirs:
        for s in sorted(sd.glob('*/summary.json')):
            r=json.loads(s.read_text(encoding='utf8'))
            if r['final_eval_accuracy'] is None:continue
            rows.append(r)
    write(root/'stage7_all_runs.csv',rows)
    lookup={(int(r['seed']),r['variant']):r for r in rows}
    if args.expected_seeds and args.expected_variants:
        expected={(seed,variant) for seed in args.expected_seeds for variant in args.expected_variants}
        if set(lookup) != expected:
            raise ValueError(f'Incomplete or unexpected study: missing={expected-set(lookup)}, extra={set(lookup)-expected}')
    for r in rows:
        if not math.isfinite(r['final_eval_accuracy']) or not math.isfinite(r['training_wall_seconds']) or r['training_wall_seconds'] <= 0:
            raise ValueError(f'Invalid completed metrics: {r["seed"]}/{r["variant"]}')
    recipes={(r['dataset'],r['optimizer'],r['epochs'],r['updates'],r['batch_size'],r['train_subset'],r['eval_subset'],r['lr'],r.get('implementation_version')) for r in rows}
    if len(recipes) != 1:
        raise ValueError('Mixed experiment recipes in one results directory')
    paired=[]
    references=list(dict.fromkeys([args.reference]+args.additional_references))
    for reference in references:
      for r in rows:
        base=lookup.get((int(r['seed']),reference))
        if not base or r['variant']==reference: continue
        paired.append({'seed':r['seed'],'variant':r['variant'],'reference':reference,
            'accuracy_delta_pp':100*(r['final_eval_accuracy']-base['final_eval_accuracy']),
            'wall_ratio':r['training_wall_seconds']/base['training_wall_seconds'],
            'wall_saving_pct':100*(1-r['training_wall_seconds']/base['training_wall_seconds']),
            'partial_fraction_delta_pp':100*(r['partial_fraction']-base['partial_fraction']),
            'q_variant':r['partial_fraction'], 'q_reference':base['partial_fraction']})
    write(root/'stage7_paired_vs_reference.csv',paired)
    variants=sorted(set((r['reference'],r['variant']) for r in paired))
    estimates=[]
    for reference,name in variants:
        subset=[r for r in paired if r['variant']==name and r['reference']==reference]
        if not subset:continue
        diff=[float(r['accuracy_delta_pp']) for r in subset]
        log_time=[math.log(float(r['wall_ratio'])) for r in subset]
        lim=ci95(diff)
        llim=ci95(log_time)
        estimates.append({'variant':name,'reference':reference,'n_paired_seeds':len(subset),
            'mean_accuracy_delta_pp':statistics.mean(diff),'ci95_accuracy_low_pp':lim[0],'ci95_accuracy_high_pp':lim[1],
            'geometric_wall_ratio':math.exp(statistics.mean(log_time)),
            'ci95_wall_ratio_low':math.exp(llim[0]) if llim[0] is not None else None,
            'ci95_wall_ratio_high':math.exp(llim[1]) if llim[1] is not None else None,
            'mean_partial_fraction_diff_pp':statistics.mean(float(r['partial_fraction_delta_pp']) for r in subset)})
    write(root/'stage7_descriptive_paired_estimates.csv',estimates)
    lines=['# Stage 7 reviewer-experiment descriptive summary','',
        '**These comparisons are NEW post-confirmation analyses. They do not modify the original Stage 6B inference.**','',
        f'Reference: `{args.reference}`; study folder: `{root}`.','',
        'Samples with missing variants are excluded pairwise; inspect n_paired_seeds.','',
        '| Variant | Reference | Paired n | Accuracy delta (pp) | Geometric wall ratio | Partial q delta (pp) |',
        '|---|---|---:|---:|---:|---:|']
    for e in estimates:
        lines.append(f"| {e['variant']} | {e['reference']} | {e['n_paired_seeds']} | {e['mean_accuracy_delta_pp']:+.3f} | {e['geometric_wall_ratio']:.3f} | {e['mean_partial_fraction_diff_pp']:+.2f} |")
    lines.extend(['','The CSV also contains exploratory paired 95% confidence intervals. Interpret intervals','as descriptive unless a separate Stage 7 analysis plan was frozen before running.'])
    (root/'stage7_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines))

if __name__=='__main__':main()
