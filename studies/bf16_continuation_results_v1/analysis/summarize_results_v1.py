"""Portable descriptive aggregation of audited BF16 continuation evidence."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

def span(values):
    values=list(values);return dict(min=min(values),median=median(values),max=max(values))

def summarize(data):
    pairs=data['pairings'];spec=data['spec'];groups=[]
    for arm in spec['arms'][1:]:
        conditions=[]
        for seed in spec['seeds']:
            rows=[r for r in pairs if r['arm']==arm and r['seed']==seed];n=len(rows);wins=sum(r['saving_percent']>0 for r in rows)
            p=sum(math.comb(n,k) for k in range(wins,n+1))/2**n
            conditions.append(dict(arm=arm,seed=seed,n=n,positive_sessions=wins,sign_p=p,
                saving_percent=span(r['saving_percent'] for r in rows),training_saving_percent=span(r['training_saving_percent'] for r in rows),
                complete_seconds_saved=span(r['reference_seconds']-r['candidate_seconds'] for r in rows),
                reference_seconds=span(r['reference_seconds'] for r in rows),candidate_seconds=span(r['candidate_seconds'] for r in rows),
                common_deficit_pp=span(r['common_deficit_pp'] for r in rows),deployed_deficit_pp=span(r['deployed_deficit_pp'] for r in rows),
                reference_accuracy=span(r['reference_accuracy'] for r in rows),candidate_accuracy=span(r['candidate_accuracy'] for r in rows),deployed_accuracy=span(r['deployed_accuracy'] for r in rows),
                common_loss_change=span(r['candidate_loss']-r['reference_loss'] for r in rows),deployed_loss_change=span(r['deployed_loss']-r['reference_loss'] for r in rows),
                historical_source_plus_adaptation_saving_percent=span(100*(r['reference_seconds']-r['candidate_seconds'])/(r['source_training_seconds']+r['reference_seconds']) for r in rows),
                reference_first_savings=span(r['saving_percent'] for r in rows if r['reference_first']),candidate_first_savings=span(r['saving_percent'] for r in rows if not r['reference_first']),
                component_seconds_saved={k:span(r['reference_components'][k]-r['candidate_components'][k] for r in rows) for k in rows[0]['reference_components']},
                baseline_ready=all(r['baseline_ready'] for r in rows),quality_pass=all(r['quality_pass'] for r in rows)))
        p=min(1,spec['candidate_policies']*max(c['sign_p'] for c in conditions))
        passed=not data['fragmented_sessions'] and p<=spec['timing_family_alpha'] and all(c['n']==spec['sessions'] and c['positive_sessions']==c['n'] and c['baseline_ready'] and c['quality_pass'] for c in conditions)
        groups.append(dict(arm=arm,conditions=conditions,adjusted_direction_p=p,prospective_screen_pass=bool(passed)))
    resources=[]
    for seed in spec['seeds']:
        for arm in spec['arms']:
            rows=[r for r in data['trajectories'] if r['seed']==seed and r['arm']==arm]
            resources.append(dict(seed=seed,arm=arm,peak_allocated_bytes=span(r['peak_allocated'] for r in rows),peak_reserved_bytes=span(r['peak_reserved'] for r in rows),
                source_training_seconds=span(r['source_training_seconds'] for r in rows),warmup_dispatch=[r['warmup_dispatch'] for r in rows]))
    return dict(groups=groups,resources=resources,scope='Fixed-checkpoint adaptation; repeats are timing checks, not extra seeds. Source-plus-adaptation percentages are accounting context, not a measured full training workflow.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('data',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    result=summarize(json.loads(args.data.read_text(encoding='utf8')));text=json.dumps(result,indent=2)+'\n'
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf8')
    else:print(text)
