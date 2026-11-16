"""Portable descriptive aggregation of ordinary timing and trace-derived accounting."""
import argparse
import json
from pathlib import Path
from statistics import median


def span(values):
    values=list(values);return dict(min=min(values),median=median(values),max=max(values))


def summarize(data):
    pairs=data['pairings'];groups=[]
    for key in dict.fromkeys((r['seed'],r['batch'],r['dropout']) for r in pairs):
        rows=[r for r in pairs if (r['seed'],r['batch'],r['dropout'])==key]
        groups.append(dict(seed=key[0],batch=key[1],dropout=key[2],n=len(rows),
            savings=span(r['control_saving_percent'] for r in rows),positive=sum(r['control_saving_percent']>0 for r in rows),
            compatible_ms=span(r['compatible_ms_per_update'] for r in rows),control_ms=span(r['control_ms_per_update'] for r in rows),
            compatible_first=span(r['control_saving_percent'] for r in rows if r['compatible_first']),
            control_first=span(r['control_saving_percent'] for r in rows if not r['compatible_first']),
            learning_exact=all(r['learning_exact'] for r in rows)))
    profiles=data['profiles'];compat=[r for r in profiles if r['arm']=='compatible']
    return dict(groups=groups,all_pair_saving_range=span(r['control_saving_percent'] for r in pairs),
        positive_pairs=sum(r['control_saving_percent']>0 for r in pairs),pairs=len(pairs),
        compatible_copy_gpu_share=span(r['compatibility_gpu_share_percent'] for r in compat),
        event_overhead=span(r['event_wall_overhead_percent'] for r in profiles),
        profiler_overhead=span(r['profiler_session_overhead_percent'] for r in profiles),
        trace_gpu_phase_shares={phase:span(r['gpu_phase_percent'].get(phase,0) for r in data['trace_reconstruction']) for phase in ('forward_loss','backward','optimizer','unresolved')},
        direct_copy_gpu_us_per_update=span(r['compatibility_gpu_us']/data['spec']['profile_active_updates'] for r in data['trace_reconstruction'] if r['arm']=='compatible'),
        scope='Descriptive fixed-batch timing; mixed order/noise and changed normal-dropout learning preclude equivalent-method or accuracy claims.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('data',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists();args.output.write_text(json.dumps(summarize(json.loads(args.data.read_text(encoding='utf8'))),indent=2)+'\n',encoding='utf8')
