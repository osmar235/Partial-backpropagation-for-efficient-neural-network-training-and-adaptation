"""Portable, standard-library-only descriptive cost analysis."""
import argparse
import json
from pathlib import Path
from statistics import median


def describe(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values))


def summarize(data):
    spec = data['spec']; records = data['records']
    by_key = {(r['session'], r['summary']['seed'], r['summary']['arm']['name']): r for r in records}
    pairs = []
    for record in records:
        s = record['summary']; session = record['session']; seed = s['seed']; arm = s['arm']
        ref = by_key[session, seed, 'full']['summary']; routed = by_key[session, seed, 'routed_full']['summary']
        row = dict(session=session, seed=seed, arm=arm['name'], boundary=arm['boundary'], interval=arm['interval'],
                   milliseconds_per_update=1000*s['seconds']/spec['updates'],
                   saving_vs_full_percent=100*(ref['seconds']-s['seconds'])/ref['seconds'],
                   saving_vs_routed_percent=100*(routed['seconds']-s['seconds'])/routed['seconds'],
                   saving_vs_faster_control_percent=100*(min(ref['seconds'], routed['seconds'])-s['seconds'])/min(ref['seconds'], routed['seconds']),
                   peak_allocated_mib=s['peak_allocated_bytes']/2**20,
                   components_mean_ms={key: 1000*sum(z['components'][key] for z in record['components'])/len(record['components']) for key in record['components'][0]['components']})
        if arm['interval'] > 1:
            suffix = by_key[session, seed, f'cut{arm["boundary"]}_r0']['summary']
            projected = (routed['seconds'] + (arm['interval']-1)*suffix['seconds'])/arm['interval']
            row['measured_minus_mixture_ms_per_update'] = 1000*(s['seconds']-projected)/spec['updates']
        pairs.append(row)
    groups = []
    for arm in dict.fromkeys(r['summary']['arm']['name'] for r in records):
        for seed in spec['seeds']:
            rows = [r for r in pairs if r['arm'] == arm and r['seed'] == seed]
            group = dict(arm=arm, seed=seed, n=len(rows))
            for key in ('milliseconds_per_update', 'saving_vs_full_percent', 'saving_vs_routed_percent',
                        'saving_vs_faster_control_percent', 'peak_allocated_mib'):
                group[key] = describe([r[key] for r in rows])
            group['positive_vs_full'] = sum(r['saving_vs_full_percent'] > 0 for r in rows)
            group['positive_vs_both_controls'] = sum(r['saving_vs_faster_control_percent'] > 0 for r in rows)
            group['components_mean_ms'] = {key: describe([r['components_mean_ms'][key] for r in rows]) for key in rows[0]['components_mean_ms']}
            if 'measured_minus_mixture_ms_per_update' in rows[0]:
                group['measured_minus_mixture_ms_per_update'] = describe([r['measured_minus_mixture_ms_per_update'] for r in rows])
            hashes = [r['summary'] for r in records if r['summary']['arm']['name'] == arm and r['summary']['seed'] == seed]
            group['identical_across_sessions'] = all(len({r[key] for r in hashes}) == 1 for key in ('anchor_model_hash', 'anchor_optimizer_hash', 'final_model_hash', 'final_optimizer_hash', 'stream_fingerprint'))
            groups.append(group)
    return dict(groups=groups, pairings=pairs, faster_control_comparison='post hoc sensitivity analysis; not a new qualification gate')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('input', type=Path); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); assert not args.output.exists()
    args.output.write_text(json.dumps(summarize(json.loads(args.input.read_text(encoding='utf8'))), indent=2)+'\n', encoding='utf8')
