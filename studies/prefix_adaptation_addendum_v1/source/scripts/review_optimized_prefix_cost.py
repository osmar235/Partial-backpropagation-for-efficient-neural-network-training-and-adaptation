"""Reconstruct paired training-block costs; no quality or novelty claim."""
import argparse
import math
from pathlib import Path
from statistics import median
import optimized_prefix_cost as p


def stats(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values))


def pair_cost(reference, candidate):
    assert reference > 0 and candidate > 0
    return 100 * (1 - candidate / reference)


def audit(out, write=False):
    out = Path(out).resolve(strict=True)
    cfg = p.read(out / 'configuration.json'); spec = cfg['spec']
    assert cfg['fingerprint'] == p.c.r.v.digest({k: v for k, v in cfg.items() if k != 'fingerprint'})
    assert cfg['sources'] == p.sources()
    p.c.r.t.verify(cfg['evidence'])
    if not cfg['smoke']: assert cfg == p.read(p.FREEZE)
    if not write: assert p.c.r.d.validate_complete(out, cfg['fingerprint'])
    rows = []; records = {}; retries = []; processes = []
    for session in range(spec['sessions']):
        folder = out / f'session_{session:02d}'
        marker = p.c.r.d.validate_complete(folder, cfg['fingerprint']); assert marker
        attempt = Path(marker['attempt']).resolve(strict=True)
        assert attempt.parent == folder and attempt.name.startswith('attempt_')
        if len(list(folder.glob('attempt_*'))) != 1: retries.append(session)
        assert set(marker['sha256']) == {str(q) for q in attempt.rglob('*.json')}
        process = p.read(attempt / 'process.json'); assert process['session'] == session
        processes.append(process['uuid'])
        for position, (seed, arm) in enumerate(p.schedule(spec, session)):
            loc = attempt / f'{position:02d}_{seed}_{arm["name"]}'
            sealed = p.c.r.d.validate_complete(loc, cfg['fingerprint']); assert sealed
            assert set(sealed['sha256']) == {str(loc / name) for name in ('summary.json', 'steps.json', 'components.json')}
            s = p.read(loc / 'summary.json'); steps = p.read(loc / 'steps.json'); comp = p.read(loc / 'components.json')
            assert s['arm'] == arm and s['seed'] == seed and s['updates'] == spec['updates']
            assert s['seconds'] > 0 and math.isfinite(s['seconds'])
            assert s['milliseconds_per_update'] == 1000*s['seconds']/spec['updates']
            assert len(steps) == spec['updates'] and len(comp) == spec['component_updates']
            for index, step in enumerate(steps):
                assert step['step'] == index and step['full'] == p.full_step(arm, index)
                assert math.isfinite(step['loss']) and step['loss'] > 0 and step['batch_examples'] > 0
                assert step['components'] == {}
            for index, row in enumerate(comp):
                assert row['step'] == index and row['full'] == p.full_step(arm, index)
                assert set(row['components']) == {'augmentation', 'routing_rng_zero_grad', 'forward_loss', 'loss_readback', 'backward', 'optimizer'}
                assert all(math.isfinite(v) and v >= 0 for v in row['components'].values())
                assert row['loss'] == steps[index]['loss'], 'Component replay differs from primary trajectory'
            assert s['integrity']['status'] == 'PASS'
            assert s['integrity']['full_updates'] == sum(p.full_step(arm, i) for i in range(spec['updates']))
            assert s['integrity']['optimizer_ages_verified'] and s['integrity']['inactive_gradients_absent']
            s['component_mean_ms'] = {key: 1000*sum(r['components'][key] for r in comp)/len(comp) for key in comp[0]['components']}
            records[session, seed, arm['name']] = (s, steps)
        for seed in spec['seeds']:
            ref, ref_steps = records[session, seed, 'full']
            routed, routed_steps = records[session, seed, 'routed_full']
            for key in ('final_model_hash', 'final_optimizer_hash'):
                assert ref[key] == routed[key], 'Routed-full control must exactly reproduce full'
            assert ref_steps == routed_steps
            for arm in p.arms(spec)[1:]:
                s, steps = records[session, seed, arm['name']]
                for key in ('anchor_model_hash', 'anchor_optimizer_hash', 'stream_fingerprint'):
                    assert s[key] == ref[key]
                assert [r['batch_examples'] for r in steps] == [r['batch_examples'] for r in ref_steps]
                row = dict(session=session, seed=seed, arm=arm['name'], boundary=arm['boundary'], interval=arm['interval'],
                           reference_seconds=ref['seconds'], candidate_seconds=s['seconds'],
                           saving_percent=pair_cost(ref['seconds'], s['seconds']),
                           component_mean_ms=s['component_mean_ms'], reference_component_mean_ms=ref['component_mean_ms'])
                if arm['interval'] > 1:
                    suffix = records[session, seed, f'cut{arm["boundary"]}_r0'][0]
                    fraction = 1/arm['interval']
                    linear = fraction*routed['seconds'] + (1-fraction)*suffix['seconds']
                    row.update(zero_extra_overhead_mixture_seconds=linear,
                               measured_minus_linear_ms_per_update=1000*(s['seconds']-linear)/spec['updates'],
                               zero_extra_overhead_mixture_saving_percent=pair_cost(ref['seconds'], linear))
                rows.append(row)
    assert len(set(processes)) == spec['sessions']
    groups = []
    for arm in p.arms(spec)[1:]:
        for seed in spec['seeds']:
            group = [r for r in rows if r['arm'] == arm['name'] and r['seed'] == seed]
            values = [r['saving_percent'] for r in group]
            groups.append(dict(arm=arm['name'], seed=seed, saving_percent=stats(values),
                               positive_sessions=sum(v > 0 for v in values), sessions=len(values),
                               descriptive_consistent_headroom=not cfg['smoke'] and not retries and all(v > 0 for v in values)))
    result = dict(status='PASS', smoke=cfg['smoke'], sessions=spec['sessions'], arms=len(p.arms(spec)),
                  timed_blocks=len(records), measured_updates=len(records)*spec['updates'],
                  paired_comparisons=len(rows), retried_sessions=retries, routed_full_exact=True,
                  scope='training-block cost only; no validation, quality qualification or complete adaptation speedup', groups=groups)
    if write:
        save = p.save
        save(out / 'audit.json', result); save(out / 'pairings.json', rows)
        lines = ['# Optimized prefix/suffix cost screen', '',
                 'Integrity PASS. This measures training-block headroom only; no quality or complete adaptation claim.',
                 'Negative savings and retries are retained. Repeated sessions are not model seeds.', '',
                 '| Arm | Source | Median saving | Range | Positive sessions |', '|---|---:|---:|---:|---:|']
        for g in groups:
            v = g['saving_percent']
            lines.append(f'| {g["arm"]} | {g["seed"]} | {v["median"]:.2f}% | {v["minimum"]:.2f} to {v["maximum"]:.2f}% | {g["positive_sessions"]}/{g["sessions"]} |')
        lines += ['', 'r0 = suffix-only, r4/r16 = a full update at steps 0, R, 2R, ... .',
                  'Cuts freeze the stem and first k blocks; final layer norm and classifier always train.',
                  'All inputs get a fresh full forward. No cached activations or adaptive gate.',
                  'Component clocks synchronize individually and are diagnostic, not a decomposition of the primary wall clock.',
                  'Mixture costs assume no additional decision overhead and are projections, not quality guarantees.',
                  f'Retried sessions: {retries}. No inferential significance or best-boundary claim.']
        (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    else:
        assert result == p.read(out / 'audit.json')
        assert rows == p.read(out / 'pairings.json')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, default=p.OUT); args = ap.parse_args()
    assert __debug__; p.c.r.m.configure(); print(audit(args.output))
