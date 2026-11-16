"""Independent arithmetic/provenance checks for the fixed token diagnostic."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import stage8_vit_v2 as v
import stage9_boundary_diagnostic as d


def screen(accuracy, full_accuracy, time_ratio, memory_ratio, spec, ready, synthetic):
    quality = 100 * (full_accuracy - accuracy) <= spec['accuracy_margin_pp'] + 1e-10
    cost = time_ratio <= 1 - spec['minimum_time_saving'] or (
        memory_ratio is not None and memory_ratio <= 1 - spec['minimum_incremental_memory_saving'])
    return dict(quality_pass=quality, resource_pass=cost,
                seed_screen_pass=bool(ready and quality and cost and not synthetic))


def audit(out, write=False):
    import stage14_diagnostic as run
    out = Path(out)
    cfg = json.loads((out / 'configuration.json').read_text())
    cfg_body = {k: val for k, val in cfg.items() if k != 'fingerprint'}
    assert v.digest(cfg_body) == cfg['fingerprint']
    spec, fake = cfg['spec'], cfg['synthetic']
    freeze = json.loads((out / 'SYNTHETIC_FREEZE.json' if fake else run.FREEZE).read_text())
    assert v.digest({k: val for k, val in freeze.items() if k != 'freeze_hash'}) == freeze['freeze_hash'] == cfg['freeze_hash']
    assert freeze['spec'] == spec and freeze['synthetic'] == fake and freeze['expected_split'] == cfg['split']
    run.verify(freeze['source_sha256']); run.verify(freeze['evidence_sha256'])
    for cp in freeze['checkpoints'].values():
        run.verify({cp['path']: cp['sha256']})
    raw = json.loads((out / 'RAW_COMPLETE.json').read_text())
    assert raw['fingerprint'] == cfg['fingerprint']
    run.verify(raw['sha256'])
    attempt = Path(raw['attempt'])
    assert {str(p) for p in attempt.iterdir() if p.is_file()} == set(raw['sha256'])
    data = json.loads((attempt / 'data.json').read_text())
    assert data['split'] == cfg['split']
    assert data['validation_examples'] == (8 if fake else 5000)
    assert data['training_examples'] == (16 if fake else 45000)
    # Reconstruct labels from the original train-only development loader.
    _, _, _, expected_labels, expected_split = v.load_data('development', torch.device('cpu'), fake=fake)
    assert expected_split == cfg['split']
    summaries, all_passes = [], {'spatial': [], 'similarity': []}
    measured_count = warmup_count = 0
    for seed_position, seed in enumerate(spec['seeds']):
        state = json.loads((attempt / f'seed_{seed}.json').read_text())
        assert state['seed'] == seed and state['state_before'] == state['state_after'] and state['optimizer_updates'] == 0
        source_model = run.make_core(seed, fake, freeze, torch.device('cpu'))
        assert run.old.state_hash(source_model.state_dict()) == state['state_before']
        del source_model
        if not fake:
            assert state['source_sha256'] == freeze['checkpoints'][str(seed)]['sha256']
        saved = json.loads((attempt / f'streams_{seed}.json').read_text())
        assert saved == run.streams(seed, spec, data['training_examples'])
        flat = [i for row in saved for i in row]
        assert len(flat) == len(set(flat))
        with np.load(attempt / f'predictions_{seed}.npz', allow_pickle=False) as pred:
            assert set(pred.files) == {'labels', 'full', 'spatial', 'similarity'}
            labels = pred['labels']
            assert np.array_equal(labels, expected_labels.numpy())
            metrics = {}
            for mode in spec['modes']:
                z = pred[mode]
                assert z.shape == (len(labels), 10) and np.isfinite(z).all()
                metrics[mode] = dict(accuracy=float(np.mean(z.argmax(1) == labels)),
                    loss=float(F.cross_entropy(torch.from_numpy(z), torch.from_numpy(labels))),
                    prediction_changes=int(np.sum(z.argmax(1) != pred['full'].argmax(1))),
                    logit_rms_change=float(np.sqrt(np.mean((z.astype(np.float64) - pred['full']) ** 2))))
        ready = metrics['full']['accuracy'] >= spec['baseline_min_accuracy']
        if not fake:
            cp = freeze['checkpoints'][str(seed)]
            assert abs(metrics['full']['accuracy'] - cp['accuracy']) < 1e-12
            assert abs(metrics['full']['loss'] - cp['loss']) <= spec['source_loss_tolerance']
            assert ready
        rows = d.records(attempt / f'measurements_{seed}.csv')
        warm = d.records(attempt / f'warmups_{seed}.csv')
        expected_order = []
        for trial in range(spec['trials']):
            kinds = spec['operations'] if trial % 2 == 0 else list(reversed(spec['operations']))
            for kind in kinds:
                for position, mode in enumerate(run.order(spec, seed_position, trial)):
                    expected_order.append((trial, kind, mode, position))
        assert [(int(r['trial']), r['operation'], r['mode'], int(r['position'])) for r in rows] == expected_order
        assert [(r['operation'], r['mode'], int(r['warmup'])) for r in warm] == [
            (kind, mode, w) for kind in spec['operations'] for mode in run.order(spec, seed_position, 0)
            for w in range(spec['warmups'])]
        for row in rows + warm:
            assert int(row['seed']) == seed
            assert math.isfinite(float(row['scalar'])) and float(row['seconds']) > 0
            assert math.isfinite(float(row['seconds']))
            base, peak, inc, reserved = [int(row[k]) for k in ('baseline_allocated', 'peak_allocated', 'incremental_allocated', 'peak_reserved')]
            assert inc == peak - base >= 0 and reserved >= peak
            if cfg['device'] == 'cuda':
                assert base > 0 and inc > 0
            else:
                assert base == peak == inc == reserved == 0
            assert row['all_gradients_finite'] == 'True'
            if row['operation'] == 'forward_backward':
                assert math.isfinite(float(row['patch_gradient_norm'])) and float(row['patch_gradient_norm']) > 0
        for row in rows:
            assert int(row['dropout_seed']) == seed + spec['dropout_seed_offset'] + int(row['trial'])
        measured_count += len(rows)
        warmup_count += len(warm)
        stats = {}
        for kind in spec['operations']:
            for mode in spec['modes']:
                selected = [r for r in rows if r['mode'] == mode and r['operation'] == kind]
                assert len(selected) == spec['trials']
                stats[kind, mode] = {key: float(np.mean([float(r[key]) for r in selected]))
                    for key in ('seconds', 'baseline_allocated', 'peak_allocated', 'incremental_allocated', 'peak_reserved')}
        for mode in spec['modes']:
            train, full = stats['forward_backward', mode], stats['forward_backward', 'full']
            tr = train['seconds'] / full['seconds']
            mr = train['incremental_allocated'] / full['incremental_allocated'] if full['incremental_allocated'] else None
            flags = screen(metrics[mode]['accuracy'], metrics['full']['accuracy'], tr, mr, spec, ready, fake)
            if mode != 'full':
                all_passes[mode].append(flags['seed_screen_pass'])
            row = dict(seed=seed, mode=mode, **metrics[mode], baseline_ready=ready,
                accuracy_delta_pp=100 * (metrics[mode]['accuracy'] - metrics['full']['accuracy']),
                forward_backward_time_ratio=tr, forward_backward_incremental_memory_ratio=mr, **flags)
            for kind in spec['operations']:
                row.update({kind + '_' + key: val for key, val in stats[kind, mode].items()})
            summaries.append(row)
    result = dict(status='PASS', synthetic=fake, measured_operations=measured_count, warmup_operations=warmup_count,
        optimizer_updates=0, source_checkpoints_unchanged=True, source_accuracy_reproduced=not fake,
        training_streams_and_validation_labels_verified=True,
        both_seed_review_flags={mode: all(flags) for mode, flags in all_passes.items()},
        interpretation='Artificial plumbing only' if fake else 'Exploratory immediate quality and operation cost; no training efficacy claim')
    if write:
        v.write_csv(out / 'summary.csv', summaries)
        v.atomic_json(out / 'audit.json', result)
        lines = ['# Stage14A token diagnostic', '', json.dumps(result, indent=2), '',
            '| Seed | Arm | Accuracy % | Delta pp | F+B time ratio | Incremental memory ratio | Seed filter |',
            '|---|---|---:|---:|---:|---:|---|']
        for r in summaries:
            mr = r['forward_backward_incremental_memory_ratio']
            lines.append(f"| {r['seed']} | {r['mode']} | {100*r['accuracy']:.2f} | {r['accuracy_delta_pp']:.2f} | {r['forward_backward_time_ratio']:.4f} | {mr if mr is not None else 'N/A'} | {r['seed_screen_pass']} |")
        lines += ['', 'Review before any training pilot. Similarity must justify its extra complexity against spatial merging.',
            'A failed filter parks this configuration only. Saved logits permit independent rescoring.',
            'F+B excludes optimizer steps; reported savings do not establish faster learning.', '']
        (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')
    elif (out / 'audit.json').exists():
        assert json.loads((out / 'audit.json').read_text()) == result
        saved_summary = d.records(out / 'summary.csv')
        assert len(saved_summary) == len(summaries)
        for observed, expected in zip(saved_summary, summaries):
            for key, value in expected.items():
                if value is None:
                    assert observed[key] == ''
                elif isinstance(value, (bool, str)):
                    assert observed[key] == str(value)
                else:
                    assert math.isclose(float(observed[key]), value, rel_tol=1e-10, abs_tol=1e-10)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    print(json.dumps(audit(args.output, write=False), indent=2))
