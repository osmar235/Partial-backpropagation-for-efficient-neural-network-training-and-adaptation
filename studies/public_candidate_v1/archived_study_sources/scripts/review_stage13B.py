"""Audit replay, optimizer ages, streams and fixed-endpoint reference qualification."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import stage13b_reference as r
p, v, d, old = r.p, r.v, r.d, r.old


def metrics(logits, labels):
    assert np.isfinite(logits).all()
    return dict(loss=float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels))),
                accuracy=float(np.mean(logits.argmax(1) == labels)))


def qualifies(reproduced, initial, final, synthetic):
    return bool(reproduced and final['loss'] < initial['loss'] and final['accuracy'] > initial['accuracy'] and not synthetic)


def audit(out, write=False):
    out = Path(out); cfg = json.loads((out / 'configuration.json').read_text())
    assert v.digest({k: val for k, val in cfg.items() if k != 'fingerprint'}) == cfg['fingerprint']
    spec, recipe, fake = cfg['spec'], cfg['recipe'], cfg['synthetic']
    record = json.loads((out / 'SYNTHETIC_FREEZE.json' if fake else r.FREEZE).read_text())
    assert v.digest({k: val for k, val in record.items() if k != 'freeze_hash'}) == record['freeze_hash'] == cfg['freeze_hash']
    assert record['spec'] == spec and record['recipe'] == recipe and record['split'] == cfg['split']
    r.verify(record['source_sha256']); r.verify(record['evidence_sha256'])
    for entry in record['checkpoints'].values():
        r.verify({entry['path']: entry['sha256']})
    if not fake:
        r.verify({record['pretrained']['path']: record['pretrained']['sha256']})
    raw = json.loads((out / 'RAW_COMPLETE.json').read_text())
    assert raw['fingerprint'] == cfg['fingerprint']; r.verify(raw['sha256'])
    attempt = Path(raw['attempt'])
    assert set(raw['sha256']) == {str(f) for f in attempt.iterdir() if f.is_file()}
    x, y, ev, ey, split = p.load_data(torch.device('cpu'), fake)
    assert split == cfg['split']
    labels = ey.numpy(); curves, comparisons, gates, reproduced_all = [], [], [], []
    for seed in spec['seeds']:
        source_path = attempt / f'source_{seed}.pt' if fake else Path(record['checkpoints'][str(seed)]['path'])
        cp = torch.load(source_path, map_location='cpu', weights_only=True)
        assert cp['seed'] == seed and cp['updates'] == 2 * recipe['batches_per_epoch'] and cp['arm'] == 'lp_full'
        bg = torch.Generator(); bg.set_state(cp['batch_rng'])
        expected_batches = [ix.tolist() for ix in v.frozen6.epoch_batches(len(y), recipe['batch_size'], bg)]
        assert len(expected_batches) == recipe['batches_per_epoch']
        assert sorted(i for b in expected_batches for i in b) == list(range(len(y)))
        ag = torch.Generator(); ag.set_state(cp['augmentation_rng'])
        for batch in expected_batches:
            # Exact RNG consumption of frozen Stage12 crop offsets and flips.
            torch.randint(0, 9, (len(batch), 2), generator=ag); torch.rand(len(batch), generator=ag)
        expected_trace = json.loads((attempt / f'expected_trace_{seed}.json').read_text())
        if fake:
            with np.load(attempt / f'golden_{seed}.npz', allow_pickle=False) as z:
                initial_expected, final_expected = metrics(z['initial'], labels), metrics(z['final'], labels)
        else:
            entry = record['checkpoints'][str(seed)]
            assert expected_trace == r.original_trace(entry, recipe)
            initial_expected = {k: float(entry['initial']['validation_' + k]) for k in ('loss', 'accuracy')}
            final_expected = {k: float(entry['expected']['validation_' + k]) for k in ('loss', 'accuracy')}
        assert len(expected_trace) == recipe['batches_per_epoch']
        endpoints, initial_logits = {}, None
        reproduced = False
        for arm in spec['arms']:
            stem = f'{seed}_{arm}'
            state = json.loads((attempt / f'state_{stem}.json').read_text())
            assert state['initial_hash'] == old.state_hash(cp['model_state'])
            net, opt, _, _ = r.restore(seed, cp, recipe, fake, torch.device('cpu'), arm)
            assert state['start_ages'] == r.ages(net, opt)
            assert all(age == (cp['updates'] if arm == 'continuation' and n.startswith('fc.') else 0)
                       for n, age in state['start_ages'].items())
            final = torch.load(attempt / f'final_{stem}.pt', map_location='cpu', weights_only=True)
            assert final['seed'] == seed and final['arm'] == arm and final['fingerprint'] == cfg['fingerprint']
            assert final['updates'] == 3 * recipe['batches_per_epoch']
            net.load_state_dict(final['model_state']); opt.load_state_dict(final['optimizer_state'])
            assert r.ages(net, opt) == state['final_ages']
            assert all(state['final_ages'][n] == age + recipe['batches_per_epoch'] for n, age in state['start_ages'].items())
            assert all(torch.isfinite(t).all() for t in net.parameters())
            for values in opt.state.values():
                assert all(torch.isfinite(value).all() for value in values.values() if torch.is_tensor(value))
            for n, t in net.named_buffers():
                assert torch.equal(t, cp['model_state'][n])
            assert state['batchnorm_unchanged'] and state['finite_gradients_all_steps']
            assert state['final_hash'] == old.state_hash(net.state_dict())
            assert torch.equal(final['batch_rng'], bg.get_state()) and torch.equal(final['augmentation_rng'], ag.get_state())
            assert torch.equal(final['initial_augmentation_rng'], cp['augmentation_rng'])
            batches = json.loads((attempt / f'batches_{stem}.json').read_text()); assert batches == expected_batches
            rows = d.records(attempt / f'steps_{stem}.csv'); assert len(rows) == len(batches)
            trace_passes, trace_errors = [], []
            for j, (row, expected, batch) in enumerate(zip(rows, expected_trace, batches), 1):
                assert int(row['trajectory_step']) == j and int(row['step']) == cp['updates'] + j == int(expected['step'])
                assert int(row['batch_examples']) == len(batch) == int(expected['batch_examples'])
                assert row['batch_sha256'] == old.state_hash({'indices': torch.tensor(batch)})
                assert math.isfinite(float(row['loss'])) and math.isfinite(float(row['update_seconds'])) and float(row['update_seconds']) > 0
                for key in ('backbone_lr', 'head_lr'):
                    assert math.isclose(float(row[key]), float(expected[key]), rel_tol=1e-12, abs_tol=1e-15)
                error = abs(float(row['loss']) - float(expected['loss']))
                trace_errors.append(error)
                trace_passes.append(error <= spec['trace_atol'] + spec['trace_rtol'] * abs(float(expected['loss'])))
            with np.load(attempt / f'scores_{stem}.npz', allow_pickle=False) as z:
                assert set(z.files) == {'labels'} | {f'step_{s}' for s in spec['score_steps']}
                assert np.array_equal(z['labels'], labels)
                if initial_logits is None:
                    initial_logits = z['step_0'].copy()
                else:
                    assert np.array_equal(initial_logits, z['step_0'])
                for step in spec['score_steps']:
                    logits = z[f'step_{step}']; assert logits.shape == (len(labels), 10)
                    row = dict(seed=seed, arm=arm, step=step, **metrics(logits, labels))
                    curves.append(row)
                initial, endpoint = metrics(z['step_0'], labels), metrics(z[f'step_{recipe["batches_per_epoch"]}'], labels)
            initial_reproduced = abs(initial['loss'] - initial_expected['loss']) <= spec['validation_loss_atol'] and abs(initial['accuracy'] - initial_expected['accuracy']) <= spec['accuracy_atol']
            endpoint_reproduced = abs(endpoint['loss'] - final_expected['loss']) <= spec['validation_loss_atol'] and abs(endpoint['accuracy'] - final_expected['accuracy']) <= spec['accuracy_atol']
            if arm == 'continuation':
                reproduced = initial_reproduced and endpoint_reproduced and all(trace_passes)
                reproduced_all.append(reproduced)
                gates.append(qualifies(reproduced, initial, endpoint, fake))
            endpoints[arm] = dict(initial=initial, final=endpoint)
            comparisons.append(dict(seed=seed, arm=arm, initial_accuracy=initial['accuracy'], final_accuracy=endpoint['accuracy'],
                initial_loss=initial['loss'], final_loss=endpoint['loss'], accuracy_gain_pp=100*(endpoint['accuracy']-initial['accuracy']),
                loss_improvement=initial['loss']-endpoint['loss'], initial_reproduced=initial_reproduced,
                endpoint_reproduced=endpoint_reproduced, trace_steps_matching=sum(trace_passes),
                trace_steps=len(rows), trace_max_abs_error=max(trace_errors),
                reference_qualified=qualifies(reproduced, initial, endpoint, fake) if arm == 'continuation' else False))
            del net, opt, final
        for row in comparisons[-2:]:
            row['reset_minus_continuation_accuracy_pp'] = 100 * (endpoints['reset']['final']['accuracy'] - endpoints['continuation']['final']['accuracy'])
            row['reset_minus_continuation_loss'] = endpoints['reset']['final']['loss'] - endpoints['continuation']['final']['loss']
        del cp
    result = dict(status='PASS', synthetic=fake, integrity_pass=True,
        reference_reproduction_pass=all(reproduced_all), both_seed_reference_qualified=all(gates),
        trajectories=2*len(spec['seeds']), measured_updates=2*len(spec['seeds'])*recipe['batches_per_epoch'],
        source_files_unchanged=True, streams_and_optimizer_ages_verified=True,
        local_learning_tested=False, automatic_next_experiment=False)
    if write:
        v.atomic_json(out / 'audit.json', result); v.write_csv(out / 'curves.csv', curves); v.write_csv(out / 'comparisons.csv', comparisons)
        lines = ['# Stage13B reference repair', '', 'Integrity and reference qualification are separate outcomes.', '',
            '```json', json.dumps(result, indent=2), '```', '',
            '| Seed | Arm | Initial accuracy % | Final accuracy % | Loss improvement | Reference qualified |',
            '|---|---|---:|---:|---:|---|']
        for row in comparisons:
            lines.append(f"| {row['seed']} | {row['arm']} | {100*row['initial_accuracy']:.2f} | {100*row['final_accuracy']:.2f} | {row['loss_improvement']:.6f} | {row['reference_qualified']} |")
        lines += ['', 'Review the full curves and original-trace reproduction before a local-learning comparison.',
            'Reset effects apply to this continued recipe only. No Stage13A result is overwritten.',
            'Synthetic outcomes are mechanics checks only; no efficacy inference.', '']
        (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')
    else:
        assert json.loads((out / 'audit.json').read_text()) == result
        for name, expected_rows in [('curves.csv', curves), ('comparisons.csv', comparisons)]:
            observed_rows = d.records(out / name); assert len(observed_rows) == len(expected_rows)
            for observed, expected in zip(observed_rows, expected_rows):
                for key, value in expected.items():
                    if isinstance(value, (str, bool)):
                        assert observed[key] == str(value)
                    else:
                        assert math.isclose(float(observed[key]), value, rel_tol=1e-10, abs_tol=1e-10)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('output', type=Path); args = ap.parse_args()
    print(json.dumps(audit(args.output, False), indent=2))
