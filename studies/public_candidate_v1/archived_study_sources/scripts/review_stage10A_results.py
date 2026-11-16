"""Post-run review; preserves frozen Stage10A and independently checks outcomes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/stage10A_representation_v1'
OUT = ROOT / 'results/stage10A_review_v1'
MUTABLE = {str(ROOT / p) for p in ['CURRENT_RELEASE.md', 'README_STAGE9.md']}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(name, rows):
    with (OUT / name).open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10), (a, b)


def crossing(rows, threshold, length):
    first = None
    streak = 0
    for row in rows:
        if float(row['validation_accuracy']) >= threshold:
            first = first or row
            streak += 1
        else:
            streak = 0
        if streak == length:
            return first, row
    return first, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rescore-cuda', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((RAW / 'configuration.json').read_text())
    spec = cfg['spec']
    assert not cfg['synthetic']
    protected = {}

    def verify(path, expected):
        assert sha(path) == expected, path
        protected[str(path)] = expected

    for p in RAW.rglob('*'):
        if p.is_file():
            verify(p, sha(p))
    prior = json.loads((ROOT / 'protocol/stage10/preservation_check_v1.json').read_text())
    for p, h in prior['prior_immutable_sha256'].items():
        verify(p, h)
    baseline = OUT / 'preserved_inputs_sha256.csv'
    for r in read(ROOT / 'protocol/stage10/preparation_sha256_v1.csv'):
        if r['path'] in MUTABLE:
            if not baseline.exists():
                assert sha(r['path']) == r['sha256'], r['path']
        else:
            verify(r['path'], r['sha256'])
    for p, h in cfg['source_sha256'].items():
        verify(p, h)
    if not (OUT / 'mutable_pointers_before.json').exists():
        dump('mutable_pointers_before.json', {p: sha(p) for p in sorted(MUTABLE)})
    if baseline.exists():
        for r in read(baseline):
            verify(r['path'], r['sha256'])
    else:
        save('preserved_inputs_sha256.csv', [dict(path=p, sha256=h) for p, h in sorted(protected.items())])

    import stage10_models as m
    import torch
    from review_stage10A import audit
    m.configure()
    audited = audit(RAW, write=False)
    assert audited == json.loads((RAW / 'audit.json').read_text())
    assert audited['audited_updates'] == 176000 and audited['runs'] == 10
    dump('frozen_audit_recheck.json', audited)

    records, targets, cp_paths = {}, {}, {}
    target_rows, comparisons, screens = [], [], []
    saved_targets = {(int(r['seed']), r['variant'], r['role']): r for r in read(RAW / 'targets.csv')}
    for seed in spec['development_seeds']:
        for variant in spec['variants']:
            marker = json.loads((RAW / f'seed_{seed}/{variant}/COMPLETE.json').read_text())
            attempt = Path(marker['attempt'])
            summary = json.loads((attempt / 'summary.json').read_text())
            records[seed, variant] = summary
            cp_paths[seed, variant] = attempt / 'final_checkpoint.pt'
            epochs = read(attempt / 'epochs.csv')
            for role in ['primary', 'secondary']:
                threshold = spec[role + '_target']
                first, confirmed = crossing(epochs, threshold, spec['sustained_epochs'])
                r = dict(seed=seed, variant=variant, role=role, threshold=threshold,
                         reached=confirmed is not None,
                         first_crossing_epoch=int(first['epoch']) if first else '',
                         confirming_epoch=int(confirmed['epoch']) if confirmed else '',
                         observed_seconds=float(confirmed['cumulative_observed_seconds']) if confirmed else '',
                         training_seconds=float(confirmed['cumulative_training_seconds']) if confirmed else '',
                         censoring_epoch='' if confirmed else spec['epochs'],
                         censoring_observed_seconds='' if confirmed else summary['observed_seconds'])
                original = saved_targets[seed, variant, role]
                for k in r:
                    if isinstance(r[k], bool) or isinstance(r[k], str):
                        assert str(r[k]) == original[k], (k, r[k], original[k])
                    else:
                        close(r[k], original[k])
                target_rows.append(r)
                if role == 'primary':
                    targets[seed, variant] = r
        full = records[seed, 'full_rgb32']
        ft = targets[seed, 'full_rgb32']
        controls = [records[seed, v] for v in ['full_rgb32', 'small_rgb32', 'coarse_rgb16']]
        best_accuracy = max(r['final_validation_accuracy'] for r in controls)
        best = min((targets[seed, r['variant']] for r in controls if targets[seed, r['variant']]['reached']),
                   key=lambda r: (r['observed_seconds'], r['variant']))
        for variant in spec['variants']:
            a, at = records[seed, variant], targets[seed, variant]
            comparisons.append(dict(seed=seed, variant=variant, parameter_count=a['parameter_count'],
                                    final_accuracy=a['final_validation_accuracy'],
                                    final_delta_vs_full_pp=100 * (a['final_validation_accuracy'] - full['final_validation_accuracy']),
                                    training_seconds=a['training_seconds'],
                                    training_ratio_to_full=a['training_seconds'] / full['training_seconds'],
                                    primary_reached=at['reached'], confirming_epoch=at['confirming_epoch'],
                                    observed_target_seconds=at['observed_seconds'],
                                    target_time_ratio_to_full=at['observed_seconds'] / ft['observed_seconds'] if at['reached'] and ft['reached'] else ''))
        a, at = records[seed, 'haar_branches'], targets[seed, 'haar_branches']
        coarse, pixel = records[seed, 'coarse_rgb16'], records[seed, 'pixel_branches']
        rf = at['observed_seconds'] / ft['observed_seconds'] if at['reached'] and ft['reached'] else None
        rb = at['observed_seconds'] / best['observed_seconds'] if at['reached'] else None
        delta = 100 * (a['final_validation_accuracy'] - best_accuracy)
        detail = 100 * (a['final_validation_accuracy'] - coarse['final_validation_accuracy'])
        structure = 100 * (a['final_validation_accuracy'] - pixel['final_validation_accuracy'])
        cost = a['training_seconds'] / pixel['training_seconds']
        practical_pass = rf is not None and rb is not None and max(rf, rb) <= 1 - spec['minimum_time_gain'] and delta >= -spec['final_accuracy_margin_pp']
        detail_pass = detail >= spec['minimum_detail_gain_pp']
        structure_pass = structure >= spec['minimum_structure_gain_pp'] and cost <= spec['maximum_structure_training_cost_ratio']
        screens.append(dict(seed=seed, full_target_ready=ft['reached'], haar_target_reached=at['reached'],
                            fastest_ordinary_control=best['variant'], target_time_ratio_to_full=rf if rf is not None else '',
                            target_time_ratio_to_best_ordinary=rb if rb is not None else '',
                            final_delta_vs_best_ordinary_pp=delta, detail_gain_vs_coarse_pp=detail,
                            structure_gain_vs_pixel_pp=structure, training_ratio_vs_pixel=cost,
                            practical_pass=practical_pass, detail_pass=detail_pass, structure_pass=structure_pass,
                            all_flags_pass=practical_pass and detail_pass and structure_pass))
    for r, original in zip(screens, read(RAW / 'screen.csv')):
        for k, value in r.items():
            if isinstance(value, bool) or isinstance(value, str):
                assert str(value) == original[k], (k, value, original[k])
            else:
                close(value, original[k])
    aggregate = []
    for variant in spec['variants']:
        a = [r for r in comparisons if r['variant'] == variant]
        aggregate.append(dict(variant=variant, n=len(a), parameter_count=a[0]['parameter_count'],
                              mean_final_accuracy=st.mean(r['final_accuracy'] for r in a),
                              mean_final_delta_vs_full_pp=st.mean(r['final_delta_vs_full_pp'] for r in a),
                              mean_training_seconds=st.mean(r['training_seconds'] for r in a),
                              primary_reached=sum(r['primary_reached'] for r in a)))
    save('independent_targets.csv', target_rows)
    save('independent_seed_results.csv', comparisons)
    save('independent_mean_results.csv', aggregate)
    save('independent_screen.csv', screens)
    if args.rescore_cuda:
        import stage8_vit_v2 as v
        assert torch.cuda.is_available()
        preflight = v.idle_preflight()
        x, y, ev, ey, split = v.load_data('development', torch.device('cuda'))
        assert split == cfg['split']
        del x, y
        rescored = []
        for (seed, variant), path in cp_paths.items():
            cp = torch.load(path, map_location='cpu', weights_only=True)
            model = m.make_model(variant, seed).to('cuda')
            model.load_state_dict(cp['model_state'])
            loss, accuracy = v.evaluate(model, ev, ey, spec['eval_batch_size'])
            expected = records[seed, variant]
            assert abs(accuracy - expected['final_validation_accuracy']) < 1e-12, (seed, variant)
            assert abs(loss - expected['final_validation_loss']) < 1e-6, (seed, variant)
            rescored.append(dict(seed=seed, variant=variant, checkpoint_sha256=sha(path),
                                 validation_accuracy=accuracy, validation_loss=loss,
                                 recorded_accuracy=expected['final_validation_accuracy'], recorded_loss=expected['final_validation_loss']))
            del model, cp
        save('checkpoint_validation_rescore.csv', rescored)
        dump('rescore_metadata.json', dict(status='PASS', device=torch.cuda.get_device_name(0),
                                         test_dataset_loaded=False, training_performed=False, preflight=preflight,
                                         execution=m.execution_settings()))
    for p, h in protected.items():
        assert sha(p) == h, p
    result = dict(status='PASS', runs=10, updates=176000, independent_targets_and_screens_match=True,
                  full_baselines_ready=all(r['full_target_ready'] for r in screens),
                  all_candidate_flags_failed_both_seeds=all(not r['practical_pass'] and not r['detail_pass'] and not r['structure_pass'] for r in screens),
                  preserved_immutable_files=len(protected),
                  final_checkpoint_validation_rescored=(OUT / 'checkpoint_validation_rescore.csv').exists(),
                  decision='Reject this tested coarse/detail design; no automatic tuning or optimizer hybrid. Publication on hold.')
    dump('independent_audit.json', result)
    print(json.dumps(result, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
