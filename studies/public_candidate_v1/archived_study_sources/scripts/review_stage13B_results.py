"""Post-run verification and saved-state rescoring only; no optimizer updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/stage13B_reference_v1'
OUT = ROOT / 'results/stage13B_review_v1'
MUTABLE = {str(ROOT / n) for n in ('CURRENT_RELEASE.md', 'README_STAGE9.md')}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    with Path(path).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def save(name, rows):
    with (OUT / name).open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def metrics(z, labels):
    a = z.astype(np.float64)
    maximum = a.max(1)
    loss = maximum + np.log(np.exp(a - maximum[:, None]).sum(1)) - a[np.arange(len(labels)), labels]
    return float(loss.mean()), float(np.mean(z.argmax(1) == labels))


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'review_artifacts_sha256.json').exists(), 'Review already sealed'
    protected = json.loads((ROOT / 'protocol/stage13b/preparation_preservation_v1.json').read_text())['prior_immutable_sha256']
    manifest = ROOT / 'protocol/stage13b/preparation_sha256_v1.json'
    entries = json.loads(manifest.read_text())
    for path, h in {**protected, **entries}.items():
        assert sha(path) == h, path
    protected.update({path: h for path, h in entries.items() if path not in MUTABLE})
    protected[str(manifest)] = sha(manifest)
    protected.update({str(path): sha(path) for path in RAW.rglob('*') if path.is_file()})
    save('preserved_inputs_sha256.csv', [dict(path=path, sha256=h) for path, h in sorted(protected.items())])
    dump('mutable_pointers_before.json', {path: sha(path) for path in sorted(MUTABLE)})

    import torch
    import stage13b_reference as r
    from review_stage13B import audit
    r.p.execution.configure()
    cfg = json.loads((RAW / 'configuration.json').read_text())
    assert not cfg['synthetic']
    r.d.validate_complete(RAW, cfg['fingerprint'])
    frozen = audit(RAW, False); dump('frozen_audit_recheck.json', frozen)
    assert frozen['reference_reproduction_pass'] and frozen['both_seed_reference_qualified']
    record = json.loads(r.FREEZE.read_text())
    attempt = Path(json.loads((RAW / 'RAW_COMPLETE.json').read_text())['attempt'])
    saved = {(int(row['seed']), row['arm'], int(row['step'])): row for row in read(RAW / 'curves.csv')}
    curves, summaries, traces = [], [], []
    for seed in cfg['spec']['seeds']:
        original = [row for row in read(record['checkpoints'][str(seed)]['trace']) if int(row['epoch']) == 3]
        for arm in cfg['spec']['arms']:
            rows = read(attempt / f'steps_{seed}_{arm}.csv')
            errors = [abs(float(a['loss']) - float(b['loss'])) for a, b in zip(rows, original)]
            assert len(rows) == len(original) == 352
            if arm == 'continuation':
                assert max(errors) == 0
            traces.append(dict(seed=seed, arm=arm, steps=len(rows), exactly_equal_losses=sum(e == 0 for e in errors), max_abs_error=max(errors)))
            points = {}
            with np.load(attempt / f'scores_{seed}_{arm}.npz', allow_pickle=False) as data:
                for step in cfg['spec']['score_steps']:
                    loss, accuracy = metrics(data[f'step_{step}'], data['labels'])
                    expected = saved[seed, arm, step]
                    assert abs(loss - float(expected['loss'])) <= 1e-6
                    assert accuracy == float(expected['accuracy'])
                    points[step] = dict(loss=loss, accuracy=accuracy)
                    curves.append(dict(seed=seed, arm=arm, step=step, loss_float64=loss, accuracy=accuracy))
            before, first, final = points[0], points[1], points[352]
            recovered = [s for s in cfg['spec']['score_steps'] if s > 0 and points[s]['loss'] < before['loss'] and points[s]['accuracy'] > before['accuracy']]
            assert first['loss'] > before['loss'] and first['accuracy'] < before['accuracy']
            assert final['loss'] < before['loss'] and final['accuracy'] > before['accuracy']
            summaries.append(dict(seed=seed, arm=arm, initial_loss=before['loss'], first_step_loss=first['loss'], final_loss=final['loss'],
                initial_accuracy=before['accuracy'], first_step_accuracy=first['accuracy'], final_accuracy=final['accuracy'],
                first_step_accuracy_change_pp=100*(first['accuracy']-before['accuracy']),
                final_accuracy_gain_pp=100*(final['accuracy']-before['accuracy']),
                first_observed_joint_recovery_step=min(recovered)))
    save('independent_curves.csv', curves); save('independent_summary.csv', summaries); save('trace_reproduction.csv', traces)

    # Reload all four saved final models and both sources; no training or updates.
    device = torch.device('cuda'); assert torch.cuda.is_available()
    x, y, ev, ey, split = r.p.load_data(device, False); del x, y
    assert split == cfg['split']
    rescored = []
    for seed in cfg['spec']['seeds']:
        for arm in ('source', 'continuation', 'reset'):
            path = Path(record['checkpoints'][str(seed)]['path']) if arm == 'source' else attempt / f'final_{seed}_{arm}.pt'
            cp = torch.load(path, map_location='cpu', weights_only=True)
            model = r.p.make_model(seed, False).to(device); model.load_state_dict(cp['model_state'], strict=True)
            before_hash = r.old.state_hash(model.state_dict())
            logits = r.predict(model, ev, cfg['recipe'])
            loss, accuracy = metrics(logits, ey.cpu().numpy())
            comparison_arm = 'continuation' if arm == 'source' else arm
            step = 0 if arm == 'source' else 352
            with np.load(attempt / f'scores_{seed}_{comparison_arm}.npz', allow_pickle=False) as data:
                assert np.array_equal(data['labels'], ey.cpu().numpy())
                error = float(np.max(np.abs(logits - data[f'step_{step}'])))
                assert error <= 1e-5
            assert accuracy == float(saved[seed, comparison_arm, step]['accuracy'])
            assert abs(loss - float(saved[seed, comparison_arm, step]['loss'])) <= 1e-6
            assert r.old.state_hash(model.state_dict()) == before_hash
            rescored.append(dict(seed=seed, arm=arm, checkpoint=str(path), sha256=sha(path),
                validation_loss_float64=loss, validation_accuracy=accuracy, max_abs_logit_error=error))
            del model, cp
    save('saved_state_rescore.csv', rescored)
    del ev, ey

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.7), sharex=True)
    for column, seed in enumerate(cfg['spec']['seeds']):
        for arm, color, label in [('continuation', '#1769aa', 'Restore Adam history'), ('reset', '#d97813', 'Reset Adam history')]:
            rows = [row for row in curves if row['seed'] == seed and row['arm'] == arm]
            steps = [row['step'] for row in rows]
            axes[0, column].plot(steps, [row['loss_float64'] for row in rows], 'o-', color=color, label=label, linewidth=1.8, markersize=4)
            axes[1, column].plot(steps, [100*row['accuracy'] for row in rows], 'o-', color=color, linewidth=1.8, markersize=4)
        initial = next(row for row in curves if row['seed'] == seed and row['step'] == 0)
        axes[0, column].axhline(initial['loss_float64'], linestyle=':', color='#666666', label='Starting level')
        axes[1, column].axhline(100*initial['accuracy'], linestyle=':', color='#666666')
        axes[0, column].set_title(f'Source seed {seed}')
        for row in (0, 1):
            ax = axes[row, column]; ax.set_xscale('symlog', linthresh=1); ax.grid(alpha=.18)
            ax.set_xticks(cfg['spec']['score_steps']); ax.set_xticklabels(cfg['spec']['score_steps'])
        axes[0, column].set_ylim(.2, 1.7); axes[1, column].set_ylim(54, 94)
        axes[1, column].set_xlabel('Full-backpropagation updates (nonlinear spacing)')
    axes[0, 0].set_ylabel('Development cross entropy'); axes[1, 0].set_ylabel('Development accuracy (%)')
    axes[0, 1].legend(fontsize=9, loc='upper right')
    fig.suptitle('A harmful first update can precede a successful training epoch', fontsize=14)
    fig.text(.5, .01, 'Same previously used 5,000-example development set; fixed 352-update endpoint. No local-learning arm.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(OUT / 'reference_recovery.png', dpi=180)
    fig.savefig(OUT / 'reference_recovery.svg')
    plt.close(fig)
    for path, h in protected.items():
        assert sha(path) == h, path
    dump('independent_audit.json', dict(status='PASS', trajectories=4, updates_verified=1408,
        original_training_losses_reproduced_exactly=704, saved_states_rescored=6,
        max_abs_rescored_logit_error=max(row['max_abs_logit_error'] for row in rescored),
        source_and_final_scores_reproduced=True, all_four_trajectories_initially_worse=True,
        all_four_trajectories_endpoint_better=True, both_seed_reference_qualified=True,
        prior_and_raw_files_preserved=len(protected), parameter_updates_performed=False,
        development_scoring_performed=True, test_scoring_performed=False, local_learning_tested=False))
    print(json.dumps(json.loads((OUT / 'independent_audit.json').read_text()), indent=2))


if __name__ == '__main__':
    main()
