"""Read-only review of the failed Stage 8 v1 development screen."""
import csv, hashlib, json, math
from pathlib import Path
import torch
import stage8_vit as v
import stage8_workflow as w


def rows(path):
    with path.open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8), (a, b)


def main():
    dev = v.ROOT/'results/stage8A_vit_development'
    out = v.ROOT/'results/stage8A_review_v1'
    assert not (dev/'calibration').exists(), 'Unexpected calibration output: review manually'
    assert not (v.ROOT/'protocol/stage8/FROZEN_EVALUATION.json').exists()
    assert not (v.ROOT/'results/stage8B_vit_evaluation').exists()
    summaries = []
    for lr in v.SPEC['learning_rates']:
        for seed in v.SPEC['development_seeds']:
            folder = dev/'screen'/w.lr_label(lr)/f'seed_{seed}'/'full_adamw'
            r = w.completed(folder, 'development', seed, 'full_adamw', lr, v.SPEC, False)
            config = json.loads((folder/'run_config.json').read_text())
            assert config['split']['kind'] == 'train_only_stratified_45000_5000'
            assert not config['tiny'] and config['device'] == 'cuda'
            steps, epochs = rows(folder/'step_trace.csv'), rows(folder/'epoch_trace.csv')
            assert len(steps) == r['updates'] == 26400 and len(epochs) == 150
            assert r['partial_updates'] == 0 and r['full_updates'] == 26400
            ema = None
            for i, s in enumerate(steps):
                assert int(s['step']) == i+1 and int(s['epoch']) == i//176+1
                assert int(s['partial']) == 0
                loss = float(s['loss']); assert math.isfinite(loss)
                if ema is None: assert s['ema_before'] == ''
                else: close(s['ema_before'], ema)
                ema = loss if ema is None else v.SPEC['rho']*ema+(1-v.SPEC['rho'])*loss
                close(s['lr'], v.learning_rate(i,26400,176,lr,v.SPEC))
                for key in ('augmentation','forward','backward','optimizer'):
                    assert float(s[key+'_seconds']) >= 0
            for e, row in enumerate(epochs):
                assert int(row['epoch']) == e+1 and int(row['updates']) == (e+1)*176
                assert int(row['full_updates']) == 176 and int(row['partial_updates']) == 0
                close(row['cumulative_training_seconds'],sum(float(x['epoch_training_seconds']) for x in epochs[:e+1]))
            close(r['training_wall_seconds'],epochs[-1]['cumulative_training_seconds'])
            close(r['final_eval_accuracy'],epochs[-1]['eval_accuracy'])
            close(r['final_eval_loss'],epochs[-1]['eval_loss'])
            for key in ('augmentation','forward','backward','optimizer'):
                close(r['component_seconds'][key],sum(float(s[key+'_seconds']) for s in steps))
            checkpoint = torch.load(folder/'final_checkpoint.pt',map_location='cpu',weights_only=True)
            assert checkpoint['fingerprint'] == r['fingerprint']
            assert checkpoint['seed'] == seed and checkpoint['variant'] == 'full_adamw'
            for state in checkpoint['optimizer_state']['state'].values():
                assert int(state['step']) == 26400
            best = max(epochs,key=lambda e:float(e['eval_accuracy']))
            summaries.append(dict(seed=seed,lr=lr,final_validation_accuracy=r['final_eval_accuracy'],
                best_validation_accuracy=float(best['eval_accuracy']),best_epoch=int(best['epoch']),
                training_seconds=r['training_wall_seconds'],updates=r['updates'],
                checkpoint_sha256=r['checkpoint_sha256']))
    assert len(list(dev.rglob('summary.json'))) == 4
    selection = json.loads((dev/'baseline_selection.json').read_text())
    assert selection['source_sha256'] == v.source_hashes() and not selection['ready']
    assert selection['selected_lr'] == .001
    for screen in selection['screen']:
        acc = [s['final_validation_accuracy'] for s in summaries if s['lr'] == screen['lr']]
        close(screen['mean_validation_accuracy'],sum(acc)/len(acc))
        close(screen['min_validation_accuracy'],min(acc))
    out.mkdir(parents=True,exist_ok=True)
    files = sorted(p for p in dev.rglob('*') if p.is_file())
    files += [Path(p) for p in v.source_hashes()]
    # Includes immutable v1 harness, protocol, raw data, checkpoint and installed model source.
    manifest = [dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    v.write_csv(out/'preserved_v1_sha256.csv',manifest)
    v.write_csv(out/'per_seed.csv',summaries)
    result = dict(status='PASS',runs=4,epochs=600,steps=105600,baseline_readiness='FAIL',
        selected_lr=.001,selected_mean_validation_accuracy=.8319,
        selected_min_validation_accuracy=.8292,source_hashes_verified=True,
        checkpoint_hashes_and_counters_verified=True,evaluation_outputs_absent=True,
        note='Audit pass verifies integrity; baseline quality did not pass readiness.')
    v.atomic_json(out/'audit.json',result)
    lines = ['# Stage 8A v1 development review','',
        'Four completed baseline runs pass source/configuration, step/epoch trace, LR schedule,',
        'checkpoint SHA-256 and AdamW-counter checks: 600 epochs / 105,600 updates.',
        'The readiness decision is correctly FAIL. No calibration, real freeze or evaluation outputs exist.','',
        '| LR | Seed | Final validation accuracy | Training seconds |','|---:|---:|---:|---:|']
    lines += [f"| {r['lr']} | {r['seed']} | {100*r['final_validation_accuracy']:.2f}% | {r['training_seconds']:.1f} |" for r in summaries]
    lines += ['', 'The selected LR 0.001 averages 83.19%, minimum 82.92%. Required: mean >=85%,',
        'minimum >=83%. LR 0.0003 averages 81.59%, minimum 81.12%. Final epochs remain',
        'the selection endpoint; diagnostic best epochs do not replace them.','',
        'This is an underperforming development recipe, not evidence about ViT LGPB performance.',
        'Learning curves do not establish the cause. Smoothed/augmented training loss and',
        'unsmoothed validation loss are not directly comparable. Retain all four runs.','',
        'Prospective v2: one fixed LR 0.001, 300 epochs and MixUp alpha 0.2, same split/seeds/model.',
        'Readiness thresholds stay unchanged. This is a bundled development amendment,',
        'not a causal ablation or independent confirmation; reuse of validation is disclosed.',
        'If v2 fails readiness, stop again; no automatic search and no evaluation.','',
        'Preservation manifest: preserved_v1_sha256.csv. Raw v1 results and code are unmodified.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
