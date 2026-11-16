"""Independent raw-record, stream, source, optimizer and decision audit."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch


def close(x, y):
    assert math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-10), (x, y)


def assess(records, spec, synthetic):
    screens = []
    for seed in spec['seeds']:
        arms = {row['mode']: row for row in records if row['seed'] == seed}
        full, spatial = arms['full'], arms['spatial']
        ready = (full['final_accuracy'] >= spec['baseline_min_accuracy'] and
                 100*(full['initial_accuracy']-full['final_accuracy']) <= spec['accuracy_margin_pp']+1e-10)
        best = max(full['initial_accuracy'], full['final_accuracy'], spatial['final_accuracy'])
        for mode in ('spatial', 'similarity'):
            row = arms[mode]
            quality = 100*(best-row['final_accuracy']) <= spec['accuracy_margin_pp']+1e-10
            tr = row['charged_recovery_seconds']/full['charged_recovery_seconds']
            mr = row['max_training_incremental_allocated']/full['max_training_incremental_allocated'] if full['max_training_incremental_allocated'] else None
            time_pass = tr <= 1-spec['minimum_time_saving']+1e-12
            memory_pass = mr is not None and mr <= 1-spec['minimum_incremental_memory_saving']+1e-12
            screens.append(dict(seed=seed, mode=mode, reference_qualified=ready, best_control_accuracy=best,
                final_delta_vs_best_pp=100*(row['final_accuracy']-best), quality_pass=quality,
                charged_time_ratio=tr, incremental_memory_ratio=mr, time_pass=time_pass,
                memory_pass=memory_pass, seed_screen_pass=bool(ready and quality and (time_pass or memory_pass) and not synthetic)))
    both = {mode: all(row['seed_screen_pass'] for row in screens if row['mode'] == mode)
            for mode in ('spatial', 'similarity')}
    advantage = True
    for seed in spec['seeds']:
        arms = {row['mode']: row for row in records if row['seed'] == seed}
        a, b = arms['similarity'], arms['spatial']
        advantage &= (100*(a['final_accuracy']-b['final_accuracy']) >= spec['similarity_advantage_pp']-1e-10 and
                      a['charged_recovery_seconds'] <= b['charged_recovery_seconds']*(1+spec['maximum_similarity_cost_penalty'])+1e-10 and
                      a['max_training_incremental_allocated'] <= b['max_training_incremental_allocated']*(1+spec['maximum_similarity_cost_penalty'])+1e-10)
    justified = bool(both['similarity'] and (not both['spatial'] or advantage))
    decision = 'prioritize_similarity' if justified else 'prefer_spatial' if both['spatial'] else 'park_fixed_recipe'
    return screens, both, justified, decision


def check_memory(row, device):
    baseline, peak, inc, reserved = [int(row[k]) for k in ('baseline_allocated','peak_allocated','incremental_allocated','peak_reserved')]
    assert peak-baseline == inc >= 0 and reserved >= peak
    if device == 'cuda': assert baseline > 0 and peak > 0
    else: assert baseline == peak == inc == reserved == 0


def compare_csv(saved, expected):
    assert len(saved) == len(expected)
    for row, ref in zip(saved, expected):
        assert set(row) == set(ref)
        for key, val in ref.items():
            if val is None: assert row[key] == ''
            elif isinstance(val, (str, bool)): assert row[key] == str(val)
            else: close(row[key], val)


def audit(out, write=False):
    import stage14_recovery as r
    out = Path(out); cfg = json.loads((out/'configuration.json').read_text())
    assert r.v.digest({k: val for k, val in cfg.items() if k != 'fingerprint'}) == cfg['fingerprint']
    spec, fake = cfg['spec'], cfg['synthetic']
    freeze = json.loads((out/'SYNTHETIC_FREEZE.json' if fake else r.FREEZE).read_text())
    assert r.v.digest({k: val for k, val in freeze.items() if k != 'freeze_hash'}) == freeze['freeze_hash'] == cfg['freeze_hash']
    assert freeze['spec'] == spec and freeze['synthetic'] == fake and freeze['expected_split'] == cfg['split']
    r.t.verify(freeze['source_sha256']); r.t.verify(freeze['evidence_sha256'])
    for cp in freeze['checkpoints'].values(): r.t.verify({cp['path']: cp['sha256']})
    _, y, _, ey, split = r.v.load_data('development', torch.device('cpu'), fake=fake)
    assert split == cfg['split']; labels = ey.numpy()
    summaries = []; total_updates = 0
    for seed, order in zip(spec['seeds'], spec['orders']):
        source_core, source_cp = r.source(seed, fake, freeze)
        source_opt = r.restore(source_core, source_cp, spec, torch.device('cpu'))
        source_model_hash = r.old.state_hash(source_core.state_dict()); source_opt_hash = r.optimizer_hash(source_opt)
        parameter_names = list(dict(source_core.named_parameters()))
        del source_opt, source_core, source_cp
        for position, mode in enumerate(order):
            parent = out/f'seed_{seed}'/mode
            rc = json.loads((parent/'run_config.json').read_text())
            assert r.v.digest({k: val for k,val in rc.items() if k != 'fingerprint'}) == rc['fingerprint']
            assert rc['seed'] == seed and rc['mode'] == mode and rc['position'] == position and rc['root_fingerprint'] == cfg['fingerprint']
            marker = r.d.validate_complete(parent, rc['fingerprint']); assert marker
            attempt = Path(marker['attempt']); assert attempt.parent.resolve() == parent.resolve()
            assert {str(path) for path in attempt.iterdir() if path.is_file()} == set(marker['sha256'])
            restoration = json.loads((attempt/'restoration.json').read_text())
            assert restoration['model_sha256'] == source_model_hash and restoration['optimizer_sha256'] == source_opt_hash
            assert restoration['parameter_names'] == parameter_names
            assert restoration['source_restored_after_warmup'] and restoration['source_optimizer_age'] == spec['source_optimizer_age']
            steps = r.d.records(attempt/'steps.csv'); curves = r.d.records(attempt/'curves.csv')
            total = spec['epochs']*spec['expected_batches_per_epoch']; assert len(steps) == total
            bg, ag, mg, pg = r.generators(seed, spec); idx = 0
            for epoch in range(1, spec['epochs']+1):
                seen = 0
                for indices in r.v.frozen6.epoch_batches(len(y), spec['batch_size'], bg):
                    row = steps[idx]; n = len(indices); seen += n
                    assert int(row['epoch']) == epoch and int(row['step']) == idx+1 and int(row['batch_examples']) == n
                    assert row['indices_sha256'] == r.tensor_hash(indices)
                    assert row['augmentation_before'] == r.tensor_hash(ag.get_state())
                    # Exact independent reconstruction of the frozen crop/flip draws.
                    torch.randint(0,9,(n,),generator=ag); torch.randint(0,9,(n,),generator=ag); torch.rand(n,generator=ag)
                    assert row['augmentation_after'] == r.tensor_hash(ag.get_state())
                    close(row['mixup_lambda'], mg.beta(spec['mixup_alpha'],spec['mixup_alpha']))
                    torch.randperm(n, generator=pg)
                    assert row['permutation_after'] == r.tensor_hash(pg.get_state()) and row['mixup_state'] == r.v.digest(mg.bit_generator.state)
                    warm = spec['lr_warmup_epochs']*spec['expected_batches_per_epoch']
                    lr = spec['learning_rate']*(idx+1)/warm if idx < warm else spec['learning_rate']*(
                        spec['min_lr_ratio']+(1-spec['min_lr_ratio'])*.5*(1+math.cos(math.pi*(idx-warm)/max(1,total-warm-1))))
                    close(row['lr'],lr)
                    assert int(row['dropout_seed']) == seed+spec['dropout_seed_offset']+idx
                    assert int(row['optimizer_age']) == spec['source_optimizer_age']+idx+1
                    assert row['gradient_checked'] == str(idx in (0,total-1))
                    for key in ('loss','transform_seconds','update_seconds'):
                        assert math.isfinite(float(row[key])) and float(row[key]) > 0
                    check_memory(row,cfg['device']); idx += 1
                assert seen == len(y)
            assert len(curves) == spec['epochs']+1
            with np.load(attempt/'predictions.npz', allow_pickle=False) as pred:
                assert set(pred.files) == {'labels'} | {f'epoch_{i}' for i in range(spec['epochs']+1)}
                assert np.array_equal(pred['labels'], labels)
                for epoch,row in enumerate(curves):
                    assert int(row['seed']) == seed and row['mode'] == mode and int(row['epoch']) == epoch
                    assert int(row['updates']) == epoch*spec['expected_batches_per_epoch']
                    z = pred[f'epoch_{epoch}']; assert z.shape == (len(labels),10) and np.isfinite(z).all()
                    # Torch cross-entropy is independent of the runner's float64 NumPy implementation.
                    ce = float(torch.nn.functional.cross_entropy(torch.from_numpy(z).double(),ey))
                    close(row['loss'],ce); close(row['accuracy'],float(np.mean(z.argmax(1)==labels)))
                    assert float(row['validation_seconds']) > 0; check_memory(row,cfg['device'])
                if not fake:
                    with np.load(freeze['checkpoints'][str(seed)]['predictions_path'],allow_pickle=False) as prior:
                        assert np.array_equal(labels,prior['labels'])
                        assert np.max(np.abs(pred['epoch_0']-prior[mode])) <= spec['source_logit_tolerance']
            cp = torch.load(attempt/'final_checkpoint.pt',map_location='cpu',weights_only=False)
            assert cp['seed'] == seed and cp['mode'] == mode and cp['updates'] == total
            assert cp['root_fingerprint'] == cfg['fingerprint'] and cp['run_fingerprint'] == rc['fingerprint']
            assert torch.equal(cp['batch_rng'],bg.get_state()) and torch.equal(cp['augmentation_rng'],ag.get_state())
            assert torch.equal(cp['permutation_rng'],pg.get_state()) and cp['mixup_rng'] == mg.bit_generator.state
            core = r.m.make_model('full_rgb32',seed,fake); core.load_state_dict(cp['model_state'],strict=True)
            opt = torch.optim.AdamW(core.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
            opt.load_state_dict(cp['optimizer_state'])
            assert len(opt.state) == len(parameter_names) and len(opt.param_groups) == 1
            for p in core.parameters():
                state = opt.state[p]; assert int(state['step']) == spec['source_optimizer_age']+total
                assert bool(torch.isfinite(p).all())
                for key in ('exp_avg','exp_avg_sq'):
                    assert state[key].shape == p.shape and bool(torch.isfinite(state[key]).all())
                assert bool((state['exp_avg_sq'] >= 0).all())
            close(opt.param_groups[0]['lr'],spec['learning_rate']*spec['min_lr_ratio'])
            inference = r.d.records(attempt/'inference.csv'); assert len(inference) == spec['inference_trials']
            gen = torch.Generator().manual_seed(seed+spec['inference_seed_offset'])
            indices = torch.randperm(len(y),generator=gen)[:spec['inference_trials']*spec['batch_size']].reshape(spec['inference_trials'],spec['batch_size'])
            for i,(row,ix) in enumerate(zip(inference,indices)):
                assert int(row['trial']) == i and row['indices_sha256'] == r.tensor_hash(ix)
                assert math.isfinite(float(row['seconds'])) and float(row['seconds']) > 0
                assert math.isfinite(float(row['scalar'])); check_memory(row,cfg['device'])
            summary = json.loads((attempt/'summary.json').read_text())
            assert summary['seed'] == seed and summary['mode'] == mode and summary['updates'] == total
            assert summary['initial_model_sha256'] == source_model_hash and summary['initial_optimizer_sha256'] == source_opt_hash
            assert summary['final_model_sha256'] == r.old.state_hash(core.state_dict()) != source_model_hash
            assert summary['final_optimizer_sha256'] == r.optimizer_hash(opt) != source_opt_hash
            for prefix,row in [('initial',curves[0]),('final',curves[-1])]:
                close(summary[prefix+'_accuracy'],row['accuracy']); close(summary[prefix+'_loss'],row['loss'])
            training = sum(float(row['transform_seconds'])+float(row['update_seconds']) for row in steps)
            close(summary['training_seconds'],training)
            for key in ('setup_seconds','warmup_restore_seconds'):
                assert math.isfinite(summary[key]) and summary[key] > 0; close(summary[key],restoration[key])
            charge = training+restoration['setup_seconds']+restoration['warmup_restore_seconds']
            close(summary['charged_recovery_seconds'],charge)
            close(summary['validation_seconds'],sum(float(row['validation_seconds']) for row in curves))
            for key in ('inference_benchmark_seconds','checkpoint_seconds','observed_arm_seconds'):
                assert math.isfinite(summary[key]) and summary[key] > 0
            assert summary['observed_arm_seconds'] >= charge+summary['validation_seconds']+summary['inference_benchmark_seconds']+summary['checkpoint_seconds']
            for key in ('setup_memory','warmup_memory'): check_memory(restoration[key],cfg['device'])
            assert summary['max_training_incremental_allocated'] == max(int(row['incremental_allocated']) for row in steps)
            assert summary['max_training_peak_allocated'] == max(int(row['peak_allocated']) for row in steps)
            assert summary['max_measured_arm_allocated'] == max([restoration[key]['peak_allocated'] for key in ('setup_memory','warmup_memory')]
                +[int(row['peak_allocated']) for row in steps+curves+inference])
            src = freeze['checkpoints'].get(str(seed),{})
            for key in ('source_training_seconds','source_total_run_seconds'): close(summary[key],src.get(key,0))
            summary.update(accuracy_recovered_pp=100*(summary['final_accuracy']-summary['initial_accuracy']),
                loss_change=summary['final_loss']-summary['initial_loss'],
                source_training_plus_recovery_seconds=summary['source_training_seconds']+charge,
                source_run_plus_observed_recovery_seconds=summary['source_total_run_seconds']+summary['observed_arm_seconds'],
                mean_inference_seconds=float(np.mean([float(row['seconds']) for row in inference])),
                mean_inference_incremental_allocated=float(np.mean([float(row['incremental_allocated']) for row in inference])))
            summaries.append(summary); total_updates += total
            del core,opt,cp
    screens,both,justified,decision = assess(summaries,spec,fake)
    result = dict(status='PASS',synthetic=fake,training_updates=total_updates,
        discarded_hardware_warmup_updates=len(summaries)*spec['hardware_warmup_steps'],
        trajectories=len(summaries),curve_points=len(summaries)*(spec['epochs']+1),
        source_states_optimizer_histories_streams_labels_and_costs_verified=True,
        both_seed_reference_qualified=all(row['reference_qualified'] for row in screens),
        both_seed_recovery_screen=both,similarity_justified_over_spatial=justified,decision=decision,
        test_scoring_performed=False,automatic_followup=False)
    if write:
        r.v.write_csv(out/'summary.csv',summaries); r.v.write_csv(out/'screen.csv',screens)
        r.v.atomic_json(out/'audit.json',result)
        lines=['# Stage14B token recovery','','```json',json.dumps(result,indent=2),'```','',
            '| Seed | Mode | Initial accuracy | Final accuracy | Charged recovery s | Source training + recovery s |',
            '|---|---|---:|---:|---:|---:|']
        for row in summaries:
            lines.append(f"| {row['seed']} | {row['mode']} | {100*row['initial_accuracy']:.2f}% | {100*row['final_accuracy']:.2f}% | {row['charged_recovery_seconds']:.3f} | {row['source_training_plus_recovery_seconds']:.3f} |")
        lines += ['', 'Only the fixed final endpoint decides. Review both seeds and all controls before proceeding.',
            'Recovery charges include setup, warmup/restoration and training, excluding evaluation/logging/inference benchmarks.',
            'Full source costs and observed arm costs are retained. No from-scratch speedup or novelty claim.',
            'Prior Stage14A failure remains unchanged. Development examples have been reused. No automatic extension.', '']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text()) == result
        compare_csv(r.d.records(out/'summary.csv'),summaries); compare_csv(r.d.records(out/'screen.csv'),screens)
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('output',type=Path); args=ap.parse_args()
    import stage10_models as m
    m.configure(); print(json.dumps(audit(args.output),indent=2))
