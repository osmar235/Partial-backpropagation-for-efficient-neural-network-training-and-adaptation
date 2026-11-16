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
    screens=[]
    for seed in spec['seeds']:
        arms={row['mode']:row for row in records if row['seed']==seed}
        full=arms['full_restore']; candidate=arms['similarity_fresh']
        controls=[arms[f'{mode}_{history}'] for mode in ('full','spatial') for history in ('restore','fresh')]
        ready=(full['final_accuracy']>=spec['baseline_min_accuracy'] and
               100*(full['initial_accuracy']-full['final_accuracy'])<=spec['accuracy_margin_pp']+1e-10)
        best=max([full['initial_accuracy']]+[row['final_accuracy'] for row in controls])
        def qualified(row):return 100*(best-row['final_accuracy'])<=spec['accuracy_margin_pp']+1e-10
        qualified_controls=[row for row in controls if qualified(row)]
        # An absent qualifying endpoint is a failed reference, never an easy time win.
        ready=bool(ready and qualified_controls)
        fastest=min([row['complete_adaptation_seconds'] for row in qualified_controls],default=float('inf'))
        full_time=min(arms['full_restore']['complete_adaptation_seconds'],arms['full_fresh']['complete_adaptation_seconds'])
        denominator=min(full_time,fastest)
        ratio=candidate['complete_adaptation_seconds']/denominator
        reset_gain=100*(candidate['final_accuracy']-arms['similarity_restore']['final_accuracy'])
        full_gain=100*(arms['full_fresh']['final_accuracy']-full['final_accuracy'])
        spatial_gain=100*(arms['spatial_fresh']['final_accuracy']-arms['spatial_restore']['final_accuracy'])
        interaction=reset_gain-full_gain
        quality=qualified(candidate);time_pass=ratio<=1-spec['minimum_time_saving']+1e-12
        mechanism=reset_gain>spec['interaction_tolerance_pp'] and interaction>spec['interaction_tolerance_pp']
        screens.append(dict(seed=seed,mode='similarity_fresh',reference_qualified=ready,
            best_control_accuracy=best,final_delta_vs_best_pp=100*(candidate['final_accuracy']-best),
            quality_pass=quality,complete_time_ratio=ratio,time_pass=time_pass,
            similarity_reset_gain_pp=reset_gain,full_reset_gain_pp=full_gain,
            interaction_pp=interaction,spatial_interaction_pp=spatial_gain-full_gain,
            mechanism_seed_support=bool(mechanism and not synthetic),
            seed_screen_pass=bool(ready and quality and time_pass and not synthetic)))
    both=dict(practical=all(row['seed_screen_pass'] for row in screens),
              mechanism=all(row['mechanism_seed_support'] for row in screens))
    decision=('plan_fresh_confirmation' if both['practical'] and both['mechanism'] else
              'practical_candidate_mechanism_not_supported' if both['practical'] else
              'mechanism_only_no_efficiency_claim' if both['mechanism'] else 'close_fixed_reset_hypothesis')
    if synthetic:decision='synthetic_validation_only'
    return screens,both,both['practical'],decision


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
    import stage14_optimizer_history as r
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
            computation,history=mode.rsplit('_',1)
            initial_age=spec['source_optimizer_age'] if history=='restore' else 0
            fresh_core,fresh_cp=r.source(seed,fake,freeze)
            assigned=r.restore(fresh_core,fresh_cp,spec,torch.device('cpu'),history)
            source_opt_hash=r.optimizer_hash(assigned)
            initial_diag=r.diagnostic_state(fresh_core,assigned)
            del fresh_core,fresh_cp,assigned
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
            assert restoration['history']==history and restoration['initial_optimizer_age']==initial_age
            assert restoration['initial_state_entries']==(len(parameter_names) if history=='restore' else 0)
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
                    assert int(row['optimizer_age']) == initial_age+idx+1
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
                        assert np.max(np.abs(pred['epoch_0']-prior[computation])) <= spec['source_logit_tolerance']
            adaptation_manifest=json.loads((attempt/'adaptation_artifacts.json').read_text())
            assert set(adaptation_manifest)=={str(attempt/name) for name in
                ('restoration.json','steps.csv','curves.csv','diagnostics.csv','telemetry.json','predictions.npz','final_checkpoint.pt')}
            r.t.verify(adaptation_manifest)
            # Exact pair equality at initialization; historical restored-trajectory reproduction.
            pair=out/f'seed_{seed}'/f'{computation}_{"fresh" if history=="restore" else "restore"}'
            pair_attempt=Path(json.loads((pair/'COMPLETE.json').read_text())['attempt'])
            with np.load(attempt/'predictions.npz') as pred, np.load(pair_attempt/'predictions.npz') as other:
                assert np.array_equal(pred['epoch_0'],other['epoch_0'])
            if not fake and history=='restore':
                previous=r.ROOT/'results/stage14B_token_recovery_v1'/f'seed_{seed}'/computation
                previous_attempt=Path(json.loads((previous/'COMPLETE.json').read_text())['attempt'])
                with np.load(attempt/'predictions.npz') as pred,np.load(previous_attempt/'predictions.npz') as reference:
                    for epoch in range(spec['epochs']+1):
                        assert np.max(np.abs(pred[f'epoch_{epoch}']-reference[f'epoch_{epoch}'])) <= spec['source_logit_tolerance'], 'Historical reproduction failed'
                for current,previous_step in zip(steps,r.d.records(previous_attempt/'steps.csv')):
                    assert abs(float(current['loss'])-float(previous_step['loss']))<=spec['reproduction_loss_tolerance']
            cp = torch.load(attempt/'final_checkpoint.pt',map_location='cpu',weights_only=False)
            assert cp['history']==history
            assert cp['seed'] == seed and cp['mode'] == mode and cp['updates'] == total
            assert cp['root_fingerprint'] == cfg['fingerprint'] and cp['run_fingerprint'] == rc['fingerprint']
            assert torch.equal(cp['batch_rng'],bg.get_state()) and torch.equal(cp['augmentation_rng'],ag.get_state())
            assert torch.equal(cp['permutation_rng'],pg.get_state()) and cp['mixup_rng'] == mg.bit_generator.state
            core = r.m.make_model('full_rgb32',seed,fake); core.load_state_dict(cp['model_state'],strict=True)
            opt = torch.optim.AdamW(core.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
            opt.load_state_dict(cp['optimizer_state'])
            assert len(opt.state) == len(parameter_names) and len(opt.param_groups) == 1
            for p in core.parameters():
                state = opt.state[p]; assert int(state['step']) == initial_age+total
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
            diagnostics=r.d.records(attempt/'diagnostics.csv')
            assert [int(row['step']) for row in diagnostics]==[1,total]
            for row in diagnostics:
                for key,value in row.items(): assert math.isfinite(float(value)) and float(value)>=0
                assert float(row['parameter_update_l2'])>0 and float(row['gradient_l2'])>0
            close(diagnostics[0]['before_moment1_l2'],initial_diag['moment1_l2'])
            close(diagnostics[0]['before_moment2_l2'],initial_diag['moment2_l2'])
            final_diag=r.diagnostic_state(core,opt)
            close(diagnostics[-1]['after_moment1_l2'],final_diag['moment1_l2'])
            close(diagnostics[-1]['after_moment2_l2'],final_diag['moment2_l2'])
            summary = json.loads((attempt/'summary.json').read_text())
            assert summary['computation']==computation and summary['history']==history and summary['initial_optimizer_age']==initial_age
            launch_path=Path(summary['launch_record'])
            assert launch_path.parent.parent.resolve()==(out/'launches').resolve()
            shared=json.loads(launch_path.read_text())
            assert shared['split']==cfg['split'] and shared['training_examples']==len(y) and shared['validation_examples']==len(labels)
            close(summary['shared_data_seconds'],shared['seconds']/(len(spec['seeds'])*len(spec['modes'])))
            assert summary['shared_data_seconds']>0
            close(summary['complete_adaptation_seconds'],summary['adaptation_seconds']+summary['shared_data_seconds'])
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
            assert summary['adaptation_seconds'] >= charge+summary['validation_seconds']+summary['checkpoint_seconds']
            assert summary['observed_arm_seconds'] >= summary['adaptation_seconds']+summary['inference_benchmark_seconds']
            for key in ('setup_memory','warmup_memory'): check_memory(restoration[key],cfg['device'])
            assert summary['max_training_incremental_allocated'] == max(int(row['incremental_allocated']) for row in steps)
            assert summary['max_training_peak_allocated'] == max(int(row['peak_allocated']) for row in steps)
            assert summary['max_measured_arm_allocated'] == max([restoration[key]['peak_allocated'] for key in ('setup_memory','warmup_memory')]
                +[int(row['peak_allocated']) for row in steps+curves+inference])
            src = freeze['checkpoints'].get(str(seed),{})
            for key in ('source_training_seconds','source_total_run_seconds'): close(summary[key],src.get(key,0))
            summary.update(accuracy_recovered_pp=100*(summary['final_accuracy']-summary['initial_accuracy']),
                loss_change=summary['final_loss']-summary['initial_loss'],
                source_training_plus_recovery_seconds=summary['source_training_seconds']+summary['complete_adaptation_seconds'],
                source_run_plus_observed_recovery_seconds=summary['source_total_run_seconds']+summary['observed_arm_seconds'],
                mean_inference_seconds=float(np.mean([float(row['seconds']) for row in inference])),
                mean_inference_incremental_allocated=float(np.mean([float(row['incremental_allocated']) for row in inference])))
            summaries.append(summary); total_updates += total
            del core,opt,cp
    screens,both,justified,decision = assess(summaries,spec,fake)
    uninterrupted_timing=len({row['launch_record'] for row in summaries})==1
    if not uninterrupted_timing:
        for row in screens:row['seed_screen_pass']=False
        both['practical']=False;justified=False
        decision='timing_interrupted_review_required' if not fake else 'synthetic_validation_only'
    result = dict(status='PASS',synthetic=fake,training_updates=total_updates,
        discarded_hardware_warmup_updates=len(summaries)*spec['hardware_warmup_steps'],
        trajectories=len(summaries),curve_points=len(summaries)*(spec['epochs']+1),
        source_states_optimizer_histories_streams_labels_and_costs_verified=True,
        paired_initial_logits_identical=True,historical_restored_trajectory_reproduced=not fake,
        retained_arms_share_one_launch=uninterrupted_timing,
        both_seed_reference_qualified=all(row['reference_qualified'] for row in screens),
        both_seed_screen=both,practical_candidate=justified,decision=decision,
        test_scoring_performed=False,automatic_followup=False)
    if write:
        r.v.write_csv(out/'summary.csv',summaries); r.v.write_csv(out/'screen.csv',screens)
        r.v.atomic_json(out/'audit.json',result)
        lines=['# Stage14C optimizer history','','```json',json.dumps(result,indent=2),'```','',
            '| Seed | Mode | Initial accuracy | Final accuracy | Complete adaptation s | Source training + adaptation s |',
            '|---|---|---:|---:|---:|---:|']
        for row in summaries:
            lines.append(f"| {row['seed']} | {row['mode']} | {100*row['initial_accuracy']:.2f}% | {100*row['final_accuracy']:.2f}% | {row['complete_adaptation_seconds']:.3f} | {row['source_training_plus_recovery_seconds']:.3f} |")
        lines += ['', 'Only the fixed final endpoint decides. Review both seeds and all controls before proceeding.',
            'Primary adaptation cost includes allocated shared data preparation plus setup, warmup/restoration, training, all validation, diagnostics, logging, checkpoint saving and artifact hashing. Post-endpoint inference and offline audit are excluded.',
            'Full source costs and observed arm costs are retained. No from-scratch speedup or novelty claim.',
            'Prior Stage14A/B failures remain unchanged. Development examples have been reused. No automatic extension.', '']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text()) == result
        compare_csv(r.d.records(out/'summary.csv'),summaries); compare_csv(r.d.records(out/'screen.csv'),screens)
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('output',type=Path); args=ap.parse_args()
    import stage10_models as m
    m.configure(); print(json.dumps(audit(args.output),indent=2))
