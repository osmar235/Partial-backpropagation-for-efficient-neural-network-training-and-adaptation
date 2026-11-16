"""Matched token recovery with retained/fresh AdamW and complete adaptation timing."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path
import numpy as np
import torch
import stage14_diagnostic as t

v, d, m, old = t.v, t.d, t.m, t.old
ROOT = v.ROOT
SPEC = ROOT / 'protocol/stage14c/optimizer_history_spec_v1.json'
FREEZE = ROOT / 'protocol/stage14c/FROZEN_OPTIMIZER_HISTORY_v1.json'
REAL_OUT = ROOT / 'results/stage14C_optimizer_history_v1'


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def optimizer_hash(opt):
    sd = opt.state_dict()
    state = {str(i): {k: tensor_hash(z) if torch.is_tensor(z) else z for k, z in row.items()}
             for i, row in sd['state'].items()}
    return v.digest(dict(state=state, groups=sd['param_groups']))


def sources():
    names = ['scripts/stage14_optimizer_history.py', 'scripts/review_stage14C.py',
             'scripts/test_stage14_optimizer_history.py', 'scripts/run_stage14C_optimizer_history.cmd',
             'protocol/stage14c/optimizer_history_spec_v1.json', 'protocol/stage14c/OPTIMIZER_HISTORY_v1.md']
    import stage14_recovery as previous
    return {**previous.sources(), **{str(ROOT / name): d.sha(ROOT / name) for name in names}}


def freeze_record(spec, fake):
    prior = json.loads(t.FREEZE.read_text())
    t.verify(prior['source_sha256']); t.verify(prior['evidence_sha256'])
    evidence = [t.FREEZE, ROOT / 'results/stage13C_stage14A_review_v1/REPORT.md',
                ROOT / 'results/stage13C_stage14A_review_v1/independent_audit.json',
                ROOT / 'protocol/stage14/POST_DIAGNOSTIC_DECISION_v1.md']
    checkpoints = {} if fake else copy.deepcopy(prior['checkpoints'])
    if not fake:
        cfg = json.loads((t.REAL_OUT / 'configuration.json').read_text())
        assert d.validate_complete(t.REAL_OUT, cfg['fingerprint'])
        attempt = Path(json.loads((t.REAL_OUT / 'RAW_COMPLETE.json').read_text())['attempt'])
        evidence += [t.REAL_OUT / 'configuration.json', t.REAL_OUT / 'COMPLETE.json',
                     t.REAL_OUT / 'RAW_COMPLETE.json', t.REAL_OUT / 'audit.json']
        for seed in spec['seeds']:
            src = checkpoints[str(seed)]
            t.verify({src['path']: src['sha256']})
            summary_path = Path(src['path']).parent / 'summary.json'
            summary = json.loads(summary_path.read_text())
            src.update(source_training_seconds=summary['training_seconds'],
                       source_total_run_seconds=summary['total_run_seconds'],
                       predictions_path=str(attempt / f'predictions_{seed}.npz'))
            evidence += [summary_path, Path(src['predictions_path'])]
    import stage14_recovery as previous
    previous_freeze=json.loads(previous.FREEZE.read_text())
    t.verify(previous_freeze['source_sha256']); t.verify(previous_freeze['evidence_sha256'])
    prior_cfg=json.loads((previous.REAL_OUT/'configuration.json').read_text())
    prior_marker=d.validate_complete(previous.REAL_OUT,prior_cfg['fingerprint']); assert prior_marker
    evidence += [previous.FREEZE, previous.REAL_OUT/'COMPLETE.json']
    evidence += [Path(p) for p in prior_marker['sha256']]
    for marker_path in previous.REAL_OUT.glob('seed_*/*/COMPLETE.json'):
        marker=json.loads(marker_path.read_text()); t.verify(marker['sha256'])
        evidence += [Path(p) for p in marker['sha256']]
    rec = dict(prepared_date='2026-10-24', synthetic=fake, spec=spec, source_sha256=sources(),
               evidence_sha256={str(path): d.sha(path) for path in evidence}, checkpoints=checkpoints,
               expected_split={'kind': 'artificial'} if fake else prior['expected_split'],
               execution=m.execution_settings(), python=platform.python_version(),
               torch=str(torch.__version__), torchvision=str(t.torchvision.__version__), numpy=np.__version__)
    rec['freeze_hash'] = v.digest(rec)
    return rec


def source(seed, fake, freeze):
    core = t.make_core(seed, fake, freeze, torch.device('cpu'))
    if fake:
        # Mature artificial Adam history generated on CPU for identical audit/device sources.
        opt = torch.optim.AdamW(core.parameters(), lr=.001, weight_decay=.05)
        gen = torch.Generator().manual_seed(seed + 247001)
        xx = torch.randn(3, 3, 32, 32, generator=gen)
        yy = torch.tensor([0, 1, 2])
        core.train()
        for i in range(2):
            old.update(core, opt, xx, yy, seed + 248001 + i, .1)
        cp = dict(model_state=copy.deepcopy(core.state_dict()), optimizer_state=copy.deepcopy(opt.state_dict()), updates=2)
    else:
        cp = torch.load(freeze['checkpoints'][str(seed)]['path'], map_location='cpu', weights_only=False)
    return core, cp


def restore(core, cp, spec, device, history="restore"):
    core.to(device); core.load_state_dict(cp['model_state'], strict=True); core.zero_grad(set_to_none=True)
    names = list(dict(core.named_parameters()))
    assert names == [key for key in cp['model_state'] if key in set(names)]
    sd = cp['optimizer_state']
    assert len(sd['param_groups']) == 1
    ids = sd['param_groups'][0]['params']
    assert ids == list(range(len(names))) and len(sd['state']) == len(names)
    assert cp['updates'] == spec['source_optimizer_age']
    for name, idx in zip(names, ids):
        for key in ('exp_avg', 'exp_avg_sq'):
            assert sd['state'][idx][key].shape == dict(core.named_parameters())[name].shape
        assert int(sd['state'][idx]['step']) == spec['source_optimizer_age']
    group = sd['param_groups'][0]
    assert group['weight_decay'] == spec['weight_decay'] and tuple(group['betas']) == (.9, .999)
    assert group['eps'] == 1e-8 and not group.get('amsgrad', False)
    opt = torch.optim.AdamW(core.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    assert history in ('restore','fresh')
    if history == 'restore':
        opt.load_state_dict(copy.deepcopy(sd))
    else:
        # Match every source hyperparameter; empty state resets moments AND counters.
        fresh_sd=copy.deepcopy(sd['param_groups'])
        opt.load_state_dict(dict(state={},param_groups=fresh_sd))
        assert not opt.state
    return opt


def diagnostic_state(core,opt):
    weights=torch.cat([p.detach().cpu().double().flatten() for p in core.parameters()])
    moments={}
    for label,key in [('moment1_l2','exp_avg'),('moment2_l2','exp_avg_sq')]:
        values=[opt.state[p][key].detach().cpu().double().flatten() for p in core.parameters() if p in opt.state]
        moments[label]=float(torch.cat(values).norm()) if values else 0.
    return dict(weights=weights,**moments)


def generators(seed, spec):
    return (torch.Generator().manual_seed(seed + spec['batch_seed_offset']),
            torch.Generator().manual_seed(seed + spec['augmentation_seed_offset']),
            np.random.default_rng(seed + spec['mixup_seed_offset']),
            torch.Generator().manual_seed(seed + spec['permutation_seed_offset']))


def memory_start(device):
    if device.type == 'cpu':
        return 0
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def memory_end(device, baseline):
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    return dict(baseline_allocated=baseline, peak_allocated=peak, incremental_allocated=peak-baseline,
                peak_reserved=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)


def score(logits, labels):
    z = logits.astype(np.float64); z -= z.max(1, keepdims=True)
    loss = float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), labels]))
    return loss, float(np.mean(logits.argmax(1) == labels))


def run_one(out, seed, mode, position, cfg, freeze, data, device, shared_data_seconds=0., launch_record=None):
    spec, fake = cfg['spec'], cfg['synthetic']
    computation,history=mode.rsplit('_',1)
    initial_age=spec['source_optimizer_age'] if history=='restore' else 0
    parent = out / f'seed_{seed}' / mode; parent.mkdir(parents=True, exist_ok=True)
    rc = dict(root_fingerprint=cfg['fingerprint'], seed=seed, mode=mode, position=position)
    rc['fingerprint'] = v.digest(rc); old.write_freeze(parent / 'run_config.json', rc)
    if d.validate_complete(parent, rc['fingerprint']):
        print('SKIP completed', seed, mode, flush=True); return
    attempt = d.next_attempt(parent); v.sync(device); started = time.perf_counter()
    x, y, ev, ey, _ = data; labels = ey.cpu().numpy()
    base = memory_start(device); tick = time.perf_counter()
    core, cp = source(seed, fake, freeze); opt = restore(core, cp, spec, device, history)
    model = t.TokenModel(core, computation, spec['after_blocks']); v.sync(device)
    setup_seconds = time.perf_counter()-tick; setup_memory = memory_end(device, base)
    initial_hash = old.state_hash(core.state_dict()); initial_opt = optimizer_hash(opt)
    curves, logits, steps, telemetry, diagnostics = [], {}, [], [], []
    validation_seconds = 0.

    def evaluate(epoch):
        nonlocal validation_seconds
        v.sync(device); tick = time.perf_counter()
        baseline = memory_start(device)
        z = t.predictions(model, ev, spec['eval_batch_size'])
        loss, accuracy = score(z, labels); v.sync(device)
        seconds = time.perf_counter()-tick; validation_seconds += seconds
        logits[f'epoch_{epoch}'] = z
        curves.append(dict(seed=seed, mode=mode, epoch=epoch,
                           updates=epoch*spec['expected_batches_per_epoch'], loss=loss,
                           accuracy=accuracy, validation_seconds=seconds, **memory_end(device, baseline)))
        return z

    initial_logits = evaluate(0)
    if not fake:
        with np.load(freeze['checkpoints'][str(seed)]['predictions_path'], allow_pickle=False) as prior:
            assert np.array_equal(labels, prior['labels'])
            assert np.max(np.abs(initial_logits-prior[computation])) <= spec['source_logit_tolerance']
            loss, accuracy = score(prior[computation], labels)
            assert abs(curves[0]['loss']-loss) <= spec['source_loss_tolerance']
            assert curves[0]['accuracy'] == accuracy
    # Hardware warmup is discarded, restored exactly and fully charged.
    base = memory_start(device); tick = time.perf_counter(); model.train()
    xx = v.frozen6.normalize_cifar(x[:spec['batch_size']]); yy = y[:len(xx)]
    for i in range(spec['hardware_warmup_steps']):
        old.update(model, opt, xx, yy, seed+249001+i, spec['label_smoothing'])
    del opt
    opt = restore(core, cp, spec, device, history); del xx, yy
    v.sync(device); warmup_restore_seconds = time.perf_counter()-tick
    warmup_memory = memory_end(device, base)
    assert old.state_hash(core.state_dict()) == initial_hash and optimizer_hash(opt) == initial_opt
    v.atomic_json(attempt / 'restoration.json', dict(model_sha256=initial_hash, optimizer_sha256=initial_opt,
        parameter_names=list(dict(core.named_parameters())), source_optimizer_age=spec['source_optimizer_age'],
        source_restored_after_warmup=True, history=history, initial_optimizer_age=initial_age,
        initial_state_entries=len(opt.state), setup_seconds=setup_seconds,
        warmup_restore_seconds=warmup_restore_seconds, setup_memory=setup_memory, warmup_memory=warmup_memory))
    del cp
    bg, ag, mg, pg = generators(seed, spec)
    batches = math.ceil(len(y)/spec['batch_size']); assert batches == spec['expected_batches_per_epoch']
    total = spec['epochs']*batches; idx = 0
    for epoch in range(1, spec['epochs']+1):
        model.train()
        if not fake:
            telemetry.append(dict(epoch=epoch, sample=v.telemetry()))
        for indices in v.frozen6.epoch_batches(len(y), spec['batch_size'], bg):
            n = len(indices); indices_sha = tensor_hash(indices); augmentation_before = tensor_hash(ag.get_state())
            opt.zero_grad(set_to_none=True); v.sync(device); tick = time.perf_counter()
            ix = indices.to(device)
            xx = v.frozen6.augment_and_normalize(x[ix], ag)
            xx, targets, lam = v.mix_batch(xx, y[ix], mg, pg, spec['mixup_alpha'], 10)
            v.sync(device); transform_seconds = time.perf_counter()-tick
            lr = v.learning_rate(idx, total, batches, spec['learning_rate'], spec)
            for group in opt.param_groups: group['lr'] = lr
            dropout = seed+spec['dropout_seed_offset']+idx
            checked = idx in (0,total-1)
            before=diagnostic_state(core,opt) if checked else None
            base = memory_start(device); tick = time.perf_counter()
            loss = old.update(model, opt, xx, targets, dropout, spec['label_smoothing'])
            v.sync(device); seconds = time.perf_counter()-tick; mem = memory_end(device, base)
            idx += 1
            checked = idx in (1, total)
            if checked:
                after=diagnostic_state(core,opt)
                diagnostics.append(dict(step=idx, before_moment1_l2=before['moment1_l2'],
                    before_moment2_l2=before['moment2_l2'], after_moment1_l2=after['moment1_l2'],
                    after_moment2_l2=after['moment2_l2'],
                    parameter_update_l2=float((after['weights']-before['weights']).norm()),
                    gradient_l2=float(torch.cat([p.grad.detach().cpu().double().flatten() for p in core.parameters()]).norm())))
                del before,after
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in core.parameters())
                assert float(core.body.conv_proj.weight.grad.norm()) > 0
            steps.append(dict(epoch=epoch, step=idx, batch_examples=n, indices_sha256=indices_sha,
                augmentation_before=augmentation_before, augmentation_after=tensor_hash(ag.get_state()),
                permutation_after=tensor_hash(pg.get_state()), mixup_state=v.digest(mg.bit_generator.state),
                mixup_lambda=lam, dropout_seed=dropout, lr=lr, loss=loss,
                optimizer_age=initial_age+idx, transform_seconds=transform_seconds,
                update_seconds=seconds, gradient_checked=checked, **mem))
            del xx, targets, ix
        assert all(int(opt.state[p]['step']) == initial_age+idx for p in core.parameters())
        model.zero_grad(set_to_none=True); evaluate(epoch)
        v.write_csv(attempt / 'steps.csv', steps); v.write_csv(attempt / 'curves.csv', curves)
        print(f'RECOVERY seed={seed} {mode} epoch={epoch}/{spec["epochs"]} accuracy={100*curves[-1]["accuracy"]:.2f}%', flush=True)
    assert idx == total and all(bool(torch.isfinite(p).all()) for p in core.parameters())
    model.zero_grad(set_to_none=True)
    model_before = old.state_hash(core.state_dict()); opt_before = optimizer_hash(opt)
    v.write_csv(attempt/'diagnostics.csv',diagnostics)
    v.atomic_json(attempt / 'telemetry.json', telemetry)
    np.savez(attempt / 'predictions.npz', labels=labels, **logits)
    tick = time.perf_counter()
    old.checkpoint(attempt / 'final_checkpoint.pt', dict(model_state=core.state_dict(), optimizer_state=opt.state_dict(),
        seed=seed, mode=mode, history=history, updates=idx, root_fingerprint=cfg['fingerprint'], run_fingerprint=rc['fingerprint'],
        batch_rng=bg.get_state(), augmentation_rng=ag.get_state(), permutation_rng=pg.get_state(), mixup_rng=mg.bit_generator.state))
    v.sync(device); checkpoint_seconds = time.perf_counter()-tick
    # Hashing/reproducibility instrumentation is conservatively charged as logging.
    training_artifacts={str(p):d.sha(p) for p in attempt.iterdir() if p.is_file()}
    v.atomic_json(attempt/'adaptation_artifacts.json',training_artifacts)
    v.sync(device); adaptation_seconds=time.perf_counter()-started
    # Independent inference sample; measured only after the fixed training endpoint.
    gen = torch.Generator().manual_seed(seed+spec['inference_seed_offset'])
    ii = torch.randperm(len(y), generator=gen)[:spec['inference_trials']*spec['batch_size']]
    assert len(ii) == spec['inference_trials']*spec['batch_size']
    inference = []
    v.sync(device); tick = time.perf_counter()
    for trial, indices in enumerate(ii.reshape(spec['inference_trials'], spec['batch_size'])):
        xx = v.frozen6.normalize_cifar(x[indices.to(device)]); yy = y[indices.to(device)]
        if trial == 0:
            for j in range(spec['inference_warmups']):
                t.operation(model, xx, yy, 'inference', seed+250001+j, device)
        row = t.operation(model, xx, yy, 'inference', seed+251001+trial, device)
        inference.append(dict(trial=trial, indices_sha256=tensor_hash(indices), **row))
        del xx, yy
    v.sync(device); inference_benchmark_seconds = time.perf_counter()-tick
    assert old.state_hash(core.state_dict()) == model_before and optimizer_hash(opt) == opt_before
    v.write_csv(attempt/'inference.csv',inference)
    training_seconds = sum(row['transform_seconds']+row['update_seconds'] for row in steps)
    src = freeze['checkpoints'].get(str(seed), {})
    summary = dict(seed=seed, mode=mode, computation=computation, history=history,
        initial_optimizer_age=initial_age, adaptation_seconds=adaptation_seconds,
        shared_data_seconds=shared_data_seconds, launch_record=str(launch_record) if launch_record else None,
        complete_adaptation_seconds=adaptation_seconds+shared_data_seconds, updates=idx, initial_accuracy=curves[0]['accuracy'],
        final_accuracy=curves[-1]['accuracy'], initial_loss=curves[0]['loss'], final_loss=curves[-1]['loss'],
        initial_model_sha256=initial_hash, initial_optimizer_sha256=initial_opt,
        final_model_sha256=model_before, final_optimizer_sha256=opt_before,
        training_seconds=training_seconds, setup_seconds=setup_seconds, warmup_restore_seconds=warmup_restore_seconds,
        charged_recovery_seconds=training_seconds+setup_seconds+warmup_restore_seconds,
        validation_seconds=validation_seconds, inference_benchmark_seconds=inference_benchmark_seconds,
        checkpoint_seconds=checkpoint_seconds, observed_arm_seconds=time.perf_counter()-started,
        source_training_seconds=src.get('source_training_seconds', 0),
        source_total_run_seconds=src.get('source_total_run_seconds', 0),
        max_training_incremental_allocated=max(row['incremental_allocated'] for row in steps),
        max_training_peak_allocated=max(row['peak_allocated'] for row in steps),
        max_measured_arm_allocated=max([setup_memory['peak_allocated'], warmup_memory['peak_allocated']]
            +[row['peak_allocated'] for row in steps]+[row['peak_allocated'] for row in inference]
            +[row['peak_allocated'] for row in curves]))
    v.atomic_json(attempt / 'summary.json', summary)
    v.atomic_json(parent / 'COMPLETE.json', dict(fingerprint=rc['fingerprint'], attempt=str(attempt),
        sha256={str(path): d.sha(path) for path in attempt.iterdir() if path.is_file()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze-only', action='store_true'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-tag', default='final_cpu'); ap.add_argument('--smoke-device', choices=['cpu','cuda'], default='cpu')
    args = ap.parse_args(); assert re.fullmatch('[a-z0-9_]{1,32}', args.smoke_tag)
    m.configure(); spec = json.loads(SPEC.read_text()); fake = args.smoke
    if fake:
        spec.update(seeds=[9961,9962], after_blocks=1, epochs=2, batch_size=3, eval_batch_size=4,
                    expected_batches_per_epoch=6, source_optimizer_age=2, hardware_warmup_steps=1,
                    inference_trials=3, inference_warmups=1)
    freeze = freeze_record(spec, fake)
    if args.freeze_only:
        assert not fake and not REAL_OUT.exists(); old.write_freeze(FREEZE, freeze)
        print('FROZEN', freeze['freeze_hash']); return
    device = torch.device(args.smoke_device if fake else 'cuda')
    assert device.type == 'cpu' or torch.cuda.is_available()
    out = ROOT / f'results/stage14C_sanity_optimizer_v1_{args.smoke_tag}' if fake else REAL_OUT
    if not fake:
        assert FREEZE.exists() and json.loads(FREEZE.read_text()) == freeze, 'Frozen inputs/environment changed'
    out.mkdir(exist_ok=True)
    if fake: old.write_freeze(out / 'SYNTHETIC_FREEZE.json', freeze)
    cfg = dict(spec=spec, synthetic=fake, freeze_hash=freeze['freeze_hash'], split=freeze['expected_split'],
               device=str(device), hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU')
    cfg['fingerprint'] = v.digest(cfg); old.write_freeze(out / 'configuration.json', cfg)
    from review_stage14C import audit
    if d.validate_complete(out, cfg['fingerprint']):
        print(json.dumps(audit(out, False), indent=2)); print('SKIP completed, re-audited recovery study'); return
    launch = d.next_attempt(out / 'launches')
    v.atomic_json(launch / 'preflight.json', {'synthetic': True} if fake else v.idle_preflight())
    tick = time.perf_counter(); data = v.load_data('development', device, fake=fake); v.sync(device)
    assert data[-1] == cfg['split']
    v.atomic_json(launch / 'data.json', dict(seconds=time.perf_counter()-tick, split=data[-1],
        training_examples=len(data[1]), validation_examples=len(data[3])))
    shared_data_seconds=json.loads((launch/'data.json').read_text())['seconds']/(len(spec['seeds'])*len(spec['modes']))
    for seed, order in zip(spec['seeds'], spec['orders']):
        for position, mode in enumerate(order):
            run_one(out, seed, mode, position, cfg, freeze, data, device, shared_data_seconds, launch/'data.json')
    del data
    audit(out, True)
    paths = [out/name for name in ('configuration.json','audit.json','summary.csv','screen.csv','REPORT.md')]
    paths += list(out.glob('seed_*/*/COMPLETE.json'))+list(out.glob('seed_*/*/run_config.json'))
    paths += list((out / 'launches').rglob('*.json'))
    v.atomic_json(out / 'COMPLETE.json', dict(fingerprint=cfg['fingerprint'], synthetic=fake,
        sha256={str(path): d.sha(path) for path in paths}))
    print('COMPLETE. Send', out, 'for review.')


if __name__ == '__main__': main()
