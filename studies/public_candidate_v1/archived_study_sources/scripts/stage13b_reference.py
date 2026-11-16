"""Replay the known full-training reference and isolate classifier Adam reset."""
import argparse
import copy
import json
import math
import platform
import re
import time
from pathlib import Path
import numpy as np
import torch
import stage12_adaptation as prior
p, v, d, old = prior.p, prior.v, prior.d, prior.old
ROOT = v.ROOT
HERE = ROOT / 'protocol/stage13b'
SPEC = HERE / 'reference_spec_v1.json'
FREEZE = HERE / 'FROZEN_REFERENCE_v1.json'
REAL = ROOT / 'results/stage13B_reference_v1'


def verify(entries):
    for path, h in entries.items():
        assert p.sha(path) == h, path


def sources():
    names = ['scripts/stage13b_reference.py', 'scripts/review_stage13B.py',
        'scripts/test_stage13b_reference.py', 'scripts/run_stage13B_reference.cmd',
        'protocol/stage13b/reference_spec_v1.json', 'protocol/stage13b/REFERENCE_REPAIR_v1.md']
    return {**prior.sources(), **{str(ROOT / n): p.sha(ROOT / n) for n in names}}


def freeze_record(spec, recipe, fake):
    raw = ROOT / 'results/stage12A_adaptation_v1'
    evidence = [raw / 'configuration.json', ROOT / 'results/stage13A_review_v1/REPORT.md',
        ROOT / 'results/stage12A_review_v1/REPORT.md']
    checkpoints = {}
    if not fake:
        for seed in spec['seeds']:
            parent = raw / f'seed_{seed}/lp_full'
            rc = json.loads((parent / 'run_config.json').read_text())
            marker = d.validate_complete(parent, rc['fingerprint']); assert marker
            attempt = Path(marker['attempt'])
            checkpoint = attempt / 'warmup_checkpoint.pt'
            epochs = d.records(attempt / 'epochs.csv')
            checkpoints[str(seed)] = dict(path=str(checkpoint), sha256=p.sha(checkpoint),
                fingerprint=rc['fingerprint'], root_fingerprint=rc['root_fingerprint'],
                trace=str(attempt / 'steps.csv'), initial=epochs[1], expected=epochs[2])
            evidence += [parent / 'COMPLETE.json', parent / 'run_config.json', attempt / 'epochs.csv', attempt / 'steps.csv']
    record = dict(date='2026-10-23', synthetic=fake, spec=spec, recipe=recipe,
        source_sha256=sources(), evidence_sha256={str(f): p.sha(f) for f in evidence},
        checkpoints=checkpoints, split={'kind': 'artificial'} if fake else json.loads((raw / 'configuration.json').read_text())['split'],
        pretrained=None if fake else dict(path=str(p.WEIGHTS), sha256=p.sha(p.WEIGHTS)),
        execution=p.execution.execution_settings(), torch=str(torch.__version__), python=platform.python_version())
    record['freeze_hash'] = v.digest(record)
    return record


def rates(opt, index, recipe):
    factor = p.rate_factor(index, recipe)
    for group in opt.param_groups:
        group['lr'] = factor * recipe['head_lr' if group['role'] == 'head' else 'backbone_lr']
    return factor * recipe['backbone_lr'], factor * recipe['head_lr']


def restore(seed, cp, recipe, fake, device, arm):
    model = p.make_model(seed, fake).to(device)
    model.load_state_dict(cp['model_state'], strict=True)
    opt = p.optimizer(model, recipe)
    opt.load_state_dict(copy.deepcopy(cp['optimizer_state']))
    assert arm in ('continuation', 'reset')
    if arm == 'reset':
        opt.state.clear()
    bg = torch.Generator(); bg.set_state(cp['batch_rng'].cpu())
    ag = torch.Generator(); ag.set_state(cp['augmentation_rng'].cpu())
    return model, opt, bg, ag


def ages(model, opt):
    return {name: int(opt.state[t]['step']) if t in opt.state else 0 for name, t in model.named_parameters()}


@torch.no_grad()
def predict(model, ev, recipe):
    model.eval()
    result = []
    for i in range(0, len(ev), recipe['eval_batch_size']):
        z = p.forward(model, p.transform(ev[i:i + recipe['eval_batch_size']], recipe['image_size']), 'full')
        result.append(z.cpu())
    z = torch.cat(result)
    assert torch.isfinite(z).all()
    return z.numpy()


def fake_source(seed, data, recipe, device):
    x, y, ev, ey, _ = data
    model = p.make_model(seed, True).to(device); opt = p.optimizer(model, recipe)
    bg = torch.Generator().manual_seed(seed + 281001); ag = torch.Generator().manual_seed(seed + 282001)
    index = 0
    for epoch in range(2):
        for ix in v.frozen6.epoch_batches(len(y), recipe['batch_size'], bg):
            ix = ix.to(device); xx = p.transform(x[ix], recipe['image_size'], ag)
            rates(opt, index, recipe); p.update(model, opt, xx, y[ix], 'head', 0.); index += 1
    cp = copy.deepcopy(dict(seed=seed, arm='lp_full', updates=index, model_state=model.state_dict(),
        optimizer_state=opt.state_dict(), batch_rng=bg.get_state(), augmentation_rng=ag.get_state()))
    initial = predict(model, ev, recipe)
    trace = []
    for ix in v.frozen6.epoch_batches(len(y), recipe['batch_size'], bg):
        ix = ix.to(device); xx = p.transform(x[ix], recipe['image_size'], ag)
        br, hr = rates(opt, index, recipe); loss = p.update(model, opt, xx, y[ix], 'full', 0.); index += 1
        trace.append(dict(step=index, batch_examples=len(ix), loss=loss, backbone_lr=br, head_lr=hr))
    return cp, initial, predict(model, ev, recipe), trace


def original_trace(entry, recipe):
    return [r for r in d.records(Path(entry['trace'])) if int(r['epoch']) == 3]


def run(out, cfg, record, device):
    spec, recipe, fake = cfg['spec'], cfg['recipe'], cfg['synthetic']
    attempt = d.next_attempt(out)
    v.atomic_json(attempt / 'preflight.json', [] if fake else v.idle_preflight())
    data = p.load_data(device, fake); x, y, ev, ey, split = data
    assert split == record['split'] and math.ceil(len(y) / recipe['batch_size']) == recipe['batches_per_epoch']
    for seed, order in zip(spec['seeds'], spec['orders']):
        if fake:
            cp, initial_expected, final_expected, expected_trace = fake_source(seed, data, recipe, device)
            np.savez(attempt / f'golden_{seed}.npz', labels=ey.cpu().numpy(), initial=initial_expected, final=final_expected)
            old.checkpoint(attempt / f'source_{seed}.pt', cp)
        else:
            entry = record['checkpoints'][str(seed)]
            cp = torch.load(entry['path'], map_location='cpu', weights_only=True)
            assert cp['fingerprint'] == entry['fingerprint'] and cp['root_fingerprint'] == entry['root_fingerprint']
            expected_trace = original_trace(entry, recipe)
        assert cp['seed'] == seed and cp['arm'] == 'lp_full' and cp['updates'] == 2 * recipe['batches_per_epoch']
        v.atomic_json(attempt / f'expected_trace_{seed}.json', expected_trace)
        initial_hash = old.state_hash(cp['model_state'])
        for arm in order:
            # Warm-up on disposable state, then restore ALL experimental state.
            net, opt, _, _ = restore(seed, cp, recipe, fake, device, arm)
            xx = p.transform(x[:recipe['batch_size']], recipe['image_size'])
            for _ in range(spec['warmup_steps']):
                p.update(net, opt, xx, y[:len(xx)], 'full', 0.)
            del net, opt
            net, opt, bg, ag = restore(seed, cp, recipe, fake, device, arm)
            assert old.state_hash(net.state_dict()) == initial_hash
            start_ages = ages(net, opt)
            buffers = {n: t.detach().cpu().clone() for n, t in net.named_buffers()}
            scores = {'step_0': predict(net, ev, recipe), 'labels': ey.cpu().numpy()}
            rows, batches = [], []
            initial_ag = ag.get_state().clone()
            for step, ix in enumerate(v.frozen6.epoch_batches(len(y), recipe['batch_size'], bg), 1):
                batches.append(ix.tolist()); idx = ix.to(device)
                xx = p.transform(x[idx], recipe['image_size'], ag)
                global_index = cp['updates'] + step - 1
                br, hr = rates(opt, global_index, recipe)
                v.sync(device); started = time.perf_counter()
                loss = p.update(net, opt, xx, y[idx], 'full', 0.)
                v.sync(device); elapsed = time.perf_counter() - started
                rows.append(dict(step=global_index + 1, trajectory_step=step, batch_examples=len(ix),
                    batch_sha256=old.state_hash({'indices': ix}), loss=loss, backbone_lr=br, head_lr=hr, update_seconds=elapsed))
                assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in net.parameters())
                if step in spec['score_steps']:
                    scores[f'step_{step}'] = predict(net, ev, recipe)
                    accuracy = np.mean(scores[f'step_{step}'].argmax(1) == scores['labels'])
                    print('REFERENCE', seed, arm, f'step={step}/{recipe["batches_per_epoch"]} accuracy={100*accuracy:.2f}%', flush=True)
            for n, t in net.named_buffers():
                assert torch.equal(t.cpu(), buffers[n]), n
            final_ages = ages(net, opt)
            assert all(final_ages[n] == start_ages[n] + recipe['batches_per_epoch'] for n in start_ages)
            net.zero_grad(set_to_none=True)
            stem = f'{seed}_{arm}'
            v.write_csv(attempt / f'steps_{stem}.csv', rows)
            v.atomic_json(attempt / f'batches_{stem}.json', batches)
            np.savez(attempt / f'scores_{stem}.npz', **scores)
            old.checkpoint(attempt / f'final_{stem}.pt', dict(model_state=net.state_dict(), optimizer_state=opt.state_dict(),
                batch_rng=bg.get_state(), augmentation_rng=ag.get_state(), initial_augmentation_rng=initial_ag,
                seed=seed, arm=arm, updates=cp['updates'] + len(rows), fingerprint=cfg['fingerprint']))
            v.atomic_json(attempt / f'state_{stem}.json', dict(initial_hash=initial_hash,
                final_hash=old.state_hash(net.state_dict()), start_ages=start_ages, final_ages=final_ages,
                batchnorm_unchanged=True, finite_gradients_all_steps=True,
                telemetry=[] if fake else v.telemetry()))
            del net, opt
        del cp
    verify(record['source_sha256']); verify(record['evidence_sha256'])
    for entry in record['checkpoints'].values():
        verify({entry['path']: entry['sha256']})
    v.atomic_json(out / 'RAW_COMPLETE.json', dict(fingerprint=cfg['fingerprint'], attempt=str(attempt),
        sha256={str(f): p.sha(f) for f in attempt.iterdir() if f.is_file()}))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--freeze-only', action='store_true')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--smoke-tag', default='final_cpu')
    ap.add_argument('--smoke-device', choices=['cpu', 'cuda'], default='cpu'); args = ap.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,32}', args.smoke_tag)
    p.execution.configure(); spec = json.loads(SPEC.read_text()); recipe = json.loads(prior.SPEC.read_text())
    if args.smoke:
        spec.update(seeds=[9971, 9972], score_steps=[0, 1, 3, 6], warmup_steps=1)
        recipe.update(image_size=32, batch_size=3, eval_batch_size=4, batches_per_epoch=6)
    record = freeze_record(spec, recipe, args.smoke)
    if args.freeze_only:
        assert not args.smoke and not REAL.exists()
        old.write_freeze(FREEZE, record); print('FROZEN', record['freeze_hash']); return
    device = torch.device(args.smoke_device if args.smoke else 'cuda')
    assert device.type == 'cpu' or torch.cuda.is_available()
    out = ROOT / f'results/stage13B_sanity_reference_v1_{args.smoke_tag}' if args.smoke else REAL
    if not args.smoke:
        assert FREEZE.exists() and json.loads(FREEZE.read_text()) == record, 'Frozen inputs/environment changed'
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        old.write_freeze(out / 'SYNTHETIC_FREEZE.json', record)
    cfg = dict(spec=spec, recipe=recipe, synthetic=args.smoke, freeze_hash=record['freeze_hash'],
        split=record['split'], device=str(device), hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu')
    cfg['fingerprint'] = v.digest(cfg); old.write_freeze(out / 'configuration.json', cfg)
    from review_stage13B import audit
    if d.validate_complete(out, cfg['fingerprint']):
        print(json.dumps(audit(out, False), indent=2)); print('Verified read-only'); return
    if not (out / 'RAW_COMPLETE.json').exists():
        run(out, cfg, record, device)
    result = audit(out, True)
    paths = [out / n for n in ['configuration.json', 'RAW_COMPLETE.json', 'audit.json', 'curves.csv', 'comparisons.csv', 'REPORT.md']]
    if args.smoke:
        paths.append(out / 'SYNTHETIC_FREEZE.json')
    v.atomic_json(out / 'COMPLETE.json', dict(status='complete', fingerprint=cfg['fingerprint'], sha256={str(f): p.sha(f) for f in paths}))
    print(json.dumps(result, indent=2)); print('COMPLETE. Review before any local comparison.')


if __name__ == '__main__':
    main()
