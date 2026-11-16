"""Frozen, no-update diagnostic for midpoint token compression."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import platform
import re
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import stage8_vit_v2 as v
import stage9_boundary_diagnostic as d
import stage10_models as m
import stage10_representation as old
from stage14_tokens import MODES, TokenModel

ROOT = v.ROOT
SPEC = ROOT / 'protocol/stage14/token_diagnostic_spec_v1.json'
FREEZE = ROOT / 'protocol/stage14/FROZEN_TOKEN_DIAGNOSTIC_v1.json'
REAL_OUT = ROOT / 'results/stage14A_token_diagnostic_v1'


def verify(entries):
    for path, expected in entries.items():
        assert d.sha(Path(path)) == expected, ('Changed file', path)


def sources():
    names = ['scripts/stage14_tokens.py', 'scripts/stage14_diagnostic.py',
             'scripts/review_stage14A.py', 'scripts/test_stage14_tokens.py',
             'scripts/run_stage14A_token_diagnostic.cmd',
             'protocol/stage14/token_diagnostic_spec_v1.json',
             'protocol/stage14/TOKEN_DIAGNOSTIC_v1.md']
    return {**old.sources(), **{str(ROOT / n): d.sha(ROOT / n) for n in names}}


def freeze_record(spec, fake):
    prior = ROOT / 'results/stage11B_growth_pilot_v1'
    evidence = [prior / 'configuration.json', ROOT / 'results/stage11B_review_v1/REPORT.md',
                ROOT / 'results/stage13A_review_v1/REPORT.md',
                ROOT / 'paper/RESEARCH_DISCUSSION_MEMORY_v1.md']
    checkpoints = {}
    if not fake:
        for seed in spec['seeds']:
            parent = prior / f'seed_{seed}/full'
            rc = json.loads((parent / 'run_config.json').read_text())
            marker = d.validate_complete(parent, rc['fingerprint'])
            assert marker
            attempt = Path(marker['attempt'])
            summary = json.loads((attempt / 'summary.json').read_text())
            cp = attempt / 'final_checkpoint.pt'
            checkpoints[str(seed)] = dict(path=str(cp), sha256=d.sha(cp), updates=17600,
                run_fingerprint=rc['fingerprint'], root_fingerprint=rc['root_fingerprint'],
                accuracy=summary['final_validation_accuracy'], loss=summary['final_validation_loss'])
            evidence += [parent / 'COMPLETE.json', parent / 'run_config.json', attempt / 'summary.json']
    record = dict(prepared_date='2026-10-23', synthetic=fake, spec=spec,
        source_sha256=sources(), evidence_sha256={str(p): d.sha(p) for p in evidence},
        checkpoints=checkpoints, expected_split={'kind': 'artificial'} if fake else
        json.loads((prior / 'configuration.json').read_text())['split'],
        execution=m.execution_settings(), torch_version=str(torch.__version__),
        torchvision_version=str(torchvision.__version__), python_version=platform.python_version())
    record['freeze_hash'] = v.digest(record)
    return record


def make_core(seed, fake, record, device):
    core = m.make_model('full_rgb32', seed, fake)
    if fake:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 219001)
            torch.nn.init.normal_(core.head.weight, std=.05)
    else:
        src = record['checkpoints'][str(seed)]
        cp = torch.load(src['path'], map_location='cpu', weights_only=False)
        for key in ('run_fingerprint', 'root_fingerprint', 'updates'):
            assert cp[key] == src[key]
        assert cp['seed'] == seed and cp['variant'] == 'full'
        core.load_state_dict(cp['model_state'], strict=True)
    return core.to(device)


def streams(seed, spec, n):
    g = torch.Generator().manual_seed(seed + spec['stream_seed_offset'])
    indices = torch.randperm(n, generator=g)[:spec['trials'] * spec['batch_size']]
    assert len(indices) == spec['trials'] * spec['batch_size']
    return indices.reshape(spec['trials'], spec['batch_size']).tolist()


def order(spec, seed_position, trial):
    k = (seed_position + trial) % len(MODES)
    return list(MODES[k:] + MODES[:k])


@torch.no_grad()
def predictions(model, x, batch):
    model.eval()
    result = torch.cat([model(x[i:i + batch]).cpu() for i in range(0, len(x), batch)])
    assert torch.isfinite(result).all()
    return result.numpy()


def operation(model, x, y, kind, seed, device):
    model.train(kind == 'forward_backward')
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    v.sync(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    started = time.perf_counter()
    with torch.enable_grad() if kind == 'forward_backward' else torch.no_grad():
        model.zero_grad(set_to_none=True)
        logits = model(x)
        scalar = F.cross_entropy(logits, y) if kind == 'forward_backward' else logits.square().mean()
        if kind == 'forward_backward':
            scalar.backward()
        value = float(scalar.detach())
    v.sync(device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
    assert np.isfinite(value) and seconds > 0
    # Validation deliberately outside the timed region and captured memory peaks.
    finite = True
    patch_norm = 0.
    if kind == 'forward_backward':
        finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        patch_norm = float(model.core.body.conv_proj.weight.grad.norm())
        assert finite and patch_norm > 0
    return dict(seconds=seconds, baseline_allocated=baseline, peak_allocated=peak,
        incremental_allocated=peak - baseline, peak_reserved=reserved,
        scalar=value, all_gradients_finite=finite, patch_gradient_norm=patch_norm)


def run(out, cfg, record, device):
    spec, fake = cfg['spec'], record['synthetic']
    attempt = d.next_attempt(out)
    v.atomic_json(attempt / 'preflight.json', {'synthetic': True} if fake else v.idle_preflight())
    start = time.perf_counter()
    data = v.load_data('development', device, fake=fake)
    v.sync(device)
    x, y, ev, ey, split = data
    assert split == record['expected_split']
    v.atomic_json(attempt / 'data.json', dict(split=split, training_examples=len(y),
        validation_examples=len(ey), load_seconds=time.perf_counter() - start))
    for seed_position, seed in enumerate(spec['seeds']):
        start = time.perf_counter()
        core = make_core(seed, fake, record, device)
        before = old.state_hash(core.state_dict())
        model = TokenModel(core, 'full', spec['after_blocks'])
        logits = {}
        for mode in MODES:
            model.mode = mode
            logits[mode] = predictions(model, ev, spec['eval_batch_size'])
            print('VALIDATION', seed, mode, f'{100 * np.mean(logits[mode].argmax(1) == ey.cpu().numpy()):.2f}%', flush=True)
        if not fake:
            expected = record['checkpoints'][str(seed)]
            accuracy = float(np.mean(logits['full'].argmax(1) == ey.cpu().numpy()))
            loss = float(F.cross_entropy(torch.from_numpy(logits['full']), ey.cpu()))
            assert abs(accuracy - expected['accuracy']) < 1e-12, 'Source accuracy not reproduced'
            assert abs(loss - expected['loss']) <= spec['source_loss_tolerance'], 'Source loss not reproduced'
            assert accuracy >= spec['baseline_min_accuracy']
        np.savez(attempt / f'predictions_{seed}.npz', labels=ey.cpu().numpy(), **logits)
        validation_seconds = time.perf_counter() - start
        indices = streams(seed, spec, len(y))
        v.atomic_json(attempt / f'streams_{seed}.json', indices)
        warm, measured, telemetry = [], [], []
        for trial in range(spec['trials']):
            idx = torch.tensor(indices[trial], device=device)
            xx, yy = v.frozen6.normalize_cifar(x[idx]), y[idx]
            if trial == 0:
                for kind in spec['operations']:
                    for mode in order(spec, seed_position, 0):
                        model.mode = mode
                        for w in range(spec['warmups']):
                            row = operation(model, xx, yy, kind, seed + 216001 + w, device)
                            warm.append(dict(seed=seed, mode=mode, operation=kind, warmup=w, **row))
            kinds = spec['operations'] if trial % 2 == 0 else list(reversed(spec['operations']))
            for kind in kinds:
                for position, mode in enumerate(order(spec, seed_position, trial)):
                    model.mode = mode
                    dropout_seed = seed + spec['dropout_seed_offset'] + trial
                    row = operation(model, xx, yy, kind, dropout_seed, device)
                    measured.append(dict(seed=seed, trial=trial, mode=mode, operation=kind,
                        position=position, dropout_seed=dropout_seed, **row))
            if not fake and trial in (0, spec['trials'] - 1):
                telemetry.append(dict(trial=trial, sample=v.telemetry()))
        model.zero_grad(set_to_none=True)
        after = old.state_hash(core.state_dict())
        assert before == after
        v.write_csv(attempt / f'measurements_{seed}.csv', measured)
        v.write_csv(attempt / f'warmups_{seed}.csv', warm)
        v.atomic_json(attempt / f'telemetry_{seed}.json', telemetry)
        v.atomic_json(attempt / f'seed_{seed}.json', dict(seed=seed, state_before=before, state_after=after,
            validation_and_model_setup_seconds=validation_seconds, optimizer_updates=0,
            source_sha256=None if fake else d.sha(Path(record['checkpoints'][str(seed)]['path']))))
        print('COST COMPLETE', seed, len(measured), 'operations; weights unchanged', flush=True)
        del model, core
    verify(record['source_sha256']); verify(record['evidence_sha256'])
    for cp in record['checkpoints'].values():
        verify({cp['path']: cp['sha256']})
    v.atomic_json(out / 'RAW_COMPLETE.json', dict(fingerprint=cfg['fingerprint'], attempt=str(attempt),
        sha256={str(p): d.sha(p) for p in attempt.iterdir() if p.is_file()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-tag', default='final_cpu')
    ap.add_argument('--smoke-device', choices=['cpu', 'cuda'], default='cpu')
    args = ap.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,32}', args.smoke_tag)
    m.configure()
    spec = json.loads(SPEC.read_text())
    if args.smoke:
        spec.update(seeds=[9951, 9952], after_blocks=1, batch_size=3, eval_batch_size=4, trials=3, warmups=1)
    record = freeze_record(spec, args.smoke)
    if args.freeze_only:
        assert not args.smoke and not REAL_OUT.exists()
        old.write_freeze(FREEZE, record)
        print('FROZEN', record['freeze_hash'])
        return
    device = torch.device(args.smoke_device if args.smoke else 'cuda')
    assert device.type == 'cpu' or torch.cuda.is_available()
    out = ROOT / f'results/stage14_sanity_tokens_v1_{args.smoke_tag}' if args.smoke else REAL_OUT
    if not args.smoke:
        assert FREEZE.exists(), 'Real execution requires the prepared freeze'
        assert json.loads(FREEZE.read_text()) == record, 'Frozen sources/environment changed; stop for review'
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        old.write_freeze(out / 'SYNTHETIC_FREEZE.json', record)
    cfg = dict(spec=spec, freeze_hash=record['freeze_hash'], synthetic=args.smoke,
        device=str(device), hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        split=record['expected_split'])
    cfg['fingerprint'] = v.digest(cfg)
    old.write_freeze(out / 'configuration.json', cfg)
    from review_stage14A import audit
    if d.validate_complete(out, cfg['fingerprint']):
        print(json.dumps(audit(out, write=False), indent=2))
        print('COMPLETE: verified read-only; no repeated measurements')
        return
    if not (out / 'RAW_COMPLETE.json').exists():
        run(out, cfg, record, device)
    result = audit(out, write=True)
    paths = [out / p for p in ('configuration.json', 'RAW_COMPLETE.json', 'audit.json', 'summary.csv', 'REPORT.md')]
    if args.smoke:
        paths.append(out / 'SYNTHETIC_FREEZE.json')
    v.atomic_json(out / 'COMPLETE.json', dict(status='complete', fingerprint=cfg['fingerprint'],
        sha256={str(p): d.sha(p) for p in paths}))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
