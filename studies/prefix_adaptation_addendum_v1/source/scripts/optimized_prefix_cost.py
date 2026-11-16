"""Bounded routing cost screen against the qualified BF16/automatic baseline."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import gc
import math
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
import torch
import bf16_continuation as c

ROOT = c.ROOT
HERE = ROOT / 'protocol/optimized_prefix_cost'
SPEC = HERE / 'cost_spec_v1.json'
FREEZE = HERE / 'FROZEN_COST_v1.json'
OUT = ROOT / 'results/optimized_prefix_cost_v1'
read = c.read
save = c.r.v.atomic_json
sha = c.r.d.sha


def sources():
    names = ['scripts/optimized_prefix_cost.py', 'scripts/review_optimized_prefix_cost.py',
             'scripts/test_optimized_prefix_cost.py', 'scripts/run_optimized_prefix_cost.cmd',
             'protocol/optimized_prefix_cost/cost_spec_v1.json',
             'protocol/optimized_prefix_cost/COST_PROTOCOL_v1.md']
    return {**c.sources(), **{str(ROOT / p): sha(ROOT / p) for p in names}}


def configuration(smoke, device):
    cfg = c.configuration(smoke, device)
    spec = read(SPEC)
    if smoke:
        spec.update(seeds=[9971, 9972], boundaries=[1, 2], sessions=2,
                    updates=16, anchor_updates=2, warmup_updates=2,
                    component_updates=2, batch_size=3)
    # Retain source age and optimizer contract from the qualified implementation.
    cfg['spec'].update(spec)
    cfg['spec']['arms'] = ['reference_bf16_auto' if device.type == 'cuda' else 'reference_fp32_math']
    cfg['sources'] = sources()
    if not smoke:
        prior = read(c.FREEZE)
        assert prior['fingerprint'] == '1fec63bf3a8821e606a773f2bdaebca8f96cdf8c7d8bc9c871ff62baa46d0518'
        review = ROOT / 'results/bf16_continuation_review_v1'
        assert read(review / 'independent_audit.json')['both_policies_pass']
        for path in [c.FREEZE, c.OUT / 'COMPLETE.json', review / 'independent_audit.json',
                     review / 'group_summary.json', review / 'REPORT.md']:
            cfg['evidence'][str(path)] = sha(path)
    cfg.pop('fingerprint')
    cfg['fingerprint'] = c.r.v.digest(cfg)
    return cfg


def arms(spec):
    result = [dict(name='full', boundary=0, interval=1, routed=False),
              dict(name='routed_full', boundary=spec['boundaries'][-1], interval=1, routed=True)]
    for k in spec['boundaries']:
        for interval in [0, *spec['refresh_intervals']]:
            result.append(dict(name=f'cut{k}_r{interval}', boundary=k, interval=interval, routed=True))
    return result


def schedule(spec, session):
    choices = arms(spec)
    half = len(choices) // 2
    if session // 2 % 2:
        choices = choices[half:] + choices[:half]
    if session % 2:
        choices = list(reversed(choices))
    seeds = spec['seeds'] if session % 2 == 0 else list(reversed(spec['seeds']))
    return [(seed, arm) for seed in seeds for arm in choices]


def full_step(arm, step):
    return arm['interval'] > 0 and step % arm['interval'] == 0


class Router:
    """Freeze stem, position/class tokens and first k blocks; preserve native forward."""
    def __init__(self, model, boundary):
        depth = len(model.body.encoder.layers)
        assert model.variant == 'full_rgb32' and not model.dual and 0 <= boundary <= depth
        suffix = ('head.', 'body.encoder.ln.') + tuple(
            f'body.encoder.layers.encoder_layer_{i}.' for i in range(boundary, depth))
        self.prefix = {n: p for n, p in model.named_parameters() if not n.startswith(suffix)} if boundary else {}
        self.active = True
        assert all(p.requires_grad for p in model.parameters())

    def select(self, active):
        if active != self.active:
            for parameter in self.prefix.values():
                parameter.requires_grad_(active)
            self.active = active


class Stream:
    def __init__(self, seed, spec, x, y):
        self.bg, self.ag, self.mg, self.pg = c.r.previous.generators(seed, spec)
        self.x, self.y, self.spec = x, y, spec
        self.order = torch.empty(0, dtype=torch.long)
        self.offset = 0
        self.indices = []
        self.lambdas = []

    def batch(self):
        if self.offset >= len(self.order):
            self.order = torch.randperm(len(self.y), generator=self.bg)
            self.offset = 0
        ids = self.order[self.offset:self.offset + self.spec['batch_size']]
        self.offset += len(ids)
        self.indices.append(ids.tolist())
        ids = ids.to(self.x.device)
        x = c.r.v.frozen6.augment_and_normalize(self.x[ids], self.ag)
        x, targets, lam = c.r.v.mix_batch(x, self.y[ids], self.mg, self.pg, self.spec['mixup_alpha'], 10)
        self.lambdas.append(float(lam))
        return x, targets

    def fingerprint(self):
        return c.r.v.digest(dict(indices=self.indices, mixup=self.lambdas))


def step(model, opt, router, arm, index, seed, stream, policy, spec, component=False):
    device = next(model.parameters()).device
    timings = {}
    def measure(name, fn):
        if component: c.r.v.sync(device)
        started = time.perf_counter() if component else 0
        value = fn()
        if component:
            c.r.v.sync(device)
            timings[name] = time.perf_counter() - started
        return value
    x, targets = measure('augmentation', stream.batch)
    def route():
        torch.manual_seed(seed + spec['dropout_seed_offset'] + index)
        opt.zero_grad(set_to_none=True)
        if arm['routed']: router.select(full_step(arm, index))
    measure('routing_rng_zero_grad', route)
    _, loss = measure('forward_loss', lambda: c.forward_loss(model, x, targets, policy, spec))
    value = measure('loss_readback', loss.item)
    if not math.isfinite(value): raise RuntimeError('Nonfinite loss; preserve this failed attempt')
    measure('backward', loss.backward)
    measure('optimizer', opt.step)
    return dict(step=index, full=full_step(arm, index), loss=value, batch_examples=len(x), components=timings)


def restore(model, opt, anchor):
    model.requires_grad_(True)
    model.load_state_dict(anchor['model'])
    model.zero_grad(set_to_none=True)
    opt.load_state_dict(copy.deepcopy(anchor['optimizer']))
    model.train()


def snapshot(model, opt):
    return dict(model={n: t.detach().cpu().clone() for n, t in model.state_dict().items()},
                optimizer=copy.deepcopy(opt.state_dict()), model_hash=c.r.old.state_hash(model.state_dict()),
                optimizer_hash=c.r.optimizer_hash(opt))


def make_anchor(seed, cfg, device, x, y):
    policy = cfg['spec']['arms'][0]
    model, opt, _ = c.make(seed, policy, cfg, device)
    for group in opt.param_groups: group['lr'] = cfg['spec']['learning_rate']
    model.train()
    stream = Stream(seed + 7000, cfg['spec'], x, y)
    with c.e.policy(policy):
        for index in range(cfg['spec']['anchor_updates']):
            step(model, opt, None, arms(cfg['spec'])[0], index, seed + 7000, stream, policy, cfg['spec'])
    anchor = snapshot(model, opt)
    del model, opt
    return anchor


def state_check(model, opt, router, arm, anchor, spec):
    prefix = set(router.prefix)
    full_count = sum(full_step(arm, i) for i in range(spec['updates']))
    for name, p in model.named_parameters():
        active = name not in prefix or full_step(arm, spec['updates'] - 1)
        assert p.requires_grad == active
        assert (p.grad is not None) == active
        assert bool(torch.isfinite(p).all())
        if active: assert bool(torch.isfinite(p.grad).all()) and p.grad.dtype == torch.float32
        state = opt.state[p]
        age = spec['anchor_updates'] + (full_count if name in prefix else spec['updates'])
        assert int(state['step']) == age, (name, state['step'], age)
        for key in ('exp_avg', 'exp_avg_sq'):
            assert state[key].dtype == torch.float32 and bool(torch.isfinite(state[key]).all())
        if name in prefix and full_count == 0:
            assert torch.equal(p.detach().cpu(), anchor['model'][name]), name
    return dict(status='PASS', prefix_tensors=len(prefix), prefix_coordinates=sum(p.numel() for p in router.prefix.values()),
                full_updates=full_count, suffix_updates=spec['updates'], all_states_finite=True,
                optimizer_ages_verified=True, inactive_gradients_absent=True,
                pure_suffix_prefix_unchanged=(full_count == 0))


def run_arm(parent, seed, arm, cfg, device, x, y, anchor):
    spec = cfg['spec']; policy = spec['arms'][0]
    model, opt, _ = c.make(seed, policy, cfg, device)
    restore(model, opt, anchor)
    router = Router(model, arm['boundary'])
    stream = Stream(seed + 9000, spec, x, y)
    with c.e.policy(policy):
        flags = c.e.settings()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            step(model, opt, router, arm, 0, seed + 9000, stream, policy, spec)
        for index in range(1, spec['warmup_updates']):
            step(model, opt, router, arm, index, seed + 9000, stream, policy, spec)
    counts = Counter(event.name for event in prof.events())
    dispatch = {k: counts[v] for k, v in c.q.frozen_backend.OPS.items()}
    assert sum(dispatch.values()) == len(model.body.encoder.layers)
    restore(model, opt, anchor)
    router = Router(model, arm['boundary'])
    assert c.r.old.state_hash(model.state_dict()) == anchor['model_hash']
    assert c.r.optimizer_hash(opt) == anchor['optimizer_hash']
    stream = Stream(seed, spec, x, y)
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    rows = []
    with c.e.policy(policy):
        c.r.v.sync(device); start = time.perf_counter()
        for index in range(spec['updates']):
            rows.append(step(model, opt, router, arm, index, seed, stream, policy, spec))
        c.r.v.sync(device); seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    integrity = state_check(model, opt, router, arm, anchor, spec)
    final_model_hash = c.r.old.state_hash(model.state_dict())
    final_optimizer_hash = c.r.optimizer_hash(opt)
    # An independent reset pass instruments components; it is not the primary clock.
    restore(model, opt, anchor); router = Router(model, arm['boundary'])
    component_stream = Stream(seed, spec, x, y); component_rows = []
    with c.e.policy(policy):
        for index in range(spec['component_updates']):
            component_rows.append(step(model, opt, router, arm, index, seed, component_stream, policy, spec, True))
    summary = dict(arm=arm, seed=seed, seconds=seconds, milliseconds_per_update=1000*seconds/spec['updates'],
                   updates=spec['updates'], anchor_model_hash=anchor['model_hash'], anchor_optimizer_hash=anchor['optimizer_hash'],
                   final_model_hash=final_model_hash, final_optimizer_hash=final_optimizer_hash,
                   stream_fingerprint=stream.fingerprint(), policy=flags, warmup_dispatch=dispatch,
                   peak_allocated_bytes=peak, integrity=integrity)
    parent.mkdir()
    save(parent / 'summary.json', summary); save(parent / 'steps.json', rows)
    save(parent / 'components.json', component_rows)
    c.seal(parent, cfg['fingerprint'], list(parent.glob('*.json')))
    print(f'COST seed={seed} {arm["name"]} {1000*seconds/spec["updates"]:.3f} ms/update', flush=True)
    del model, opt; gc.collect()


def session_run(out, session, cfg, device):
    folder = out / f'session_{session:02d}'; folder.mkdir(exist_ok=True)
    if c.r.d.validate_complete(folder, cfg['fingerprint']):
        print('SKIP completed session', session, flush=True); return
    # Retry the entire session; never splice earlier successful arms into a new process.
    attempt = c.r.d.next_attempt(folder)
    save(attempt / 'process.json', dict(uuid=str(uuid.uuid4()), pid=os.getpid(), session=session))
    try:
        save(attempt / 'preflight.json', {'smoke': True} if cfg['smoke'] else c.r.v.idle_preflight())
        x, y, ev, ey, split = c.r.v.load_data('development', device, fake=cfg['smoke'])
        assert split == cfg['split']; del ev, ey
        anchors = {seed: make_anchor(seed, cfg, device, x, y) for seed in cfg['spec']['seeds']}
        for position, (seed, arm) in enumerate(schedule(cfg['spec'], session)):
            run_arm(attempt / f'{position:02d}_{seed}_{arm["name"]}', seed, arm, cfg, device, x, y, anchors[seed])
        save(attempt / 'telemetry_after.json', {'smoke': True} if cfg['smoke'] else c.r.v.telemetry())
        paths = [p for p in attempt.rglob('*.json') if p.is_file()]
        save(folder / 'COMPLETE.json', dict(fingerprint=cfg['fingerprint'], attempt=str(attempt), sha256={str(p): sha(p) for p in paths}))
    except Exception as error:
        save(attempt / 'FAILED.json', dict(type=type(error).__name__, message=str(error)))
        raise


def main():
    ap = argparse.ArgumentParser(); action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze-only', action='store_true'); action.add_argument('--run-all', action='store_true')
    action.add_argument('--session', type=int)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--tag', default='v1'); args = ap.parse_args()
    assert args.tag.replace('_', '').isalnum()
    assert sys.flags.optimize == 0, 'Run without -O: integrity assertions must remain enabled'
    c.r.m.configure(); device = torch.device(args.device)
    assert args.smoke or device.type == 'cuda'
    cfg = configuration(args.smoke, device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists()
        c.r.old.write_freeze(FREEZE, cfg); print('FROZEN', cfg['fingerprint']); return
    if not args.smoke: assert cfg == read(FREEZE), 'Frozen inputs/environment changed'
    out = ROOT / f'results/optimized_prefix_cost_smoke_{args.tag}_{device.type}' if args.smoke else OUT
    out.mkdir(exist_ok=True); c.r.old.write_freeze(out / 'configuration.json', cfg)
    from review_optimized_prefix_cost import audit
    if c.r.d.validate_complete(out, cfg['fingerprint']):
        print(audit(out)); return
    if args.session is not None:
        assert 0 <= args.session < cfg['spec']['sessions']
        session_run(out, args.session, cfg, device); return
    for session in range(cfg['spec']['sessions']):
        argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--session', str(session), '--device', args.device, '--tag', args.tag]
        if args.smoke: argv.append('--smoke')
        subprocess.run(argv, cwd=ROOT, check=True)
    result = audit(out, write=True)
    c.seal(out, cfg['fingerprint'], [out / p for p in ('configuration.json', 'audit.json', 'pairings.json', 'REPORT.md')] + list(out.glob('session_*/COMPLETE.json')))
    print(result); print('COMPLETE. Send', out, 'for review.')


if __name__ == '__main__': main()
