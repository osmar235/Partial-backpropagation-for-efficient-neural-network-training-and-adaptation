"""Short frozen diagnostic at existing checkpoints; no new training trajectory."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import math
import platform
import re
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import stage8_vit_v2 as v
import stage9_boundary_diagnostic as d
import stage10_models as m
import stage10_representation as old
import stage11_growth as g

ROOT = v.ROOT
SPEC_PATH = ROOT / 'protocol/stage11/growth_feasibility_spec_v1.json'
FREEZE_PATH = ROOT / 'protocol/stage11/FROZEN_GROWTH_FEASIBILITY_v1.json'
REAL_OUT = ROOT / 'results/stage11A_growth_feasibility_v1'
RAW = ROOT / 'results/stage10A_representation_v1'


def sources():
    names = ['scripts/stage11_growth.py', 'scripts/stage11_growth_feasibility.py',
             'scripts/test_stage11_growth.py', 'scripts/review_stage11A.py',
             'scripts/run_stage11A_growth_feasibility.cmd',
             'protocol/stage11/growth_feasibility_spec_v1.json',
             'protocol/stage11/GROWTH_FEASIBILITY_v1.md']
    return {**old.sources(), **{str(ROOT / n): d.sha(ROOT / n) for n in names}}


def freeze_record(spec, fake):
    checkpoints, evidence = {}, {}
    if fake:
        split = {'kind': 'artificial'}
    else:
        cfg = json.loads((RAW / 'configuration.json').read_text())
        assert not cfg['synthetic']
        d.validate_complete(RAW, cfg['fingerprint'])
        for p, h in cfg['source_sha256'].items():
            assert d.sha(p) == h, p
        for seed in spec['source_seeds']:
            for variant in spec['source_variants']:
                parent = RAW / f'seed_{seed}' / variant
                run = json.loads((parent / 'run_config.json').read_text())
                marker = d.validate_complete(parent, run['fingerprint'])
                attempt = Path(marker['attempt'])
                cp, summary = attempt / 'final_checkpoint.pt', attempt / 'summary.json'
                checkpoints[f'{seed}:{variant}'] = dict(path=str(cp), sha256=d.sha(cp),
                                                       summary_path=str(summary), summary_sha256=d.sha(summary))
        for rel in ['configuration.json', 'COMPLETE.json']:
            evidence[str(RAW / rel)] = d.sha(RAW / rel)
        for rel in ['results/stage10A_review_v1/REPORT.md', 'results/stage10A_review_v1/independent_audit.json',
                    'protocol/stage10/POST_SCREEN_DECISION_v1.md']:
            evidence[str(ROOT / rel)] = d.sha(ROOT / rel)
        split = cfg['split']
    record = dict(version=spec['version'],prepared_date='2026-10-22',spec=spec,synthetic=fake,
                  source_sha256=sources(),evidence_sha256=evidence,checkpoints=checkpoints,
                  expected_split=split,required_execution=m.execution_settings())
    record['freeze_hash'] = v.digest(record)
    return record


def batch(data, seed, index, indices, spec):
    x, y = data[:2]
    ix = torch.tensor(indices, device=x.device)
    ag = torch.Generator().manual_seed(seed + 221001 + index)
    mg = np.random.default_rng(seed + 222001 + index)
    pg = torch.Generator().manual_seed(seed + 223001 + index)
    xx = v.frozen6.augment_and_normalize(x[ix], ag)
    return v.mix_batch(xx, y[ix], mg, pg, spec['mixup_alpha'], 10)


def step(model, opt, data, seed, index, indices, spec):
    xx, targets, lam = batch(data, seed, index, indices, spec)
    loss = old.update(model, opt, xx, targets, seed + 224001 + index, spec['label_smoothing'])
    return loss, lam


def inputs(record, seed, data, device, fake):
    models = {}
    for variant in record['spec']['source_variants']:
        model = m.make_model(variant, seed, fake).to(device)
        if fake:
            opt = g.reset_adam(model)
            x, y = data[:2]
            targets = F.one_hot(y[:3], 10).float()
            for i in range(3):
                old.update(model, opt, x[:3], targets, seed + 225001 + i)
        else:
            entry = record['checkpoints'][f'{seed}:{variant}']
            cp = torch.load(entry['path'], map_location='cpu', weights_only=True)
            assert cp['seed'] == seed and cp['variant'] == variant and cp['updates'] == 17600
            model.load_state_dict(cp['model_state'])
        models['small' if variant == 'small_rgb32' else 'full'] = model
    return models


@torch.no_grad()
def equivalence(models, seed, data, spec):
    small, grown = models['small'], models['grown']
    small.eval(); grown.eval()
    ev, ey = data[2:4]
    rows = []
    for start in range(0, len(ey), spec['eval_batch_size']):
        x = ev[start:start + spec['eval_batch_size']]
        labels = ey[start:start + spec['eval_batch_size']]
        a, b = small(x), grown(x)
        error = (a - b).abs()
        scaled = error / (spec['logit_atol'] + spec['logit_rtol'] * a.abs())
        values = zip(labels.tolist(), a.argmax(1).tolist(), b.argmax(1).tolist(),
                     F.cross_entropy(a, labels, reduction='none').tolist(),
                     F.cross_entropy(b, labels, reduction='none').tolist(),
                     error.amax(1).tolist(), scaled.amax(1).tolist())
        for i, (label, ap, bp, al, bl, ae, se) in enumerate(values):
            rows.append(dict(seed=seed,index=start+i,label=label,small_prediction=ap,grown_prediction=bp,
                             small_loss=al,grown_loss=bl,max_abs_logit_error=ae,max_scaled_logit_error=se))
    small.train(); grown.train()
    return rows


def liveness(models, initial, seed, data, spec, indices, attempt):
    rows, summary = [], []
    for path, model in models.items():
        model.load_state_dict(initial[path]); model.train()
        opt = g.reset_adam(model, spec['learning_rate'], spec['weight_decay'])
        before = old.state_hash(model.state_dict())
        for i in range(spec['liveness_steps']):
            loss, lam = step(model, opt, data, seed, i, indices[i % len(indices)], spec)
            rows.append(dict(seed=seed,path=path,step=i+1,loss=loss,mixup_lambda=lam))
        finite = all(torch.isfinite(p).all() and p.grad is not None and torch.isfinite(p.grad).all()
                     and torch.isfinite(opt.state[p]['exp_avg']).all() and torch.isfinite(opt.state[p]['exp_avg_sq']).all()
                     for p in model.parameters())
        counters = [int(opt.state[p]['step']) for p in model.parameters()]
        after = old.state_hash(model.state_dict())
        twins = g.twin_difference(model) if path == 'grown' else None
        assert finite and min(counters) == max(counters) == spec['liveness_steps']
        assert before != after
        assert path != 'grown' or twins > 1e-8
        summary.append(dict(seed=seed,path=path,finite=True,optimizer_reset=True,minimum_step=min(counters),
                            maximum_step=max(counters),state_before=before,state_after=after,
                            grown_embedding_twin_difference=twins))
        old.checkpoint(attempt/f'liveness_{seed}_{path}.pt',dict(seed=seed,path=path,steps=spec['liveness_steps'],
                       model_state=model.state_dict(),optimizer_state=opt.state_dict()))
        model.load_state_dict(initial[path]); model.zero_grad(set_to_none=True)
    return rows, summary


def benchmark(models, initial, seed, data, spec, indices):
    paths = spec['paths']
    optimizers = {p:g.reset_adam(models[p],spec['learning_rate'],spec['weight_decay']) for p in paths}

    def restore(path):
        models[path].load_state_dict(initial[path]); models[path].zero_grad(set_to_none=True)
        for state in optimizers[path].state.values():
            for key in ['step','exp_avg','exp_avg_sq']:
                state[key].zero_()

    def one(path,index):
        restore(path)
        (loss,lam),seconds = d.timed(data[0].device,lambda:step(models[path],optimizers[path],data,seed,index,
                                                            indices[index % len(indices)],spec))
        assert all(int(s['step']) == 1 for s in optimizers[path].state.values())
        return loss,lam,seconds

    cold = []
    for p in paths:
        loss,lam,seconds = one(p,0)
        cold.append(dict(seed=seed,path=p,seconds=seconds,loss=loss,mixup_lambda=lam))
    for i in range(spec['warmup_steps_per_path']):
        for p in paths:
            one(p,i % len(indices))
    rows,telemetry = [],[]
    for i in range(spec['probe_batches']):
        order = paths[i % 3:] + paths[:i % 3]
        if (i // 3) % 2:
            order = list(reversed(order))
        if data[0].device.type == 'cuda' and i % 6 == 0:
            telemetry.append(dict(seed=seed,probe=i,sample=json.dumps(v.telemetry())))
        for position,p in enumerate(order):
            loss,lam,seconds = one(p,i)
            rows.append(dict(seed=seed,probe=i,path=p,order_position=position,seconds=seconds,
                             loss=loss,mixup_lambda=lam,batch_examples=len(indices[i]),
                             dropout_seed=seed+224001+i))
    for p in paths:
        restore(p)
        assert old.state_hash(models[p].state_dict()) == old.state_hash(initial[p])
    return rows,cold,telemetry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-tag',default='final_cpu')
    p.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu')
    args = p.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag)
    m.configure(); fake = args.smoke
    spec = json.loads(SPEC_PATH.read_text())
    if fake:
        spec.update(source_seeds=[9961,9962],batch_size=3,eval_batch_size=4,probe_batches=6,
                    warmup_steps_per_path=2,liveness_steps=4)
    record = freeze_record(spec,fake)
    if args.freeze_only:
        assert not fake and not REAL_OUT.exists()
        old.write_freeze(FREEZE_PATH,record)
        print('FROZEN',record['freeze_hash']); return
    out = ROOT/'results'/('stage11_sanity_growth_v1_'+args.smoke_tag) if fake else REAL_OUT
    if fake:
        out.mkdir(exist_ok=True); old.write_freeze(out/'SYNTHETIC_FREEZE.json',record)
    else:
        assert FREEZE_PATH.exists() and json.loads(FREEZE_PATH.read_text()) == record,'Freeze changed or missing'
    device = torch.device(args.smoke_device if fake else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cfg = dict(spec=spec,synthetic=fake,freeze_hash=record['freeze_hash'],source_sha256=record['source_sha256'],
               split=record['expected_split'],execution=m.execution_settings(),checkpoints=record['checkpoints'],
               python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
               torchvision=__import__('torchvision').__version__,cuda=torch.version.cuda,device=str(device),
               gpu='cpu' if device.type=='cpu' else torch.cuda.get_device_name(device))
    cfg['fingerprint'] = v.digest(cfg); out.mkdir(exist_ok=True)
    cp = out/'configuration.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == cfg,'Changed environment/source/config; do not overwrite'
    else:
        v.atomic_json(cp,cfg)
    from review_stage11A import audit
    if (out/'COMPLETE.json').exists():
        audit(out,write=False); print('SKIP matching completed feasibility diagnostic'); return
    attempt = d.next_attempt(out)
    v.atomic_json(attempt/'preflight.json',v.idle_preflight() if device.type=='cuda' else [])
    t = time.perf_counter(); data = v.load_data('development',device,fake=fake)
    assert data[-1] == cfg['split']
    v.atomic_json(attempt/'data_loading.json',dict(seconds=time.perf_counter()-t))
    eq,live,live_summary,measurements,cold,telemetry,manifests,states = [],[],[],[],[],[],[],[]
    for seed in spec['source_seeds']:
        models = inputs(record,seed,data,device,fake)
        models['grown'],growth_seconds = d.timed(device,lambda:g.expand(models['small'],seed,fake))
        initial = {p:{n:t.detach().clone() for n,t in model.state_dict().items()} for p,model in models.items()}
        states.append(dict(seed=seed,growth_seconds=growth_seconds,
                           parameter_counts={p:sum(x.numel() for x in model.parameters()) for p,model in models.items()},
                           initial_state_sha256={p:old.state_hash(state) for p,state in initial.items()}))
        eq.extend(equivalence(models,seed,data,spec))
        gen = torch.Generator().manual_seed(seed+220001)
        ix = [torch.randperm(len(data[1]),generator=gen)[:spec['batch_size']].tolist() for _ in range(spec['probe_batches'])]
        manifests.append(dict(seed=seed,indices=ix))
        a,b = liveness(models,initial,seed,data,spec,ix,attempt); live.extend(a); live_summary.extend(b)
        a,b,c = benchmark(models,initial,seed,data,spec,ix); measurements.extend(a); cold.extend(b); telemetry.extend(c)
        print(f'GROWTH FEASIBILITY seed={seed}: preservation, disposable updates and operation costs recorded',flush=True)
        del models,initial
    for name,rows in [('equivalence.csv',eq),('liveness.csv',live),('measurements.csv',measurements),('cold_steps.csv',cold)]:
        v.write_csv(attempt/name,rows)
    if telemetry:
        v.write_csv(attempt/'telemetry.csv',telemetry)
    v.atomic_json(attempt/'liveness_summary.json',live_summary)
    v.atomic_json(attempt/'input_states.json',states)
    v.atomic_json(attempt/'probe_indices.json',manifests)
    # Audit incomplete attempt before sealing; completed audit is strictly read-only.
    audit(out,write=True,attempt=attempt)
    paths = [cp]+[q for q in attempt.iterdir() if q.is_file()]
    v.atomic_json(out/'COMPLETE.json',dict(status='complete',fingerprint=cfg['fingerprint'],synthetic=fake,
                                        attempt=str(attempt),sha256={str(q):d.sha(q) for q in paths}))
    print('COMPLETE. Send',out,'for review. No efficacy pilot has been launched.')


if __name__ == '__main__':
    main()
