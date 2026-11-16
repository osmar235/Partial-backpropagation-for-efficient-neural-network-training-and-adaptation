"""Matched local/full trajectories with restored classifier and auxiliary Adam."""
import argparse
import copy
import json
import math
import re
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import stage13b_reference as b
import stage13_diagnostic as a
import stage13_local as l
p, v, d, old = b.p, b.v, b.d, b.old
ROOT = v.ROOT
HERE = ROOT / 'protocol/stage13c'
SPEC = HERE / 'trajectory_spec_v1.json'
FREEZE = HERE / 'FROZEN_TRAJECTORY_v1.json'
REAL = ROOT / 'results/stage13C_local_trajectory_v1'


def sources():
    names = ['scripts/stage13c_trajectory.py', 'scripts/review_stage13C.py', 'scripts/test_stage13c_trajectory.py',
        'scripts/run_stage13C_local_trajectory.cmd', 'protocol/stage13c/trajectory_spec_v1.json',
        'protocol/stage13c/LOCAL_TRAJECTORY_v1.md']
    return {**b.sources(), **a.sources(), **{str(ROOT/n): p.sha(ROOT/n) for n in names}}


def freeze_record(spec, recipe, fake):
    record = b.freeze_record(spec, recipe, fake)
    record.pop('freeze_hash')
    record['source_sha256'] = sources(); record['auxiliary_sources'] = {}
    paths = [ROOT/'results/stage13B_review_v1/REPORT.md', ROOT/'results/stage13B_review_v1/independent_audit.json',
        ROOT/'protocol/stage13b/POST_REFERENCE_DECISION_v1.md']
    if not fake:
        folder = ROOT/'results/stage13A_local_diagnostic_v1'
        cfg = json.loads((folder/'configuration.json').read_text()); d.validate_complete(folder, cfg['fingerprint'])
        raw = json.loads((folder/'RAW_COMPLETE.json').read_text()); b.verify(raw['sha256'])
        attempt = Path(raw['attempt'])
        for seed in spec['seeds']:
            cp, stream = attempt/f'calibrated_{seed}.pt', attempt/f'streams_{seed}.json'
            record['auxiliary_sources'][str(seed)] = dict(path=str(cp), sha256=p.sha(cp), stream=str(stream))
            paths += [cp, stream]
        paths += [folder/'configuration.json', folder/'COMPLETE.json', folder/'RAW_COMPLETE.json']
    record['evidence_sha256'].update({str(path): p.sha(path) for path in paths})
    record['freeze_hash'] = v.digest(record)
    return record


def tensors_cpu(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:tensors_cpu(x) for k,x in value.items()}
    if isinstance(value, list): return [tensors_cpu(x) for x in value]
    return copy.deepcopy(value)


def equal_state(x, y):
    if torch.is_tensor(x): return torch.equal(x.cpu(), y.cpu())
    if isinstance(x, dict): return x.keys() == y.keys() and all(equal_state(x[k], y[k]) for k in x)
    if isinstance(x, list): return len(x) == len(y) and all(equal_state(u,w) for u,w in zip(x,y))
    return x == y


def restore(seed, cp, calibrated, recipe, fake, device, arm):
    net, opt, bg, ag = b.restore(seed, cp, recipe, fake, device, 'continuation')
    model = l.LocalModel(net, seed, fake).to(device)
    model.load_state_dict(calibrated['model_state'], strict=True)
    assert old.state_hash(model.net.state_dict()) == old.state_hash(cp['model_state'])
    if arm in ('local', 'full_aux'):
        params = list(model.aux0.parameters()) + list(model.aux1.parameters())
        auxiliary = torch.optim.AdamW(params, lr=.001, weight_decay=recipe['weight_decay'])
        auxiliary.load_state_dict(copy.deepcopy(calibrated['aux_optimizer_state']))
        opt.add_param_group(dict(params=params, lr=recipe['head_lr'], role='aux'))
        for param in params:
            opt.state[param] = copy.deepcopy(auxiliary.state[param])
        del auxiliary
    assert arm in ('full','full_aux','local','suffix')
    return model, opt, bg, ag


def rates(opt, index, recipe):
    factor = p.rate_factor(index, recipe)
    for group in opt.param_groups:
        group['lr'] = factor * recipe['head_lr' if group['role'] in ('head','aux') else 'backbone_lr']
    return factor*recipe['backbone_lr'], factor*recipe['head_lr']


def calibrate(seed, cp, data, spec, recipe, record, fake, device, path):
    x,y,_,_,_ = data
    net = p.make_model(seed, fake).to(device); net.load_state_dict(cp['model_state'])
    if fake:
        g = torch.Generator().manual_seed(seed+314001)
        batches = torch.randint(len(y), (spec['calibration_steps'], recipe['batch_size']), generator=g).tolist()
    else:
        batches = json.loads(Path(record['auxiliary_sources'][str(seed)]['stream']).read_text())['calibration_batches']
    assert len(batches) == spec['calibration_steps']
    v.sync(device); started = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    model = l.LocalModel(net, seed, fake).to(device)
    opt = torch.optim.AdamW(list(model.aux0.parameters())+list(model.aux1.parameters()), lr=spec['calibration_lr'], weight_decay=recipe['weight_decay'])
    losses = []
    for i, batch in enumerate(batches):
        ix = torch.tensor(batch, device=device)
        xx = p.transform(x[ix], recipe['image_size'], torch.Generator().manual_seed(seed+315001+i))
        opt.zero_grad(set_to_none=True)
        with torch.no_grad(): z0 = model.block0(xx); z1 = model.block1(z0)
        loss0 = F.cross_entropy(model.auxiliary(z0,0), y[ix]); loss1 = F.cross_entropy(model.auxiliary(z1,1), y[ix])
        (loss0+loss1).backward(); opt.step()
        losses.append(dict(step=i+1, loss0=float(loss0.detach()), loss1=float(loss1.detach())))
    model.zero_grad(set_to_none=True)
    calibrated = dict(model_state=tensors_cpu(model.state_dict()), aux_optimizer_state=tensors_cpu(opt.state_dict()), batches=batches)
    old.checkpoint(path, calibrated); v.sync(device)
    seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    assert old.state_hash(model.net.state_dict()) == old.state_hash(cp['model_state'])
    if not fake:
        original = torch.load(record['auxiliary_sources'][str(seed)]['path'], map_location='cpu', weights_only=True)
        assert old.state_hash(calibrated['model_state']) == old.state_hash(original['model_state']), 'Calibration weights did not reproduce'
        assert equal_state(calibrated['aux_optimizer_state'], original['aux_optimizer_state']), 'Calibration moments did not reproduce'
    return calibrated, losses, dict(seconds=seconds, peak_allocated=peak, reproduced=not fake)


def run(out, cfg, record, device):
    spec, recipe, fake = cfg['spec'], cfg['recipe'], cfg['synthetic']
    attempt = d.next_attempt(out)
    v.atomic_json(attempt/'preflight.json', [] if fake else v.idle_preflight())
    data = p.load_data(device, fake); x,y,ev,ey,split = data
    assert split == cfg['split']
    for seed, order in zip(spec['seeds'],spec['orders']):
        if fake:
            cp, z0, z1, trace = b.fake_source(seed,data,recipe,device)
            cp = tensors_cpu(cp); old.checkpoint(attempt/f'source_{seed}.pt',cp)
            np.savez(attempt/f'golden_{seed}.npz',initial=z0,final=z1,labels=ey.cpu().numpy())
        else:
            entry = record['checkpoints'][str(seed)]
            cp = torch.load(entry['path'],map_location='cpu',weights_only=True)
            assert cp['fingerprint']==entry['fingerprint'] and cp['root_fingerprint']==entry['root_fingerprint']
            trace = b.original_trace(entry,recipe)
        assert cp['updates']==2*recipe['batches_per_epoch'] and cp['seed']==seed
        v.atomic_json(attempt/f'expected_trace_{seed}.json',trace)
        calibrated, losses, calibration = calibrate(seed,cp,data,spec,recipe,record,fake,device,attempt/f'calibrated_{seed}.pt')
        v.write_csv(attempt/f'calibration_losses_{seed}.csv',losses); v.atomic_json(attempt/f'calibration_{seed}.json',calibration)
        for arm in order:
            model,opt,_,_ = restore(seed,cp,calibrated,recipe,fake,device,arm)
            xx = p.transform(x[:recipe['batch_size']],recipe['image_size'])
            for _ in range(spec['warmup_steps']): l.step(model,opt,xx,y[:len(xx)],arm)
            del model,opt
            model,opt,bg,ag = restore(seed,cp,calibrated,recipe,fake,device,arm)
            start_ages = b.ages(model,opt); initial_hash = old.state_hash(model.state_dict())
            initial = calibrated['model_state']; arm_start=time.perf_counter()
            scores = dict(labels=ey.cpu().numpy(),step_0=b.predict(model.net,ev,recipe))
            rows,batches = [],[]
            for step,ix in enumerate(v.frozen6.epoch_batches(len(y),recipe['batch_size'],bg),1):
                batches.append(ix.tolist()); model.zero_grad(set_to_none=True); v.sync(device)
                segment = time.perf_counter()
                idx=ix.to(device); xx=p.transform(x[idx],recipe['image_size'],ag)
                br,hr=rates(opt,cp['updates']+step-1,recipe); v.sync(device)
                baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else 0
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                began=time.perf_counter(); values=l.step(model,opt,xx,y[idx],arm); v.sync(device)
                op_seconds=time.perf_counter()-began; segment_seconds=time.perf_counter()-segment
                peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
                reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0
                for name,t in model.named_parameters():
                    assert (t.grad is not None and bool(torch.isfinite(t.grad).all())) if l.active(name,arm) else t.grad is None, name
                rows.append(dict(step=step,global_step=cp['updates']+step,batch_examples=len(ix),
                    batch_sha256=old.state_hash({'indices':ix}),final_loss=values[-1],objective_sum=sum(values),
                    backbone_lr=br,head_aux_lr=hr,operation_seconds=op_seconds,training_seconds=segment_seconds,
                    baseline_allocated=baseline,peak_allocated=peak,incremental_allocated=peak-baseline,peak_reserved=reserved))
                if step in spec['score_steps']:
                    scores[f'step_{step}']=b.predict(model.net,ev,recipe)
                    print('TRAJECTORY',seed,arm,f'{step}/{recipe["batches_per_epoch"]}',f'accuracy={100*np.mean(scores[f"step_{step}"].argmax(1)==scores["labels"]):.2f}%',flush=True)
            model.zero_grad(set_to_none=True)
            for name,t in model.named_buffers(): assert torch.equal(t.cpu(),initial[name]),name
            for name,t in model.named_parameters():
                assert torch.isfinite(t).all()
                if not l.active(name,arm): assert torch.equal(t.detach().cpu(),initial[name]) and t not in opt.state,name
            final_ages=b.ages(model,opt)
            assert all(final_ages[n]==start_ages[n]+(len(rows) if l.active(n,arm) else 0) for n in start_ages)
            stem=f'{seed}_{arm}'
            old.checkpoint(attempt/f'final_{stem}.pt',dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),
                batch_rng=bg.get_state(),augmentation_rng=ag.get_state(),seed=seed,arm=arm,updates=len(rows),fingerprint=cfg['fingerprint']))
            np.savez(attempt/f'scores_{stem}.npz',**scores)
            v.write_csv(attempt/f'steps_{stem}.csv',rows); v.atomic_json(attempt/f'batches_{stem}.json',batches)
            v.atomic_json(attempt/f'state_{stem}.json',dict(initial_hash=initial_hash,final_hash=old.state_hash(model.state_dict()),
                start_ages=start_ages,final_ages=final_ages,arm_elapsed_seconds=time.perf_counter()-arm_start,
                calibration_charge_seconds=calibration['seconds'] if arm in ('full_aux','local') else 0,
                all_gradients_verified=True,telemetry=[] if fake else v.telemetry()))
            del model,opt
        del cp,calibrated
    b.verify(record['source_sha256']); b.verify(record['evidence_sha256'])
    v.atomic_json(out/'RAW_COMPLETE.json',dict(fingerprint=cfg['fingerprint'],attempt=str(attempt),
        sha256={str(f):p.sha(f) for f in attempt.iterdir() if f.is_file()}))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--freeze-only',action='store_true'); ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--smoke-tag',default='final_cpu'); ap.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu'); args=ap.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag)
    p.execution.configure(); spec=json.loads(SPEC.read_text()); recipe=json.loads(b.prior.SPEC.read_text())
    if args.smoke:
        spec.update(seeds=[9981,9982],score_steps=[0,1,3,6],warmup_steps=1,calibration_steps=4)
        recipe.update(image_size=32,batch_size=3,eval_batch_size=4,batches_per_epoch=6)
    record=freeze_record(spec,recipe,args.smoke)
    if args.freeze_only:
        assert not args.smoke and not REAL.exists(); old.write_freeze(FREEZE,record); print('FROZEN',record['freeze_hash']); return
    device=torch.device(args.smoke_device if args.smoke else 'cuda')
    assert device.type=='cpu' or torch.cuda.is_available()
    out=ROOT/f'results/stage13C_sanity_trajectory_v1_{args.smoke_tag}' if args.smoke else REAL
    if not args.smoke: assert FREEZE.exists() and json.loads(FREEZE.read_text())==record,'Frozen inputs changed'
    out.mkdir(parents=True,exist_ok=True)
    if args.smoke: old.write_freeze(out/'SYNTHETIC_FREEZE.json',record)
    cfg=dict(spec=spec,recipe=recipe,synthetic=args.smoke,freeze_hash=record['freeze_hash'],split=record['split'],device=str(device),
        hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu')
    cfg['fingerprint']=v.digest(cfg); old.write_freeze(out/'configuration.json',cfg)
    from review_stage13C import audit
    if d.validate_complete(out,cfg['fingerprint']): print(json.dumps(audit(out,False),indent=2)); print('Verified read-only'); return
    if not (out/'RAW_COMPLETE.json').exists(): run(out,cfg,record,device)
    result=audit(out,True)
    files=[out/n for n in ['configuration.json','RAW_COMPLETE.json','audit.json','curves.csv','summary.csv','REPORT.md']]
    if args.smoke: files.append(out/'SYNTHETIC_FREEZE.json')
    v.atomic_json(out/'COMPLETE.json',dict(status='complete',fingerprint=cfg['fingerprint'],sha256={str(f):p.sha(f) for f in files}))
    print(json.dumps(result,indent=2)); print('COMPLETE. Review before any further experiment.')


if __name__=='__main__': main()
