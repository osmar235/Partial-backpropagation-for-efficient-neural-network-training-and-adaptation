"""Cost-only comparison at saved states: efficient predecided vs boundary decisions."""
from __future__ import annotations
import argparse, copy, csv, json, math, platform, re, statistics as st
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import stage8_vit_v2 as v
import stage9_boundary_diagnostic as d
from stage9_split_backward_proof import split_forward

ROOT=v.ROOT
SPEC=ROOT/'protocol/stage9/deployment_cost_spec_v1.json'
PATHS=['ordinary_full','predecided_freeze','boundary_full','boundary_freeze']

def predecided_forward(model,x,cut):
    # A blind or augmentation-only schedule knows the choice before forward.
    # Prefix graph construction is unnecessary on its partial updates.
    with torch.no_grad():
        tokens=model._process_input(x)
        tokens=torch.cat([model.class_token.expand(len(x),-1,-1),tokens],dim=1)
        h=model.encoder.dropout(tokens+model.encoder.pos_embedding)
        for layer in list(model.encoder.layers)[:cut]: h=layer(h)
    for layer in list(model.encoder.layers)[cut:]: h=layer(h)
    return model.heads(model.encoder.ln(h)[:,0])

def prepare(data,seed,epoch,spec,device):
    x,y=data[:2]; n=spec['probe_batches']; b=spec['batch_size']
    g=torch.Generator().manual_seed(seed+epoch*1000+141001)
    ix=torch.randperm(len(y),generator=g)[:n*b]
    assert len(ix)==n*b
    ag=torch.Generator().manual_seed(seed+epoch*1000+142001)
    mg=np.random.default_rng(seed+epoch*1000+143001)
    pg=torch.Generator().manual_seed(seed+epoch*1000+144001)
    probes=[]
    for i in range(n):
        indices=ix[i*b:(i+1)*b].to(device)
        def augment():
            xx=v.frozen6.augment_and_normalize(x[indices],ag)
            return v.mix_batch(xx,y[indices],mg,pg,.2,10)
        (xx,target,lam),seconds=d.timed(device,augment)
        probes.append(dict(x=xx,target=target,lam=lam,augmentation_seconds=seconds,
            dropout_seed=seed+epoch*10000+145001+i,indices=indices.cpu().tolist()))
    return probes

def bench_state(state,seed,epoch,data,spec,device,fake):
    cut=spec['cut']; model=v.make_model(tiny=fake).to(device); model.train()
    weights={n:t.to(device) for n,t in state['model_state'].items()}
    opt_state=copy.deepcopy(state['optimizer_state'])
    batches=math.ceil(len(data[1])/(4 if fake else 256))
    lr=v.learning_rate(state['updates'],(10 if fake else 300)*batches,batches,.001,v.SPEC)
    for group in opt_state['param_groups']: group['lr']=lr
    for value in opt_state['state'].values():
        for key,t in value.items():
            if key!='step' and isinstance(t,torch.Tensor): value[key]=t.to(device)
    opt=d.optimizer(model,lr); prefix=set(d.named_partition(model,cut))
    probes=prepare(data,seed,epoch,spec,device); telemetry=[]; rows=[]
    initial_steps={n:int(s['step']) for (n,_),s in zip(model.named_parameters(),opt_state['state'].values())}

    def one(probe,path,check=False):
        d.restore(model,opt,weights,opt_state); v.seed_all(probe['dropout_seed'])
        def operation():
            opt.zero_grad(set_to_none=True)
            if path.startswith('boundary'):
                logits,h,bridge=split_forward(model,probe['x'],cut)
            elif path=='predecided_freeze':
                logits=predecided_forward(model,probe['x'],cut)
            else: logits=model(probe['x'])
            loss=F.cross_entropy(logits,probe['target'],label_smoothing=.1)
            loss_value=loss.item()  # Shared with the actual training recipe.
            loss.backward()
            signal=None
            if path.startswith('boundary'):
                # Only RMS is needed; no entropy/activation feature bundle.
                signal=bridge.grad.detach().square().mean().sqrt().item()
                # Decision is FORCED for cost isolation, not a trained gate.
                if path=='boundary_full': h.backward(bridge.grad)
            opt.step()
            return loss_value,signal
        (loss,signal),seconds=d.timed(device,operation)
        assert math.isfinite(loss) and seconds>0
        if signal is not None: assert math.isfinite(signal)
        partial=path in ['predecided_freeze','boundary_freeze']
        if check:
            for name,param in model.named_parameters():
                assert int(opt.state[param]['step'])==initial_steps[name]+int(not (partial and name in prefix))
                if partial and name in prefix:
                    assert param.grad is None
                    torch.testing.assert_close(param,weights[name],rtol=0,atol=0)
            snapshot=dict(loss=loss,parameters={n:p.detach().clone() for n,p in model.named_parameters()},
                gradients={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()})
        else: snapshot=None
        return dict(path=path,operation_seconds=seconds,augmentation_seconds=probe['augmentation_seconds'],
            total_seconds=seconds+probe['augmentation_seconds'],loss=loss,boundary_rms='' if signal is None else signal),snapshot

    for i in range(spec['warmup_batches']):
        for path in PATHS: one(probes[i%len(probes)],path)
    # Validate paired full and partial behavior outside deployment timers. No
    # parameter/gradient clones or integrity reductions enter measured samples.
    snapshots={path:one(probes[0],path,check=True)[1] for path in PATHS}
    for a,b in [('ordinary_full','boundary_full'),('predecided_freeze','boundary_freeze')]:
        sa,sb=snapshots[a],snapshots[b]
        assert math.isclose(sa['loss'],sb['loss'],rel_tol=1e-5,abs_tol=1e-6)
        for name in sa['parameters']:
            torch.testing.assert_close(sa['parameters'][name],sb['parameters'][name],rtol=1e-4,atol=1e-6)
            ga,gb=sa['gradients'][name],sb['gradients'][name]
            if ga is None or gb is None: assert ga is None and gb is None
            else: torch.testing.assert_close(ga,gb,rtol=1e-4,atol=1e-6)
    assert any(float(g.norm())>0 for n,g in snapshots['ordinary_full']['gradients'].items() if n in prefix)
    del snapshots
    for i,probe in enumerate(probes):
        order=PATHS[i%4:]+PATHS[:i%4]
        if (i//4)%2: order=list(reversed(order))
        if not fake and i%8==0: telemetry.append(dict(seed=seed,epoch=epoch,probe=i,sample=json.dumps(v.telemetry())))
        for position,path in enumerate(order):
            r,_=one(probe,path)
            rows.append(dict(seed=seed,epoch=epoch,probe=i,order_position=position,cut=cut,batch_size=spec['batch_size'],mixup_lambda=probe['lam'],**r))
    manifest=[dict(probe=i,indices=p['indices'],dropout_seed=p['dropout_seed']) for i,p in enumerate(probes)]
    return rows,manifest,telemetry

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-tag',default='final');args=p.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag)
    fake=args.smoke;spec=json.loads(SPEC.read_text());device=torch.device('cpu' if fake else 'cuda')
    if fake:
        torch.set_num_threads(1)
        spec.update(development_seeds=[9901],checkpoint_epochs=[1],batch_size=2,probe_batches=4,warmup_batches=1,cut=1)
    if not fake and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    v.configure()
    sources={**d.source_hashes(),str(Path(__file__).resolve()):d.sha(__file__),str(SPEC.resolve()):d.sha(SPEC),
        str((ROOT/'protocol/stage9/DEPLOYMENT_COST_v1.md').resolve()):d.sha(ROOT/'protocol/stage9/DEPLOYMENT_COST_v1.md')}
    raw=ROOT/'results/stage9C_boundary_diagnostic_v1'; source_checkpoints={}
    if not fake:
        original=json.loads((raw/'configuration.json').read_text())
        assert not original['fake'] and original['spec']['development_seeds']==spec['development_seeds']
        for path,h in original['source_sha256'].items(): assert d.sha(path)==h
        assert d.validate_complete(raw,original['fingerprint'])['status']=='complete'
        for seed in spec['development_seeds']:
            marker=d.validate_complete(raw/f'seed_{seed}/training',original['fingerprint']); assert marker
            for epoch in spec['checkpoint_epochs']:
                filename=marker['checkpoints'][str(epoch)];source_checkpoints[f'{seed}:{epoch}']=dict(path=filename,sha256=d.sha(filename))
        preflight=v.idle_preflight()
    else: preflight=None
    data=v.load_data('development',device,fake=fake)
    config=dict(spec=spec,synthetic=fake,source_sha256=sources,checkpoints=source_checkpoints,split=data[-1],
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        gpu='cpu' if fake else torch.cuda.get_device_name(device))
    fingerprint=v.digest(config);config['fingerprint']=fingerprint
    out=ROOT/'results'/('stage9_sanity_deployment_cost_v1_'+args.smoke_tag if fake else 'stage9D_deployment_cost_v1')
    out.mkdir(exist_ok=True)
    cfg=out/'configuration.json'
    if cfg.exists(): assert json.loads(cfg.read_text())==config,'Changed configuration: do not overwrite completed evidence'
    else:v.atomic_json(cfg,config)
    done=d.validate_complete(out,fingerprint)
    if done: print('SKIP matching completed cost-only benchmark');return
    attempt=d.next_attempt(out)
    if preflight is not None:v.atomic_json(attempt/'preflight.json',preflight)
    rows=[];manifests=[];telemetry=[]
    for seed in spec['development_seeds']:
        for epoch in spec['checkpoint_epochs']:
            if fake:
                v.seed_all(seed);m=v.make_model(tiny=True);opt=d.optimizer(m,.001)
                # Initialize nonzero prefix gradients and realistic Adam state.
                for _ in range(3):
                    opt.zero_grad(set_to_none=True);F.cross_entropy(m(data[0][:4]),data[1][:4],label_smoothing=.1).backward();opt.step()
                state=dict(model_state=m.state_dict(),optimizer_state=opt.state_dict(),updates=3)
            else:
                state=torch.load(source_checkpoints[f'{seed}:{epoch}']['path'],map_location='cpu',weights_only=True)
                assert state['seed']==seed and state['epoch']==epoch and state['updates']==epoch*176
                assert all(int(s['step'])==state['updates'] for s in state['optimizer_state']['state'].values())
            a,b,c=bench_state(state,seed,epoch,data,spec,device,fake)
            rows.extend(a);manifests.append(dict(seed=seed,epoch=epoch,probes=b));telemetry.extend(c)
            print(f'COST ONLY seed={seed} epoch={epoch} complete; no trajectory trained',flush=True)
    expected=len(spec['development_seeds'])*len(spec['checkpoint_epochs'])*spec['probe_batches']*4
    assert len(rows)==expected
    v.write_csv(attempt/'measurements.csv',rows);v.atomic_json(attempt/'probe_indices.json',manifests)
    if telemetry:v.write_csv(attempt/'telemetry.csv',telemetry)
    summary=[]
    for seed in spec['development_seeds']:
        for epoch in spec['checkpoint_epochs']:
            means={path:st.mean(r['total_seconds'] for r in rows if r['seed']==seed and r['epoch']==epoch and r['path']==path) for path in PATHS}
            for q in spec['assumed_omission_fractions']:
                pre=(1-q)*means['ordinary_full']+q*means['predecided_freeze']
                boundary=(1-q)*means['boundary_full']+q*means['boundary_freeze']
                summary.append(dict(seed=seed,epoch=epoch,assumed_omission_fraction=q,
                    predecided_projected_ratio=pre/means['ordinary_full'],boundary_projected_ratio=boundary/means['ordinary_full'],
                    boundary_over_predecided_ratio=boundary/pre,**{k+'_seconds':value for k,value in means.items()}))
    v.write_csv(attempt/'cost_summary.csv',summary)
    report=['# Stage9D deployment operation costs','',
        'COST ONLY: forced decisions at reset shared states, not a controller or training trajectory.',
        'Batch256 in real mode. Hypothetical mixtures do not establish accuracy or time-to-quality.',
        'Predecided partial forward avoids prefix graph construction; boundary paths pay for RMS and synchronization.',
        'Predecided path approximates the efficient known-decision kernel; calibration, scheduling, logging and data loading still need end-to-end measurement.',
        'Inspect cost_summary.csv for incremental gate cost versus efficient predecided scheduling.',
        'All paired-gradient/parameter and freeze-counter checks passed before measured samples.',
        'No test dataset, no efficacy claim, no automatic pilot launch. Send this entire directory for review.','']
    (attempt/'REPORT.md').write_text('\n'.join(report),encoding='utf8')
    files=[p for p in attempt.iterdir() if p.is_file()]
    v.atomic_json(out/'COMPLETE.json',dict(status='complete',synthetic=fake,fingerprint=fingerprint,rows=expected,attempt=str(attempt),sha256={str(p):d.sha(p) for p in files}))
    print('COMPLETE:',out)

if __name__=='__main__':main()
