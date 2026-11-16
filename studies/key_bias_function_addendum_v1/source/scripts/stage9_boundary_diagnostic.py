"""Shared-checkpoint ViT cut-location, prefix-update utility and cost diagnostic."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,platform,re,statistics as st,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import stage8_vit_v2 as v
from stage9_split_backward_proof import split_forward

ROOT=v.ROOT
SPEC_PATH=ROOT/'protocol/stage9/boundary_diagnostic_spec_v1.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def records(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def csv_atomic(path,rows):
    tmp=Path(path).with_suffix('.csv.tmp');v.write_csv(tmp,rows);tmp.replace(path)
def source_hashes():
    paths=[Path(__file__),SPEC_PATH,Path(__file__).with_name('stage9_split_backward_proof.py'),
           ROOT/'protocol/stage9/BOUNDARY_DIAGNOSTIC_v1.md']
    return {**v.source_hashes(),**{str(p.resolve()):sha(p) for p in paths}}
def named_partition(model,cut):
    assert 0<=cut<len(model.encoder.layers)
    suffix=tuple(f'encoder.layers.encoder_layer_{i}.' for i in range(cut,len(model.encoder.layers)))+('encoder.ln.','heads.')
    prefix_names=[n for n,p in model.named_parameters() if not n.startswith(suffix)]
    return prefix_names
def restore(model,optimizer,weights,state):
    model.load_state_dict(weights);model.zero_grad(set_to_none=True)
    optimizer.load_state_dict(copy.deepcopy(state))
    for group in optimizer.param_groups:
        if not group.get('capturable',False) and not group.get('fused',False):
            for p in group['params']:
                if 'step' in optimizer.state[p]:assert optimizer.state[p]['step'].device.type=='cpu'
def timed(device,fn):
    v.sync(device);start=time.perf_counter();result=fn();v.sync(device)
    return result,time.perf_counter()-start
def optimizer(model,lr):return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=v.SPEC['weight_decay'])
def make_model(fake):return v.make_model(tiny=fake)
def next_attempt(parent):
    parent.mkdir(parents=True,exist_ok=True)
    number=1
    while (parent/f'attempt_{number:03d}').exists():number+=1
    path=parent/f'attempt_{number:03d}';path.mkdir();return path
def checkpoint_save(path,value):
    tmp=path.with_suffix('.pt.tmp');torch.save(value,tmp);tmp.replace(path)
def validate_complete(path,fingerprint):
    marker=Path(path)/'COMPLETE.json'
    if not marker.exists():return None
    r=json.loads(marker.read_text());assert r['fingerprint']==fingerprint
    for p,h in r['sha256'].items():assert sha(p)==h,('Modified completed input',p)
    return r


def train_seed(folder,seed,data,spec,device,fake,fingerprint):
    parent=folder/'training';done=validate_complete(parent,fingerprint)
    if done:print('SKIP completed training',seed,flush=True);return done
    attempt=next_attempt(parent);x,y,ev,ey,_=data
    v.configure();v.seed_all(seed)
    model=make_model(fake).to(device);opt=optimizer(model,spec['learning_rate'])
    # Collection uses independent streams; diagnostics run only AFTER all checkpoints exist.
    v.seed_all(seed+95001)
    bg=torch.Generator().manual_seed(seed+91001);ag=torch.Generator().manual_seed(seed+92001)
    mg=np.random.default_rng(seed+98001);pg=torch.Generator().manual_seed(seed+99001)
    batch=spec['training_batch_size'];batches=math.ceil(len(y)/batch)
    schedule_total=spec['lr_schedule_horizon_epochs']*batches
    checkpoints={};rows=[];telemetry=[];idx=0
    start=time.perf_counter()
    for epoch in range(1,spec['training_epochs']+1):
        if not fake:telemetry.append(dict(epoch=epoch,sample=json.dumps(v.telemetry())))
        v.sync(device);t0=time.perf_counter();loss_total=count=0
        for ix in v.frozen6.epoch_batches(len(y),batch,bg):
            ix=ix.to(device);xx=v.frozen6.augment_and_normalize(x[ix],ag)
            xx,target,_=v.mix_batch(xx,y[ix],mg,pg,.2,10)
            lr=v.learning_rate(idx,schedule_total,batches,spec['learning_rate'],v.SPEC)
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(xx),target,label_smoothing=.1)
            value=loss.item();assert math.isfinite(value)
            loss.backward();opt.step();idx+=1;loss_total+=value*len(ix);count+=len(ix)
        v.sync(device);seconds=time.perf_counter()-t0
        row=dict(epoch=epoch,updates=idx,mean_training_loss=loss_total/count,training_seconds=seconds,lr=lr)
        if epoch in spec['checkpoint_epochs']:
            val_loss,acc=v.evaluate(model,ev,ey,128 if fake else 512)
            path=attempt/f'epoch_{epoch:03d}.pt'
            checkpoint_save(path,dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),
                seed=seed,epoch=epoch,updates=idx,lr=lr,fingerprint=fingerprint))
            checkpoints[str(epoch)]=str(path)
            row.update(validation_loss=val_loss,validation_accuracy=acc)
            print(f'BASELINE seed={seed} epoch={epoch}/{spec["training_epochs"]} validation={100*acc:.2f}% checkpoint saved',flush=True)
        else:row.update(validation_loss='',validation_accuracy='')
        rows.append(row)
    csv_atomic(attempt/'epochs.csv',rows)
    if telemetry:csv_atomic(attempt/'telemetry.csv',telemetry)
    files=[*map(Path,checkpoints.values()),attempt/'epochs.csv']
    if telemetry:files.append(attempt/'telemetry.csv')
    result=dict(fingerprint=fingerprint,checkpoints=checkpoints,updates=idx,
        elapsed_seconds=time.perf_counter()-start,attempt=str(attempt),sha256={str(p):sha(p) for p in files})
    parent.mkdir(parents=True,exist_ok=True);v.atomic_json(parent/'COMPLETE.json',result)
    del model,opt
    return result


def role_pools(y):
    """Class-stratified disjoint calibration/scoring pools, independent of method."""
    g=torch.Generator().manual_seed(129001);pools=[[],[]];cpu=y.detach().cpu()
    for c in sorted(cpu.unique().tolist()):
        ix=torch.where(cpu==c)[0];ix=ix[torch.randperm(len(ix),generator=g)]
        n=len(ix)//2;pools[0].extend(ix[:n].tolist());pools[1].extend(ix[n:].tolist())
    return [torch.tensor(p,dtype=torch.long) for p in pools]


def prepare_probes(data,seed,epoch,spec,device,fake):
    x,y,ev,ey,_=data
    train_pools=role_pools(y);eval_pools=role_pools(ey)
    if fake: # Balanced positions suffice for synthetic diagnostics with very small random classes.
        train_pools=[torch.arange(0,len(y)//2),torch.arange(len(y)//2,len(y))]
        eval_pools=[torch.arange(0,len(ey)//2),torch.arange(len(ey)//2,len(ey))]
    half=spec['probe_batches_per_checkpoint']//2;train_batch=spec['probe_train_batch_size'];eval_batch=spec['probe_validation_batch_size']
    probes=[]
    for role_i,role in enumerate(['calibration','scoring']):
        g=torch.Generator().manual_seed(seed+epoch*1000+120001+role_i)
        tp=train_pools[role_i];vp=eval_pools[role_i]
        assert half*train_batch<=len(tp) and half*eval_batch<=len(vp)
        tr=tp[torch.randperm(len(tp),generator=g)[:half*train_batch]]
        va=vp[torch.randperm(len(vp),generator=g)[:half*eval_batch]]
        ag=torch.Generator().manual_seed(seed+epoch*1000+121001+role_i)
        mg=np.random.default_rng(seed+epoch*1000+122001+role_i)
        pg=torch.Generator().manual_seed(seed+epoch*1000+123001+role_i)
        for j in range(half):
            ti=tr[j*train_batch:(j+1)*train_batch].to(device);vi=va[j*eval_batch:(j+1)*eval_batch].to(device)
            def augment():
                xx=v.frozen6.augment_and_normalize(x[ti],ag)
                return v.mix_batch(xx,y[ti],mg,pg,.2,10)
            (xx,target,lam),seconds=timed(device,augment)
            probes.append(dict(role=role,probe=j,dropout_seed=seed+epoch*10000+role_i*100+j+124001,
                x=xx,targets=target,lam=lam,vx=ev[vi],vy=ey[vi],augmentation_seconds=seconds,
                training_indices=ti.detach().cpu().tolist(),validation_indices=vi.detach().cpu().tolist()))
    # Iterating alternate roles makes timing conditions less confounded with the calibration role.
    return [probes[r*half+j] for j in range(half) for r in range(2)]


@torch.no_grad()
def validation_loss(model,x,y):
    model.eval();loss=F.cross_entropy(model(x),y).item();model.train();return loss


def features(logits_loss,h,bridge,target):
    grad=bridge.grad.detach();activation=h.detach()
    smooth=.9*target+.1/10
    entropy=-(smooth*smooth.log()).sum(1).mean()
    return dict(raw_loss=float(logits_loss),excess_loss=float(logits_loss)-float(entropy.item()),
        boundary_grad_rms=float(grad.square().mean().sqrt().item()),
        activation_grad_rms=float((activation*grad).square().mean().sqrt().item()))


def probe_state(folder,seed,epoch,checkpoint,data,spec,device,fake,fingerprint):
    parent=folder/f'probes/epoch_{epoch:03d}';done=validate_complete(parent,fingerprint)
    if done:assert done['checkpoint_sha256']==sha(checkpoint);print('SKIP completed probes',seed,epoch,flush=True);return
    # AdamW's non-capturable step counters belong on CPU. Mapping the whole
    # checkpoint to CUDA would add per-parameter synchronizations to the benchmark.
    attempt=next_attempt(parent);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['fingerprint']==fingerprint and state['seed']==seed and state['epoch']==epoch
    reference=make_model(fake).to(device);work=make_model(fake).to(device)
    batches=math.ceil(len(data[1])/spec['training_batch_size'])
    assert state['updates']==epoch*batches
    assert all(int(item['step'])==state['updates'] for item in state['optimizer_state']['state'].values())
    probe_lr=v.learning_rate(state['updates'],spec['lr_schedule_horizon_epochs']*batches,
                             batches,spec['learning_rate'],v.SPEC)
    opt_ref=optimizer(reference,probe_lr);opt_work=optimizer(work,probe_lr)
    weights={n:t.to(device) for n,t in state['model_state'].items()}
    opt_state=state['optimizer_state']
    for group in opt_state['param_groups']:group['lr']=probe_lr
    for item in opt_state['state'].values():
        for name,value in item.items():
            if name!='step' and isinstance(value,torch.Tensor):item[name]=value.to(device)
    probes=prepare_probes(data,seed,epoch,spec,device,fake)
    cuts=spec['prefix_blocks'];rows=[];telemetry=[];manifest=[]
    started=time.perf_counter()

    def one(probe,cut_order,check=False):
        restore(reference,opt_ref,weights,opt_state);v.seed_all(probe['dropout_seed']);reference.train()
        def ref_forward():
            opt_ref.zero_grad(set_to_none=True)
            loss=F.cross_entropy(reference(probe['x']),probe['targets'],label_smoothing=.1)
            return loss,loss.item()
        (loss,lv),ref_forward_s=timed(device,ref_forward)
        _,ref_backward_s=timed(device,loss.backward)
        # Clones and restore operations are diagnostic overhead outside deployment timers.
        grads={n:p.grad.detach().clone() for n,p in reference.named_parameters()}
        _,ref_optimizer_s=timed(device,opt_ref.step)
        full_loss=validation_loss(reference,probe['vx'],probe['vy'])
        full_params={n:p.detach().clone() for n,p in reference.named_parameters()}
        ref_s=probe['augmentation_seconds']+ref_forward_s+ref_backward_s+ref_optimizer_s
        measured=[]
        for cut in cut_order:
            prefix_names=named_partition(work,cut);prefix_set=set(prefix_names)
            prefix_count=sum(p.numel() for n,p in work.named_parameters() if n in prefix_set)
            restore(work,opt_work,weights,opt_state);v.seed_all(probe['dropout_seed']);work.train()
            def split_fw():
                opt_work.zero_grad(set_to_none=True)
                logits,h,bridge=split_forward(work,probe['x'],cut)
                loss=F.cross_entropy(logits,probe['targets'],label_smoothing=.1)
                return logits,h,bridge,loss,loss.item()
            (logits,h,bridge,loss,actual_lv),forward_s=timed(device,split_fw)
            _,suffix_s=timed(device,loss.backward)
            assert all(p.grad is None for n,p in work.named_parameters() if n in prefix_set)
            signals,feature_s=timed(device,lambda:features(actual_lv,h,bridge,probe['targets']))
            _,prefix_s=timed(device,lambda:h.backward(bridge.grad))
            if check:
                torch.testing.assert_close(torch.tensor(actual_lv),torch.tensor(lv),rtol=1e-5,atol=1e-6)
                for n,p in work.named_parameters():torch.testing.assert_close(p.grad,grads[n],rtol=1e-4,atol=1e-6)
            _,full_opt_s=timed(device,opt_work.step)
            if check:
                for n,p in work.named_parameters():torch.testing.assert_close(p,full_params[n],rtol=1e-4,atol=1e-6)
            common=probe['augmentation_seconds']+forward_s+suffix_s+feature_s
            for mode in spec['omission_modes']:
                restore(work,opt_work,weights,opt_state)
                for n,p in work.named_parameters():p.grad=grads[n]
                def omit_step():
                    for n,p in work.named_parameters():
                        if n in prefix_set:p.grad=torch.zeros_like(p) if mode=='zero' else None
                    opt_work.step()
                _,omit_opt_s=timed(device,omit_step)
                omit_loss=validation_loss(work,probe['vx'],probe['vy'])
                with torch.no_grad():
                    error=sum((p-full_params[n]).square().sum() for n,p in work.named_parameters() if n in prefix_set)
                    displacement=sum((full_params[n]-weights[n]).square().sum() for n in prefix_names)
                    difference=float(error.sqrt().item());relative=difference/max(float(displacement.sqrt().item()),1e-30)
                measured.append(dict(seed=seed,epoch=epoch,cut=cut,role=probe['role'],probe=probe['probe'],mode=mode,
                    prefix_parameters=prefix_count,model_parameters=sum(p.numel() for p in work.parameters()),
                    probe_learning_rate=probe_lr,
                    cut_order_position=cut_order.index(cut),
                    mixup_lambda=probe['lam'],**signals,validation_full_loss=full_loss,validation_omitted_loss=omit_loss,
                    validation_omit_minus_full=omit_loss-full_loss,prefix_update_error_l2=difference,
                    relative_prefix_update_error=relative,augmentation_seconds=probe['augmentation_seconds'],
                    normal_full_seconds=ref_s,split_forward_seconds=forward_s,suffix_backward_seconds=suffix_s,
                    features_seconds=feature_s,prefix_backward_seconds=prefix_s,full_optimizer_seconds=full_opt_s,
                    omitted_optimizer_seconds=omit_opt_s,split_full_seconds=common+prefix_s+full_opt_s,
                    split_partial_seconds=common+omit_opt_s))
            del h,bridge,loss,logits
        return measured

    for j in range(spec['probe_warmup_batches']):one(probes[j%len(probes)],cuts,check=j==0)
    for i,probe in enumerate(probes):
        order=cuts[i%len(cuts):]+cuts[:i%len(cuts)]
        if (i//len(cuts))%2:order=list(reversed(order))
        if not fake and i%8==0:telemetry.append(dict(probe=i,sample=json.dumps(v.telemetry())))
        rows.extend(one(probe,order,check=i==0))
        manifest.append({k:probe[k] for k in ['role','probe','dropout_seed','training_indices','validation_indices']})
        if i%16==0:print(f'PROBES seed={seed} epoch={epoch} batch={i+1}/{len(probes)} cuts={cuts}',flush=True)
    for row in rows:
        for value in row.values():
            if isinstance(value,float):assert math.isfinite(value)
    csv_atomic(attempt/'measurements.csv',rows);v.atomic_json(attempt/'probe_indices.json',manifest)
    files=[attempt/'measurements.csv',attempt/'probe_indices.json']
    if telemetry:csv_atomic(attempt/'telemetry.csv',telemetry);files.append(attempt/'telemetry.csv')
    result=dict(fingerprint=fingerprint,checkpoint_sha256=sha(checkpoint),rows=len(rows),
        elapsed_seconds=time.perf_counter()-started,attempt=str(attempt),sha256={str(p):sha(p) for p in files})
    parent.mkdir(parents=True,exist_ok=True);v.atomic_json(parent/'COMPLETE.json',result)
    del reference,work,opt_ref,opt_work,probes,state


def ranked(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);ranks=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):ranks[order[k]]=(i+j-1)/2
        i=j
    return ranks
def spearman(a,b):
    a=ranked(a);b=ranked(b);ma=st.mean(a);mb=st.mean(b)
    xy=sum((x-ma)*(y-mb) for x,y in zip(a,b));xx=sum((x-ma)**2 for x in a);yy=sum((y-mb)**2 for y in b)
    return xy/math.sqrt(xx*yy) if xx and yy else ''

def threshold_selection(cal,score,feature,q):
    threshold=float(np.quantile([float(r[feature]) for r in cal],q))
    return threshold,[r for r in score if float(r[feature])<=threshold]


def summarize(out,spec,fingerprint):
    all_rows=[]
    for seed in spec['development_seeds']:
        for epoch in spec['checkpoint_epochs']:
            done=validate_complete(out/f'seed_{seed}/probes/epoch_{epoch:03d}',fingerprint);assert done
            all_rows.extend(records(Path(done['attempt'])/'measurements.csv'))
    expected=len(spec['development_seeds'])*len(spec['checkpoint_epochs'])*len(spec['prefix_blocks'])*spec['probe_batches_per_checkpoint']*len(spec['omission_modes'])
    assert len(all_rows)==expected,(len(all_rows),expected)
    comparisons=[];costs=[]
    for seed in spec['development_seeds']:
        for epoch in spec['checkpoint_epochs']:
            for cut in spec['prefix_blocks']:
                for mode in spec['omission_modes']:
                    group=[r for r in all_rows if int(r['seed'])==seed and int(r['epoch'])==epoch and int(r['cut'])==cut and r['mode']==mode]
                    cal=[r for r in group if r['role']=='calibration'];score=sorted([r for r in group if r['role']=='scoring'],key=lambda r:int(r['probe']))
                    base=st.mean(float(r['normal_full_seconds']) for r in score)
                    for q in spec['calibration_quantiles']:
                        projected=st.mean((1-q)*float(r['split_full_seconds'])+q*float(r['split_partial_seconds']) for r in score)
                        costs.append(dict(seed=seed,epoch=epoch,cut=cut,mode=mode,assumed_partial_fraction=q,
                            prefix_parameters=int(score[0]['prefix_parameters']),model_parameters=int(score[0]['model_parameters']),
                            normal_full_seconds=base,projected_split_seconds=projected,projected_ratio=projected/base,
                            mean_features_seconds=st.mean(float(r['features_seconds']) for r in score)))
                    for feature in spec['features']:
                        values=[float(r[feature]) for r in score]
                        correlation=spearman(values,[float(r['prefix_update_error_l2']) for r in score])
                        for q in spec['calibration_quantiles']:
                            threshold,chosen=threshold_selection(cal,score,feature,q);k=len(chosen);n=len(score)
                            periodic=[score[min(n-1,int((i+.5)*n/k))] for i in range(k)] if k else []
                            result=dict(seed=seed,epoch=epoch,cut=cut,mode=mode,feature=feature,calibration_quantile=q,
                                threshold=threshold,calibration_n=len(cal),scoring_n=n,selected_n=k,
                                realized_eligibility_fraction=k/n,spearman_score_update_error=correlation)
                            for metric in ['prefix_update_error_l2','relative_prefix_update_error','validation_omit_minus_full']:
                                result['selected_'+metric]=st.mean(float(r[metric]) for r in chosen) if chosen else ''
                                result['blind_expected_'+metric]=st.mean(float(r[metric]) for r in score)
                                result['periodic_'+metric]=st.mean(float(r[metric]) for r in periodic) if periodic else ''
                                oracle=sorted(score,key=lambda r:float(r[metric]))[:k]
                                result['oracle_'+metric]=st.mean(float(r[metric]) for r in oracle) if oracle else ''
                            comparisons.append(result)
    csv_atomic(out/'all_measurements.csv',all_rows);csv_atomic(out/'signal_comparisons.csv',comparisons);csv_atomic(out/'projected_costs.csv',costs)
    means=[]
    for cut in spec['prefix_blocks']:
        for mode in spec['omission_modes']:
            rows=[r for r in costs if r['cut']==cut and r['mode']==mode and r['assumed_partial_fraction']==.4]
            if not rows:continue
            means.append(dict(cut=cut,mode=mode,prefix_parameters=rows[0]['prefix_parameters'],
                model_parameters=rows[0]['model_parameters'],mean_projected_ratio=st.mean(r['projected_ratio'] for r in rows)))
    csv_atomic(out/'boundary_cost_summary.csv',means)
    lines=['# Stage9C boundary/utility diagnostic','',
        'DEVELOPMENT DIAGNOSTIC ONLY. No adaptive controller was trained; no test dataset was loaded.',
        'All cuts use shared model/optimizer states. Lower projected cost is not established time-to-quality improvement.','',
        '| Prefix blocks | State handling | Prefix parameters | Mean projected ratio at40% omission |',
        '|---:|---|---:|---:|']
    lines += [f"| {r['cut']} | {r['mode']} | {r['prefix_parameters']} | {r['mean_projected_ratio']:.4f} |" for r in means]
    lines += ['', 'Review signal_comparisons.csv by seed/epoch/cut/mode: thresholds fit calibration batches only.',
        'Positive validation_omit_minus_full is immediate one-step harm, not final accuracy loss.',
        'Blind expected values are uniform-subset expectations at matched selected count; periodic subsets are count matched.',
        'Oracle subsets use unavailable outcome labels and are diagnostic bounds only. No best cut is automatically selected.',
        'Projected mixtures count the whole measured feature bundle but omit diagnostic restoring/cloning/oracle work.',
        'A complete online policy, long-run accuracy, net end-to-end GPU savings and novelty remain unestablished.',
        'Send this complete directory for review before any controlled efficacy pilot.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    v.atomic_json(out/'COMPLETE.json',dict(status='complete',fingerprint=fingerprint,measurements=expected,
        synthetic='sanity' in str(out),sha256={str(out/name):sha(out/name) for name in ['all_measurements.csv','signal_comparisons.csv','projected_costs.csv','REPORT.md']}))
    print('DIAGNOSTIC COMPLETE:',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-tag',default='final',help='Separate artificial-data rehearsal name; scientific outputs are fixed')
    p.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu',help='Artificial-data device only')
    args=p.parse_args();spec=json.loads(SPEC_PATH.read_text());fake=args.smoke
    if fake:
        torch.set_num_threads(1)
        spec.update(development_seeds=[9001,9002],training_epochs=2,lr_schedule_horizon_epochs=10,
            checkpoint_epochs=[1,2],prefix_blocks=[0,1],training_batch_size=4,probe_batches_per_checkpoint=4,
            probe_train_batch_size=2,probe_validation_batch_size=2,probe_warmup_batches=1)
    device=torch.device(args.smoke_device if fake else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag),'Invalid artificial rehearsal tag'
    out=ROOT/'results'/('stage9_sanity_boundary_v1_'+args.smoke_tag if fake else 'stage9C_boundary_diagnostic_v1')
    out.mkdir(parents=True,exist_ok=True);v.configure()
    preflight=v.idle_preflight() if device.type=='cuda' else None
    data=v.load_data('development',device,fake=fake)
    config=dict(spec=spec,source_sha256=source_hashes(),fake=fake,device=str(device),split=data[-1],
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        torchvision=__import__('torchvision').__version__,cuda=torch.version.cuda,
        gpu='cpu' if device.type=='cpu' else torch.cuda.get_device_name(device))
    fingerprint=v.digest(config);config['fingerprint']=fingerprint
    path=out/'configuration.json'
    if path.exists():assert json.loads(path.read_text())==config,'Code/settings/environment changed: use an amended version/output directory'
    else:v.atomic_json(path,config)
    if preflight is not None:v.atomic_json(out/'preflight_latest.json',preflight)
    for seed in spec['development_seeds']:
        folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
        trained=train_seed(folder,seed,data,spec,device,fake,fingerprint)
        for epoch in spec['checkpoint_epochs']:
            probe_state(folder,seed,epoch,trained['checkpoints'][str(epoch)],data,spec,device,fake,fingerprint)
    summarize(out,spec,fingerprint)


if __name__=='__main__':main()
