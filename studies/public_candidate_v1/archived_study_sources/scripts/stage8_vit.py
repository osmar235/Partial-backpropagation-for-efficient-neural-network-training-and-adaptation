"""New architecture harness. No old scientific code or outcomes are modified."""
from __future__ import annotations
import copy, csv, hashlib, json, math, platform, random, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.models.vision_transformer import VisionTransformer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import agrls_stage6a_cifar10_augmented_schedule as frozen6

SPEC_PATH=ROOT/'protocol/stage8/study_spec_v1.json'
SPEC=json.loads(SPEC_PATH.read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def source_hashes():
    import torchvision.models.vision_transformer as vit_source
    files=[Path(__file__).resolve(),Path(__file__).with_name('stage8_workflow.py'),
           SPEC_PATH,Path(frozen6.__file__),ROOT/'agrls_stage5a1_cifar10_fggr_microprobe.py',
           Path(vit_source.__file__),ROOT/'protocol/stage8/STAGE8_PROTOCOL_v1.md']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def configure():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def make_model(spec=SPEC,*,tiny=False):
    cfg={k:v for k,v in spec['architecture'].items() if k!='prefix_blocks'}
    if tiny:cfg.update(num_layers=2,num_heads=2,hidden_dim=32,mlp_dim=64)
    return VisionTransformer(**cfg)

def partition(model,blocks):
    suffix_roots=tuple(f'encoder.layers.encoder_layer_{i}.' for i in range(blocks,len(model.encoder.layers)))+('encoder.ln.','heads.')
    prefix=[];suffix=[];prefix_names=[]
    for name,p in model.named_parameters():
        if name.startswith(suffix_roots):suffix.append(p)
        else:prefix.append(p);prefix_names.append(name)
    assert prefix and suffix and 0<blocks<len(model.encoder.layers)
    assert set(map(id,prefix)).isdisjoint(set(map(id,suffix)))
    return prefix,suffix,prefix_names

def split_indices(targets,spec=SPEC):
    g=torch.Generator().manual_seed(spec['split_seed']);tr=[];va=[]
    y=torch.as_tensor(targets)
    for c in range(10):
        indices=torch.where(y==c)[0]
        indices=indices[torch.randperm(len(indices),generator=g)]
        n=spec['validation_per_class']
        assert len(indices)>n
        va.extend(indices[:n].tolist());tr.extend(indices[n:].tolist())
    return torch.tensor(tr),torch.tensor(va)

def load_data(phase,device,*,fake=False,tiny=False,data_dir=ROOT/'data'):
    if fake:
        g=torch.Generator().manual_seed(98001)
        x=torch.rand(16,3,32,32,generator=g);y=torch.randint(0,10,(16,),generator=g)
        ev=torch.rand(8,3,32,32,generator=g);ey=torch.randint(0,10,(8,),generator=g)
        return x.to(device),y.to(device),frozen6.normalize_cifar(ev.to(device)),ey.to(device),{'kind':'artificial'}
    # Development NEVER constructs CIFAR10(train=False).
    train=CIFAR10(str(data_dir),train=True,download=False)
    x=torch.from_numpy(train.data).permute(0,3,1,2).contiguous().float().div_(255)
    y=torch.tensor(train.targets)
    if phase=='development':
        tr,va=split_indices(train.targets)
        ev,ey=x[va],y[va];x,y=x[tr],y[tr]
        split={'kind':'train_only_stratified_45000_5000',
               'train_index_sha256':hashlib.sha256(tr.numpy().tobytes()).hexdigest(),
               'validation_index_sha256':hashlib.sha256(va.numpy().tobytes()).hexdigest()}
    elif phase=='evaluation':
        test=CIFAR10(str(data_dir),train=False,download=False)
        ev=torch.from_numpy(test.data).permute(0,3,1,2).contiguous().float().div_(255)
        ey=torch.tensor(test.targets);split={'kind':'full_50000_train_10000_test'}
    else:raise ValueError(phase)
    return x.to(device),y.to(device),frozen6.normalize_cifar(ev.to(device)),ey.to(device),split

def telemetry():
    executable=shutil.which('nvidia-smi')
    if not executable:return {'available':False,'reason':'nvidia-smi missing'}
    fields=['timestamp','index','name','driver_version','pstate','temperature.gpu','utilization.gpu',
            'clocks.sm','clocks.mem','power.draw','power.limit']
    try:
        kwargs={'capture_output':True,'text':True,'timeout':10}
        if sys.platform=='win32':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW
        p=subprocess.run([executable,'--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],**kwargs)
        if p.returncode:return {'available':False,'reason':p.stderr.strip()}
        rows=list(csv.reader(p.stdout.splitlines()))
        return {'available':True,'gpus':[dict(zip(fields,[v.strip() for v in row])) for row in rows]}
    except (OSError,subprocess.TimeoutExpired) as e:return {'available':False,'reason':str(e)}

def idle_preflight():
    samples=[]
    for _ in range(3):
        sample=telemetry();samples.append(sample)
        if not sample['available']:raise RuntimeError('GPU telemetry unavailable; inspect the environment before scientific runs')
        time.sleep(.5)
    for i in range(len(samples[0]['gpus'])):
        utilization=[float(s['gpus'][i]['utilization.gpu']) for s in samples]
        if sorted(utilization)[1]>20:
            raise RuntimeError(f'GPU is already busy ({utilization}% utilization). Stop other GPU jobs and rerun the same launcher.')
    return samples

def learning_rate(step,total,batches,lr,spec):
    warm=spec['lr_warmup_epochs']*batches
    if warm and step<warm:return lr*(step+1)/warm
    progress=(step-warm)/max(1,total-warm-1)
    return lr*(spec['min_lr_ratio']+(1-spec['min_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))

def periodic_indices(epochs,batches,profile):
    assert len(profile)==epochs
    result=set()
    for epoch,q in enumerate(profile):
        k=round(q*batches)
        if not 0<=k<=batches//2:raise ValueError('Infeasible archived partial count')
        local={int((j+.5)*(batches-1)/k)+1 for j in range(k)} if k else set()
        assert len(local)==k and 0 not in local and all(j+1 not in local for j in local)
        result.update(epoch*batches+j for j in local)
    assert all(j+1 not in result for j in result)
    return result

def compute_backward(loss,prefix,suffix,partial,variant):
    if partial:
        gradients=torch.autograd.grad(loss,suffix)
        for p,g in zip(suffix,gradients):p.grad=g
        for p in prefix:p.grad=None if variant=='replay_prefix' else torch.zeros_like(p)
    else:loss.backward()
def optimizer_step(optimizer,prefix,partial,variant,last_delta):
    before=[p.detach().clone() for p in prefix] if variant=='replay_prefix' and not partial else None
    optimizer.step()
    if variant=='replay_prefix':
        if partial:
            assert last_delta is not None
            with torch.no_grad():
                for p,d in zip(prefix,last_delta):p.add_(d)
        else:last_delta=[p.detach()-b for p,b in zip(prefix,before)]
    return last_delta

def update(model,optimizer,loss,prefix,suffix,partial,variant,last_delta):
    compute_backward(loss,prefix,suffix,partial,variant)
    return optimizer_step(optimizer,prefix,partial,variant,last_delta)

@torch.no_grad()
def evaluate(model,x,y,batch):
    model.eval();loss=correct=0
    for i in range(0,len(y),batch):
        logits=model(x[i:i+batch]);labels=y[i:i+batch]
        loss+=F.cross_entropy(logits,labels,reduction='sum').item()
        correct+=(logits.argmax(1)==labels).sum().item()
    model.train()
    return loss/len(y),correct/len(y)

def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def atomic_json(path,value):
    pending=path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');pending.replace(path)

def run_trajectory(out,phase,seed,variant,lr,device,data,spec=SPEC,profile=None,*,tiny=False,fake=False,order=0,freeze_hash=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    xtr,ytr,xev,yev,split=data
    config=dict(phase=phase,seed=seed,variant=variant,lr=lr,spec=spec,split=split,
        tiny=tiny,fake=fake,device=str(device),torch=torch.__version__,python=platform.python_version(),
        torchvision=__import__('torchvision').__version__,numpy=np.__version__,cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        source_sha256=source_hashes(),profile=profile,order_position=order,freeze_hash=freeze_hash)
    fingerprint=digest(config);config['fingerprint']=fingerprint
    cp=out/'run_config.json';sp=out/'summary.json'
    if cp.exists():
        old=json.loads(cp.read_text())
        if old['fingerprint']!=fingerprint:raise ValueError(f'Incompatible code/settings/environment: use a new output directory: {out}')
    else:atomic_json(cp,config)
    if sp.exists():
        old=json.loads(sp.read_text());assert old['fingerprint']==fingerprint and old['status']=='complete'
        print(f'SKIP {out}',flush=True);return old
    configure();seed_all(seed)
    model=make_model(spec,tiny=tiny).to(device)
    prefix,suffix,prefix_names=partition(model,1 if tiny else spec['architecture']['prefix_blocks'])
    initial=copy.deepcopy(model.state_dict())
    state_hash=hashlib.sha256()
    for name,tensor in initial.items():
        state_hash.update(name.encode());state_hash.update(tensor.detach().cpu().numpy().tobytes())
    initial_hash=state_hash.hexdigest()
    batch=spec['batch_size'];batches=math.ceil(len(ytr)/batch);total=spec['epochs']*batches
    planned=periodic_indices(spec['epochs'],batches,profile) if variant=='periodic_matched' else set()
    telemetry_rows=[]
    def snapshot(tag,epoch):
        if device.type=='cuda':telemetry_rows.append({'tag':tag,'epoch':epoch,'sample':json.dumps(telemetry())})
    started_utc=datetime.now(timezone.utc).isoformat()
    process_start=time.perf_counter();snapshot('before_warmup',0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=spec['weight_decay'])
    bg=torch.Generator().manual_seed(seed+93001);ag=torch.Generator().manual_seed(seed+94001)
    warm_iter=iter(frozen6.epoch_batches(len(ytr),batch,bg));ema=None;previous=False;last=None
    sync(device);start=time.perf_counter()
    for t in range(spec['algorithm_warmup_steps']):
        try:ix=next(warm_iter)
        except StopIteration:
            warm_iter=iter(frozen6.epoch_batches(len(ytr),batch,bg));ix=next(warm_iter)
        ix=ix.to(device);x=frozen6.augment_and_normalize(xtr[ix],ag)
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x),ytr[ix],label_smoothing=spec['label_smoothing']);lv=loss.item()
        partial=(t%2==1) if variant=='periodic_matched' else (variant!='full_adamw' and ema is not None and not previous and lv<=ema)
        last=update(model,optimizer,loss,prefix,suffix,partial,variant,last)
        ema=lv if ema is None else spec['rho']*ema+(1-spec['rho'])*lv;previous=partial
    sync(device);warm_seconds=time.perf_counter()-start
    model.load_state_dict(initial);model.zero_grad(set_to_none=True)
    del optimizer
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=spec['weight_decay'])
    seed_all(seed+95001)  # Identical dropout stream per seed, independent of method warm-up.
    bg=torch.Generator().manual_seed(seed+91001);ag=torch.Generator().manual_seed(seed+92001)
    ema=None;previous=False;last=None;idx=0;partial_total=0;wall=ev_seconds=0.0
    epochs=[];steps=[];clocks={k:0.0 for k in ('augmentation','forward','backward','optimizer','backward_full','backward_partial')}
    for epoch in range(1,spec['epochs']+1):
        snapshot('before_training',epoch)
        losses=examples=ep_partial=0;mode_backward={'full':[],'partial':[]}
        sync(device);epoch_start=time.perf_counter()
        for ix in frozen6.epoch_batches(len(ytr),batch,bg):
            ix=ix.to(device);raw=xtr[ix];labels=ytr[ix]
            sync(device);t0=time.perf_counter();x=frozen6.augment_and_normalize(raw,ag);sync(device);aug=time.perf_counter()-t0
            step_lr=learning_rate(idx,total,batches,lr,spec)
            for group in optimizer.param_groups:group['lr']=step_lr
            optimizer.zero_grad(set_to_none=True)
            sync(device);t0=time.perf_counter()
            loss=F.cross_entropy(model(x),labels,label_smoothing=spec['label_smoothing'])
            sync(device);forward=time.perf_counter()-t0
            lv=loss.item()  # Exactly ONE scalar extraction per timed update in every variant.
            if not math.isfinite(lv):raise FloatingPointError((variant,epoch,idx,lv))
            partial=(idx in planned) if variant=='periodic_matched' else (
                variant!='full_adamw' and ema is not None and not previous and lv<=ema)
            assert not(partial and previous)
            sync(device);t0=time.perf_counter()
            compute_backward(loss,prefix,suffix,partial,variant)
            sync(device);backward=time.perf_counter()-t0
            sync(device);t0=time.perf_counter()
            last=optimizer_step(optimizer,prefix,partial,variant,last)
            sync(device);optim=time.perf_counter()-t0
            steps.append(dict(epoch=epoch,step=idx+1,partial=int(partial),loss=lv,ema_before='' if ema is None else ema,
                lr=step_lr,augmentation_seconds=aug,forward_seconds=forward,backward_seconds=backward,optimizer_seconds=optim))
            for key,value in [('augmentation',aug),('forward',forward),('backward',backward),('optimizer',optim),
                              ('backward_partial' if partial else 'backward_full',backward)]:clocks[key]+=value
            mode_backward['partial' if partial else 'full'].append(backward)
            losses+=lv*len(labels);examples+=len(labels);ep_partial+=int(partial)
            ema=lv if ema is None else spec['rho']*ema+(1-spec['rho'])*lv
            previous=partial;idx+=1
        sync(device);epoch_wall=time.perf_counter()-epoch_start;wall+=epoch_wall;partial_total+=ep_partial
        snapshot('after_training',epoch)
        sync(device);ev_start=time.perf_counter();evloss,acc=evaluate(model,xev,yev,spec['eval_batch_size']);sync(device)
        ev=time.perf_counter()-ev_start;ev_seconds+=ev
        epochs.append(dict(epoch=epoch,updates=idx,training_loss=losses/examples,eval_loss=evloss,eval_accuracy=acc,
            epoch_training_seconds=epoch_wall,cumulative_training_seconds=wall,evaluation_seconds=ev,
            partial_updates=ep_partial,full_updates=batches-ep_partial,partial_fraction=ep_partial/batches,
            mean_backward_full_seconds=sum(mode_backward['full'])/max(1,len(mode_backward['full'])),
            mean_backward_partial_seconds=sum(mode_backward['partial'])/max(1,len(mode_backward['partial']))))
        print(f'{phase} seed={seed} {variant} epoch={epoch}/{spec["epochs"]} accuracy={100*acc:.2f}% q={ep_partial/batches:.3f} train_s={wall:.1f}',flush=True)
    write_csv(out/'epoch_trace.csv',epochs);write_csv(out/'step_trace.csv',steps);write_csv(out/'gpu_telemetry.csv',telemetry_rows)
    # Final weights and complete AdamW state enable an additional state audit.
    sync(device);checkpoint_start=time.perf_counter()
    checkpoint=out/'final_checkpoint.pt';pending=out/'final_checkpoint.pt.tmp'
    torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),seed=seed,
                    variant=variant,ema=ema,fingerprint=fingerprint),pending);pending.replace(checkpoint)
    checkpoint_seconds=time.perf_counter()-checkpoint_start
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary=dict(status='complete',phase=phase,seed=seed,variant=variant,optimizer='adamw',
        fingerprint=fingerprint,lr=lr,epochs=spec['epochs'],updates=idx,partial_updates=partial_total,
        initial_model_sha256=initial_hash,started_utc=started_utc,finished_utc=datetime.now(timezone.utc).isoformat(),
        full_updates=idx-partial_total,partial_fraction=partial_total/idx,max_consecutive_partials=1 if partial_total else 0,
        final_eval_accuracy=epochs[-1]['eval_accuracy'],final_eval_loss=epochs[-1]['eval_loss'],
        training_wall_seconds=wall,evaluation_seconds=ev_seconds,warmup_seconds=warm_seconds,
        checkpoint_seconds=checkpoint_seconds,total_experiment_elapsed_seconds=time.perf_counter()-process_start,
        checkpoint_sha256=checkpoint_hash,prefix_parameter_names=prefix_names,
        prefix_parameter_count=sum(p.numel() for p in prefix),suffix_parameter_count=sum(p.numel() for p in suffix),
        component_seconds=clocks,timing_unattributed_seconds=wall-sum(clocks[k] for k in ('augmentation','forward','backward','optimizer')),
        scalar_loss_item_calls=idx,timed_explicit_cuda_sync_calls=(8*idx+2*spec['epochs']) if device.type=='cuda' else 0,
        order_position=order,freeze_hash=freeze_hash)
    for target in spec['targets']:
        first=next((r for r in epochs if r['eval_accuracy']>=target),None)
        summary[f'training_only_time_to_{round(100*target)}pct']=first['cumulative_training_seconds'] if first else None
        summary[f'epoch_to_{round(100*target)}pct']=first['epoch'] if first else None
    # Timed explicit synchronization: four start/end pairs per update,
    # plus epoch start/end. Scalar extraction may also synchronize implicitly.
    atomic_json(sp,summary)
    del model,optimizer
    return summary
