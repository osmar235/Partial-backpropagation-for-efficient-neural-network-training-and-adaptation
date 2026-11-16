"""Portable Oxford Pet baseline qualification. Official-test access is absent."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT/'protocol/pets_transfer/baseline_spec_v1.json'
DATA = ROOT/'data/pets_transfer'
WEIGHTS = ROOT/'data/pretrained/resnet18-f37072fd.pth'
REAL = ROOT/'results/pets_baseline_v1'
SOURCE_NAMES = ['scripts/pets_transfer.py', 'scripts/test_pets_transfer.py',
    'scripts/run_pets_baseline.cmd', 'protocol/pets_transfer/baseline_spec_v1.json',
    'protocol/pets_transfer/PROTOCOL_v1.md', 'README_PETS_TRANSFER.md']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def write(p, value):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    tmp.replace(p)

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def state_hash(state):
    return digest({k: {'shape':list(v.shape), 'dtype':str(v.dtype), 'hash':tensor_hash(v)} for k,v in state.items()})

def sources():
    return {p:sha(ROOT/p) for p in SOURCE_NAMES}

def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)

def configure(device):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    if device.type=='cuda':
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA with BF16 required'

@contextmanager
def lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'RUN.lock').open('a+b') as stream:
        stream.seek(0,2)
        if not stream.tell(): stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream, fcntl.LOCK_UN)

def split_indices(labels, spec):
    rng=np.random.default_rng(spec['split_seed']); training=[]; validation=[]
    labels=np.asarray(labels)
    for c in range(spec['classes']):
        ix=np.flatnonzero(labels==c); assert len(ix)>=2
        ix=rng.permutation(ix); n=max(1,round(len(ix)*spec['validation_fraction_per_class']))
        validation.extend(ix[:n].tolist()); training.extend(ix[n:].tolist())
    return sorted(training), sorted(validation)

def prepare():
    from torchvision.datasets import OxfordIIITPet
    spec=read(SPEC); DATA.mkdir(parents=True,exist_ok=True)
    with lock(DATA):
        metadata=DATA/'cache_metadata.json'
        if metadata.exists():
            record=read(metadata)
            assert record['spec_hash']==sha(SPEC), 'Cache belongs to a different recipe'
            assert sha(DATA/'trainval_cache.pt')==record['cache_sha256']
            assert sha(WEIGHTS)==record['weights_sha256']
            print('Verified existing trainval cache and weights.',flush=True)
            return
        start=time.perf_counter()
        # The archive includes test files, but this object only reads trainval.txt.
        ds=OxfordIIITPet(str(DATA),split='trainval',target_types='category',download=True)
        download_seconds=time.perf_counter()-start
        cache_start=time.perf_counter(); images=[]; labels=[]; ids=[]; image_hashes={}
        for i in range(len(ds)):
            im,label=ds[i]
            im=TVF.center_crop(TVF.resize(im,[spec['cache_size']],antialias=True),[spec['cache_size']]*2)
            images.append(TVF.pil_to_tensor(im));labels.append(int(label))
            path=Path(ds._images[i]);ids.append(path.stem);image_hashes[path.name]=sha(path)
            if (i+1)%250==0: print(f'CACHE trainval {i+1}/{len(ds)}',flush=True)
        train,val=split_indices(labels,spec)
        payload=dict(images=torch.stack(images),labels=torch.tensor(labels),ids=ids,train=train,validation=val)
        pending=DATA/'trainval_cache.pending.pt';torch.save(payload,pending);pending.replace(DATA/'trainval_cache.pt')
        WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
        if not WEIGHTS.exists():
            torch.hub.download_url_to_file(ResNet18_Weights.IMAGENET1K_V1.url,str(WEIGHTS)+'.pending',hash_prefix='f37072fd')
            Path(str(WEIGHTS)+'.pending').replace(WEIGHTS)
        assert sha(WEIGHTS).startswith('f37072fd')
        write(metadata,dict(spec_hash=sha(SPEC),dataset='OxfordIIITPet/trainval',test_read=False,
            examples=len(labels),train=len(train),validation=len(val),classes=spec['classes'],
            split_hash=digest({'train':train,'validation':val}),cache_sha256=sha(DATA/'trainval_cache.pt'),
            trainval_image_hashes=image_hashes,trainval_annotation_sha256=sha(DATA/'oxford-iiit-pet/annotations/trainval.txt'),
            weights_sha256=sha(WEIGHTS),weights_url=ResNet18_Weights.IMAGENET1K_V1.url,
            download_seconds=download_seconds,cache_and_weights_preparation_seconds=time.perf_counter()-cache_start,
            dataset_source='https://www.robots.ox.ac.uk/~vgg/data/pets/',license='CC-BY-SA-4.0; original image owners retain copyright'))
        print(f'Prepared {len(train)} training / {len(val)} development-validation images. Test unopened.',flush=True)

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,8,3,padding=1);self.bn=nn.BatchNorm2d(8);self.fc=nn.Linear(8,3)
    def forward(self,x): return self.fc(self.bn(self.conv(x)).relu().mean((2,3)))

def make_model(seed,cfg,device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(291030)
        model=Tiny() if cfg['tiny'] else resnet18(weights=None)
        if not cfg['tiny'] and not cfg['smoke']:
            model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
        torch.manual_seed(seed+290000)
        model.fc=nn.Linear(model.fc.in_features,cfg['spec']['classes'])
    return model.eval().to(device=device,memory_format=torch.channels_last)

def optimizer(model,spec,device):
    prefix=[];head=[]
    for name,p in model.named_parameters(): (head if name.startswith('fc.') else prefix).append(p)
    return torch.optim.AdamW([{'params':prefix,'lr':spec['backbone_lr'],'base_lr':spec['backbone_lr']},
        {'params':head,'lr':spec['head_lr'],'base_lr':spec['head_lr']}],weight_decay=spec['weight_decay'],fused=device.type=='cuda')

def data(cfg):
    spec=cfg['spec']
    if not cfg['smoke']: return torch.load(DATA/'trainval_cache.pt',map_location='cpu',weights_only=True)
    rng=torch.Generator().manual_seed(291031); n=18 if cfg['tiny'] else 8
    return dict(images=torch.randint(0,256,(n,3,spec['cache_size'],spec['cache_size']),dtype=torch.uint8,generator=rng),
        labels=torch.arange(n)%spec['classes'],ids=[str(i) for i in range(n)],train=list(range(n-3)),validation=list(range(n-3,n)))

def stream(seed,epoch,indices,spec):
    gen=torch.Generator().manual_seed(seed*10000+epoch)
    order=torch.tensor(indices)[torch.randperm(len(indices),generator=gen)]
    d=spec['cache_size']-spec['image_size']
    offsets=torch.randint(0,d+1,(len(indices),2),generator=gen)
    flips=torch.rand(len(indices),generator=gen)<.5
    return order,offsets,flips

def transform(raw,spec,device,offsets=None,flips=None):
    x=raw.to(device);n=len(x);size=spec['image_size']
    if offsets is None:
        d=(spec['cache_size']-size)//2;x=x[:,:,d:d+size,d:d+size]
    else:
        ix=torch.arange(size,device=device);offsets=offsets.to(device)
        x=x.permute(0,2,3,1)[torch.arange(n,device=device)[:,None,None],offsets[:,0,None,None]+ix[None,:,None],offsets[:,1,None,None]+ix[None,None,:]].permute(0,3,1,2)
        x=torch.where(flips.to(device)[:,None,None,None],x.flip(-1),x)
    x=x.float()/255
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return ((x-mean)/std).contiguous(memory_format=torch.channels_last)

def factor(index,batches,spec):
    warm=batches*spec['lr_warmup_epochs'];total=batches*spec['epochs']
    if index<warm:return (index+1)/max(1,warm)
    return spec['min_lr_ratio']+(1-spec['min_lr_ratio'])*.5*(1+math.cos(math.pi*(index-warm)/max(1,total-warm-1)))

@contextmanager
def arithmetic(policy,device):
    old_conv=torch.backends.cudnn.allow_tf32;old_mat=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32=policy!='common_fp32';torch.backends.cuda.matmul.allow_tf32=policy!='common_fp32'
    try:
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=policy=='bf16'):yield
    finally:
        torch.backends.cudnn.allow_tf32=old_conv;torch.backends.cuda.matmul.allow_tf32=old_mat

def update(model,opt,x,y,policy,device,smoothing):
    opt.zero_grad(set_to_none=True)
    with arithmetic(policy,device):logits=model(x)
    loss=F.cross_entropy(logits.float(),y,label_smoothing=smoothing)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward();opt.step();return float(loss.detach())

@torch.no_grad()
def evaluate(model,cache,policy,cfg,device):
    spec=cfg['spec'];outputs=[];labels=[];ix=cache['validation']
    for start in range(0,len(ix),spec['eval_batch_size']):
        ids=ix[start:start+spec['eval_batch_size']];x=transform(cache['images'][ids],spec,device)
        with arithmetic(policy,device):logits=model(x)
        outputs.append(logits.float().cpu());labels.append(cache['labels'][ids])
    logits=torch.cat(outputs);y=torch.cat(labels)
    assert torch.isfinite(logits).all()
    return dict(logits=logits,labels=y,accuracy=float((logits.argmax(1)==y).float().mean()),
                loss=float(F.cross_entropy(logits,y)))

def configuration(smoke,tiny,device):
    spec=read(SPEC)
    if smoke:
        spec.update(seeds=[1,2],epochs=2,batch_size=4,eval_batch_size=4,warmup_steps=1,
                    classes=3,cache_size=20 if tiny else 256,image_size=16 if tiny else 224)
    cfg=dict(spec=spec,smoke=smoke,tiny=tiny,source_hashes=sources(),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),
            numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None))
    if not smoke:
        meta=read(DATA/'cache_metadata.json');assert meta['spec_hash']==sha(SPEC) and not meta['test_read']
        assert sha(DATA/'trainval_cache.pt')==meta['cache_sha256'] and sha(WEIGHTS)==meta['weights_sha256']
        cfg['inputs']={k:meta[k] for k in ('cache_sha256','weights_sha256','split_hash','examples','train','validation')}
        cfg['cache_metadata_sha256']=sha(DATA/'cache_metadata.json')
    cfg['fingerprint']=digest(cfg)
    return cfg

def verify_complete(parent,cfg):
    marker=parent/'COMPLETE.json'
    if not marker.exists():return None
    record=read(marker);assert record['fingerprint']==cfg['fingerprint']
    attempt=parent/record['attempt'];assert attempt.parent==parent and attempt.is_dir()
    for name,h in record['sha256'].items():
        assert Path(name).name==name and sha(attempt/name)==h,(attempt,name)
    return attempt

def run_one(root,cfg,seed,policy,position,device):
    parent=root/f'seed_{seed}'/policy;parent.mkdir(parents=True,exist_ok=True)
    if verify_complete(parent,cfg):print('SKIP verified',seed,policy,flush=True);return
    attempt=parent/('attempt_'+uuid.uuid4().hex);attempt.mkdir()
    spec=cfg['spec'];sync(device);start=time.perf_counter()
    cache=data(cfg);model=make_model(seed,cfg,device);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_hash=state_hash(initial);opt=optimizer(model,spec,device)
    warmids=cache['train'][:spec['batch_size']];xx=transform(cache['images'][warmids],spec,device);yy=cache['labels'][warmids].to(device)
    for _ in range(spec['warmup_steps']):update(model,opt,xx,yy,policy,device,spec['label_smoothing'])
    model.load_state_dict(initial);model.zero_grad(set_to_none=True);opt=optimizer(model,spec,device)
    assert state_hash(model.state_dict())==initial_hash and len(opt.state)==0
    sync(device);setup_seconds=time.perf_counter()-start
    batches=math.ceil(len(cache['train'])/spec['batch_size']);steps=[];epochs=[];streams=[];idx=0;training_seconds=0;validation_seconds=0
    for epoch in range(spec['epochs']):
        sync(device);t=time.perf_counter();order,offsets,flips=stream(seed,epoch,cache['train'],spec)
        streams.append(dict(epoch=epoch,order=tensor_hash(order),offsets=tensor_hash(offsets),flips=tensor_hash(flips)))
        weighted_loss=seen=0
        for offset in range(0,len(order),spec['batch_size']):
            ids=order[offset:offset+spec['batch_size']];x=transform(cache['images'][ids],spec,device,offsets[offset:offset+len(ids)],flips[offset:offset+len(ids)])
            y=cache['labels'][ids].to(device);lr=factor(idx,batches,spec)
            for group in opt.param_groups:group['lr']=group['base_lr']*lr
            loss=update(model,opt,x,y,policy,device,spec['label_smoothing']);idx+=1
            steps.append(dict(step=idx,epoch=epoch,examples=len(ids),loss=loss,lr_factor=lr))
            weighted_loss+=loss*len(ids);seen+=len(ids)
        sync(device);train_s=time.perf_counter()-t;training_seconds+=train_s
        t=time.perf_counter();common=evaluate(model,cache,'common_fp32',cfg,device);sync(device);val_s=time.perf_counter()-t;validation_seconds+=val_s
        epochs.append(dict(epoch=epoch+1,updates=idx,training_loss=weighted_loss/seen,common_accuracy=common['accuracy'],
            common_loss=common['loss'],training_seconds=train_s,validation_seconds=val_s))
        print(f'BASELINE seed={seed} {policy} epoch={epoch+1}/{spec["epochs"]} val={100*common["accuracy"]:.2f}% train_s={training_seconds:.1f}',flush=True)
    t=time.perf_counter();own=evaluate(model,cache,policy,cfg,device);sync(device);validation_seconds+=time.perf_counter()-t
    assert idx==batches*spec['epochs']
    ages={name:int(opt.state[p]['step'].item()) for name,p in model.named_parameters()}
    assert set(ages.values())=={idx} and all(torch.isfinite(p).all() for p in model.parameters())
    for name,value in model.state_dict().items():
        if 'running_' in name or 'num_batches_tracked' in name:assert torch.equal(value.cpu(),initial[name]),name
    t=time.perf_counter()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),ages=ages,seed=seed,policy=policy,
        fingerprint=cfg['fingerprint'],initial_hash=initial_hash,updates=idx),attempt/'final.pt')
    np.savez_compressed(attempt/'logits.npz',common=common['logits'].numpy(),own=own['logits'].numpy(),labels=common['labels'].numpy())
    for name,rows in [('steps.csv',steps),('epochs.csv',epochs)]:
        with (attempt/name).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(attempt/'streams.json',streams);sync(device);artifact_seconds=time.perf_counter()-t;complete=time.perf_counter()-start
    summary=dict(seed=seed,policy=policy,position=position,fingerprint=cfg['fingerprint'],initial_hash=initial_hash,updates=idx,
        common_accuracy=common['accuracy'],common_loss=common['loss'],own_accuracy=own['accuracy'],own_loss=own['loss'],
        setup_seconds=setup_seconds,training_seconds=training_seconds,validation_seconds=validation_seconds,artifact_seconds=artifact_seconds,
        complete_seconds=complete,other_charged_seconds=complete-setup_seconds-training_seconds-validation_seconds-artifact_seconds,
        ages=ages,smoke=cfg['smoke'],test_access=False)
    write(attempt/'summary.json',summary)
    hashes={p.name:sha(p) for p in attempt.iterdir() if p.is_file()}
    write(parent/'COMPLETE.json',dict(fingerprint=cfg['fingerprint'],attempt=attempt.name,sha256=hashes))
    del model,opt,cache,initial,xx,yy,x,y,common,own
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()

def audit(root,device,rescore=False):
    cfg=read(root/'baseline_freeze.json');spec=cfg['spec'];assert sources()==cfg['source_hashes']
    cache=data(cfg);rows=[];rescoring=[];stream_records={};initial_records={}
    for seed in spec['seeds']:
        for policy in spec['policies']:
            parent=root/f'seed_{seed}'/policy;attempt=verify_complete(parent,cfg);assert attempt, (seed,policy)
            row=read(attempt/'summary.json');cp=torch.load(attempt/'final.pt',map_location='cpu',weights_only=True)
            assert cp['fingerprint']==cfg['fingerprint'] and cp['seed']==seed and cp['policy']==policy
            assert set(cp['ages'].values())=={row['updates']}
            assert len(cp['optimizer']['state'])==len(cp['ages'])
            for state in cp['optimizer']['state'].values():
                assert int(state['step'])==row['updates']
                assert all(torch.isfinite(state[k]).all() for k in ('exp_avg','exp_avg_sq'))
            with np.load(attempt/'logits.npz') as logits:
                labels=logits['labels'];assert np.array_equal(labels,cache['labels'][cache['validation']].numpy())
                for name in ('common','own'):
                    x=logits[name].astype(np.float64);assert np.isfinite(x).all()
                    accuracy=float(np.mean(x.argmax(1)==labels));loss=float(np.mean(np.logaddexp.reduce(x,axis=1)-x[np.arange(len(labels)),labels]))
                    assert abs(accuracy-row[name+'_accuracy'])<1e-7 and abs(loss-row[name+'_loss'])<1e-5
                if rescore:
                    model=make_model(seed,cfg,device);model.load_state_dict(cp['model'])
                    for mode,name in [('common_fp32','common'),(policy,'own')]:
                        pred=evaluate(model,cache,mode,cfg,device)['logits'].numpy()
                        error=float(np.max(np.abs(pred-logits[name])))
                        assert error<=1e-5 and np.array_equal(pred.argmax(1),logits[name].argmax(1)),(seed,policy,name,error)
                        rescoring.append(dict(seed=seed,policy=policy,score=name,max_error=error))
                    del model
            observed=read(attempt/'streams.json');expected=[]
            for epoch in range(spec['epochs']):
                order,offsets,flips=stream(seed,epoch,cache['train'],spec)
                expected.append(dict(epoch=epoch,order=tensor_hash(order),offsets=tensor_hash(offsets),flips=tensor_hash(flips)))
            assert observed==expected
            stream_records[(seed,policy)]=observed;initial_records[(seed,policy)]=row['initial_hash']
            with (attempt/'steps.csv').open(newline='',encoding='utf8') as f:steps=list(csv.DictReader(f))
            batches=math.ceil(len(cache['train'])/spec['batch_size']);assert len(steps)==row['updates']==batches*spec['epochs']
            for i,s in enumerate(steps):
                assert int(s['step'])==i+1 and int(s['epoch'])==i//batches
                assert math.isfinite(float(s['loss'])) and abs(float(s['lr_factor'])-factor(i,batches,spec))<1e-12
                assert int(s['examples'])==min(spec['batch_size'],len(cache['train'])-(i%batches)*spec['batch_size'])
            assert row['complete_seconds']>row['training_seconds']>0 and row['other_charged_seconds']>=-1e-6
            rows.append(row);del cp
    pairs=[]
    for seed in spec['seeds']:
        full=next(r for r in rows if r['seed']==seed and r['policy']=='fp32_tf32');bf=next(r for r in rows if r['seed']==seed and r['policy']=='bf16')
        assert initial_records[(seed,'fp32_tf32')]==initial_records[(seed,'bf16')]
        assert stream_records[(seed,'fp32_tf32')]==stream_records[(seed,'bf16')]
        pairs.append(dict(seed=seed,fp32_common_percent=100*full['common_accuracy'],bf16_common_percent=100*bf['common_accuracy'],
            deficit_common_pp=100*(full['common_accuracy']-bf['common_accuracy']),deficit_own_pp=100*(full['own_accuracy']-bf['own_accuracy']),
            complete_saving_percent=100*(1-bf['complete_seconds']/full['complete_seconds'])))
    quality=all(p['deficit_common_pp']<=spec['precision_max_deficit_pp']+1e-5 and p['deficit_own_pp']<=spec['precision_max_deficit_pp']+1e-5 for p in pairs)
    ready=min(p['fp32_common_percent'] for p in pairs)>=spec['baseline_min_seed_accuracy_percent'] and np.mean([p['fp32_common_percent'] for p in pairs])>=spec['baseline_min_mean_accuracy_percent']
    positive=all(p['complete_saving_percent']>0 for p in pairs)
    launches=read(root/'launches.json');fragmented=len(launches)>1 or any(len(list((root/f'seed_{s}'/p).glob('attempt_*')))>1 for s in spec['seeds'] for p in spec['policies'])
    result=dict(integrity='PASS',smoke=cfg['smoke'],test_access=False,independent_development_seeds=len(spec['seeds']),
        full_baseline_ready=bool(ready),bf16_quality_ready=bool(quality),bf16_faster_every_pair=positive,fragmented=fragmented,
        qualified=bool(not cfg['smoke'] and ready and quality and positive and not fragmented),pairs=pairs,
        rescores=rescoring,source_fingerprint=cfg['fingerprint'],next_phase='review baseline; official-test evaluation remains disabled')
    write(root/'audit.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result

def run(args):
    device=torch.device(args.device);configure(device)
    root=Path(args.output) if args.output else REAL
    if args.smoke:
        assert args.output and root.resolve()!=REAL.resolve(), 'Smoke needs a separate output folder'
    else: assert device.type=='cuda', 'Real baseline requires CUDA'
    with lock(root):
        cfg=configuration(args.smoke,not args.native,device);freeze=root/'baseline_freeze.json'
        if freeze.exists():assert read(freeze)==cfg, 'Frozen inputs/runtime differ; preserve output and review'
        else:write(freeze,cfg)
        if all(verify_complete(root/f'seed_{s}'/p,cfg) for s in cfg['spec']['seeds'] for p in cfg['spec']['policies']):
            audit(root,device,rescore=True);return
        launch=root/'launches.json';launches=read(launch) if launch.exists() else []
        launches.append(dict(id=uuid.uuid4().hex,unix_time=time.time()));write(launch,launches)
        for i,seed in enumerate(cfg['spec']['seeds']):
            policies=cfg['spec']['policies'][::1 if i%2==0 else -1]
            for position,policy in enumerate(policies):run_one(root,cfg,seed,policy,position,device)
        audit(root,device,rescore=True)
        print('COMPLETE. Send '+str(root.resolve())+' for review. Official test was not accessed.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','baseline','audit'])
    parser.add_argument('--device',default='cuda');parser.add_argument('--output');parser.add_argument('--smoke',action='store_true');parser.add_argument('--native',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='audit':
        device=torch.device(args.device);configure(device)
        with lock(Path(args.output) if args.output else REAL):audit(Path(args.output) if args.output else REAL,device,rescore=True)
    else:
        if not args.smoke:args.native=True
        run(args)

if __name__=='__main__':main()
