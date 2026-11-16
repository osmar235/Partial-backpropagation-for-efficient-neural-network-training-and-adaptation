"""Fixed Pet full/frozen/periodic development. No official-test access or tuning."""
import argparse
import csv
import gc
import itertools
import math
from pathlib import Path
import time
import uuid
import numpy as np
import torch
import torch.nn.functional as F
import pets_transfer as b

SPEC=b.ROOT/'protocol/pets_selective/development_spec_v1.json'
PREPARED=b.ROOT/'protocol/pets_selective/PREPARED_v1.json'
REAL=b.ROOT/'results/pets_selective_development_v1'
FILES=b.SOURCE_NAMES+['scripts/pets_selective.py','scripts/test_pets_selective.py',
    'scripts/run_pets_selective.cmd','protocol/pets_selective/development_spec_v1.json',
    'protocol/pets_selective/PROTOCOL_v1.md','README_PETS_SELECTIVE.md']


def is_prefix(name,tiny=False):
    return name.startswith(('conv.','bn.')) if tiny else name.startswith(('conv1.','bn1.','layer1.','layer2.'))


def active(arm,index):
    assert arm in ('full','frozen','periodic_r4')
    return arm=='full' or (arm=='periodic_r4' and index%4==0)


def forward(model,x,prefix_active,tiny=False):
    if prefix_active:return model(x)
    with torch.no_grad():
        if tiny:x=model.bn(model.conv(x)).relu()
        else:x=model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x))))))
    if tiny:return model.fc(x.mean((2,3)))
    x=model.layer4(model.layer3(x))
    return model.fc(torch.flatten(model.avgpool(x),1))


def update(model,opt,x,y,arm,index,cfg,device):
    opt.zero_grad(set_to_none=True)
    with b.arithmetic('bf16',device):logits=forward(model,x,active(arm,index),cfg['tiny'])
    loss=F.cross_entropy(logits.float(),y,label_smoothing=cfg['spec']['label_smoothing'])
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward();opt.step()
    return float(loss.detach())


def configuration(smoke,tiny,device):
    cfg=b.configuration(smoke,tiny,device)
    plan=b.read(SPEC)
    assert plan['refresh_period']==4 and plan['policy']=='bf16' and not plan['test_access']
    assert sorted(map(tuple,plan['orders']))==sorted(itertools.permutations(plan['arms']))
    if not smoke:
        prepared=b.read(PREPARED)
        assert prepared['validation_status']=='PASS'
        for rel,h in {**prepared['sources'],**prepared['evidence']}.items():assert b.sha(b.ROOT/rel)==h,rel
        baseline=b.read(b.REAL/'baseline_freeze.json')
        assert cfg['runtime']==baseline['runtime'] and cfg['inputs']==baseline['inputs']
        assert b.read(b.REAL/'audit.json')['qualified']
    cfg['spec']['seeds']=[1,2] if smoke else plan['seeds']
    cfg['plan']=plan
    cfg['source_hashes']={rel:b.sha(b.ROOT/rel) for rel in FILES}
    cfg.pop('fingerprint')
    cfg['fingerprint']=b.digest(cfg)
    return cfg


def ages(model,opt):
    return {name:int(opt.state[p]['step'].item()) if p in opt.state else 0 for name,p in model.named_parameters()}


def check_state(model,opt,initial,arm,total,cfg):
    actual=ages(model,opt)
    prefix_steps=sum(active(arm,i) for i in range(total))
    for name,p in model.named_parameters():
        expected=prefix_steps if is_prefix(name,cfg['tiny']) else total
        assert actual[name]==expected,(name,actual[name],expected)
        assert torch.isfinite(p).all()
        if expected==0:assert p not in opt.state and torch.equal(p.detach().cpu(),initial[name]),name
        else:
            for key in ('exp_avg','exp_avg_sq'):assert torch.isfinite(opt.state[p][key]).all()
    for name,value in model.state_dict().items():
        if 'running_' in name or 'num_batches_tracked' in name:assert torch.equal(value.cpu(),initial[name]),name
    return actual


def run_one(root,cfg,seed,arm,position,device):
    parent=root/f'seed_{seed}'/arm;parent.mkdir(parents=True,exist_ok=True)
    if b.verify_complete(parent,cfg):print('SKIP verified',seed,arm,flush=True);return
    attempt=parent/('attempt_'+uuid.uuid4().hex);attempt.mkdir()
    spec=cfg['spec'];b.sync(device);start=time.perf_counter()
    cache=b.data(cfg);model=b.make_model(seed,cfg,device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_hash=b.state_hash(initial);opt=b.optimizer(model,spec,device)
    ids=cache['train'][:spec['batch_size']];xx=b.transform(cache['images'][ids],spec,device);yy=cache['labels'][ids].to(device)
    for i in range(spec['warmup_steps']):update(model,opt,xx,yy,arm,i,cfg,device)
    model.load_state_dict(initial);model.zero_grad(set_to_none=True);opt=b.optimizer(model,spec,device)
    assert b.state_hash(model.state_dict())==initial_hash and len(opt.state)==0
    b.sync(device);setup_seconds=time.perf_counter()-start
    batches=math.ceil(len(cache['train'])/spec['batch_size'])
    steps=[];epochs=[];streams=[];idx=0;training_seconds=validation_seconds=0
    for epoch in range(spec['epochs']):
        b.sync(device);t=time.perf_counter();order,offsets,flips=b.stream(seed,epoch,cache['train'],spec)
        streams.append(dict(epoch=epoch,order=b.tensor_hash(order),offsets=b.tensor_hash(offsets),flips=b.tensor_hash(flips)))
        weighted_loss=seen=0
        for offset in range(0,len(order),spec['batch_size']):
            ids=order[offset:offset+spec['batch_size']]
            x=b.transform(cache['images'][ids],spec,device,offsets[offset:offset+len(ids)],flips[offset:offset+len(ids)])
            y=cache['labels'][ids].to(device);lr=b.factor(idx,batches,spec)
            for group in opt.param_groups:group['lr']=group['base_lr']*lr
            prefix_active=active(arm,idx);loss=update(model,opt,x,y,arm,idx,cfg,device);idx+=1
            steps.append(dict(step=idx,epoch=epoch,examples=len(ids),loss=loss,lr_factor=lr,prefix_active=int(prefix_active)))
            weighted_loss+=loss*len(ids);seen+=len(ids)
        b.sync(device);train_s=time.perf_counter()-t;training_seconds+=train_s
        t=time.perf_counter();common=b.evaluate(model,cache,'common_fp32',cfg,device);b.sync(device)
        val_s=time.perf_counter()-t;validation_seconds+=val_s
        epochs.append(dict(epoch=epoch+1,updates=idx,training_loss=weighted_loss/seen,common_accuracy=common['accuracy'],common_loss=common['loss'],training_seconds=train_s,validation_seconds=val_s))
        print(f'PETS seed={seed} {arm} epoch={epoch+1}/{spec["epochs"]} val={100*common["accuracy"]:.2f}% train_s={training_seconds:.1f}',flush=True)
    t=time.perf_counter();own=b.evaluate(model,cache,'bf16',cfg,device);b.sync(device);validation_seconds+=time.perf_counter()-t
    assert idx==batches*spec['epochs']
    actual=check_state(model,opt,initial,arm,idx,cfg)
    t=time.perf_counter()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),ages=actual,seed=seed,arm=arm,fingerprint=cfg['fingerprint'],initial_hash=initial_hash,updates=idx),attempt/'final.pt')
    np.savez_compressed(attempt/'logits.npz',common=common['logits'].numpy(),own=own['logits'].numpy(),labels=common['labels'].numpy())
    for name,rows in [('steps.csv',steps),('epochs.csv',epochs)]:
        with (attempt/name).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    b.write(attempt/'streams.json',streams);b.sync(device);artifact_seconds=time.perf_counter()-t;complete=time.perf_counter()-start
    b.write(attempt/'summary.json',dict(seed=seed,arm=arm,position=position,fingerprint=cfg['fingerprint'],initial_hash=initial_hash,updates=idx,ages=actual,
        common_accuracy=common['accuracy'],common_loss=common['loss'],own_accuracy=own['accuracy'],own_loss=own['loss'],
        training_seconds=training_seconds,complete_seconds=complete,setup_seconds=setup_seconds,validation_seconds=validation_seconds,artifact_seconds=artifact_seconds,
        other_charged_seconds=complete-setup_seconds-training_seconds-validation_seconds-artifact_seconds,
        prefix_refreshes=sum(active(arm,i) for i in range(idx)),smoke=cfg['smoke'],test_access=False))
    b.write(parent/'COMPLETE.json',dict(fingerprint=cfg['fingerprint'],attempt=attempt.name,sha256={p.name:b.sha(p) for p in attempt.iterdir() if p.is_file()}))
    del model,opt,cache,initial,xx,yy,x,y,common,own
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()


def audit(root,device):
    cfg=b.read(root/'development_freeze.json');spec=cfg['spec']
    assert {p:b.sha(b.ROOT/p) for p in FILES}==cfg['source_hashes']
    assert cfg['fingerprint']==b.digest({k:v for k,v in cfg.items() if k!='fingerprint'})
    if not cfg['smoke']:
        assert b.sha(b.DATA/'trainval_cache.pt')==cfg['inputs']['cache_sha256']
        assert b.sha(b.WEIGHTS)==cfg['inputs']['weights_sha256']
    cache=b.data(cfg);rows=[];rescores=[];stream_records={}
    batches=math.ceil(len(cache['train'])/spec['batch_size']);total=batches*spec['epochs']
    for seed_index,seed in enumerate(spec['seeds']):
        expected_stream=[]
        for epoch in range(spec['epochs']):
            order,offsets,flips=b.stream(seed,epoch,cache['train'],spec)
            expected_stream.append(dict(epoch=epoch,order=b.tensor_hash(order),offsets=b.tensor_hash(offsets),flips=b.tensor_hash(flips)))
        for arm in cfg['plan']['arms']:
            parent=root/f'seed_{seed}'/arm;attempt=b.verify_complete(parent,cfg);assert attempt
            row=b.read(attempt/'summary.json');cp=torch.load(attempt/'final.pt',map_location='cpu',weights_only=True)
            assert row['seed']==cp['seed']==seed and row['arm']==cp['arm']==arm
            assert row['fingerprint']==cp['fingerprint']==cfg['fingerprint']
            assert row['position']==cfg['plan']['orders'][seed_index].index(arm)
            assert row['updates']==cp['updates']==total and not row['test_access']
            assert row['prefix_refreshes']==sum(active(arm,i) for i in range(total))
            model=b.make_model(seed,cfg,device);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            assert b.state_hash(initial)==row['initial_hash']==cp['initial_hash']
            model.load_state_dict(cp['model']);opt=b.optimizer(model,spec,device);opt.load_state_dict(cp['optimizer'])
            assert check_state(model,opt,initial,arm,total,cfg)==cp['ages']==row['ages']
            assert b.read(attempt/'streams.json')==expected_stream
            with (attempt/'steps.csv').open(newline='',encoding='utf8') as f:steps=list(csv.DictReader(f))
            assert len(steps)==total
            for i,s in enumerate(steps):
                assert int(s['step'])==i+1 and int(s['epoch'])==i//batches and int(s['prefix_active'])==active(arm,i)
                assert math.isfinite(float(s['loss'])) and abs(float(s['lr_factor'])-b.factor(i,batches,spec))<1e-12
                assert int(s['examples'])==min(spec['batch_size'],len(cache['train'])-(i%batches)*spec['batch_size'])
            with (attempt/'epochs.csv').open(newline='',encoding='utf8') as f:epochs=list(csv.DictReader(f))
            assert len(epochs)==spec['epochs'] and int(epochs[-1]['updates'])==total
            assert abs(float(epochs[-1]['common_accuracy'])-row['common_accuracy'])<1e-8
            assert abs(sum(float(e['training_seconds']) for e in epochs)-row['training_seconds'])<1e-6
            assert row['complete_seconds']>row['training_seconds']>0 and row['other_charged_seconds']>=-1e-6
            with np.load(attempt/'logits.npz') as z:
                y=z['labels'];assert np.array_equal(y,cache['labels'][cache['validation']].numpy())
                for mode,name in [('common_fp32','common'),('bf16','own')]:
                    x=z[name].astype(np.float64);assert np.isfinite(x).all()
                    acc=float(np.mean(x.argmax(1)==y));loss=float(np.mean(np.logaddexp.reduce(x,axis=1)-x[np.arange(len(y)),y]))
                    assert abs(acc-row[name+'_accuracy'])<1e-7 and abs(loss-row[name+'_loss'])<1e-5
                    pred=b.evaluate(model,cache,mode,cfg,device)['logits'].numpy();error=float(np.max(np.abs(pred-z[name])))
                    assert error<=1e-5 and np.array_equal(pred.argmax(1),z[name].argmax(1)),(seed,arm,mode,error)
                    rescores.append(dict(seed=seed,arm=arm,mode=mode,max_error=error))
            rows.append(row);del model,opt,cp,initial
    pairs=[]
    for seed in spec['seeds']:
        full=next(r for r in rows if r['seed']==seed and r['arm']=='full')
        for arm in ('frozen','periodic_r4'):
            r=next(r for r in rows if r['seed']==seed and r['arm']==arm)
            pairs.append(dict(seed=seed,arm=arm,accuracy_difference_pp=100*(r['common_accuracy']-full['common_accuracy']),
                own_accuracy_difference_pp=100*(r['own_accuracy']-full['own_accuracy']),
                complete_saving_percent=100*(1-r['complete_seconds']/full['complete_seconds']),
                training_saving_percent=100*(1-r['training_seconds']/full['training_seconds'])))
    fragmented=len(b.read(root/'launches.json'))>1 or any(len(list((root/f'seed_{s}'/a).glob('attempt_*')))>1 for s in spec['seeds'] for a in cfg['plan']['arms'])
    result=dict(integrity='PASS',smoke=cfg['smoke'],test_access=False,fragmented=fragmented,rows=rows,pairs=pairs,rescores=rescores,
        paired_accuracy_sd_pp={arm:float(np.std([p['accuracy_difference_pp'] for p in pairs if p['arm']==arm],ddof=1)) for arm in ('frozen','periodic_r4')},
        next_phase='Review development; no automated selection or official-test evaluation')
    b.write(root/'audit.json',result)
    print('AUDIT PASS; trajectories=',len(rows),'rescores=',len(rescores),'fragmented=',fragmented,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['development','audit'])
    parser.add_argument('--device',default='cuda');parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--native',action='store_true')
    args=parser.parse_args();device=torch.device(args.device);b.configure(device)
    root=args.output or REAL
    if args.smoke:assert args.output and root.resolve()!=REAL.resolve()
    elif args.action=='development':assert device.type=='cuda'
    with b.lock(root):
        if args.action=='audit':audit(root,device);return
        cfg=configuration(args.smoke,args.smoke and not args.native,device)
        freeze=root/'development_freeze.json'
        if freeze.exists():assert b.read(freeze)==cfg,'Frozen source/data/runtime differs; preserve results and review'
        else:b.write(freeze,cfg)
        if all(b.verify_complete(root/f'seed_{s}'/a,cfg) for s in cfg['spec']['seeds'] for a in cfg['plan']['arms']):
            audit(root,device);print('All completed trajectories verified; no training rerun.',flush=True);return
        path=root/'launches.json';launches=b.read(path) if path.exists() else []
        launches.append(dict(id=uuid.uuid4().hex,unix_time=time.time()));b.write(path,launches)
        for i,seed in enumerate(cfg['spec']['seeds']):
            for position,arm in enumerate(cfg['plan']['orders'][i]):run_one(root,cfg,seed,arm,position,device)
        audit(root,device)
        print('COMPLETE. Send '+str(root.resolve())+' for review. Official test not accessed.',flush=True)


if __name__=='__main__':main()
