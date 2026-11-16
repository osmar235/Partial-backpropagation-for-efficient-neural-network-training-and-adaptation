"""Matched-gradient AdamW replay diagnostic; no performance/quality benchmark."""
import argparse
import collections
import copy
import math
from pathlib import Path
import os
import torch
import optimized_execution as e

ROOT=e.ROOT;HERE=ROOT/'protocol/optimizer_sensitivity'
SPEC=HERE/'diagnostic_spec_v1.json';FREEZE=HERE/'FROZEN_DIAGNOSTIC_v1.json'
OUT=ROOT/'results/optimizer_sensitivity_v1'
read=e.read;save=e.save

def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)

def sources():
    names=['scripts/optimizer_sensitivity.py','scripts/review_optimizer_sensitivity.py','scripts/test_optimizer_sensitivity.py',
           'scripts/run_optimizer_sensitivity.cmd','protocol/optimizer_sensitivity/diagnostic_spec_v1.json',
           'protocol/optimizer_sensitivity/DIAGNOSTIC_PROTOCOL_v1.md','scripts/stage11_growth_pilot.py','scripts/stage11_growth.py']
    return {**e.sources(),**{str(ROOT/p):e.q.d.sha(ROOT/p) for p in names}}

def configuration(smoke,device):
    e.q.m.configure();spec=read(SPEC)
    if smoke:spec.update(initialized_seeds=[99301],trained_seeds=[99302],batch_size=3,top_coordinates=8)
    cfg=e.q.freeze(spec,smoke,device);cfg['sources']=sources();cfg['checkpoints']={};cfg['evidence']={}
    if not smoke:
        path=ROOT/'protocol/qkv_training/FROZEN_TRAINING_v1.json';old=read(path)
        cfg['evidence'][str(path)]=e.q.d.sha(path);cfg['checkpoints']=old['checkpoints'];cfg['expected_split']=old['split']
        for record in cfg['checkpoints'].values():
            assert e.q.d.sha(record['path'])==record['sha256'];cfg['evidence'][record['path']]=record['sha256']
        for name in ['batches.meta']+[f'data_batch_{i}' for i in range(1,6)]:
            p=ROOT/'data/cifar-10-batches-py'/name;cfg['evidence'][str(p)]=e.q.d.sha(p)
    cfg.pop('fingerprint');cfg['fingerprint']=e.q.v.digest(cfg);return cfg

def trajectories(cfg):
    spec=cfg['spec'];rows=[]
    for kind,seeds in [('initialized',spec['initialized_seeds']),('trained',spec['trained_seeds'])]:
        for seed in seeds:
            for domain in (['synthetic'] if kind=='initialized' or cfg['smoke'] else ['synthetic','training']):
                rows.append(dict(kind=kind,seed=seed,domain=domain,key=f'{kind}_{seed}_{domain}'))
    return rows

def inputs(task,cfg,device):
    spec=cfg['spec'];batch=spec['batch_size']
    if task['domain']=='synthetic':
        x,y=e.prior.inputs(task['seed'],batch,spec,device);return x,y,dict(kind='synthetic',seed=task['seed'])
    train=e.q.v.CIFAR10(str(ROOT/'data'),train=True,download=False)
    tr,va=e.q.v.split_indices(train.targets)
    split=dict(kind='train_only_stratified_45000_5000',train_index_sha256=e.q.tensor_hash(tr),validation_index_sha256=e.q.tensor_hash(va))
    assert split==cfg['expected_split']
    gen=torch.Generator().manual_seed(spec['training_batch_seed']);indices=tr[torch.randperm(len(tr),generator=gen)[:batch]]
    assert not set(indices.tolist()).intersection(va.tolist())
    x=torch.from_numpy(train.data[indices.numpy()]).permute(0,3,1,2).float()/255
    x=e.q.v.frozen6.normalize_cifar(x).to(device);y=torch.tensor(train.targets)[indices].to(device)
    return x,y,dict(kind='training_only',indices=indices.tolist(),split=split,augmentation=False,mixup=False)

def optimizer(parameters,spec):
    return torch.optim.AdamW(parameters,lr=spec['learning_rate'],betas=tuple(spec['betas']),eps=spec['epsilon'],weight_decay=spec['weight_decay'])

def validate_state(state,weights,age=None):
    group=state['param_groups'];assert len(group)==1 and group[0]['params']==list(range(len(weights)))
    assert not group[0]['amsgrad'] and not group[0]['maximize']
    assert not state['state'] or set(state['state'])==set(range(len(weights)))
    for i,(name,value) in enumerate(weights.items()):
        if i not in state['state']:continue
        entry=state['state'][i]
        assert set(entry)=={'step','exp_avg','exp_avg_sq'}
        if age is not None:assert int(entry['step'])==age
        for key in ('exp_avg','exp_avg_sq'):
            assert entry[key].shape==value.shape and entry[key].dtype==torch.float32 and torch.isfinite(entry[key]).all(),name
        assert (entry['exp_avg_sq']>=0).all()

def opt_hash(state):
    return e.q.v.digest(dict(groups=state['param_groups'],state={str(i):{k:e.q.tensor_hash(v) for k,v in s.items()} for i,s in state['state'].items()}))

def replay(weights,state,gradient,spec,device):
    assert list(weights)==list(gradient);validate_state(state,weights)
    params=[torch.nn.Parameter(v.to(device).clone()) for v in weights.values()]
    opt=optimizer(params,spec);opt.load_state_dict(cpu(state))
    before=opt_hash(cpu(opt.state_dict()));assert before==opt_hash(state)
    for p,g in zip(params,gradient.values()):
        assert g.dtype==torch.float32 and torch.isfinite(g).all();p.grad=g.to(device).clone()
    with e.policy('reference_fp32_math'):opt.step()
    after=cpu(opt.state_dict());delta={n:p.detach().cpu()-v for (n,v),p in zip(weights.items(),params)}
    for i,p in enumerate(params):
        assert int(after['state'][i]['step'])==(int(state['state'][i]['step']) if state['state'] else 0)+1
        assert torch.isfinite(p).all() and torch.isfinite(after['state'][i]['exp_avg']).all() and torch.isfinite(after['state'][i]['exp_avg_sq']).all()
    return dict(delta=delta,start_model_sha256=e.q.old.state_hash(weights),start_optimizer_sha256=before,
                end_optimizer_sha256=opt_hash(after),optimizer_age=int(after['state'][0]['step'])),after,{n:cpu(p) for n,p in zip(weights,params)}

def capture(weights,task,arm,cfg,x,y,device):
    reason=e.supported(arm,device)
    if reason:return dict(status='UNSUPPORTED',reason=reason)
    model=e.make(task['seed'],arm,cfg,device,qualification=True);model.load_state_dict(weights,strict=True);model.zero_grad(set_to_none=True)
    assert list(dict(model.named_parameters()))==list(weights)
    try:
        with e.policy(arm):
            torch.manual_seed(task['seed']+cfg['spec']['dropout_seed_offset']);flags=e.settings()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                logits,loss=e.forward_loss(model,x,y,e.properties(arm)[1],cfg['spec']);loss.backward();e.q.v.sync(device)
        gradients={n:cpu(p.grad) for n,p in model.named_parameters()}
        if not bool(torch.isfinite(loss)) or not all(g is not None and g.dtype==torch.float32 and torch.isfinite(g).all() for g in gradients.values()):
            return dict(status='NONFINITE')
        counts=collections.Counter(event.name for event in prof.events());dispatch={name:counts[op] for name,op in e.q.frozen_backend.OPS.items()}
        assert sum(dispatch.values())==(2 if cfg['smoke'] else 6)
        if arm.endswith('_math'):assert dispatch['math']==sum(dispatch.values())
        return dict(status='FINITE',gradient=gradients,logits=cpu(logits.float()),loss=float(loss.detach()),dispatch=dispatch,policy=flags)
    except RuntimeError as error:
        if not (e.q.frozen_backend.unsupported(error) or 'cudnn_status_not_supported' in str(error).lower()):raise
        return dict(status='UNSUPPORTED',reason=str(error))

def load_origin(task,cfg,x,y,device):
    model=e.make(task['seed'],'native_fp32_math',cfg,torch.device('cpu'),qualification=True)
    weights=cpu(model.state_dict());fresh=cpu(optimizer(model.parameters(),cfg['spec']).state_dict())
    if task['kind']=='initialized':return weights,fresh,None,dict(kind='initialized')
    if cfg['smoke']:
        # Artificial saved-state fixture only: never described as trained evidence.
        history=fresh
        for step in range(2):
            cap=capture(weights,task,'reference_fp32_math',cfg,x,y,device);assert cap['status']=='FINITE'
            rec,history,weights=replay(weights,history,cap['gradient'],cfg['spec'],device)
        return weights,fresh,history,dict(kind='artificial_trained_fixture',age=2)
    record=cfg['checkpoints'][str(task['seed'])];assert e.q.d.sha(record['path'])==record['sha256']
    cp=torch.load(record['path'],map_location='cpu',weights_only=False)
    assert cp['seed']==task['seed'] and cp['variant']=='full'
    for key in ('root_fingerprint','run_fingerprint','updates'):assert cp[key]==record[key]
    model.load_state_dict(cp['model_state'],strict=True)
    assert list(cp['model_state'])==list(dict(model.named_parameters()))
    weights=cpu(model.state_dict());history=cpu(cp['optimizer_state']);validate_state(history,weights,record['updates'])
    original=cpu(history['param_groups']);canonical=cpu(fresh['param_groups'])
    for key in canonical[0]:
        if key!='lr':assert original[0][key]==canonical[0][key],key
    history['param_groups']=canonical
    return weights,fresh,history,dict(kind='archived_trained',checkpoint=record,original_optimizer_groups=original,diagnostic_optimizer_groups=canonical)

def analytic(theta,g,m,v,step,group):
    b1,b2=group['betas'];age=step+1
    mm=b1*m+(1-b1)*g;vv=b2*v+(1-b2)*g*g
    return -group['lr']*group['weight_decay']*theta-group['lr']*(mm/(1-b1**age))/(math.sqrt(vv/(1-b2**age))+group['eps'])

def analyze(weights,state,ref,candidate,ref_delta,delta,spec):
    gradient=e.aggregate(ref,candidate);updates=e.aggregate(ref_delta,delta)
    boundaries=spec['gradient_bin_upper_bounds'];bins=[dict(lower=0 if i==0 else boundaries[i-1],upper=boundaries[i] if i<len(boundaries) else None,
        coordinates=0,sign_flips=0,zero_disagreements=0,update_difference_squared=0.) for i in range(len(boundaries)+1)]
    top=[];total=flips=zeros=0
    for index,name in enumerate(weights):
        a=ref[name].flatten().double();b=candidate[name].flatten().double();u=ref_delta[name].flatten().double();v=delta[name].flatten().double();diff=(u-v).abs()
        sign=(a*b)<0;zero=(a==0)!=(b==0);total+=len(a);flips+=int(sign.sum());zeros+=int(zero.sum())
        bucket=torch.bucketize(a.abs(),torch.tensor(boundaries,dtype=torch.float64),right=True)
        for i,entry in enumerate(bins):
            mask=bucket==i;entry['coordinates']+=int(mask.sum());entry['sign_flips']+=int(sign[mask].sum());entry['zero_disagreements']+=int(zero[mask].sum())
            entry['update_difference_squared']+=float((diff[mask]**2).sum())
        count=min(spec['top_coordinates'],len(a));indices=torch.argsort(diff,descending=True,stable=True)[:count]
        for j in indices.tolist():top.append((float(diff[j]),name,j,index))
    top.sort(key=lambda z:(-z[0],z[3],z[2]));coordinates=[];group=state['param_groups'][0]
    for difference,name,j,index in top[:spec['top_coordinates']]:
        entry=state['state'].get(index);step=int(entry['step']) if entry else 0
        m=float(entry['exp_avg'].flatten()[j]) if entry else 0.;v=float(entry['exp_avg_sq'].flatten()[j]) if entry else 0.
        theta=float(weights[name].flatten()[j]);a=float(ref[name].flatten()[j]);b=float(candidate[name].flatten()[j])
        u=float(ref_delta[name].flatten()[j]);w=float(delta[name].flatten()[j])
        coordinates.append(dict(name=name,flat_index=j,parameter=theta,reference_gradient=a,candidate_gradient=b,initial_first_moment=m,initial_second_moment=v,
            initial_step=step,reference_delta=u,candidate_delta=w,absolute_delta_difference=difference,
            reference_analytic_delta=analytic(theta,a,m,v,step,group),candidate_analytic_delta=analytic(theta,b,m,v,step,group)))
    mass=sum(z['update_difference_squared'] for z in bins)
    for z in bins:z['update_difference_mass_fraction']=z['update_difference_squared']/mass if mass else 0.
    return dict(gradient=gradient,update=updates,coordinates=total,strict_sign_flips=flips,zero_disagreements=zeros,
        sign_flip_fraction=flips/total,bins=bins,top_coordinates=coordinates)

def trajectory(parent,task,cfg,device):
    parent.mkdir(parents=True,exist_ok=True);fp=e.q.v.digest(dict(root=cfg['fingerprint'],task=task))
    if e.q.d.validate_complete(parent,fp):print('SKIP',task['key'],flush=True);return
    attempt=e.q.d.next_attempt(parent);save(attempt/'process.json',dict(pid=os.getpid(),task=task,root_fingerprint=cfg['fingerprint']))
    x,y,provenance=inputs(task,cfg,device);e.q.old.checkpoint(attempt/'inputs.pt',dict(x=cpu(x),y=cpu(y),provenance=provenance))
    weights,fresh,archived,source=load_origin(task,cfg,x,y,device);save(attempt/'source.json',source)
    history=cpu(fresh);spec=cfg['spec'];reference=spec['reference'];bridge={}
    for anchor in spec['anchors']:
        if anchor==2:
            mid=capture(weights,task,reference,cfg,x,y,device);assert mid['status']=='FINITE'
            bridge['intermediate_gradient']=mid['gradient']
            rec,history,weights=replay(weights,history,mid['gradient'],spec,device)
            bridge['second_step']=rec;bridge['second_weights']=cpu(weights)
        contexts={'fresh':cpu(fresh)}
        if anchor==0 and archived is not None:contexts['archived']=cpu(archived)
        if anchor==2:contexts['retained_two_updates']=cpu(history)
        captures={arm:capture(weights,task,arm,cfg,x,y,device) for arm in spec['arms']}
        ref=captures[reference];assert ref['status']=='FINITE'
        repeat=capture(weights,task,reference,cfg,x,y,device)
        assert repeat['status']=='FINITE' and ref['loss']==repeat['loss'] and torch.equal(ref['logits'],repeat['logits'])
        assert all(torch.equal(g,repeat['gradient'][n]) for n,g in ref['gradient'].items())
        records={};summaries=[]
        for context,state in contexts.items():
            records[context]={}
            for arm,cap in captures.items():
                if cap['status']!='FINITE':continue
                record,_,_=replay(weights,state,cap['gradient'],spec,device);records[context][arm]=record
            for arm,record in records[context].items():
                summary=analyze(weights,state,ref['gradient'],captures[arm]['gradient'],records[context][reference]['delta'],record['delta'],spec)
                summaries.append(dict(context=context,arm=arm,**summary))
                print('DIAGNOSTIC',task['key'],'anchor',anchor,context,arm,'max_update_difference',summary['update']['max_abs'],flush=True)
        raw=dict(task=task,anchor=anchor,weights=weights,contexts=contexts,captures=captures,replays=records,reference_repeat_exact=True)
        e.q.old.checkpoint(attempt/f'anchor_{anchor}.pt',raw);save(attempt/f'anchor_{anchor}_summary.json',summaries)
        if anchor==0:
            rec,history,weights=replay(weights,history,ref['gradient'],spec,device)
            bridge['first_step']=rec;bridge['first_weights']=cpu(weights)
    e.q.old.checkpoint(attempt/'trajectory_bridge.pt',bridge)
    e.prior.complete(parent,fp,attempt)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--tag',default='preparation1');ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
    assert args.tag.replace('_','').isalnum();device=torch.device(args.device);assert args.smoke or device.type=='cuda'
    cfg=configuration(args.smoke,device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();e.q.old.write_freeze(FREEZE,cfg);print('FROZEN',cfg['fingerprint']);return
    out=ROOT/f'results/optimizer_sensitivity_smoke_{args.tag}_{device.type}' if args.smoke else OUT
    if not args.smoke:assert cfg==read(FREEZE),'Frozen source/spec/evidence/environment changed'
    out.mkdir(exist_ok=True);e.q.old.write_freeze(out/'configuration.json',cfg)
    from review_optimizer_sensitivity import audit
    if e.q.d.validate_complete(out,cfg['fingerprint']):print(audit(out));return
    for task in trajectories(cfg):trajectory(out/task['key'],task,cfg,device)
    result=audit(out,write=True);e.prior.complete(out,cfg['fingerprint']);print(result);print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
