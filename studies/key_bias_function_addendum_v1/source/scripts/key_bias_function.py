"""Saved-update component swaps, with matched forward-only evaluations."""
import argparse
import collections
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import optimizer_sensitivity as d

e=d.e;ROOT=d.ROOT;HERE=ROOT/'protocol/key_bias_function'
SPEC=HERE/'functional_spec_v1.json';FREEZE=HERE/'FROZEN_FUNCTIONAL_v1.json';OUT=ROOT/'results/key_bias_function_v1'
read=d.read;save=d.save;cpu=d.cpu

def sources():
    names=['scripts/key_bias_function.py','scripts/review_key_bias_function.py','scripts/test_key_bias_function.py','scripts/run_key_bias_function.cmd',
           'protocol/key_bias_function/functional_spec_v1.json','protocol/key_bias_function/FUNCTIONAL_PROTOCOL_v1.md']
    return {**d.sources(),**{str(ROOT/p):e.q.d.sha(ROOT/p) for p in names}}

def configuration(smoke,device):
    e.q.m.configure();spec=read(SPEC)
    source=ROOT/f'results/optimizer_sensitivity_smoke_preparation1_{device.type}' if smoke else d.OUT
    old=read(source/'configuration.json');assert old['smoke']==smoke and old['device']==str(device)
    assert e.q.d.validate_complete(source,old['fingerprint'])
    for p,h in old['sources'].items():assert e.q.d.sha(p)==h,p
    if smoke:spec.update(trained_seeds=old['spec']['trained_seeds'],batch_size=old['spec']['batch_size'])
    cfg=e.q.freeze(spec,smoke,device);cfg['sources']=sources();cfg['parent_config']=old;cfg['records']=[]
    cfg['evidence']={str(source/'configuration.json'):e.q.d.sha(source/'configuration.json'),str(source/'COMPLETE.json'):e.q.d.sha(source/'COMPLETE.json'),**old['evidence']}
    for seed in spec['trained_seeds']:
        task=f'trained_{seed}_'+('synthetic' if smoke else 'training');parent=source/task
        marker=read(parent/'COMPLETE.json');assert e.q.d.validate_complete(parent,marker['fingerprint'])
        cfg['evidence'][str(parent/'COMPLETE.json')]=e.q.d.sha(parent/'COMPLETE.json');cfg['evidence'].update(marker['sha256'])
        attempt=Path(marker['attempt'])
        for anchor in spec['anchors']:cfg['records'].append(dict(seed=seed,anchor=anchor,key=f'{task}_anchor_{anchor}',raw_path=str(attempt/f'anchor_{anchor}.pt'),inputs_path=str(attempt/'inputs.pt')))
    for p,h in cfg['evidence'].items():assert e.q.d.sha(p)==h,p
    cfg.pop('fingerprint');cfg['fingerprint']=e.q.v.digest(cfg);return cfg

def batches(record,cfg):
    original=torch.load(record['inputs_path'],map_location='cpu',weights_only=False)
    task=dict(kind='trained',seed=record['seed'],domain='synthetic' if cfg['smoke'] else 'training')
    x,y,provenance=d.inputs(task,cfg['parent_config'],torch.device('cpu'))
    assert torch.equal(original['x'],x) and torch.equal(original['y'],y) and original['provenance']==provenance
    if cfg['smoke']:
        gen=torch.Generator().manual_seed(cfg['spec']['second_batch_seed']);n=cfg['spec']['batch_size']
        second=dict(x=torch.randn(n,3,32,32,generator=gen),y=torch.randint(0,10,(n,),generator=gen),provenance=dict(kind='artificial_second'))
    else:
        original_indices=original['provenance']['indices'];assert len(set(original_indices))==cfg['spec']['batch_size']
        train=e.q.v.CIFAR10(str(ROOT/'data'),train=True,download=False);tr,va=e.q.v.split_indices(train.targets)
        split=dict(kind='train_only_stratified_45000_5000',train_index_sha256=e.q.tensor_hash(tr),validation_index_sha256=e.q.tensor_hash(va))
        assert split==cfg['parent_config']['expected_split']==original['provenance']['split']
        excluded=set(original_indices);assert excluded.issubset(set(tr.tolist()))
        remaining=torch.tensor([i for i in tr.tolist() if i not in excluded]);gen=torch.Generator().manual_seed(cfg['spec']['second_batch_seed'])
        indices=remaining[torch.randperm(len(remaining),generator=gen)[:cfg['spec']['batch_size']]]
        assert not set(indices.tolist()).intersection(excluded|set(va.tolist()))
        x=torch.from_numpy(train.data[indices.numpy()]).permute(0,3,1,2).float()/255
        second=dict(x=e.q.v.frozen6.normalize_cifar(x),y=torch.tensor(train.targets)[indices],provenance=dict(kind='disjoint_training_only',indices=indices.tolist(),split=split))
    return dict(original=original,second=second)

def swapped(reference,candidate):
    assert list(reference)==list(candidate)
    key_only=cpu(reference);key_restored=cpu(candidate);count=0
    for name,value in reference.items():
        assert value.shape==candidate[name].shape and value.dtype==candidate[name].dtype==torch.float32
        if name.endswith('self_attention.in_proj_bias'):
            assert value.ndim==1 and len(value)%3==0;width=len(value)//3;region=slice(width,2*width);count+=width
            key_only[name][region]=candidate[name][region];key_restored[name][region]=value[region]
            for outside in (slice(0,width),slice(2*width,None)):
                assert torch.equal(key_only[name][outside],value[outside]) and torch.equal(key_restored[name][outside],candidate[name][outside])
        else:
            assert torch.equal(key_only[name],value) and torch.equal(key_restored[name],candidate[name])
    assert count>0
    return dict(key_only=key_only,key_restored=key_restored),count

def recovered(raw,context,cfg,device):
    states={};proof={};arms=['reference_fp32_math',*cfg['spec']['candidate_arms']]
    for arm in arms:
        if arm not in raw['replays'][context]:continue
        rec,_,weights=d.replay(raw['weights'],raw['contexts'][context],raw['captures'][arm]['gradient'],cfg['parent_config']['spec'],device)
        old=raw['replays'][context][arm]
        for key in ('start_model_sha256','start_optimizer_sha256','end_optimizer_sha256','optimizer_age'):assert rec[key]==old[key]
        assert all(torch.equal(t,old['delta'][n]) for n,t in rec['delta'].items())
        states[arm]=weights;proof[arm]=dict(model_sha256=e.q.old.state_hash(weights),saved_delta_exact=True,end_optimizer_sha256=rec['end_optimizer_sha256'])
    assert 'reference_fp32_math' in states
    return states,proof

def evaluate(weights,record,policy,cfg,batch,device):
    reason=e.supported(policy,device)
    if reason:return dict(status='UNSUPPORTED',reason=reason)
    model=e.make(record['seed'],policy,dict(spec=cfg['parent_config']['spec'],smoke=cfg['smoke']),device,qualification=True)
    model.load_state_dict(weights,strict=True);model.train();x=batch['x'].to(device);y=batch['y'].to(device)
    try:
        with e.policy(policy),torch.no_grad():
            flags=e.settings()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                logits,loss=e.forward_loss(model,x,y,e.properties(policy)[1],cfg['spec']);e.q.v.sync(device)
            # Score the exact returned logits in explicit FP32, outside autocast.
            scoring_loss=F.cross_entropy(logits.float(),y,label_smoothing=cfg['spec']['label_smoothing'])
        assert all(p.grad is None for p in model.parameters())
        if not torch.isfinite(logits).all() or not torch.isfinite(loss) or not torch.isfinite(scoring_loss):return dict(status='NONFINITE')
        counts=collections.Counter(event.name for event in prof.events());dispatch={name:counts[op] for name,op in e.q.frozen_backend.OPS.items()}
        assert sum(dispatch.values())==(2 if cfg['smoke'] else 6)
        if policy.endswith('_math'):assert dispatch['math']==sum(dispatch.values())
        return dict(status='FINITE',logits=cpu(logits.float()),execution_loss=float(loss),scoring_loss=float(scoring_loss),policy=flags,dispatch=dispatch,model_sha256=e.q.old.state_hash(weights))
    except RuntimeError as error:
        if not (e.q.frozen_backend.unsupported(error) or 'cudnn_status_not_supported' in str(error).lower()):raise
        return dict(status='UNSUPPORTED',reason=str(error))

def score(output,labels,spec):
    logits=output['logits'].double();loss=float(F.cross_entropy(logits,labels,label_smoothing=spec['label_smoothing']))
    assert math.isfinite(output['execution_loss'])
    assert abs(loss-output['scoring_loss'])<=spec['score_reconstruction_atol']+spec['score_reconstruction_rtol']*abs(loss)
    return dict(loss=loss,execution_loss=output['execution_loss'],execution_minus_uniform_loss=output['execution_loss']-loss,
                correct=int((logits.argmax(1)==labels).sum()),examples=len(labels))

def difference(a,b,labels,spec):
    # b minus a, preserving absolute and relative magnitudes.
    x=a['logits'].double();y=b['logits'].double();delta=y-x
    aa=score(a,labels,spec);bb=score(b,labels,spec);norm=float(torch.linalg.vector_norm(delta))
    return dict(logit_max_abs=float(delta.abs().max()),logit_l2=norm,logit_rms=float(torch.sqrt((delta*delta).mean())),
        logit_relative_l2=norm/max(float(torch.linalg.vector_norm(x)),1e-12),loss_change=bb['loss']-aa['loss'],
        prediction_disagreements=int((x.argmax(1)!=y.argmax(1)).sum()),correct_change=bb['correct']-aa['correct'])

def summarize(packet,cfg):
    rows=[];runtime=[];repeats=[];spec=cfg['spec']
    for context,blocks in packet['contexts'].items():
        for batch_name,policies in blocks['outputs'].items():
            y=packet['batches'][batch_name]['y']
            fp32=policies['reference_fp32_math']['reference']
            for policy,block in policies.items():
                ref=block['reference'];again=block['repeat'];unchanged=packet['unchanged'][batch_name][policy]
                if ref['status']!='FINITE' or again['status']!='FINITE':continue
                repeat=difference(ref,again,y,spec);repeats.append(dict(context=context,batch=batch_name,policy=policy,**repeat))
                if policy.endswith('_math'):assert torch.equal(ref['logits'],again['logits']) and ref['execution_loss']==again['execution_loss']
                if fp32['status']=='FINITE':runtime.append(dict(context=context,batch=batch_name,policy=policy,**difference(fp32,ref,y,spec)))
                for arm,variants in block['candidates'].items():
                    if any(v['status']!='FINITE' for v in variants.values()) or unchanged['status']!='FINITE':continue
                    full=difference(ref,variants['candidate'],y,spec);key=difference(ref,variants['key_only'],y,spec);rest=difference(ref,variants['key_restored'],y,spec)
                    interaction=variants['candidate']['logits'].double()-variants['key_only']['logits'].double()-variants['key_restored']['logits'].double()+ref['logits'].double()
                    rows.append(dict(context=context,batch=batch_name,policy=policy,arm=arm,full=full,key_only=key,key_restored=rest,
                        interaction_logit_l2=float(torch.linalg.vector_norm(interaction)),interaction_logit_max_abs=float(interaction.abs().max()),
                        interaction_loss_change=full['loss_change']-key['loss_change']-rest['loss_change'],
                        key_to_full_logit_l2=key['logit_l2']/full['logit_l2'] if full['logit_l2'] else None,
                        reference_from_unchanged=difference(unchanged,ref,y,spec),candidate_from_unchanged=difference(unchanged,variants['candidate'],y,spec),
                        reference_score=score(ref,y,spec),candidate_score=score(variants['candidate'],y,spec),
                        key_only_score=score(variants['key_only'],y,spec),key_restored_score=score(variants['key_restored'],y,spec)))
    return dict(comparisons=rows,runtime_reference_differences=runtime,repeat_differences=repeats)

def run_one(out,record,cfg,device):
    parent=out/record['key'];parent.mkdir(exist_ok=True);fp=e.q.v.digest(dict(root=cfg['fingerprint'],record=record))
    if e.q.d.validate_complete(parent,fp):print('SKIP',record['key'],flush=True);return
    attempt=e.q.d.next_attempt(parent);save(attempt/'process.json',dict(pid=os.getpid(),root_fingerprint=cfg['fingerprint'],record=record))
    raw=torch.load(record['raw_path'],map_location='cpu',weights_only=False);data=batches(record,cfg)
    packet=dict(record=record,batches=data,unchanged={},contexts={})
    for batch_name,batch in data.items():
        packet['unchanged'][batch_name]={p:evaluate(raw['weights'],record,p,cfg,batch,device) for p in cfg['spec']['evaluation_policies']}
    for context in raw['contexts']:
        states,proof=recovered(raw,context,cfg,device);reference=states['reference_fp32_math'];case=dict(proof=proof,swaps={},outputs={})
        variants={}
        for arm in cfg['spec']['candidate_arms']:
            if arm not in states:continue
            components,count=swapped(reference,states[arm]);assert count==(48 if cfg['smoke'] else 1152)
            variants[arm]=dict(candidate=states[arm],**components)
            case['swaps'][arm]=dict(key_coordinates=count,hashes={k:e.q.old.state_hash(v) for k,v in variants[arm].items()})
        for batch_name,batch in data.items():
            case['outputs'][batch_name]={}
            for policy in cfg['spec']['evaluation_policies']:
                ref=evaluate(reference,record,policy,cfg,batch,device)
                results={arm:{name:evaluate(weights,record,policy,cfg,batch,device) for name,weights in group.items()} for arm,group in variants.items()}
                repeat=evaluate(reference,record,policy,cfg,batch,device)
                case['outputs'][batch_name][policy]=dict(reference=ref,candidates=results,repeat=repeat)
                print('FORWARD',record['key'],context,batch_name,policy,ref['status'],flush=True)
        packet['contexts'][context]=case
    e.q.old.checkpoint(attempt/'outputs.pt',packet);save(attempt/'summary.json',summarize(packet,cfg));e.prior.complete(parent,fp,attempt)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--tag',default='preparation1');ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
    assert args.tag.replace('_','').isalnum();device=torch.device(args.device);assert args.smoke or device.type=='cuda';cfg=configuration(args.smoke,device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists();e.q.old.write_freeze(FREEZE,cfg);print('FROZEN',cfg['fingerprint']);return
    out=ROOT/f'results/key_bias_function_smoke_{args.tag}_{device.type}' if args.smoke else OUT
    if not args.smoke:assert cfg==read(FREEZE),'Frozen source/evidence/environment changed'
    out.mkdir(exist_ok=True);e.q.old.write_freeze(out/'configuration.json',cfg)
    from review_key_bias_function import audit
    if e.q.d.validate_complete(out,cfg['fingerprint']):print(audit(out));return
    for record in cfg['records']:run_one(out,record,cfg,device)
    result=audit(out,write=True);e.prior.complete(out,cfg['fingerprint']);print(result);print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
