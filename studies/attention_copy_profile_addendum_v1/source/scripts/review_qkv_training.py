"""Reconstruct learning streams, rescore final checkpoints and audit paired timing."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from review_qkv_confirmation import statistics

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)

def paired_identity(a,b):
    fields=('initial_model_sha256','initial_optimizer_sha256','final_model_sha256','final_optimizer_sha256','final_global_rng')
    return (all(a['summary'][k]==b['summary'][k] for k in fields)
        and a['steps']==b['steps']
        and a['stream_hash']==b['stream_hash']
        and all(np.array_equal(a['predictions'][k],b['predictions'][k]) for k in a['predictions']))

def qualified(values,identity,reference,unfragmented,smoke):
    stats=statistics(values,2)
    return dict(statistics=stats,identity_pass=identity,reference_pass=reference,
        unfragmented=unfragmented,confirmed=bool(not smoke and identity and reference and unfragmented and stats['directional_pass']))

def manifest(parent,fingerprint,expected=None):
    import qkv_training as e
    marker=e.r.d.validate_complete(parent,fingerprint);assert marker
    if expected is not None:assert set(marker['sha256'])=={str(p) for p in expected}
    return marker

def audit(out,write=False):
    import qkv_training as e
    r=e.r;out=Path(out).resolve(strict=True);cfg=read(out/'configuration.json');spec=cfg['spec']
    assert cfg['fingerprint']==r.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})
    assert cfg['sources']==e.sources();r.t.verify(cfg['evidence'])
    if not cfg['smoke']:assert cfg==read(e.FREEZE)
    if not write:
        manifest(out,cfg['fingerprint'],[out/p for p in ('configuration.json','audit.json','pairings.json','REPORT.md')]+list(out.glob('session_*/COMPLETE.json')))
    device=torch.device(cfg['device']);assert device.type=='cpu' or torch.cuda.is_available()
    e.q.policy()
    data=r.v.load_data('development',torch.device('cpu'),fake=cfg['smoke'])
    _,y,ev,ey,split=data;assert split==cfg['split'];labels=ey.numpy();ev=ev.to(device)
    del data
    references={}
    if not cfg['smoke']:
        for seed in spec['seeds']:
            path=Path(cfg['references'][str(seed)])
            with np.load(path/'predictions.npz',allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
            assert np.array_equal(pred['labels'],labels)
            with np.load(cfg['checkpoints'][str(seed)]['predictions_path'],allow_pickle=False) as z:source=z['full'].copy()
            references[seed]=dict(steps=r.d.records(path/'steps.csv'),predictions=pred,source=source)
    all_records={};pairings=[];fragmented=[];uuids=[]
    for session in range(spec['sessions']):
        folder=out/f'session_{session:02d}'
        paths=list(folder.glob('seed_*/*/COMPLETE.json'))+list(folder.glob('seed_*/*/run_config.json'))+list((folder/'launches').rglob('*.json'))
        manifest(folder,cfg['fingerprint'],paths)
        launches=sorted((folder/'launches').glob('attempt_*'));assert launches
        fragmented.append(len(launches)!=1)
        last=launches[-1];uuids.append(read(last/'process.json')['uuid'])
        if not cfg['smoke']:
            samples=read(last/'preflight.json');assert len(samples)==3 and all(z['available'] for z in samples)
            for gpu in range(len(samples[0]['gpus'])):
                assert sorted(float(z['gpus'][gpu]['utilization.gpu']) for z in samples)[1]<=20
        for position,(seed,arm) in enumerate(e.schedule(spec,session)):
            trace=read(last/f'trajectory_{position:02d}.json');assert (trace['seed'],trace['arm'])==(seed,arm)
            parent=folder/f'seed_{seed}'/arm;rc=read(parent/'run_config.json')
            assert rc['fingerprint']==r.v.digest({k:v for k,v in rc.items() if k!='fingerprint'})
            assert (rc['root_fingerprint'],rc['session'],rc['seed'],rc['arm'],rc['position'])==(cfg['fingerprint'],session,seed,arm,position)
            marker=manifest(parent,rc['fingerprint']);attempt=Path(marker['attempt']).resolve(strict=True);assert attempt.parent==parent
            assert set(marker['sha256'])=={str(p) for p in attempt.iterdir() if p.is_file()}
            if len(list(parent.glob('attempt_*')))!=1:fragmented[-1]=True
            summary=read(attempt/'summary.json');steps=r.d.records(attempt/'steps.csv');curves=r.d.records(attempt/'curves.csv')
            total=spec['epochs']*spec['expected_batches_per_epoch'];assert summary['updates']==len(steps)==total
            assert (summary['session'],summary['seed'],summary['arm'],summary['position'])==(session,seed,arm,position)
            components=('data_seconds','setup_seconds','warmup_restore_seconds','training_seconds','validation_seconds','artifacts_seconds','residual_seconds')
            assert all(math.isfinite(summary[k]) and summary[k]>=0 for k in components)
            close(summary['complete_workflow_seconds'],sum(summary[k] for k in components))
            assert summary['complete_workflow_seconds']>0 and summary['all_full_gradients_finite']
            assert summary['peak_allocated']>=summary['baseline_allocated']>=0
            assert summary['incremental_allocated']==summary['peak_allocated']-summary['baseline_allocated']
            streams=torch.load(attempt/'streams.pt',map_location='cpu',weights_only=False)
            bg,ag,mg,pg=r.previous.generators(seed,spec);idx=0
            assert len(streams['epoch_orders'])==spec['epochs']
            for epoch in range(1,spec['epochs']+1):
                order=torch.randperm(len(y),generator=bg);assert torch.equal(order,streams['epoch_orders'][epoch-1])
                for start in range(0,len(y),spec['batch_size']):
                    row=steps[idx];n=min(spec['batch_size'],len(y)-start)
                    assert (int(row['epoch']),int(row['step']),int(row['batch_examples']))==(epoch,idx+1,n)
                    torch.randint(0,9,(n,),generator=ag);torch.randint(0,9,(n,),generator=ag);torch.rand(n,generator=ag)
                    close(row['mixup_lambda'],mg.beta(spec['mixup_alpha'],spec['mixup_alpha']));torch.randperm(n,generator=pg)
                    close(row['lr'],r.v.learning_rate(idx,total,spec['expected_batches_per_epoch'],spec['learning_rate'],spec))
                    assert int(row['dropout_seed'])==seed+spec['dropout_seed_offset']+idx
                    assert math.isfinite(float(row['loss'])) and float(row['loss'])>0;idx+=1
            cp=torch.load(attempt/'final_checkpoint.pt',map_location='cpu',weights_only=False)
            assert (cp['seed'],cp['arm'],cp['updates'],cp['root_fingerprint'],cp['run_fingerprint'])==(seed,arm,total,cfg['fingerprint'],rc['fingerprint'])
            for k,expected in e.stream_states(bg,ag,mg,pg).items():
                if torch.is_tensor(expected):assert torch.equal(cp[k],expected) and torch.equal(streams[k],expected)
                else:assert cp[k]==streams[k]==expected
            stream_hash=r.v.digest({k:e.q.tensor_hash(v) if torch.is_tensor(v) else v for k,v in e.stream_states(bg,ag,mg,pg).items()})
            assert streams['final_global_rng']==summary['final_global_rng']
            model,opt,_=e.make(seed,arm,cfg,device)
            assert r.old.state_hash(model.state_dict())==summary['initial_model_sha256']
            assert r.optimizer_hash(opt)==summary['initial_optimizer_sha256']
            assert summary['trainable_parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.load_state_dict(cp['model_state'],strict=True);opt.load_state_dict(cp['optimizer_state'])
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            r.validate_optimizer(model,opt,'full',total,spec)
            assert r.old.state_hash(model.state_dict())==summary['final_model_sha256']
            assert r.optimizer_hash(opt)==summary['final_optimizer_sha256']
            with np.load(attempt/'predictions.npz',allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
            assert set(pred)=={'labels'}|{f'epoch_{i}' for i in range(spec['epochs']+1)} and np.array_equal(pred['labels'],labels)
            assert len(curves)==spec['epochs']+1
            for epoch,row in enumerate(curves):
                assert int(row['epoch'])==epoch and float(row['seconds'])>0
                logits=pred[f'epoch_{epoch}'];assert logits.shape==(len(labels),10) and np.isfinite(logits).all()
                loss,acc=r.score(logits,labels);close(row['loss'],loss);close(row['accuracy'],acc)
            close(summary['validation_seconds'],sum(float(z['seconds']) for z in curves))
            for prefix,epoch in [('initial',0),('final',spec['epochs'])]:
                loss,acc=r.score(pred[f'epoch_{epoch}'],labels);close(summary[prefix+'_loss'],loss);close(summary[prefix+'_accuracy'],acc)
            with e.q.sdpa_kernel(e.q.SDPBackend.MATH):rescored=r.t.predictions(model,ev,spec['eval_batch_size'])
            assert np.array_equal(rescored,pred[f'epoch_{spec["epochs"]}']),'Saved final predictions do not reproduce'
            reference_pass=True;reference_max_loss=0.;reference_max_logits=0.
            if not cfg['smoke']:
                ref=references[seed]
                assert len(ref['steps'])==len(steps)
                reference_max_loss=max(abs(float(a['loss'])-float(b['loss'])) for a,b in zip(steps,ref['steps']))
                reference_max_logits=max(float(np.max(np.abs(pred[f'epoch_{i}']-ref['predictions'][f'epoch_{i}']))) for i in range(spec['epochs']+1))
                labels_match=all(np.array_equal(pred[f'epoch_{i}'].argmax(1),ref['predictions'][f'epoch_{i}'].argmax(1)) for i in range(spec['epochs']+1))
                source_error=float(np.max(np.abs(pred['epoch_0']-ref['source'])))
                reference_pass=bool(reference_max_loss<=spec['reproduction_loss_tolerance'] and reference_max_logits<=spec['source_logit_tolerance'] and labels_match
                    and source_error<=spec['source_logit_tolerance'] and np.array_equal(pred['epoch_0'].argmax(1),ref['source'].argmax(1)))
            all_records[session,seed,arm]=dict(summary=summary,steps=steps,predictions=pred,stream_hash=stream_hash,
                reference_pass=reference_pass,reference_max_loss=reference_max_loss,reference_max_logits=reference_max_logits)
            del cp,model,opt,streams
        for seed in spec['seeds']:
            a,b=[all_records[session,seed,arm] for arm in spec['arms']]
            sa,sb=a['summary'],b['summary']
            ready=cfg['smoke'] or (sa['final_accuracy']>=spec['baseline_min_accuracy'] and 100*(sa['initial_accuracy']-sa['final_accuracy'])<=spec['accuracy_margin_pp']+1e-10)
            pairings.append(dict(session=session,seed=seed,native_first=sa['position']<sb['position'],
                native_seconds=sa['complete_workflow_seconds'],candidate_seconds=sb['complete_workflow_seconds'],
                saving_percent=100*(1-sb['complete_workflow_seconds']/sa['complete_workflow_seconds']),
                training_saving_percent=100*(1-sb['training_seconds']/sa['training_seconds']),
                paired_identity=paired_identity(a,b),reference_pass=a['reference_pass'],baseline_ready=bool(ready),
                native_reference_max_loss_error=a['reference_max_loss'],native_reference_max_logit_error=a['reference_max_logits'],
                final_accuracy=sa['final_accuracy'],candidate_accuracy=sb['final_accuracy'],
                peak_memory_delta_bytes=sb['peak_allocated']-sa['peak_allocated'],
                native_components={k:sa[k] for k in components},candidate_components={k:sb[k] for k in components}))
    assert len(set(uuids))==spec['sessions']
    results=[]
    for seed in spec['seeds']:
        selected=[p for p in pairings if p['seed']==seed]
        repeated=all(paired_identity(all_records[0,seed,arm],all_records[s,seed,arm]) for arm in spec['arms'] for s in range(spec['sessions']))
        verdict=qualified([p['saving_percent'] for p in selected],repeated and all(p['paired_identity'] for p in selected),
            all(p['reference_pass'] and p['baseline_ready'] for p in selected),not any(fragmented),cfg['smoke'])
        results.append(dict(seed=seed,deterministic_across_sessions=repeated,**verdict))
    confirmed=all(z['confirmed'] for z in results)
    result=dict(status='PASS',smoke=cfg['smoke'],sessions=spec['sessions'],trajectories=len(all_records),
        training_updates=len(all_records)*spec['epochs']*spec['expected_batches_per_epoch'],
        discarded_warmup_updates=len(all_records)*spec['hardware_warmup_steps'],
        final_checkpoints_rescored=len(all_records),fragmented_sessions=[i for i,z in enumerate(fragmented) if z],conditions=results,
        complete_adaptation_saving_confirmed=confirmed,
        decision='smoke_only' if cfg['smoke'] else 'review_complete_adaptation_saving' if confirmed else 'complete_adaptation_confirmation_not_established',
        original_ten_percent_screen='FAILED_UNCHANGED',automatic_followup=False)
    if write:
        # Recover only identical derived outputs if interrupted after analysis and before sealing.
        r.old.write_freeze(out/'audit.json',result);r.old.write_freeze(out/'pairings.json',pairings)
        lines=['# QKV real-data adaptation results','',f'Decision: {result["decision"]}. Integrity: PASS.','',
            '| Source seed | Median complete-workflow saving | Median interval | Positive sessions | Adjusted p | Identity | Reference | Confirmed |',
            '|---|---:|---|---:|---:|---|---|---|']
        for z in results:
            s=z['statistics'];lo,hi=s['median_interval_percent']
            lines.append(f'| {z["seed"]} | {s["median_saving_percent"]:.3f}% | [{lo:.3f}, {hi:.3f}]% | {s["positive_sessions"]}/{s["sessions"]} | {s["bonferroni_p"]:.6f} | {z["identity_pass"]} | {z["reference_pass"]} | {z["confirmed"]} |')
        lines+=['','Primary clock includes per-arm data loading, model/source setup, warmup, training/augmentation, all validations and final artifacts. Imports, preflight, provenance hashing and post-run audits are excluded.',
            'One pair per source per session is the inferential unit. Full-study median intervals have 99.21875% marginal and >=98.4375% simultaneous coverage under independent identically distributed continuous session effects. Shared-machine drift limits this assumption; smoke coverages differ.',
            'Inspect pairings.json for every value, order, quality/reference checks, component times and memory. Repeated sessions reproduce two fixed learning trajectories, not additional independent model seeds.',
            'Original >=10% screen remains failed; no new 5% minimum was selected. No novelty, other-architecture/hardware or faster pretraining claim. No automatic follow-up.','']
        content='\n'.join(lines);p=out/'REPORT.md'
        if p.exists():assert p.read_text(encoding='utf8')==content
        else:p.write_text(content,encoding='utf8')
    else:assert read(out/'audit.json')==result and read(out/'pairings.json')==pairings
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    import qkv_training as e
    e.r.m.configure();print(json.dumps(audit(args.output),indent=2))
