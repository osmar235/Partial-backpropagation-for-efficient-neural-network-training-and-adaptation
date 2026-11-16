"""Reconstruct learning streams, rescore final checkpoints and audit paired timing."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from statistics import median

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)

def verify_saved_result(saved,current):
    # Each audit independently gates fresh inference; preserve the first measured
    # rescore diagnostics without requiring later valid errors to be identical.
    assert set(saved)==set(current)
    assert {k:v for k,v in saved.items() if k!='rescores'}=={k:v for k,v in current.items() if k!='rescores'}

def paired_identity(a,b):
    fields=('initial_model_sha256','initial_optimizer_sha256','final_model_sha256','final_optimizer_sha256','final_global_rng')
    return (all(a['summary'][k]==b['summary'][k] for k in fields)
        and a['steps']==b['steps']
        and a['stream_hash']==b['stream_hash']
        and all(np.array_equal(a['predictions'][k],b['predictions'][k]) for k in a['predictions']))

def quality_deficits(reference,candidate,spec):
    common=100*(reference['final_accuracy']-candidate['final_accuracy'])
    own=100*(reference['deployed_accuracy']-candidate['deployed_accuracy'])
    return common,own,bool(common<=spec['accuracy_margin_pp']+1e-10 and own<=spec['accuracy_margin_pp']+1e-10)

def decision(pairings,spec,fragmented,smoke):
    results=[]
    for arm in spec['arms'][1:]:
        conditions=[]
        for seed in spec['seeds']:
            rows=[p for p in pairings if p['arm']==arm and p['seed']==seed];values=[p['saving_percent'] for p in rows]
            n=len(values);wins=sum(v>0 for v in values);p=sum(math.comb(n,k) for k in range(wins,n+1))/2**n
            conditions.append(dict(seed=seed,sessions=n,positive_sessions=wins,median_saving_percent=median(values),saving_range_percent=[min(values),max(values)],
                one_sided_sign_p=p,baseline_ready=all(r['baseline_ready'] for r in rows),quality_pass=all(r['quality_pass'] for r in rows),
                max_common_accuracy_deficit_pp=max(r['common_deficit_pp'] for r in rows),max_deployed_accuracy_deficit_pp=max(r['deployed_deficit_pp'] for r in rows)))
        adjusted=min(1.,spec['candidate_policies']*max(c['one_sided_sign_p'] for c in conditions))
        qualified=not smoke and not fragmented and adjusted<=spec['timing_family_alpha'] and all(c['sessions']==spec['sessions'] and c['positive_sessions']==c['sessions'] and c['baseline_ready'] and c['quality_pass'] for c in conditions)
        results.append(dict(arm=arm,conditions=conditions,adjusted_both_source_direction_p=adjusted,practical_screen_pass=bool(qualified)))
    return results

def refresh_assessment(pairings,conditions,spec,fragmented,smoke):
    by_key={(p['session'],p['seed'],p['arm']):p for p in pairings}
    comparisons=[]
    for session in range(spec['sessions']):
        for seed in spec['seeds']:
            frozen=by_key[session,seed,'frozen_cut3'];periodic=by_key[session,seed,'periodic_cut3_r4']
            comparisons.append(dict(session=session,seed=seed,
                common_accuracy_gain_pp=100*(periodic['candidate_accuracy']-frozen['candidate_accuracy']),
                own_policy_accuracy_gain_pp=100*(periodic['deployed_accuracy']-frozen['deployed_accuracy']),
                time_ratio_periodic_over_frozen=periodic['candidate_seconds']/frozen['candidate_seconds'],
                frozen_quality_pass=frozen['quality_pass'],periodic_quality_pass=periodic['quality_pass']))
    verdicts={v['arm']:v['practical_screen_pass'] for v in conditions}
    # Scope: necessity for THIS margin on THESE checkpoints, not superiority inference.
    needs_refresh=not smoke and not fragmented and verdicts['periodic_cut3_r4'] and all(not c['frozen_quality_pass'] for c in comparisons)
    if smoke:verdict='smoke_only'
    elif fragmented:verdict='timing_fragmented_review_required'
    elif verdicts['frozen_cut3']:verdict='frozen_prefix_already_qualifies_refresh_not_required_for_screen'
    elif needs_refresh:verdict='periodic_qualifies_frozen_fails_quality_in_every_pair'
    elif verdicts['periodic_cut3_r4']:verdict='periodic_qualifies_but_consistent_refresh_necessity_not_established'
    else:verdict='periodic_does_not_qualify_no_automatic_rescue'
    return dict(verdict=verdict,refresh_needed_for_fixed_screen=bool(needs_refresh),comparisons=comparisons,
                significance_or_population_superiority_claim=False)

def manifest(parent,fingerprint,expected=None):
    import prefix_adaptation as e
    marker=e.r.d.validate_complete(parent,fingerprint);assert marker
    if expected is not None:assert set(marker['sha256'])=={str(p) for p in expected}
    return marker

def audit(out,write=False):
    import prefix_adaptation as e
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
            qualified=Path(cfg['qualified_references'][str(seed)])
            with np.load(qualified/'predictions.npz',allow_pickle=False) as z:qual_pred={k:z[k].copy() for k in z.files}
            assert np.array_equal(qual_pred['labels'],labels)
            references[seed]=dict(predictions=qual_pred,source=source)
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
            assert summary['complete_workflow_seconds']>0 and summary['integrity']['all_states_finite']
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
                    assert int(row['full_update'])==int(e.full_update(arm,idx,spec))
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
            assert summary['parameter_coordinates']==sum(p.numel() for p in model.parameters())
            initial_state={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
            model.load_state_dict(cp['model_state'],strict=True);opt.load_state_dict(cp['optimizer_state'])
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            router=e.router_for(model,arm,spec);router.select(e.full_update(arm,total-1,spec))
            integrity=e.validate_state(model,opt,arm,total,spec,initial_state,check_gradients=False)
            assert integrity==summary['integrity']
            assert summary['last_step_trainable_parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert r.old.state_hash(model.state_dict())==summary['final_model_sha256']
            assert r.optimizer_hash(opt)==summary['final_optimizer_sha256']
            with np.load(attempt/'predictions.npz',allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
            assert set(pred)=={'labels','deployed_final'}|{f'epoch_{i}' for i in range(spec['epochs']+1)} and np.array_equal(pred['labels'],labels)
            assert len(curves)==spec['epochs']+1
            for epoch,row in enumerate(curves):
                assert int(row['epoch'])==epoch and float(row['seconds'])>0
                logits=pred[f'epoch_{epoch}'];assert logits.shape==(len(labels),10) and np.isfinite(logits).all()
                loss,acc=r.score(logits,labels);close(row['loss'],loss);close(row['accuracy'],acc)
            close(summary['validation_seconds'],sum(float(z['seconds']) for z in curves)+summary['deployment_seconds'])
            assert summary['deployment_seconds']>0
            deployed_loss,deployed_acc=r.score(pred['deployed_final'],labels)
            close(summary['deployed_loss'],deployed_loss);close(summary['deployed_accuracy'],deployed_acc)
            for prefix,epoch in [('initial',0),('final',spec['epochs'])]:
                loss,acc=r.score(pred[f'epoch_{epoch}'],labels);close(summary[prefix+'_loss'],loss);close(summary[prefix+'_accuracy'],acc)
            rescore={}
            for key,policy in [(f'epoch_{spec["epochs"]}','reference_fp32_math'),('deployed_final',spec['execution_policy'])]:
                reconstructed=e.predictions(model,ev,spec['eval_batch_size'],policy)
                error=float(np.max(np.abs(reconstructed-pred[key])))
                assert error<=spec['reproduction_logit_tolerance'] and np.array_equal(reconstructed.argmax(1),pred[key].argmax(1)),('Final logits do not reproduce',key,error)
                rescore[key]=dict(max_abs=error,exact=bool(np.array_equal(reconstructed,pred[key])))
            flags=summary['policy'];assert summary['loss_arithmetic']=='explicit_fp32_outside_autocast'
            assert flags['deterministic']==flags['cudnn_deterministic']==spec['execution_policy'].endswith('_math')
            assert not any(flags[k] for k in ('tf32_matmul','tf32_cudnn','cudnn_benchmark','warn_only','fp16_reduced_reduction','bf16_reduced_reduction','math_reduced_reduction'))
            dispatch=summary['warmup_dispatch'];assert sum(dispatch.values())==(2 if cfg['smoke'] else 6)
            if spec['execution_policy'].endswith('_math'):assert dispatch['math']==sum(dispatch.values())
            reference_pass=True;source_error=0.;historical_accuracy=None;historical_deployed_accuracy=None
            if not cfg['smoke']:
                ref=references[seed];source_error=float(np.max(np.abs(pred['epoch_0']-ref['source'])))
                reference_pass=bool(source_error<=spec['source_logit_tolerance'] and np.array_equal(pred['epoch_0'].argmax(1),ref['source'].argmax(1)))
                historical_accuracy=r.score(ref['predictions'][f'epoch_{spec["epochs"]}'],labels)[1]
                historical_deployed_accuracy=r.score(ref['predictions']['deployed_final'],labels)[1]
                assert reference_pass,('Archived source predictions mismatch',source_error)
            all_records[session,seed,arm]=dict(summary=summary,steps=steps,predictions=pred,stream_hash=stream_hash,
                reference_pass=reference_pass,source_max_logit_error=source_error,historical_accuracy=historical_accuracy,historical_deployed_accuracy=historical_deployed_accuracy,rescore=rescore)
            del cp,model,opt,streams
        for seed in spec['seeds']:
            a=all_records[session,seed,'full'];sa=a['summary']
            ready=cfg['smoke'] or (sa['final_accuracy']>=spec['baseline_min_accuracy'] and
                100*(sa['initial_accuracy']-sa['final_accuracy'])<=spec['accuracy_margin_pp']+1e-10 and
                100*(a['historical_accuracy']-sa['final_accuracy'])<=spec['accuracy_margin_pp']+1e-10 and
                sa['deployed_accuracy']>=spec['baseline_min_accuracy'] and
                100*(a['historical_deployed_accuracy']-sa['deployed_accuracy'])<=spec['accuracy_margin_pp']+1e-10)
            for arm in spec['arms'][1:]:
                b=all_records[session,seed,arm];sb=b['summary']
                assert sa['initial_model_sha256']==sb['initial_model_sha256'] and sa['initial_optimizer_sha256']==sb['initial_optimizer_sha256']
                assert np.array_equal(a['predictions']['epoch_0'],b['predictions']['epoch_0']) and a['stream_hash']==b['stream_hash']
                assert [{k:v for k,v in row.items() if k not in ('loss','full_update')} for row in a['steps']]==[{k:v for k,v in row.items() if k not in ('loss','full_update')} for row in b['steps']]
                common,deployed,quality=quality_deficits(sa,sb,spec)
                pairings.append(dict(session=session,seed=seed,arm=arm,reference_first=sa['position']<sb['position'],
                    reference_seconds=sa['complete_workflow_seconds'],candidate_seconds=sb['complete_workflow_seconds'],
                    saving_percent=100*(1-sb['complete_workflow_seconds']/sa['complete_workflow_seconds']),
                    training_saving_percent=100*(1-sb['training_seconds']/sa['training_seconds']),baseline_ready=bool(ready),quality_pass=bool(quality),
                    common_deficit_pp=common,deployed_deficit_pp=deployed,reference_accuracy=sa['final_accuracy'],candidate_accuracy=sb['final_accuracy'],
                    deployed_accuracy=sb['deployed_accuracy'],reference_deployed_accuracy=sa['deployed_accuracy'],reference_loss=sa['final_loss'],candidate_loss=sb['final_loss'],deployed_loss=sb['deployed_loss'],reference_deployed_loss=sa['deployed_loss'],
                    historical_reference_accuracy=a['historical_accuracy'],historical_reference_deployed_accuracy=a['historical_deployed_accuracy'],peak_memory_delta_bytes=sb['peak_allocated']-sa['peak_allocated'],
                    source_training_seconds=sa['source_training_seconds'],reference_components={k:sa[k] for k in components},candidate_components={k:sb[k] for k in components}))
    assert len(set(uuids))==spec['sessions']
    fragmented_ids=[i for i,z in enumerate(fragmented) if z]
    results=decision(pairings,spec,fragmented_ids,cfg['smoke'])
    repeated={arm:all(paired_identity(all_records[0,seed,arm],all_records[s,seed,arm]) for seed in spec['seeds'] for s in range(spec['sessions'])) for arm in spec['arms']}
    rescores={f'{session}/{seed}/{arm}':record['rescore'] for (session,seed,arm),record in all_records.items()}
    result=dict(status='PASS',smoke=cfg['smoke'],sessions=spec['sessions'],trajectories=len(all_records),training_updates=len(all_records)*spec['epochs']*spec['expected_batches_per_epoch'],
        discarded_warmup_updates=len(all_records)*spec['hardware_warmup_steps'],final_checkpoints_rescored=len(all_records),final_policy_rescores=2*len(all_records),
        fragmented_sessions=fragmented_ids,conditions=results,refresh_assessment=refresh_assessment(pairings,results,spec,fragmented_ids,cfg['smoke']),learning_exact_across_sessions=repeated,rescores=rescores,
        decision='smoke_only' if cfg['smoke'] else 'review_prospective_quality_and_complete_cost',historical_bf16_parameter_gate='FAILED_UNCHANGED',original_ten_percent_screen='FAILED_UNCHANGED',automatic_followup=False)
    if write:
        if (out/'audit.json').exists():verify_saved_result(read(out/'audit.json'),result)
        else:r.old.write_freeze(out/'audit.json',result)
        r.old.write_freeze(out/'pairings.json',pairings)
        lines=['# Prefix adaptation quality and complete-cost results','',f'Integrity: PASS. Decision: {result["decision"]}.','',
               'See audit.json for each update-schedule qualification and pairings.json for every source/session quality and timing comparison.',
               'The clock includes data/model setup, profiled warmup/restoration, training, epoch validations, final own-policy validation and artifacts.',
               'Fresh common AdamW reset. Full, frozen-cut3 and periodic-cut3/r4 under BF16/automatic; explicit FP32 loss. No test-set evaluation.',
               'Periodic refresh is globally phased from update 0; no epoch reset. Prefix moments/counters/decay pause on omissions. Repeated sessions are timing/numerical checks, not additional model seeds. The old BF16 numerical gate and 10% QKV screen remain failed. No automatic follow-up.','']
        content='\n'.join(lines);path=out/'REPORT.md'
        if path.exists():assert path.read_text(encoding='utf8')==content
        else:path.write_text(content,encoding='utf8')
    else:
        verify_saved_result(read(out/'audit.json'),result)
        assert read(out/'pairings.json')==pairings
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    import prefix_adaptation as e
    e.r.m.configure();assert __debug__;print(json.dumps(audit(args.output),indent=2))
