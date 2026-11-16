"""Reconstruct side-study streams, score saved predictions, audit states and gates."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from review_stage14C import close,check_memory,compare_csv

def assess(records,spec,synthetic,single_launch=True):
    screens=[]
    for seed in spec['seeds']:
        arms={row['mode']:row for row in records if row['seed']==seed}
        full=arms['full'];candidate=arms['multi_side']
        controls=[arms[mode] for mode in ('full','head','lp_full','final_side')]
        best=max([full['initial_accuracy']]+[row['final_accuracy'] for row in controls])
        qualified=lambda row:100*(best-row['final_accuracy'])<=spec['accuracy_margin_pp']+1e-10
        eligible=[row for row in controls if qualified(row)]
        ready=bool(full['final_accuracy']>=spec['baseline_min_accuracy'] and
            100*(full['initial_accuracy']-full['final_accuracy'])<=spec['accuracy_margin_pp']+1e-10 and eligible)
        fastest=min(eligible,key=lambda row:row['complete_adaptation_seconds']) if eligible else None
        denominator=min(full['complete_adaptation_seconds'],fastest['complete_adaptation_seconds'] if fastest else float('inf'))
        ratio=candidate['complete_adaptation_seconds']/denominator
        quality=qualified(candidate);timing=ratio<=1-spec['minimum_time_saving']+1e-12
        screens.append(dict(seed=seed,reference_qualified=ready,best_control_accuracy=best,
            quality_pass=quality,final_delta_vs_best_pp=100*(candidate['final_accuracy']-best),
            delta_vs_final_side_pp=100*(candidate['final_accuracy']-arms['final_side']['final_accuracy']),
            final_side_quality_qualified=qualified(arms['final_side']),
            fastest_qualified_control=fastest['mode'] if fastest else '',complete_time_ratio=ratio,time_pass=timing,
            timing_eligible=single_launch,seed_screen_pass=bool(ready and quality and timing and single_launch and not synthetic)))
    passed=all(row['seed_screen_pass'] for row in screens)
    decision=('synthetic_validation_only' if synthetic else 'timing_interrupted_review_required' if not single_launch
        else 'reference_review_required' if not all(row['reference_qualified'] for row in screens)
        else 'candidate_for_independent_confirmation' if passed else 'close_fixed_side_recipe')
    return screens,passed,decision

def audit(out,write=False):
    import stage15_side_adaptation as r
    out=Path(out);cfg=json.loads((out/'configuration.json').read_text())
    assert r.v.digest({k:val for k,val in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    spec,fake=cfg['spec'],cfg['synthetic']
    freeze=json.loads((out/'SYNTHETIC_FREEZE.json' if fake else r.FREEZE).read_text())
    assert r.v.digest({k:val for k,val in freeze.items() if k!='freeze_hash'})==freeze['freeze_hash']==cfg['freeze_hash']
    assert freeze['spec']==spec and freeze['synthetic']==fake and freeze['expected_split']==cfg['split']
    r.t.verify(freeze['source_sha256']);r.t.verify(freeze['evidence_sha256'])
    for cp in freeze['checkpoints'].values():r.t.verify({cp['path']:cp['sha256']})
    _,y,_,ey,split=r.v.load_data('development',torch.device('cpu'),fake=fake)
    assert split==cfg['split'];labels=ey.numpy();summaries=[];total_updates=0
    for seed,order in zip(spec['seeds'],spec['orders']):
        initials={};initial_states={}
        for position,mode in enumerate(order):
            core,source_cp=r.source(seed,fake,freeze)
            source_body_hash=r.old.state_hash(core.body.state_dict());source_head_hash=r.old.state_hash(core.head.state_dict())
            model,opt=r.make_condition(core,source_cp,seed,mode,spec,torch.device('cpu'));del core,source_cp
            initial_hash=r.old.state_hash(model.state_dict());initial_opt_hash=r.optimizer_hash(opt)
            initial_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
            initial_states[mode]=initial_hash
            parent=out/f'seed_{seed}'/mode;rc=json.loads((parent/'run_config.json').read_text())
            assert r.v.digest({k:val for k,val in rc.items() if k!='fingerprint'})==rc['fingerprint']
            assert rc['seed']==seed and rc['mode']==mode and rc['position']==position and rc['root_fingerprint']==cfg['fingerprint']
            marker=r.d.validate_complete(parent,rc['fingerprint']);assert marker
            attempt=Path(marker['attempt']);assert attempt.parent.resolve()==parent.resolve()
            assert {str(p) for p in attempt.iterdir() if p.is_file()}==set(marker['sha256'])
            restoration=json.loads((attempt/'restoration.json').read_text())
            assert restoration['model_sha256']==initial_hash and restoration['optimizer_sha256']==initial_opt_hash
            assert restoration['source_body_sha256']==source_body_hash and restoration['source_head_sha256']==source_head_hash
            assert restoration['parameter_names']==list(dict(model.named_parameters()))
            assert restoration['initial_trainable_parameters']==initial_trainable
            assert restoration['initial_state_entries']==restoration['initial_optimizer_age']==0
            assert restoration['source_optimizer_age']==spec['source_optimizer_age'] and restoration['source_restored_after_warmup']
            steps=r.d.records(attempt/'steps.csv');curves=r.d.records(attempt/'curves.csv')
            batches=spec['expected_batches_per_epoch'];total=batches*spec['epochs'];assert len(steps)==total
            checked={1,2,total,spec['head_epochs']*batches,spec['head_epochs']*batches+1}
            bg=torch.Generator().manual_seed(seed+spec['batch_seed_offset'])
            ag=torch.Generator().manual_seed(seed+spec['augmentation_seed_offset'])
            mg=np.random.default_rng(seed+spec['mixup_seed_offset']);pg=torch.Generator().manual_seed(seed+spec['permutation_seed_offset'])
            idx=0
            for epoch in range(1,spec['epochs']+1):
                indices=torch.randperm(len(y),generator=bg);assert math.ceil(len(y)/spec['batch_size'])==batches
                for start in range(0,len(y),spec['batch_size']):
                    ix=indices[start:start+spec['batch_size']];n=len(ix);row=steps[idx]
                    assert int(row['epoch'])==epoch and int(row['step'])==idx+1 and int(row['batch_examples'])==n
                    assert row['indices_sha256']==r.tensor_hash(ix) and row['augmentation_before']==r.tensor_hash(ag.get_state())
                    torch.randint(0,9,(n,),generator=ag);torch.randint(0,9,(n,),generator=ag);torch.rand(n,generator=ag)
                    assert row['augmentation_after']==r.tensor_hash(ag.get_state())
                    close(row['mixup_lambda'],mg.beta(spec['mixup_alpha'],spec['mixup_alpha']));torch.randperm(n,generator=pg)
                    assert row['permutation_after']==r.tensor_hash(pg.get_state()) and row['mixup_state']==r.v.digest(mg.bit_generator.state)
                    warm=spec['lr_warmup_epochs']*batches
                    lr=spec['learning_rate']*(idx+1)/warm if idx<warm else spec['learning_rate']*(spec['min_lr_ratio']+
                        (1-spec['min_lr_ratio'])*.5*(1+math.cos(math.pi*(idx-warm)/max(1,total-warm-1))))
                    close(row['lr'],lr);assert int(row['dropout_seed'])==seed+spec['dropout_seed_offset']+idx
                    assert int(row['head_age'])==idx+1
                    body_age=idx+1 if mode=='full' else max(0,idx+1-spec['head_epochs']*batches) if mode=='lp_full' else 0
                    assert int(row['body_age'])==body_age and row['gradient_checked']==str(idx+1 in checked)
                    for key in ('loss','transform_seconds','update_seconds'):assert math.isfinite(float(row[key])) and float(row[key])>0
                    check_memory(row,cfg['device']);idx+=1
            with np.load(attempt/'predictions.npz',allow_pickle=False) as pred:
                assert set(pred.files)=={'labels'}|{f'epoch_{i}' for i in range(spec['epochs']+1)}
                assert np.array_equal(pred['labels'],labels) and len(curves)==spec['epochs']+1
                initials[mode]=pred['epoch_0'].copy()
                for epoch,row in enumerate(curves):
                    assert int(row['seed'])==seed and row['mode']==mode and int(row['epoch'])==epoch and int(row['updates'])==epoch*batches
                    z=pred[f'epoch_{epoch}'];assert z.shape==(len(labels),10) and np.isfinite(z).all()
                    close(row['loss'],float(torch.nn.functional.cross_entropy(torch.from_numpy(z).double(),ey)))
                    close(row['accuracy'],float(np.mean(z.argmax(1)==labels)))
                    assert float(row['validation_seconds'])>0;check_memory(row,cfg['device'])
                if not fake:
                    with np.load(freeze['checkpoints'][str(seed)]['predictions_path'],allow_pickle=False) as prior:
                        assert np.array_equal(labels,prior['labels'])
                        assert np.max(np.abs(pred['epoch_0']-prior['full']))<=spec['source_logit_tolerance']
                        assert np.array_equal(pred['epoch_0'].argmax(1),prior['full'].argmax(1))
                        close(curves[0]['accuracy'],float(np.mean(prior['full'].argmax(1)==labels)))
                        source_loss=float(torch.nn.functional.cross_entropy(torch.from_numpy(prior['full']).double(),ey))
                        assert abs(float(curves[0]['loss'])-source_loss)<=spec['source_loss_tolerance']
                    if mode=='full':
                        previous=Path(freeze['full_reproduction'][str(seed)])
                        with np.load(previous/'predictions.npz',allow_pickle=False) as ref:
                            assert np.array_equal(labels,ref['labels'])
                            for epoch in range(spec['epochs']+1):
                                assert np.max(np.abs(pred[f'epoch_{epoch}']-ref[f'epoch_{epoch}']))<=spec['source_logit_tolerance'],'Full trajectory reproduction failed'
                        old_steps=r.d.records(previous/'steps.csv');assert len(old_steps)==len(steps)
                        for now,ref in zip(steps,old_steps):assert abs(float(now['loss'])-float(ref['loss']))<=spec['reproduction_loss_tolerance']
            manifest=json.loads((attempt/'adaptation_artifacts.json').read_text())
            assert set(manifest)=={str(attempt/p) for p in ('restoration.json','steps.csv','curves.csv','diagnostics.csv',
                'telemetry.json','transitions.json','predictions.npz','final_checkpoint.pt')};r.t.verify(manifest)
            cp=torch.load(attempt/'final_checkpoint.pt',map_location='cpu',weights_only=False)
            assert cp['seed']==seed and cp['mode']==mode and cp['updates']==total
            assert cp['root_fingerprint']==cfg['fingerprint'] and cp['run_fingerprint']==rc['fingerprint']
            assert torch.equal(cp['batch_rng'],bg.get_state()) and torch.equal(cp['augmentation_rng'],ag.get_state())
            assert torch.equal(cp['permutation_rng'],pg.get_state()) and cp['mixup_rng']==mg.bit_generator.state
            if mode=='lp_full':model.body.requires_grad_(True)
            model.load_state_dict(cp['model_state'],strict=True);opt.load_state_dict(cp['optimizer_state'])
            assert len(opt.param_groups)==1 and opt.param_groups[0]['params']==list(model.parameters())
            for name,p in model.named_parameters():
                assert bool(torch.isfinite(p).all())
                if name.startswith('body.') and mode in ('head','final_side','multi_side'):assert p not in opt.state;continue
                age=total-spec['head_epochs']*batches if name.startswith('body.') and mode=='lp_full' else total
                state=opt.state[p];assert int(state['step'])==age
                for key in ('exp_avg','exp_avg_sq'):assert state[key].shape==p.shape and bool(torch.isfinite(state[key]).all())
                assert bool((state['exp_avg_sq']>=0).all())
            close(opt.param_groups[0]['lr'],spec['learning_rate']*spec['min_lr_ratio'])
            if mode in ('head','final_side','multi_side'):assert r.old.state_hash(model.body.state_dict())==source_body_hash
            else:assert r.old.state_hash(model.body.state_dict())!=source_body_hash
            transitions=json.loads((attempt/'transitions.json').read_text());assert len(transitions)==int(mode=='lp_full')
            for row in transitions:
                assert row['epoch']==spec['head_epochs']+1 and row['updates']==row['head_age']==spec['head_epochs']*batches
                assert row['optimizer_before']==row['optimizer_after'] and row['body_state_entries']==0
                assert row['body_sha256']==source_body_hash and row['seconds']>0
            diagnostics=r.d.records(attempt/'diagnostics.csv');assert [int(row['step']) for row in diagnostics]==sorted(checked)
            for row in diagnostics:
                step=int(row['step']);body_active=mode=='full' or (mode=='lp_full' and step>spec['head_epochs']*batches)
                assert row['backbone_gradient_present']==str(body_active)
                for key in ('gradient_l2','head_gradient_l2'):assert math.isfinite(float(row[key])) and float(row[key])>0
                output,upstream=float(row['side_output_gradient_l2']),float(row['upstream_side_gradient_l2'])
                if mode in ('final_side','multi_side'):assert output>0 and (upstream==0 if step==1 else upstream>0)
                else:assert output==upstream==0
            inference=r.d.records(attempt/'inference.csv');assert len(inference)==spec['inference_trials']
            gen=torch.Generator().manual_seed(seed+spec['inference_seed_offset'])
            ii=torch.randperm(len(y),generator=gen)[:spec['inference_trials']*spec['batch_size']].reshape(spec['inference_trials'],spec['batch_size'])
            for i,(row,ix) in enumerate(zip(inference,ii)):
                assert int(row['trial'])==i and row['indices_sha256']==r.tensor_hash(ix)
                assert math.isfinite(float(row['seconds'])) and float(row['seconds'])>0 and math.isfinite(float(row['scalar']))
                check_memory(row,cfg['device'])
            summary=json.loads((attempt/'summary.json').read_text());launch=Path(summary['launch_record'])
            assert launch.parent.parent.resolve()==(out/'launches').resolve();shared=json.loads(launch.read_text())
            assert shared['split']==cfg['split'] and shared['training_examples']==len(y) and shared['validation_examples']==len(labels)
            close(summary['shared_data_seconds'],shared['seconds']/(len(spec['seeds'])*len(spec['modes'])))
            assert summary['shared_data_seconds']>0
            close(summary['complete_adaptation_seconds'],summary['adaptation_seconds']+summary['shared_data_seconds'])
            assert summary['seed']==seed and summary['mode']==mode and summary['updates']==total
            assert summary['initial_model_sha256']==initial_hash and summary['initial_optimizer_sha256']==initial_opt_hash
            assert summary['final_model_sha256']==r.old.state_hash(model.state_dict())!=initial_hash
            assert summary['final_optimizer_sha256']==r.optimizer_hash(opt)!=initial_opt_hash
            assert summary['initial_trainable_parameters']==initial_trainable
            assert summary['final_trainable_parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert summary['total_parameters']==sum(p.numel() for p in model.parameters())
            for prefix,row in [('initial',curves[0]),('final',curves[-1])]:
                close(summary[prefix+'_accuracy'],row['accuracy']);close(summary[prefix+'_loss'],row['loss'])
            training=sum(float(row['transform_seconds'])+float(row['update_seconds']) for row in steps)
            close(summary['training_seconds'],training)
            for key in ('setup_seconds','warmup_restore_seconds'):
                assert summary[key]>0;close(summary[key],restoration[key])
            close(summary['validation_seconds'],sum(float(row['validation_seconds']) for row in curves))
            for key in ('inference_benchmark_seconds','checkpoint_seconds','observed_arm_seconds'):assert math.isfinite(summary[key]) and summary[key]>0
            assert summary['adaptation_seconds']>=training+summary['setup_seconds']+summary['warmup_restore_seconds']+summary['validation_seconds']+summary['checkpoint_seconds']
            assert summary['observed_arm_seconds']>=summary['adaptation_seconds']+summary['inference_benchmark_seconds']
            for key in ('setup_memory','warmup_memory'):check_memory(restoration[key],cfg['device'])
            assert summary['max_training_incremental_allocated']==max(int(row['incremental_allocated']) for row in steps)
            assert summary['max_training_peak_allocated']==max(int(row['peak_allocated']) for row in steps)
            assert summary['max_measured_arm_allocated']==max([restoration[key]['peak_allocated'] for key in ('setup_memory','warmup_memory')]+[int(row['peak_allocated']) for row in steps+curves+inference])
            for key in ('source_training_seconds','source_total_run_seconds'):close(summary[key],freeze['checkpoints'].get(str(seed),{}).get(key,0))
            summary.update(accuracy_recovered_pp=100*(summary['final_accuracy']-summary['initial_accuracy']),
                source_training_plus_adaptation_seconds=summary['source_training_seconds']+summary['complete_adaptation_seconds'],
                mean_inference_seconds=float(np.mean([float(row['seconds']) for row in inference])))
            summaries.append(summary);total_updates+=total;del model,opt,cp
        assert initial_states['final_side']==initial_states['multi_side']
        assert initial_states['full']==initial_states['head']==initial_states['lp_full']
        for mode in ('head','lp_full','final_side','multi_side'):
            assert np.array_equal(initials['head'],initials[mode])
            assert np.array_equal(initials['full'].argmax(1),initials[mode].argmax(1))
            assert np.max(np.abs(initials['full']-initials[mode]))<=spec['source_logit_tolerance']
    one_launch=len({row['launch_record'] for row in summaries})==1
    screens,passed,decision=assess(summaries,spec,fake,one_launch)
    result=dict(status='PASS',synthetic=fake,trajectories=len(summaries),training_updates=total_updates,
        discarded_hardware_warmup_updates=len(summaries)*spec['hardware_warmup_steps'],curve_points=len(summaries)*(spec['epochs']+1),
        frozen_backbones_and_optimizer_ages_verified=True,streams_labels_predictions_and_costs_verified=True,
        paired_side_initial_states_identical=True,frozen_initial_logits_identical=True,initial_classes_identical=True,
        historical_full_trajectory_reproduced=not fake,retained_arms_share_one_launch=one_launch,
        both_seed_reference_qualified=all(row['reference_qualified'] for row in screens),practical_candidate=passed,decision=decision,
        test_scoring_performed=False,automatic_followup=False)
    if write:
        r.v.write_csv(out/'summary.csv',summaries);r.v.write_csv(out/'screen.csv',screens);r.v.atomic_json(out/'audit.json',result)
        lines=['# Stage15A side adaptation','','```json',json.dumps(result,indent=2),'```','',
            '| Seed | Arm | Initial accuracy | Final accuracy | Complete adaptation s | Source training + adaptation s |',
            '|---|---|---:|---:|---:|---:|']
        for row in summaries:lines.append(f"| {row['seed']} | {row['mode']} | {100*row['initial_accuracy']:.2f}% | {100*row['final_accuracy']:.2f}% | {row['complete_adaptation_seconds']:.3f} | {row['source_training_plus_adaptation_seconds']:.3f} |")
        lines+=['','Fixed final endpoint only. Primary clock includes shared data preparation, setup, warmup/restoration, training, all validation, transition checks, diagnostics, logs, checkpoint/prediction saving and hashes.',
            'Post-endpoint inference, offline audit and final summary/completion bookkeeping are separate. Costs are marginal to an existing source; no from-scratch claim.',
            'Final-side is a required capacity control. Reused development states; exploratory only. Prior failures and prior art retained. No automatic follow-up.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text())==result
        compare_csv(r.d.records(out/'summary.csv'),summaries);compare_csv(r.d.records(out/'screen.csv'),screens)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    import stage10_models as m
    m.configure();print(json.dumps(audit(args.output),indent=2))
