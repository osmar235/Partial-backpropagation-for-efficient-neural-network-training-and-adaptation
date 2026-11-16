"""Trace, state-transition, matched-history and fixed outcome audit."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import stage8_vit_v2 as v
import stage9_boundary_diagnostic as d
import stage10_models as m
import stage10_representation as old
import stage11_growth as g
import stage11_growth_pilot as run
from review_stage10A import target

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10),(a,b)

def opt_equal(a,b):
    assert a['param_groups']==b['param_groups'] and a['state'].keys()==b['state'].keys()
    for i,state in a['state'].items():
        for k,t in state.items():assert torch.equal(t,b['state'][i][k]),(i,k)

def check_state(cp,model,age,updates,seed,arm,cfg,rc):
    assert cp['seed']==seed and cp['variant']==arm and cp['updates']==updates
    assert cp['root_fingerprint']==cfg['fingerprint'] and cp['run_fingerprint']==rc['fingerprint']
    model.load_state_dict(cp['model_state'],strict=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    opt=cp['optimizer_state'];ids=[i for group in opt['param_groups'] for i in group['params']]
    assert len(ids)==len(list(model.parameters())) and len(set(ids))==len(ids)
    if age==0:assert not opt['state'];return
    assert len(opt['state'])==len(ids)
    for p,i in zip(model.parameters(),ids):
        state=opt['state'][i];assert int(state['step'])==age
        for k in ['exp_avg','exp_avg_sq']:assert state[k].shape==p.shape and torch.isfinite(state[k]).all()

def same_stream(a,b):
    for k in ['batch_rng','augmentation_rng','permutation_rng']:assert torch.equal(a[k],b[k]),k
    assert a['mixup_rng']==b['mixup_rng']

def audit(out,write=False):
    m.configure();out=Path(out);cfg=json.loads((out/'configuration.json').read_text());spec=cfg['spec'];fake=cfg['synthetic']
    assert v.digest({k:x for k,x in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    record=json.loads((out/'SYNTHETIC_FREEZE.json' if fake else run.FREEZE).read_text())
    assert record==run.freeze_record(spec,fake) and record['freeze_hash']==cfg['freeze_hash']
    assert cfg['source_sha256']==record['source_sha256'] and cfg['split']==record['expected_split']
    assert cfg['execution']==record['required_execution']==m.execution_settings()
    if (out/'COMPLETE.json').exists():assert not write;d.validate_complete(out,cfg['fingerprint'])
    mid=spec['transition_epoch']*spec['expected_batches_per_epoch'];total=spec['expected_updates_per_run']
    flat=[];targets=[];screens=[];audited=0
    for seed,order in zip(spec['development_seeds'],spec['orders']):
        records={};histories={};initials={};before_states={};stream=None
        for position,arm in enumerate(order):
            parent=out/f'seed_{seed}'/arm;rc=json.loads((parent/'run_config.json').read_text())
            assert v.digest({k:x for k,x in rc.items() if k!='fingerprint'})==rc['fingerprint']
            assert rc['root_fingerprint']==cfg['fingerprint'] and rc['seed']==seed and rc['variant']==arm and rc['order_position']==position
            marker=d.validate_complete(parent,rc['fingerprint']);assert marker
            attempt=Path(marker['attempt']);summary=json.loads((attempt/'summary.json').read_text())
            assert summary['fingerprint']==rc['fingerprint'] and summary['root_fingerprint']==cfg['fingerprint']
            assert summary['seed']==seed and summary['variant']==arm and summary['order_position']==position
            steps=d.records(attempt/'steps.csv');epochs=d.records(attempt/'epochs.csv')
            assert len(steps)==summary['updates']==total and len(epochs)==summary['epochs']==spec['epochs']
            mg=np.random.default_rng(seed+98001);train=val=observed=0.
            small_width=12 if fake else 96;full_width=2*small_width
            for i,r in enumerate(steps):
                epoch=i//spec['expected_batches_per_epoch']+1;bi=i%spec['expected_batches_per_epoch']
                n=min(spec['batch_size'],(16 if fake else 45000)-bi*spec['batch_size'])
                assert int(r['step'])==i+1 and int(r['epoch'])==epoch and int(r['batch_examples'])==n
                assert int(r['dropout_seed'])==seed+spec['dropout_seed_offset']+i
                age=i+1-(mid if run.resets(arm) and i>=mid else 0)
                assert int(r['optimizer_step'])==age
                width=full_width if arm.startswith('full') or (arm=='grow' and i>=mid) else small_width
                assert int(r['width'])==width
                close(r['mixup_lambda'],mg.beta(spec['mixup_alpha'],spec['mixup_alpha']))
                close(r['lr'],v.learning_rate(i,spec['schedule_horizon_epochs']*spec['expected_batches_per_epoch'],spec['expected_batches_per_epoch'],spec['learning_rate'],spec))
                assert math.isfinite(float(r['loss'])) and float(r['loss'])>=0
            for i,er in enumerate(epochs):
                group=steps[i*spec['expected_batches_per_epoch']:(i+1)*spec['expected_batches_per_epoch']]
                seen=sum(int(r['batch_examples']) for r in group);assert seen==(16 if fake else 45000)
                assert int(er['epoch'])==i+1 and int(er['updates'])==(i+1)*spec['expected_batches_per_epoch']
                close(er['training_loss'],sum(float(r['loss'])*int(r['batch_examples']) for r in group)/seen)
                assert 0<=float(er['validation_accuracy'])<=1 and math.isfinite(float(er['validation_loss']))
                train+=float(er['epoch_training_seconds']);val+=float(er['validation_seconds'])
                close(er['cumulative_training_seconds'],train);close(er['cumulative_validation_seconds'],val)
                assert float(er['cumulative_observed_seconds'])>observed
                observed=float(er['cumulative_observed_seconds']);assert observed>=train+val-1e-6
            close(train,summary['training_seconds']);close(val,summary['validation_seconds']);close(observed,summary['observed_seconds'])
            close(epochs[-1]['validation_accuracy'],summary['final_validation_accuracy']);close(epochs[-1]['validation_loss'],summary['final_validation_loss'])
            assert math.isfinite(summary['total_run_seconds']) and summary['total_run_seconds']>=observed+summary['hardware_warmup_seconds']+summary['checkpoint_seconds']-1e-6
            model=m.make_model(run.initial_variant(arm),seed,fake)
            assert old.state_hash(model.state_dict())==summary['initial_state_sha256']
            assert sum(p.numel() for p in model.parameters())==summary['initial_parameter_count']
            initials[arm]=summary['initial_state_sha256']
            before=torch.load(attempt/'transition_before.pt',map_location='cpu',weights_only=True)
            after=torch.load(attempt/'transition_after.pt',map_location='cpu',weights_only=True)
            final=torch.load(attempt/'final_checkpoint.pt',map_location='cpu',weights_only=True)
            info=json.loads((attempt/'transition.json').read_text())
            check_state(before,model,mid,mid,seed,arm,cfg,rc)
            before_states[arm]=before
            if arm=='grow':
                expected=g.lift_state(before['model_state']);model=m.make_model('full_rgb32',seed,fake)
                for n,t in expected.items():assert torch.equal(t,after['model_state'][n]),n
            else:
                for n,t in before['model_state'].items():assert torch.equal(t,after['model_state'][n]),n
            check_state(after,model,0 if run.resets(arm) else mid,mid,seed,arm,cfg,rc)
            if not run.resets(arm):opt_equal(before['optimizer_state'],after['optimizer_state'])
            same_stream(before,after)
            assert info['action']==('grow_and_reset' if arm=='grow' else 'reset' if run.resets(arm) else 'unchanged')
            assert info['after_epoch']==spec['transition_epoch'] and info['updates']==mid
            assert info['model_sha_before']==old.state_hash(before['model_state']) and info['model_sha_after']==old.state_hash(after['model_state'])
            assert info['parameter_count_before']==summary['initial_parameter_count'] and info['parameter_count_after']==summary['final_parameter_count']
            assert 0<=info['max_scaled_logit_error']<=1 and info['prediction_changes']==0
            assert info['optimizer_entries_after']==len(after['optimizer_state']['state'])
            close(info['total_transition_seconds'],summary['transition_seconds'])
            assert summary['transition_seconds']>0 and observed>=train+val+summary['transition_seconds']-1e-6
            age=total-mid if run.resets(arm) else total
            check_state(final,model,age,total,seed,arm,cfg,rc)
            assert summary['final_optimizer_steps']==age and summary['final_parameter_count']==sum(p.numel() for p in model.parameters())
            if stream is None:stream=final
            else:same_stream(stream,final)
            histories[arm]=([r['loss'] for r in steps[:mid]],[(r['training_loss'],r['validation_loss'],r['validation_accuracy']) for r in epochs[:spec['transition_epoch']]])
            for role in ['readiness','secondary']:
                first,last=target(epochs,spec[role+'_target'],spec['sustained_epochs'])
                tr=dict(seed=seed,variant=arm,role=role,threshold=spec[role+'_target'],reached=last is not None,
                        first_crossing_epoch=int(first['epoch']) if first else '',confirming_epoch=int(last['epoch']) if last else '',
                        observed_seconds=float(last['cumulative_observed_seconds']) if last else '',
                        censoring_epoch=spec['epochs'] if last is None else '')
                targets.append(tr)
                if role=='readiness':summary['readiness_reached']=tr['reached']
            records[arm]=summary;flat.append({k:summary[k] for k in ['seed','variant','initial_parameter_count','final_parameter_count',
                'final_validation_accuracy','training_seconds','validation_seconds','transition_seconds','hardware_warmup_seconds',
                'checkpoint_seconds','total_run_seconds','readiness_reached']});audited+=len(steps)
        for group in [['small','small_reset','grow'],['full','full_reset']]:
            reference=group[0]
            for arm in group[1:]:
                assert initials[arm]==initials[reference] and histories[arm]==histories[reference],('Pre-transition mismatch',seed,arm)
                for n,t in before_states[arm]['model_state'].items():assert torch.equal(t,before_states[reference]['model_state'][n])
                opt_equal(before_states[arm]['optimizer_state'],before_states[reference]['optimizer_state'])
                same_stream(before_states[arm],before_states[reference])
        screens.append(dict(seed=seed,**run.practical_screen(records,spec)))
    result=dict(status='PASS',synthetic=fake,fingerprint=cfg['fingerprint'],runs=10,audited_updates=audited,
                all_transition_states_and_counters_verified=True,matched_histories_bitwise_before_transition=True,data_rng_streams_matched=True,
                full_baselines_ready=all(r['baseline_ready'] for r in screens),both_seed_quality_pass=all(r['quality_pass'] for r in screens),
                both_seed_speed_pass=all(r['speed_pass'] for r in screens),promising_development_hypothesis=all(r['seed_screen_pass'] for r in screens),
                inference=spec['inference'])
    if write:
        for name,rows in [('all_runs.csv',flat),('targets.csv',targets),('screen.csv',screens)]:v.write_csv(out/name,rows)
        v.atomic_json(out/'audit.json',result)
        lines=['# Stage11B fixed midpoint-growth pilot','',spec['inference'],'',
               '| Seed | Arm | Final accuracy | Complete-run seconds | Sustained80% readiness |','|---:|---|---:|---:|---|']
        for r in flat:lines.append(f'| {r["seed"]} | {r["variant"]} | {100*r["final_validation_accuracy"]:.2f}% | {r["total_run_seconds"]:.2f} | {r["readiness_reached"]} |')
        lines+=['',f'Synthetic: {fake}. Both-seed primary screen: {result["promising_development_hypothesis"]}.',
                'Complete-run time includes setup, warmup, growth/reset/checks, validation and checkpoints.',
                'Matched initial histories, exact pre-growth states, mapped transition weights and fresh/continued Adam counters verified.',
                'Two development seeds and reused validation do not establish generalization, significance or novelty.',
                'All100 epochs retained. No automatic extension, target change or transition search. Review before next steps.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text())==result
        for name,rows in [('all_runs.csv',flat),('targets.csv',targets),('screen.csv',screens)]:
            saved=d.records(out/name);assert len(saved)==len(rows)
            for a,b in zip(rows,saved):
                for k,value in a.items():
                    if isinstance(value,(bool,str)):assert str(value)==b[k]
                    else:close(value,b[k])
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=run.REAL_OUT)
    audit(p.parse_args().root,write=False)
