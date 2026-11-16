"""Read-only trace, frozen-prefix, optimizer and matched-warmup audit."""
import argparse,json,math
from pathlib import Path
import torch
import stage12_adaptation as r
p=r.p;v=r.v;d=r.d;old=r.old

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10),(a,b)
def role(name):return 'head' if name.startswith('fc.') else 'suffix' if name.startswith('layer4.') else 'prefix'
def expected_mode(arm,i,spec):
    first=spec['head_warmup_epochs']*spec['batches_per_epoch']
    if arm=='full':return 'full'
    if arm=='head' or i<first:return 'head'
    if arm=='lp_full' or (arm=='lp_periodic' and (i-first)%spec['prefix_period']==0):return 'full'
    return 'suffix'
def counts_at(arm,steps,spec):
    modes=[expected_mode(arm,i,spec) for i in range(steps)]
    return dict(prefix=modes.count('full'),suffix=sum(m!='head' for m in modes),head=steps)
def state_check(cp,initial,model,arm,seed,updates,spec,cfg,rc):
    counts=counts_at(arm,updates,spec)
    assert cp['updates']==updates and cp['seed']==seed and cp['arm']==arm and cp['group_updates']==counts
    assert cp['root_fingerprint']==cfg['fingerprint'] and cp['fingerprint']==rc['fingerprint']
    assert cp['model_state'].keys()==initial.keys()
    for n,t in cp['model_state'].items():assert t.shape==initial[n].shape and torch.isfinite(t).all()
    for n,_ in model.named_buffers():assert torch.equal(initial[n],cp['model_state'][n]),('BN buffer changed',n)
    opt=cp['optimizer_state'];assert len(opt['param_groups'])==3
    expected_names={key:[n for n,_ in model.named_parameters() if role(n)==key] for key in ['prefix','suffix','head']}
    expected_ids=[]
    for group,key in zip(opt['param_groups'],['prefix','suffix','head']):
        assert group['role']==key and group['weight_decay']==spec['weight_decay']
        close(group['lr'],p.rate_factor(updates-1,spec)*(spec['head_lr'] if key=='head' else spec['backbone_lr']))
        names=expected_names[key];assert len(group['params'])==len(names)
        for n,idx in zip(names,group['params']):
            if counts[key]==0:
                assert idx not in opt['state'];assert torch.equal(initial[n],cp['model_state'][n]),('Frozen parameter changed',n)
            else:
                expected_ids.append(idx);state=opt['state'][idx];assert int(state['step'])==counts[key]
                for field in ['exp_avg','exp_avg_sq']:assert state[field].shape==initial[n].shape and torch.isfinite(state[field]).all()
    assert set(expected_ids)==set(opt['state'])

def same_state(a,b):
    for n,t in a['model_state'].items():assert torch.equal(t,b['model_state'][n]),n
    assert a['optimizer_state']['param_groups']==b['optimizer_state']['param_groups']
    assert a['optimizer_state']['state'].keys()==b['optimizer_state']['state'].keys()
    for k,state in a['optimizer_state']['state'].items():
        for n,t in state.items():assert torch.equal(t,b['optimizer_state']['state'][k][n])
    for key in ['batch_rng','augmentation_rng']:assert torch.equal(a[key],b[key])

def target(epochs,threshold,length):
    first=None;streak=0
    for er in epochs:
        if float(er['validation_accuracy'])>=threshold:first=first or er;streak+=1
        else:streak=0
        if streak==length:return first,er
    return first,None

def audit(out,write=False):
    p.execution.configure();out=Path(out);cfg=json.loads((out/'configuration.json').read_text());spec=cfg['spec'];fake=cfg['synthetic']
    assert cfg['fingerprint']==v.digest({k:x for k,x in cfg.items() if k!='fingerprint'})
    record=json.loads((out/'SYNTHETIC_FREEZE.json' if fake else r.FREEZE).read_text())
    assert record==r.freeze_record(spec,fake) and cfg['freeze_hash']==record['freeze_hash']
    assert cfg['source_sha256']==record['source_sha256'] and cfg['split']==record['split'] and cfg['pretrained']==record['pretrained']
    assert cfg['execution']==record['required_execution']==p.execution.execution_settings()
    if (out/'COMPLETE.json').exists():assert not write;d.validate_complete(out,cfg['fingerprint'])
    flat=[];targets=[];screens=[];audited=0;warm_steps=spec['head_warmup_epochs']*spec['batches_per_epoch']
    for seed,order in zip(spec['seeds'],spec['orders']):
        initial_model=p.make_model(seed,fake);initial=initial_model.state_dict();records={};histories={};warm_states={};last_rng=None
        for position,arm in enumerate(order):
            parent=out/f'seed_{seed}'/arm;rc=json.loads((parent/'run_config.json').read_text())
            assert rc['fingerprint']==v.digest({k:x for k,x in rc.items() if k!='fingerprint'})
            assert rc['seed']==seed and rc['arm']==arm and rc['position']==position and rc['root_fingerprint']==cfg['fingerprint']
            marker=d.validate_complete(parent,rc['fingerprint']);assert marker;attempt=Path(marker['attempt'])
            summary=json.loads((attempt/'summary.json').read_text());steps=d.records(attempt/'steps.csv');epochs=d.records(attempt/'epochs.csv')
            assert summary['fingerprint']==rc['fingerprint'] and summary['root_fingerprint']==cfg['fingerprint']
            assert summary['seed']==seed and summary['arm']==arm and summary['position']==position
            assert summary['initial_hash']==old.state_hash(initial)
            assert len(steps)==summary['updates']==spec['updates_per_run'] and len(epochs)==summary['epochs']==spec['epochs']
            counts=dict(prefix=0,suffix=0,head=0)
            for i,row in enumerate(steps):
                epoch=i//spec['batches_per_epoch']+1;bi=i%spec['batches_per_epoch'];mode=expected_mode(arm,i,spec)
                assert int(row['step'])==i+1 and int(row['epoch'])==epoch and row['mode']==mode
                assert int(row['batch_examples'])==min(spec['batch_size'],(16 if fake else 45000)-bi*spec['batch_size'])
                assert math.isfinite(float(row['loss'])) and float(row['loss'])>=0
                close(row['backbone_lr'],p.rate_factor(i,spec)*spec['backbone_lr']);close(row['head_lr'],p.rate_factor(i,spec)*spec['head_lr'])
                counts['prefix']+=int(mode=='full');counts['suffix']+=int(mode!='head');counts['head']+=1
                for key,num in counts.items():assert int(row[key+'_updates'])==num
            assert summary['group_updates']==counts
            train=validation=observed=0.
            for i,er in enumerate(epochs):
                ss=steps[i*spec['batches_per_epoch']:(i+1)*spec['batches_per_epoch']];seen=sum(int(x['batch_examples']) for x in ss)
                assert seen==(16 if fake else 45000)
                assert int(er['epoch'])==i+1 and int(er['updates'])==(i+1)*spec['batches_per_epoch']
                close(er['training_loss'],sum(float(x['loss'])*int(x['batch_examples']) for x in ss)/seen)
                assert 0<=float(er['validation_accuracy'])<=1 and math.isfinite(float(er['validation_loss']))
                assert float(er['epoch_training_seconds'])>0 and float(er['epoch_validation_seconds'])>0
                train+=float(er['epoch_training_seconds']);validation+=float(er['epoch_validation_seconds'])
                close(er['training_seconds'],train);close(er['validation_seconds'],validation)
                assert float(er['observed_seconds'])>observed;observed=float(er['observed_seconds']);assert observed>=train+validation-1e-6
            close(summary['training_seconds'],train);close(summary['validation_seconds'],validation);close(summary['observed_seconds'],observed)
            close(summary['final_accuracy'],epochs[-1]['validation_accuracy']);close(summary['final_loss'],epochs[-1]['validation_loss'])
            assert math.isfinite(summary['total_run_seconds']) and summary['total_run_seconds']>=observed+summary['hardware_warmup_seconds']+summary['final_checkpoint_seconds']-1e-6
            warm=torch.load(attempt/'warmup_checkpoint.pt',map_location='cpu',weights_only=True)
            final=torch.load(attempt/'final_checkpoint.pt',map_location='cpu',weights_only=True)
            state_check(warm,initial,initial_model,arm,seed,warm_steps,spec,cfg,rc)
            state_check(final,initial,initial_model,arm,seed,len(steps),spec,cfg,rc)
            warm_states[arm]=warm
            if last_rng is not None:
                for k in ['batch_rng','augmentation_rng']:assert torch.equal(last_rng[k],final[k])
            last_rng={k:final[k] for k in ['batch_rng','augmentation_rng']}
            histories[arm]=([x['loss'] for x in steps[:warm_steps]],[(e['training_loss'],e['validation_loss'],e['validation_accuracy']) for e in epochs[:spec['head_warmup_epochs']]])
            for target_role in ['readiness','secondary']:
                threshold=spec[target_role+'_target'];first,last=target(epochs,threshold,spec['sustained_epochs'])
                targets.append(dict(seed=seed,arm=arm,role=target_role,threshold=threshold,reached=last is not None,
                    first_crossing_epoch=int(first['epoch']) if first else '',confirming_epoch=int(last['epoch']) if last else '',
                    observed_seconds=float(last['observed_seconds']) if last else '',censoring_epoch='' if last else spec['epochs']))
                if target_role=='readiness':summary['ready']=last is not None
            records[arm]=summary
            flat.append({k:summary[k] for k in ['seed','arm','final_accuracy','training_seconds','validation_seconds','total_run_seconds','ready']})
            audited+=len(steps);del final
        for arm in ['lp_full','lp_suffix','lp_periodic']:
            assert histories[arm]==histories['head'],('Matched warmup mismatch',seed,arm)
            same_state(warm_states['head'],warm_states[arm])
        screens.append(dict(seed=seed,**r.screen(records,spec)))
    result=dict(status='PASS',synthetic=fake,fingerprint=cfg['fingerprint'],runs=10,audited_updates=audited,
        pretrained_source_verified=not fake,frozen_bn_and_inactive_parameters_verified=True,group_optimizer_counters_verified=True,
        matched_head_warmups_bitwise=True,data_rng_streams_matched=True,
        baselines_ready_both_seeds=all(s['baseline_ready'] for s in screens),quality_pass_both_seeds=all(s['quality_pass'] for s in screens),
        speed_pass_both_seeds=all(s['speed_pass'] for s in screens),opportunity_both_seeds=all(s['opportunity_pass'] for s in screens),inference=spec['inference'])
    if write:
        v.atomic_json(out/'audit.json',result)
        for name,rows in [('all_runs.csv',flat),('targets.csv',targets),('screen.csv',screens)]:v.write_csv(out/name,rows)
        lines=['# Stage12A pretrained selective-adaptation screen','',spec['inference'],'',
            '| Seed | Arm | Final accuracy | Complete-run seconds |','|---:|---|---:|---:|']
        for x in flat:lines.append(f'| {x["seed"]} | {x["arm"]} | {100*x["final_accuracy"]:.2f}% | {x["total_run_seconds"]:.2f} |')
        lines+=['',f'Synthetic: {fake}. Both-seed opportunity screen: {result["opportunity_both_seeds"]}.',
            'Frozen-BN semantics and all optimizer counters checked. Head-warmup states/histories match exactly.',
            'Readiness failure is inconclusive. No extension, tuning or novelty claim. Review before more experiments.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text())==result
        for name,rows in [('all_runs.csv',flat),('targets.csv',targets),('screen.csv',screens)]:
            saved=d.records(out/name);assert len(saved)==len(rows)
            for a,b in zip(rows,saved):
                for k,value in a.items():
                    if isinstance(value,(str,bool)):assert str(value)==b[k]
                    else:close(value,b[k])
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=r.REAL);audit(ap.parse_args().root,False)
