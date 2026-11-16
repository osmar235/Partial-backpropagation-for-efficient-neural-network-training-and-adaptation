"""Validate all frozen Stage 8 traces, source provenance and final AdamW counters."""
import argparse,csv,json,math
from pathlib import Path
import torch
import stage8_vit_v2 as v

def records(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-8),(a,b)

def audit(root,freeze_path):
    record=json.loads(Path(freeze_path).read_text());spec=record['spec'];root=Path(root)
    assert record['freeze_hash']==v.digest({k:x for k,x in record.items() if k!='freeze_hash'})
    assert record['source_sha256']==v.source_hashes()
    expected={f'seed_{s}' for s in spec['evaluation_seeds']}
    assert {d.name for d in root.glob('seed_*')}==expected
    verified=total_steps=0
    for seed,order in zip(spec['evaluation_seeds'],spec['evaluation_orders']):
        initial=set()
        assert {d.name for d in (root/f'seed_{seed}').iterdir() if d.is_dir()}==set(spec['variants'])
        for variant in order:
            folder=root/f'seed_{seed}'/variant
            r=json.loads((folder/'summary.json').read_text());config=json.loads((folder/'run_config.json').read_text())
            assert r['status']=='complete' and r['fingerprint']==config['fingerprint']
            assert config['fingerprint']==v.digest({k:x for k,x in config.items() if k!='fingerprint'})
            assert config['source_sha256']==record['source_sha256'] and config['spec']==spec
            assert config['lr']==record['selected_lr'] and config['freeze_hash']==record['freeze_hash']
            assert config['seed']==seed and config['variant']==variant and config['order_position']==order.index(variant)
            assert r['freeze_hash']==record['freeze_hash'] and r['order_position']==order.index(variant)
            initial.add(r['initial_model_sha256'])
            steps=records(folder/'step_trace.csv');epochs=records(folder/'epoch_trace.csv')
            batches=4 if record['fake'] else 196;total=batches*spec['epochs']
            assert len(steps)==total==r['updates'] and len(epochs)==spec['epochs']
            mask=v.periodic_indices(spec['epochs'],batches,record['periodic_epoch_q'])
            ema=None;previous=False
            mix_rng=v.np.random.default_rng(seed+98001)
            for i,s in enumerate(steps):
                lv=float(s['loss']);partial=bool(int(s['partial']))
                assert math.isfinite(lv) and not(partial and previous)
                assert int(s['step'])==i+1 and int(s['epoch'])==i//batches+1
                if ema is None:assert s['ema_before']==''
                else:close(s['ema_before'],ema)
                chosen=(i in mask) if variant=='periodic_matched' else (
                    variant!='full_adamw' and ema is not None and not previous and lv<=ema)
                assert partial==chosen,(seed,variant,i)
                close(s['lr'],v.learning_rate(i,total,batches,record['selected_lr'],spec))
                close(s['mixup_lambda'],float(mix_rng.beta(spec['mixup_alpha'],spec['mixup_alpha'])))
                ema=lv if ema is None else spec['rho']*ema+(1-spec['rho'])*lv;previous=partial
                for name in ('augmentation','forward','backward','optimizer'):assert float(s[name+'_seconds'])>=0
            assert sum(int(s['partial']) for s in steps)==r['partial_updates']
            assert r['full_updates']+r['partial_updates']==total
            close(r['partial_fraction'],r['partial_updates']/total)
            for e,epoch in enumerate(epochs):
                selected=steps[e*batches:(e+1)*batches]
                assert int(epoch['partial_updates'])==sum(int(s['partial']) for s in selected)
                assert int(epoch['full_updates'])+int(epoch['partial_updates'])==batches
                close(epoch['cumulative_training_seconds'],sum(float(x['epoch_training_seconds']) for x in epochs[:e+1]))
            close(r['training_wall_seconds'],sum(float(e['epoch_training_seconds']) for e in epochs))
            close(r['final_eval_accuracy'],epochs[-1]['eval_accuracy']);close(r['final_eval_loss'],epochs[-1]['eval_loss'])
            for key in ('augmentation','forward','backward','optimizer'):
                close(r['component_seconds'][key],sum(float(s[key+'_seconds']) for s in steps))
            for flag,label in ((0,'full'),(1,'partial')):
                close(r['component_seconds']['backward_'+label],sum(float(s['backward_seconds']) for s in steps if int(s['partial'])==flag))
            close(r['timing_unattributed_seconds'],r['training_wall_seconds']-sum(r['component_seconds'][key] for key in ('augmentation','forward','backward','optimizer')))
            assert r['scalar_loss_item_calls']==total
            assert r['timed_explicit_cuda_sync_calls']==(8*total+2*spec['epochs'] if config['device']=='cuda' else 0)
            for target in spec['targets']:
                label=round(target*100);first=next((e for e in epochs if float(e['eval_accuracy'])>=target),None)
                assert r[f'epoch_to_{label}pct']==(int(first['epoch']) if first else None)
                if first:close(r[f'training_only_time_to_{label}pct'],first['cumulative_training_seconds'])
                else:assert r[f'training_only_time_to_{label}pct'] is None
            checkpoint=folder/'final_checkpoint.pt'
            assert v.hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
            ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert ck['fingerprint']==r['fingerprint'] and ck['variant']==variant and ck['seed']==seed
            # Optimizer parameter IDs correspond to this preserved named-parameter order.
            model=v.make_model(spec,tiny=record['tiny'])
            model.load_state_dict(ck['model_state'],strict=True)
            names=[name for name,p in model.named_parameters()]
            ids=[i for group in ck['optimizer_state']['param_groups'] for i in group['params']]
            for name,param_id in zip(names,ids):
                expected_steps=r['full_updates'] if variant=='replay_prefix' and name in r['prefix_parameter_names'] else total
                assert int(ck['optimizer_state']['state'][param_id]['step'])==expected_steps,(variant,name)
            verified+=1;total_steps+=total
        assert len(initial)==1,'Initialization differs across paired variants'
    out=dict(status='PASS',runs=verified,steps=total_steps,checkpoint_counters_verified=True,
             identical_initialization_per_seed=True,freeze_hash=record['freeze_hash'],synthetic=record['fake'])
    v.atomic_json(root/'stage8_integrity_audit.json',out)
    print('INTEGRITY AUDIT:',json.dumps(out),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--freeze-file',type=Path,required=True);a=p.parse_args();audit(a.root,a.freeze_file)
