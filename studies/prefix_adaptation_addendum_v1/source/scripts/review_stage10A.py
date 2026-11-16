"""Trace/checkpoint audit and fixed practical/detail/structure screen."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse, csv, hashlib, json, math

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-11),(a,b)
def completion(path):
    marker=json.loads((path/'COMPLETE.json').read_text())
    for p,h in marker['sha256'].items():assert sha(p)==h,p
    return marker
def state_hash(state):
    h=hashlib.sha256()
    for name,tensor in state.items():h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def target(rows,value,n):
    streak=0;first=None
    for row in rows:
        if float(row['validation_accuracy'])>=value:
            if first is None:first=row
            streak+=1
        else:streak=0
        if streak>=n:return first,row
    return first,None

def audit(out,write=False):
    import numpy as np
    import torch
    import stage10_models as m
    m.configure();out=Path(out);cfg=json.loads((out/'configuration.json').read_text());spec=cfg['spec']
    assert digest({k:v for k,v in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    freeze=out/'SYNTHETIC_FREEZE.json' if cfg['synthetic'] else ROOT/'protocol/stage10/FROZEN_REPRESENTATION_v1.json'
    frozen=json.loads(freeze.read_text())
    assert digest({k:v for k,v in frozen.items() if k!='freeze_hash'})==cfg['freeze_hash']==frozen['freeze_hash']
    assert spec==frozen['spec'] and cfg['source_sha256']==frozen['source_sha256'] and cfg['split']==frozen['expected_split']
    assert cfg['execution']==frozen['required_execution']==m.execution_settings()
    assert cfg['execution']['deterministic'] and not cfg['execution']['warn_only']
    assert cfg['execution']['math_sdp'] and not any(cfg['execution'][k] for k in ['flash_sdp','mem_efficient_sdp','cudnn_sdp'])
    for p,h in {**frozen['source_sha256'],**frozen['evidence_sha256']}.items():assert sha(p)==h,p
    if (out/'COMPLETE.json').exists():assert completion(out)['fingerprint']==cfg['fingerprint']
    if write:assert not (out/'COMPLETE.json').exists()
    runs={};targets=[];flat=[];components={};streams={};compatible_rng={};steps_count=0
    for seed,order in zip(spec['development_seeds'],spec['orders']):
        for position,variant in enumerate(order):
            parent=out/f'seed_{seed}'/variant;marker=completion(parent);attempt=Path(marker['attempt'])
            rc=json.loads((parent/'run_config.json').read_text());summary=json.loads((attempt/'summary.json').read_text())
            assert digest({k:v for k,v in rc.items() if k!='fingerprint'})==rc['fingerprint']==marker['fingerprint']==summary['fingerprint']
            assert rc['root_fingerprint']==summary['root_fingerprint']==cfg['fingerprint']
            assert rc['seed']==summary['seed']==seed and rc['variant']==summary['variant']==variant and rc['order_position']==summary['order_position']==position
            rows=read(attempt/'steps.csv');epochs=read(attempt/'epochs.csv')
            assert len(rows)==summary['updates']==spec['expected_updates_per_run'] and len(epochs)==spec['epochs']==summary['epochs']
            mg=np.random.default_rng(seed+98001);train=val=0.;last_observed=0.
            for e,er in enumerate(epochs,1):
                group=rows[(e-1)*spec['expected_batches_per_epoch']:e*spec['expected_batches_per_epoch']]
                loss_sum=examples=0
                for batch_index,r in enumerate(group):
                    step=int(r['step'])-1;assert int(r['epoch'])==e and step//spec['expected_batches_per_epoch']==e-1
                    expected_examples=min(spec['batch_size'],(16 if cfg['synthetic'] else 45000)-batch_index*spec['batch_size'])
                    assert int(r['batch_examples'])==expected_examples
                    assert int(r['dropout_seed'])==seed+spec['dropout_seed_offset']+step
                    lam=float(mg.beta(spec['mixup_alpha'],spec['mixup_alpha']));close(lam,r['mixup_lambda'])
                    total=spec['schedule_horizon_epochs']*spec['expected_batches_per_epoch'];warm=spec['lr_warmup_epochs']*spec['expected_batches_per_epoch']
                    lr=spec['learning_rate']*(step+1)/warm if step<warm else spec['learning_rate']*(spec['min_lr_ratio']+(1-spec['min_lr_ratio'])*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1))))
                    close(lr,r['lr']);assert math.isfinite(float(r['loss'])) and 0<int(r['batch_examples'])<=spec['batch_size']
                    loss_sum+=float(r['loss'])*int(r['batch_examples']);examples+=int(r['batch_examples'])
                assert examples==(16 if cfg['synthetic'] else 45000)
                assert int(er['epoch'])==e and int(er['updates'])==e*spec['expected_batches_per_epoch']
                close(loss_sum/examples,er['training_loss']);assert 0<=float(er['validation_accuracy'])<=1 and math.isfinite(float(er['validation_loss']))
                train+=float(er['epoch_training_seconds']);val+=float(er['validation_seconds'])
                close(train,er['cumulative_training_seconds']);close(val,er['cumulative_validation_seconds'])
                observed=float(er['cumulative_observed_seconds']);assert observed>last_observed and observed>=train+val-1e-6;last_observed=observed
            assert [int(r['step']) for r in rows]==list(range(1,len(rows)+1))
            close(train,summary['training_seconds']);close(val,summary['validation_seconds']);close(last_observed,summary['observed_seconds'])
            close(epochs[-1]['validation_accuracy'],summary['final_validation_accuracy']);close(epochs[-1]['validation_loss'],summary['final_validation_loss'])
            model=m.make_model(variant,seed,cfg['synthetic'])
            assert state_hash(model.state_dict())==summary['initial_state_sha256']
            for name,module in model.named_children():assert state_hash(module.state_dict())==summary['initial_component_sha256'][name]
            cp=torch.load(attempt/'final_checkpoint.pt',map_location='cpu',weights_only=True)
            assert cp['seed']==seed and cp['variant']==variant and cp['updates']==len(rows)
            assert cp['root_fingerprint']==cfg['fingerprint'] and cp['run_fingerprint']==rc['fingerprint']
            names=list(cp['model_state']);ids=[i for g in cp['optimizer_state']['param_groups'] for i in g['params']]
            assert names==list(model.state_dict()) and len(names)==len(ids)==len(cp['optimizer_state']['state'])
            for name,i in zip(names,ids):
                tensor=cp['model_state'][name];state=cp['optimizer_state']['state'][i]
                assert tensor.shape==model.state_dict()[name].shape and torch.isfinite(tensor).all()
                assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all() and (state['exp_avg_sq']>=0).all()
                assert state['exp_avg'].shape==state['exp_avg_sq'].shape==tensor.shape and int(state['step'])==len(rows)
            for g in cp['optimizer_state']['param_groups']:close(g['lr'],rows[-1]['lr'])
            assert summary['parameter_count']==sum(p.numel() for p in model.parameters())
            data_state={k:cp[k] for k in ['batch_rng','augmentation_rng','permutation_rng','mixup_rng']}
            if seed in streams:
                for key in ['batch_rng','augmentation_rng','permutation_rng']:assert torch.equal(streams[seed][key],data_state[key]),key
                assert streams[seed]['mixup_rng']==data_state['mixup_rng']
            else:streams[seed]=data_state
            if variant in ['coarse_rgb16','pixel_branches','haar_branches']:
                components.setdefault(seed,{});components[seed][variant]=summary['initial_component_sha256']
                rng=(cp['torch_rng'],cp['cuda_rng'])
                if seed in compatible_rng:
                    a,b=compatible_rng[seed];assert torch.equal(a,rng[0]) and len(b)==len(rng[1])
                    for x,y in zip(b,rng[1]):assert torch.equal(x,y)
                else:compatible_rng[seed]=rng
            for role,threshold in [('primary',spec['primary_target']),('secondary',spec['secondary_target'])]:
                first,confirmed=target(epochs,threshold,spec['sustained_epochs'])
                record=dict(seed=seed,variant=variant,role=role,threshold=threshold,reached=confirmed is not None,
                    first_crossing_epoch=int(first['epoch']) if first else '',confirming_epoch=int(confirmed['epoch']) if confirmed else '',
                    observed_seconds=float(confirmed['cumulative_observed_seconds']) if confirmed else '',
                    training_seconds=float(confirmed['cumulative_training_seconds']) if confirmed else '',
                    censoring_epoch=spec['epochs'] if not confirmed else '',censoring_observed_seconds=summary['observed_seconds'] if not confirmed else '')
                targets.append(record)
                if role=='primary':summary['primary_reached']=record['reached'];summary['primary_observed_seconds']=record['observed_seconds']
            runs[seed,variant]=summary;steps_count+=len(rows)
            flat.append({k:summary[k] for k in ['seed','variant','order_position','parameter_count','updates','final_validation_accuracy','training_seconds','validation_seconds',
                'observed_seconds','hardware_warmup_seconds','checkpoint_seconds','experiment_elapsed_seconds','primary_reached','primary_observed_seconds']})
            del cp,model
    for seed,a in components.items():
        assert len({a[k]['body'] for k in a})==1
        for field in ['detail','detail_norm','head']:assert a['pixel_branches'][field]==a['haar_branches'][field]
        assert runs[seed,'pixel_branches']['parameter_count']==runs[seed,'haar_branches']['parameter_count']
    screen=[]
    for seed in spec['development_seeds']:
        a=runs[seed,'haar_branches'];full=runs[seed,'full_rgb32'];coarse=runs[seed,'coarse_rgb16'];pixel=runs[seed,'pixel_branches']
        controls=[runs[seed,k] for k in ['full_rgb32','small_rgb32','coarse_rgb16']]
        reached=[r for r in controls if r['primary_reached']];best=min(reached,key=lambda r:(r['primary_observed_seconds'],r['variant'])) if reached else None
        best_accuracy=max(r['final_validation_accuracy'] for r in controls)
        ratio_full=a['primary_observed_seconds']/full['primary_observed_seconds'] if a['primary_reached'] and full['primary_reached'] else None
        ratio_best=a['primary_observed_seconds']/best['primary_observed_seconds'] if a['primary_reached'] and best else None
        delta=100*(a['final_validation_accuracy']-best_accuracy)
        practical=ratio_full is not None and ratio_best is not None and max(ratio_full,ratio_best)<=1-spec['minimum_time_gain'] and delta>=-spec['final_accuracy_margin_pp']
        detail_gain=100*(a['final_validation_accuracy']-coarse['final_validation_accuracy'])
        structure_gain=100*(a['final_validation_accuracy']-pixel['final_validation_accuracy']);cost=a['training_seconds']/pixel['training_seconds']
        detail_pass=detail_gain>=spec['minimum_detail_gain_pp']
        structure_pass=structure_gain>=spec['minimum_structure_gain_pp'] and cost<=spec['maximum_structure_training_cost_ratio']
        screen.append(dict(seed=seed,full_target_ready=full['primary_reached'],haar_target_reached=a['primary_reached'],fastest_ordinary_control=best['variant'] if best else '',
            target_time_ratio_to_full=ratio_full if ratio_full is not None else '',target_time_ratio_to_best_ordinary=ratio_best if ratio_best is not None else '',
            final_delta_vs_best_ordinary_pp=delta,detail_gain_vs_coarse_pp=detail_gain,structure_gain_vs_pixel_pp=structure_gain,
            training_ratio_vs_pixel=cost,practical_pass=practical,detail_pass=detail_pass,structure_pass=structure_pass,all_flags_pass=practical and detail_pass and structure_pass))
    result=dict(status='PASS',synthetic=cfg['synthetic'],fingerprint=cfg['fingerprint'],runs=10,audited_updates=steps_count,
        all_parameter_counters_equal_updates=True,data_rng_streams_matched=True,compatible_coarse_dropout_streams_matched=True,
        initial_coarse_bodies_and_dual_components_matched=True,full_baseline_ready=all(r['full_target_ready'] for r in screen),
        both_seed_practical_pass=all(r['practical_pass'] for r in screen),both_seed_detail_pass=all(r['detail_pass'] for r in screen),
        both_seed_structure_pass=all(r['structure_pass'] for r in screen),promising_development_hypothesis=all(r['all_flags_pass'] for r in screen),
        inference='DEVELOPMENT REPRESENTATION SCREEN ONLY; no optimizer hybrid, confirmatory test, novelty or publication result')
    if write:
        def save(name,rows):
            with (out/name).open('w',encoding='utf8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        save('all_runs.csv',flat);save('targets.csv',targets);save('screen.csv',screen)
        (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        lines=['# Stage10A coarse/detail representation screen','',result['inference'],'',
            '| Seed | Arm | Parameters | Final validation accuracy | Sustained primary target | Observed seconds to target |',
            '|---:|---|---:|---:|---|---:|']
        for r in flat:
            label=f'{r["primary_observed_seconds"]:.2f}' if r['primary_reached'] else 'not reached (censored)'
            lines.append(f'| {r["seed"]} | {r["variant"]} | {r["parameter_count"]} | {100*r["final_validation_accuracy"]:.2f}% | {r["primary_reached"]} | {label} |')
        lines+=['',f'Synthetic: {cfg["synthetic"]}. Full-baseline readiness: {result["full_baseline_ready"]}.',
            f'Both-seed practical/detail/structure flags: {result["both_seed_practical_pass"]}/{result["both_seed_detail_pass"]}/{result["both_seed_structure_pass"]}.',
            f'Promising development hypothesis: {result["promising_development_hypothesis"]}.',
            'All parameters received every update; transform/seed/backprop costs are included.',
            'All censoring retained. Two seeds and reused validation cannot establish broad superiority or significance.',
            'The pixel control matches trainable dual-branch capacity; representation conditioning/allocation remain part of the comparison.',
            'Do not automatically change thresholds, architecture, duration or optimizer. Review all outputs first.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','synthetic','runs','audited_updates','full_baseline_ready','both_seed_practical_pass','both_seed_detail_pass','both_seed_structure_pass']},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT/'results/stage10A_representation_v1')
    audit(parser.parse_args().root,write=False)
