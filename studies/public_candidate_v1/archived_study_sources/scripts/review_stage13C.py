"""Frozen audit for matched local-learning trajectories and charged costs."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import stage13c_trajectory as r
from review_stage13B import metrics
p,v,d,old=r.p,r.v,r.d,r.old


def screen(records,spec,reference_ok,synthetic):
    controls=['full','full_aux','suffix']; best=max(records[c]['accuracy'] for c in controls)
    qualified=[c for c in controls if 100*(best-records[c]['accuracy'])<=spec['accuracy_margin_pp']+1e-10]
    fastest=min(qualified,key=lambda c:(records[c]['charged_training_seconds'],c))
    smallest=min(qualified,key=lambda c:(records[c]['max_incremental_allocated'],c))
    local,full=records['local'],records['full']
    tr=max(local['charged_training_seconds']/full['charged_training_seconds'],local['charged_training_seconds']/records[fastest]['charged_training_seconds'])
    denom=min(full['max_incremental_allocated'],records[smallest]['max_incremental_allocated'])
    mr=local['max_incremental_allocated']/denom if denom else None
    quality=100*(best-local['accuracy'])<=spec['accuracy_margin_pp']+1e-10
    improves=local['accuracy']>local['initial_accuracy'] and local['loss']<local['initial_loss']
    time_pass=tr<=1-spec['minimum_time_saving']; memory_pass=mr is not None and mr<=1-spec['minimum_memory_saving']
    return dict(best_control_accuracy=best,qualified_controls=';'.join(qualified),fastest_qualified_control=fastest,
        lowest_memory_qualified_control=smallest,local_delta_vs_best_pp=100*(local['accuracy']-best),
        local_time_ratio_to_required_controls=tr,local_memory_ratio_to_required_controls=mr,
        local_quality_pass=quality,local_improves=improves,local_time_pass=time_pass,local_memory_pass=memory_pass,
        seed_screen_pass=bool(reference_ok and quality and improves and (time_pass or memory_pass) and not synthetic))


def audit(out,write=False):
    out=Path(out); cfg=json.loads((out/'configuration.json').read_text())
    assert v.digest({k:val for k,val in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    spec,recipe,fake=cfg['spec'],cfg['recipe'],cfg['synthetic']
    record=json.loads((out/'SYNTHETIC_FREEZE.json' if fake else r.FREEZE).read_text())
    assert v.digest({k:val for k,val in record.items() if k!='freeze_hash'})==record['freeze_hash']==cfg['freeze_hash']
    assert record['spec']==spec and record['recipe']==recipe and record['split']==cfg['split']
    r.b.verify(record['source_sha256']); r.b.verify(record['evidence_sha256'])
    for cp in record['checkpoints'].values(): r.b.verify({cp['path']:cp['sha256']})
    if not fake: r.b.verify({record['pretrained']['path']:record['pretrained']['sha256']})
    raw=json.loads((out/'RAW_COMPLETE.json').read_text()); assert raw['fingerprint']==cfg['fingerprint']
    r.b.verify(raw['sha256']); attempt=Path(raw['attempt'])
    assert set(raw['sha256'])=={str(f) for f in attempt.iterdir() if f.is_file()}
    x,y,ev,ey,split=p.load_data(torch.device('cpu'),fake); assert split==cfg['split']
    labels=ey.numpy(); del x,ev
    curves,summaries,reference_flags,screen_flags=[],[],[],[]
    for seed in spec['seeds']:
        cp=torch.load(attempt/f'source_{seed}.pt' if fake else record['checkpoints'][str(seed)]['path'],map_location='cpu',weights_only=True)
        assert cp['seed']==seed and cp['updates']==2*recipe['batches_per_epoch']
        calibrated=torch.load(attempt/f'calibrated_{seed}.pt',map_location='cpu',weights_only=True)
        calibration=json.loads((attempt/f'calibration_{seed}.json').read_text())
        assert math.isfinite(calibration['seconds']) and calibration['seconds']>0
        assert len(d.records(attempt/f'calibration_losses_{seed}.csv'))==spec['calibration_steps']
        for row in d.records(attempt/f'calibration_losses_{seed}.csv'):
            assert all(math.isfinite(float(row[k])) for k in ('loss0','loss1'))
        if not fake:
            original=torch.load(record['auxiliary_sources'][str(seed)]['path'],map_location='cpu',weights_only=True)
            assert r.equal_state(calibrated['model_state'],original['model_state'])
            assert r.equal_state(calibrated['aux_optimizer_state'],original['aux_optimizer_state'])
            stream=json.loads(Path(record['auxiliary_sources'][str(seed)]['stream']).read_text())
            assert calibrated['batches']==stream['calibration_batches']
            del original
        assert len(calibrated['batches'])==spec['calibration_steps']
        bg=torch.Generator(); bg.set_state(cp['batch_rng'])
        batches=[ix.tolist() for ix in v.frozen6.epoch_batches(len(y),recipe['batch_size'],bg)]
        assert sorted(i for batch in batches for i in batch)==list(range(len(y)))
        ag=torch.Generator(); ag.set_state(cp['augmentation_rng'])
        for batch in batches:
            torch.randint(0,9,(len(batch),2),generator=ag); torch.rand(len(batch),generator=ag)
        expected=json.loads((attempt/f'expected_trace_{seed}.json').read_text())
        if not fake:
            assert expected==r.b.original_trace(record['checkpoints'][str(seed)],recipe)
            entry=record['checkpoints'][str(seed)]
            expected0={k:float(entry['initial']['validation_'+k]) for k in ('loss','accuracy')}
            expected1={k:float(entry['expected']['validation_'+k]) for k in ('loss','accuracy')}
        else:
            with np.load(attempt/f'golden_{seed}.npz',allow_pickle=False) as z:
                expected0,expected1=metrics(z['initial'],labels),metrics(z['final'],labels)
        records={}; full_ok=False; initial_logits=None
        for arm in spec['arms']:
            stem=f'{seed}_{arm}'; state=json.loads((attempt/f'state_{stem}.json').read_text())
            model,opt,_,_=r.restore(seed,cp,calibrated,recipe,fake,torch.device('cpu'),arm)
            assert state['initial_hash']==old.state_hash(model.state_dict())
            assert state['start_ages']==r.b.ages(model,opt)
            for name,age in state['start_ages'].items():
                target=cp['updates'] if name.startswith('net.fc.') else spec['calibration_steps'] if name.startswith('aux') and arm in ('local','full_aux') else 0
                assert age==target,(name,age,target)
            final=torch.load(attempt/f'final_{stem}.pt',map_location='cpu',weights_only=True)
            assert final['seed']==seed and final['arm']==arm and final['fingerprint']==cfg['fingerprint']
            model.load_state_dict(final['model_state']); opt.load_state_dict(final['optimizer_state'])
            assert state['final_hash']==old.state_hash(model.state_dict())
            assert state['final_ages']==r.b.ages(model,opt)
            for name,t in model.named_parameters():
                assert torch.isfinite(t).all()
                assert state['final_ages'][name]==state['start_ages'][name]+(len(batches) if r.l.active(name,arm) else 0)
                if not r.l.active(name,arm): assert torch.equal(t,calibrated['model_state'][name]) and t not in opt.state
            for name,t in model.named_buffers(): assert torch.equal(t,calibrated['model_state'][name])
            for values in opt.state.values():
                assert all(torch.isfinite(z).all() for z in values.values() if torch.is_tensor(z))
            assert torch.equal(final['batch_rng'],bg.get_state()) and torch.equal(final['augmentation_rng'],ag.get_state())
            assert final['updates']==len(batches)==recipe['batches_per_epoch']
            assert json.loads((attempt/f'batches_{stem}.json').read_text())==batches
            rows=d.records(attempt/f'steps_{stem}.csv'); assert len(rows)==len(expected)==len(batches)
            trace_match=[]
            for i,(row,batch,original) in enumerate(zip(rows,batches,expected),1):
                assert int(row['step'])==i and int(row['global_step'])==cp['updates']+i==int(original['step'])
                assert int(row['batch_examples'])==len(batch)==int(original['batch_examples'])
                assert row['batch_sha256']==old.state_hash({'indices':torch.tensor(batch)})
                for k,origkey in [('backbone_lr','backbone_lr'),('head_aux_lr','head_lr')]:
                    assert math.isclose(float(row[k]),float(original[origkey]),rel_tol=1e-12,abs_tol=1e-15)
                for k in ('final_loss','objective_sum','operation_seconds','training_seconds'):
                    assert math.isfinite(float(row[k])) and float(row[k])>0
                assert float(row['training_seconds'])>=float(row['operation_seconds'])
                base,peak,inc,reserved=[int(row[k]) for k in ('baseline_allocated','peak_allocated','incremental_allocated','peak_reserved')]
                assert peak-base==inc>=0 and reserved>=peak
                if cfg['device']=='cuda': assert base>0 and inc>0
                else: assert base==peak==inc==reserved==0
                trace_match.append(abs(float(row['final_loss'])-float(original['loss']))<=spec['trace_atol']+spec['trace_rtol']*abs(float(original['loss'])))
            with np.load(attempt/f'scores_{stem}.npz',allow_pickle=False) as z:
                assert set(z.files)=={'labels'}|{f'step_{s}' for s in spec['score_steps']}
                assert np.array_equal(z['labels'],labels)
                if initial_logits is None: initial_logits=z['step_0'].copy()
                else: assert np.array_equal(initial_logits,z['step_0'])
                for step in spec['score_steps']:
                    assert z[f'step_{step}'].shape==(len(labels),10)
                    curves.append(dict(seed=seed,arm=arm,step=step,**metrics(z[f'step_{step}'],labels)))
                initial,endpoint=metrics(z['step_0'],labels),metrics(z[f'step_{len(batches)}'],labels)
            ref0=abs(initial['accuracy']-expected0['accuracy'])<=spec['accuracy_atol'] and abs(initial['loss']-expected0['loss'])<=spec['score_loss_atol']
            ref1=abs(endpoint['accuracy']-expected1['accuracy'])<=spec['accuracy_atol'] and abs(endpoint['loss']-expected1['loss'])<=spec['score_loss_atol']
            if arm=='full': full_ok=ref0 and ref1 and all(trace_match) and endpoint['loss']<initial['loss'] and endpoint['accuracy']>initial['accuracy']
            charge=calibration['seconds'] if arm in ('full_aux','local') else 0
            assert charge==state['calibration_charge_seconds'] and state['all_gradients_verified']
            training=sum(float(row['training_seconds']) for row in rows)
            assert state['arm_elapsed_seconds']>=training
            records[arm]=dict(seed=seed,arm=arm,**endpoint,initial_loss=initial['loss'],initial_accuracy=initial['accuracy'],
                training_seconds=training,operation_seconds=sum(float(row['operation_seconds']) for row in rows),
                calibration_charge_seconds=charge,charged_training_seconds=training+charge,
                max_incremental_allocated=max(int(row['incremental_allocated']) for row in rows),
                max_absolute_allocated=max(int(row['peak_allocated']) for row in rows),
                calibration_peak_allocated=calibration['peak_allocated'] if charge else 0,
                full_trace_steps_matching=sum(trace_match) if arm=='full' else '',reference_initial_reproduced=ref0)
            del model,opt,final
        flags=screen(records,spec,full_ok,fake); reference_flags.append(full_ok); screen_flags.append(flags['seed_screen_pass'])
        for arm in spec['arms']: summaries.append(dict(**records[arm],reference_qualified=full_ok,**flags))
        del cp,calibrated
    result=dict(status='PASS',synthetic=fake,integrity_pass=True,trajectories=8,
        trajectory_updates=8*recipe['batches_per_epoch'],calibration_updates=2*spec['calibration_steps'],
        both_seed_reference_qualified=all(reference_flags),both_seed_local_screen_pass=all(screen_flags),
        streams_optimizer_states_and_costs_verified=True,source_files_unchanged=True,automatic_followup=False)
    if write:
        v.atomic_json(out/'audit.json',result); v.write_csv(out/'curves.csv',curves); v.write_csv(out/'summary.csv',summaries)
        lines=['# Stage13C local trajectory screen','','```json',json.dumps(result,indent=2),'```','',
            '| Seed | Arm | Accuracy % | CE | Charged training s | Max incremental MiB |', '|---|---|---:|---:|---:|---:|']
        for row in summaries:
            lines.append(f"| {row['seed']} | {row['arm']} | {100*row['accuracy']:.2f} | {row['loss']:.5f} | {row['charged_training_seconds']:.3f} | {row['max_incremental_allocated']/2**20:.2f} |")
        lines+=['','Review before further training. Memory-only passes are not speedup claims; inspect calibration peaks too.',
            'All outcomes use previously examined development data. Synthetic results have no scientific interpretation.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text())==result
        for name,rows in [('curves.csv',curves),('summary.csv',summaries)]:
            saved=d.records(out/name); assert len(saved)==len(rows)
            for a,b in zip(saved,rows):
                for k,val in b.items():
                    if val is None: assert a[k]==''
                    elif isinstance(val,(str,bool)): assert a[k]==str(val)
                    else: assert math.isclose(float(a[k]),val,rel_tol=1e-10,abs_tol=1e-10)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('output',type=Path); args=ap.parse_args()
    print(json.dumps(audit(args.output,False),indent=2))
