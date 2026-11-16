"""Audit fixed-source local objective diagnostic and reconstruct review flags."""
import argparse,json,math,statistics as st
from pathlib import Path
import torch
import stage13_diagnostic as r
l=r.l;p=r.p;v=r.v;d=r.d;old=r.old
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-10),(a,b)
def audit(out,write=False):
    p.execution.configure();out=Path(out);cfg=json.loads((out/'configuration.json').read_text());spec=cfg['spec'];fake=cfg['synthetic']
    assert cfg['fingerprint']==v.digest({k:x for k,x in cfg.items() if k!='fingerprint'})
    frozen=json.loads((out/'SYNTHETIC_FREEZE.json' if fake else r.FREEZE).read_text())
    assert frozen==r.freeze_record(spec,fake) and cfg['freeze_hash']==frozen['freeze_hash']
    assert cfg['source_sha256']==frozen['source_sha256'] and cfg['checkpoints']==frozen['checkpoints'] and cfg['split']==frozen['split']
    assert cfg['execution']==frozen['required_execution']==p.execution.execution_settings()
    if (out/'COMPLETE.json').exists():assert not write;d.validate_complete(out,cfg['fingerprint'])
    marker=json.loads((out/'RAW_COMPLETE.json').read_text());assert marker['fingerprint']==cfg['fingerprint']
    for path,h in marker['sha256'].items():assert p.sha(path)==h,path
    attempt=Path(marker['attempt']);summaries=[];flags=[];operations=0
    # Reconstruct index pools from the same train-only labels; no scoring/training.
    x,y,split=r.data(torch.device('cpu'),fake);assert split==cfg['split'];del x
    for seed in spec['source_seeds']:
        stream=json.loads((attempt/f'streams_{seed}.json').read_text());assert stream==r.streams(y,seed,spec)
        pool=[set(stream['pools'][key]) for key in ['calibration','probe','score']]
        assert all(not pool[a]&pool[b] for a,b in [(0,1),(0,2),(1,2)])
        meta=json.loads((attempt/f'seed_{seed}.json').read_text());assert meta['seed']==seed and meta['fingerprint']==cfg['fingerprint']
        assert meta['source_backbone_unchanged'] and meta['probe_trials']==spec['probe_trials']
        assert meta['calibration_overhead_seconds']>0
        model=l.LocalModel(p.make_model(seed,fake),seed,fake)
        cp=torch.load(attempt/f'calibrated_{seed}.pt',map_location='cpu',weights_only=True)
        assert cp['seed']==seed and cp['fingerprint']==cfg['fingerprint'] and cp['calibration_steps']==spec['calibration_steps']
        model.load_state_dict(cp['model_state'],strict=True);initial=cp['model_state']
        assert old.state_hash(initial)==meta['calibrated_hash']
        assert old.state_hash(model.net.state_dict())==cp['source_hash']==meta['source_hash']
        if not fake:
            source=torch.load(frozen['checkpoints'][str(seed)]['path'],map_location='cpu',weights_only=True)
            assert old.state_hash(source['model_state'])==cp['source_hash']
            del source
        assert all(torch.isfinite(t).all() for t in initial.values())
        aux=list(model.aux0.parameters())+list(model.aux1.parameters());opt=cp['aux_optimizer_state']
        ids=opt['param_groups'][0]['params'];assert len(ids)==len(aux)==len(opt['state'])
        for idx,t in zip(ids,aux):
            assert int(opt['state'][idx]['step'])==spec['calibration_steps']
            for key in ['exp_avg','exp_avg_sq']:assert opt['state'][idx][key].shape==t.shape and torch.isfinite(opt['state'][idx][key]).all()
        cal=d.records(attempt/f'calibration_{seed}.csv');assert len(cal)==spec['calibration_steps']
        for i,row in enumerate(cal):
            assert int(row['seed'])==seed and int(row['step'])==i+1
            for key in ['loss0','loss1']:assert math.isfinite(float(row[key])) and float(row[key])>=0
        cold=d.records(attempt/f'cold_{seed}.csv');assert len(cold)==4
        for mode,row in zip(spec['modes'],cold):assert row['mode']==mode and int(row['seed'])==seed and float(row['seconds'])>0
        rows=d.records(attempt/f'measurements_{seed}.csv');assert len(rows)==spec['probe_trials']*4==meta['measured_operations']
        index={(int(a['trial']),a['mode']):a for a in rows};assert len(index)==len(rows)
        for trial in range(spec['probe_trials']):
            order=spec['modes'][trial%4:]+spec['modes'][:trial%4]
            if (trial//4)%2:order=list(reversed(order))
            before=float(index[trial,'full']['score_loss_before'])
            for position,mode in enumerate(order):
                row=index[trial,mode]
                assert int(row['seed'])==seed and int(row['order_position'])==position and int(row['optimizer_age'])==1
                assert math.isfinite(float(row['seconds'])) and float(row['seconds'])>0
                close(row['score_loss_before'],before)
                close(row['score_improvement'],float(row['score_loss_before'])-float(row['score_loss_after']))
                for key in ['score_loss_before','score_loss_after','update_loss_sum','final_head_training_loss']:assert math.isfinite(float(row[key])) and float(row[key])>=0
                assert int(row['peak_allocated_bytes'])>=int(row['baseline_allocated_bytes'])>=0
                assert int(row['incremental_peak_bytes'])==int(row['peak_allocated_bytes'])-int(row['baseline_allocated_bytes'])
                close(row['final_head_training_loss'],index[trial,'full']['final_head_training_loss'])
        for mode in spec['modes']:
            step=torch.load(attempt/f'one_step_{seed}_{mode}.pt',map_location='cpu',weights_only=True)
            assert step['seed']==seed and step['mode']==mode and step['fingerprint']==cfg['fingerprint']
            model.load_state_dict(step['model_state'],strict=True)
            assert all(torch.isfinite(t).all() for t in model.state_dict().values())
            for name,t in model.named_buffers():assert torch.equal(t,initial[name]),name
            expected=l.optimizer(model,spec);actual=step['optimizer_state'];states=actual['state'];active_ids=[]
            names={id(t):name for name,t in model.named_parameters()}
            for group,saved in zip(expected.param_groups,actual['param_groups']):
                assert len(group['params'])==len(saved['params']);close(group['lr'],saved['lr']);close(group['weight_decay'],saved['weight_decay'])
                for param,idx in zip(group['params'],saved['params']):
                    name=names[id(param)]
                    if l.active(name,mode):
                        active_ids.append(idx);assert int(states[idx]['step'])==1
                        for key in ['exp_avg','exp_avg_sq']:assert states[idx][key].shape==param.shape and torch.isfinite(states[idx][key]).all()
                    else:assert idx not in states and torch.equal(param,initial[name]),name
            assert set(active_ids)==set(states)
            del step,expected
        alignment=d.records(attempt/f'alignment_{seed}.csv');assert len(alignment)==spec['probe_trials']*12
        seen=set()
        for a in alignment:
            key=(int(a['trial']),a['mode'],a['block']);assert key not in seen;seen.add(key)
            assert int(a['seed'])==seed and 0<=key[0]<spec['probe_trials'] and key[1] in spec['modes'] and key[2] in ['block0','block1','block2']
            aa,bb,ab=map(float,[a['norm_squared'],a['full_norm_squared'],a['dot']]);assert all(math.isfinite(t) for t in [aa,bb,ab]) and min(aa,bb)>=0
            den=math.sqrt(aa*bb);assert a['defined']==str(den>0)
            if den>0:close(a['cosine'],ab/den);assert abs(float(a['cosine']))<=1+1e-7
            else:assert a['cosine']==''
            if key[1]=='suffix' and key[2]!='block2':assert aa==0
            if key[2]=='block2' or key[1]=='full':close(aa,bb);close(ab,bb)
        aggregate={}
        for mode in spec['modes']:
            subset=[a for a in rows if a['mode']==mode]
            aggregate[mode]=dict(seed=seed,mode=mode,mean_step_seconds=st.mean(float(a['seconds']) for a in subset),
                mean_score_improvement=st.mean(float(a['score_improvement']) for a in subset),
                positive_score_fraction=st.mean(float(a['score_improvement'])>0 for a in subset),
                mean_incremental_peak_bytes=st.mean(int(a['incremental_peak_bytes']) for a in subset),
                mean_absolute_peak_bytes=st.mean(int(a['peak_allocated_bytes']) for a in subset))
        full=aggregate['full'];local=aggregate['local'];ratio=local['mean_step_seconds']/full['mean_step_seconds']
        memory=local['mean_incremental_peak_bytes']/full['mean_incremental_peak_bytes'] if full['mean_incremental_peak_bytes']>0 else None
        utility=full['mean_score_improvement']>0 and local['mean_score_improvement']>0 and local['mean_score_improvement']>=spec['minimum_local_full_utility_ratio']*full['mean_score_improvement']
        resources=ratio<=1-spec['minimum_time_saving'] or (memory is not None and memory<=1-spec['minimum_incremental_memory_saving'])
        saved_per_step=full['mean_step_seconds']-local['mean_step_seconds']
        for mode in spec['modes']:
            row=aggregate[mode];row.update(local_full_time_ratio=ratio,local_full_incremental_memory_ratio=memory if memory is not None else '',
                local_utility_review_flag=utility,local_resource_review_flag=resources,qualifies_for_pilot_review=utility and resources,
                calibration_overhead_seconds=meta['calibration_overhead_seconds'],
                optimistic_calibration_break_even_steps=math.ceil(meta['calibration_overhead_seconds']/saved_per_step) if saved_per_step>0 else '')
            summaries.append(row)
        flags.append(utility and resources);operations+=len(rows)
    result=dict(status='PASS',synthetic=fake,fingerprint=cfg['fingerprint'],measured_operations=operations,
        calibration_steps=len(spec['source_seeds'])*spec['calibration_steps'],warmup_operations=len(spec['source_seeds'])*4*spec['warmup_steps_per_mode'],
        disjoint_training_pools_verified=True,source_backbones_unchanged=True,one_step_states_and_counters_verified=True,
        both_source_states_qualify_for_pilot_review=all(flags),inference=spec['inference'])
    if write:
        v.atomic_json(out/'audit.json',result);v.write_csv(out/'summary.csv',summaries)
        lines=['# Stage13A local-objective diagnostic','',spec['inference'],'',
            '| Source seed | Mode | Mean step ms | Mean score-loss improvement | Incremental peak MiB |',
            '|---:|---|---:|---:|---:|']
        for a in summaries:lines.append(f'| {a["seed"]} | {a["mode"]} | {1000*a["mean_step_seconds"]:.3f} | {a["mean_score_improvement"]:.6f} | {a["mean_incremental_peak_bytes"]/2**20:.2f} |')
        lines+=['',f'Synthetic: {fake}. Both-source review flag: {all(flags)}.',
            'One-step utility is not convergence evidence. Timing excludes state restore, score evaluation and gradient diagnostics.',
            'Calibration overhead and optimistic break-even are reported. No automatic efficacy training or retuning.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((out/'audit.json').read_text())==result
        saved=d.records(out/'summary.csv');assert len(saved)==len(summaries)
        for a,b in zip(summaries,saved):
            for key,value in a.items():
                if isinstance(value,(str,bool)):assert str(value)==b[key]
                else:close(value,b[key])
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=r.REAL);audit(ap.parse_args().root,False)
