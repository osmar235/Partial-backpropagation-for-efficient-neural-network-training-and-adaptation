"""Independent, read-only review of completed Stage9C; outputs a separate audit."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/stage9C_boundary_diagnostic_v1'
OUT = ROOT / 'results/stage9C_review_v1'
FEATURES = ['raw_loss', 'excess_loss', 'boundary_grad_rms', 'activation_grad_rms']
METRICS = ['prefix_update_error_l2', 'relative_prefix_update_error', 'validation_omit_minus_full']

def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write(name, rows):
    with (OUT/name).open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def close(x, y):
    assert math.isclose(float(x), float(y), rel_tol=1e-10, abs_tol=1e-12), (x, y)

def quantile(a, q):
    a = sorted(a); pos = (len(a)-1)*q; lo = math.floor(pos); hi = math.ceil(pos)
    return a[lo] + (a[hi]-a[lo])*(pos-lo)

def ranks(a):
    return [sum(x < v for x in a) + (sum(x == v for x in a)-1)/2 for v in a]

def rho(a, b):
    a, b = ranks(a), ranks(b)
    return st.correlation(a, b) if len(set(a))>1 and len(set(b))>1 else ''

def main():
    p = argparse.ArgumentParser(); p.add_argument('--check-checkpoints', action='store_true'); p.add_argument('--verify-preservation', action='store_true'); args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    updates_path=OUT/'documentation_updates_sha256.json'
    updates=json.loads(updates_path.read_text()) if updates_path.exists() else {}
    allowed_updates={str(ROOT/'README_STAGE9.md'),str(ROOT/'CURRENT_RELEASE.md')}
    assert set(updates)<=allowed_updates
    if args.verify_preservation:
        original=read(OUT/'preserved_inputs_sha256.csv')
        for r in original: assert sha(r['path'])==updates.get(r['path'],r['sha256']),r['path']
        result=dict(status='PASS',files=len(original),intentional_documentation_updates=updates)
        (OUT/'preservation_after_review.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2)); return
    config = json.loads((RAW/'configuration.json').read_text()); fp=config.pop('fingerprint')
    assert hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()==fp
    assert not config['fake'] and config['device']=='cuda'
    assert config['split']['kind']=='train_only_stratified_45000_5000'
    spec=config['spec']; protected={}
    def protect(path, expected=None):
        path=Path(path); actual=sha(path)
        if expected is not None: assert actual==updates.get(str(path),expected.lower()), str(path)
        protected[str(path)]=actual
    for path, h in config['source_sha256'].items(): protect(path, h)
    for path in RAW.rglob('*'):
        if path.is_file(): protect(path)
    markers=list(RAW.rglob('COMPLETE.json')); assert len(markers)==11
    for path in markers:
        marker=json.loads(path.read_text()); assert marker['fingerprint']==fp
        for filename,h in marker['sha256'].items(): protect(filename,h)
    complete=json.loads((RAW/'COMPLETE.json').read_text())
    assert complete['status']=='complete' and complete['measurements']==6144 and not complete['synthetic']

    expected_prefix=[22080,466944,911808,1356672,1801536,2246400]
    all_rows=[]; training=[]; checkpoints=[]; telemetry=[]
    for seed in spec['development_seeds']:
        folder=RAW/f'seed_{seed}'; tr=json.loads((folder/'training/COMPLETE.json').read_text())
        assert tr['updates']==5280
        epochs=read(Path(tr['attempt'])/'epochs.csv'); assert len(epochs)==30
        for i,r in enumerate(epochs,1):
            assert int(r['epoch'])==i and int(r['updates'])==i*176
            for k in ['mean_training_loss','training_seconds','lr']: assert math.isfinite(float(r[k])) and float(r[k])>0
            assert bool(r['validation_accuracy'])==(i in spec['checkpoint_epochs'])
            if r['validation_accuracy']:
                training.append(dict(seed=seed,epoch=i,updates=int(r['updates']),validation_accuracy=float(r['validation_accuracy']),validation_loss=float(r['validation_loss'])))
        for epoch in spec['checkpoint_epochs']:
            checkpoint=Path(tr['checkpoints'][str(epoch)]); checkpoints.append((seed,epoch,checkpoint))
            marker=json.loads((folder/f'probes/epoch_{epoch:03d}/COMPLETE.json').read_text())
            assert marker['checkpoint_sha256']==sha(checkpoint) and marker['rows']==768
            attempt=Path(marker['attempt']); rows=read(attempt/'measurements.csv'); assert len(rows)==768
            indices=json.loads((attempt/'probe_indices.json').read_text()); assert len(indices)==64
            keys={(r['role'],r['probe']) for r in indices}; assert len(keys)==64
            assert keys=={(role,i) for role in ['calibration','scoring'] for i in range(32)}
            for key,limit,batch in [('training_indices',45000,128),('validation_indices',5000,64)]:
                lists=[[i for r in indices if r['role']==role for i in r[key]] for role in ['calibration','scoring']]
                assert all(len(a)==32*batch and len(set(a))==len(a) and min(a)>=0 and max(a)<limit for a in lists)
                assert not set(lists[0]) & set(lists[1])
            for r in indices:
                role_i=int(r['role']=='scoring')
                assert r['dropout_seed']==seed+epoch*10000+role_i*100+r['probe']+124001
            row_keys=set()
            for r in rows:
                assert int(r['seed'])==seed and int(r['epoch'])==epoch
                cut=int(r['cut']); assert cut in range(6) and r['mode'] in ['zero','freeze']
                key=(r['role'],int(r['probe']),cut,r['mode']); assert key not in row_keys; row_keys.add(key)
                assert (key[0],key[1]) in keys
                assert int(r['prefix_parameters'])==expected_prefix[cut] and int(r['model_parameters'])==2693578
                for k,val in r.items():
                    if k not in ['role','mode']: assert math.isfinite(float(val)), (k,val)
                close(r['validation_omit_minus_full'],float(r['validation_omitted_loss'])-float(r['validation_full_loss']))
                common=sum(float(r[k]) for k in ['augmentation_seconds','split_forward_seconds','suffix_backward_seconds','features_seconds'])
                close(r['split_full_seconds'],common+float(r['prefix_backward_seconds'])+float(r['full_optimizer_seconds']))
                close(r['split_partial_seconds'],common+float(r['omitted_optimizer_seconds']))
                if r['mode']=='freeze': close(r['relative_prefix_update_error'],1)
                assert 0<=float(r['raw_loss'])-float(r['excess_loss'])<=math.log(10)+1e-6
                for k,val in r.items():
                    if k.endswith('_seconds'): assert float(val)>0
            assert len(row_keys)==768
            grouped=collections.defaultdict(list)
            for r in rows: grouped[(r['role'],r['probe'])].append(r)
            for branches in grouped.values():
                for k in ['raw_loss','excess_loss','mixup_lambda','normal_full_seconds','augmentation_seconds','validation_full_loss','probe_learning_rate']:
                    assert len({r[k] for r in branches})==1, k
                for cut in range(6):
                    pair=[r for r in branches if int(r['cut'])==cut]; assert len(pair)==2
                    for k in FEATURES+['split_full_seconds','prefix_backward_seconds','cut_order_position']:
                        assert pair[0][k]==pair[1][k]
            all_rows.extend(rows)
        for path in (folder).rglob('telemetry.csv'):
            samples=read(path)
            for row in samples:
                data=json.loads(row['sample']); assert data['available']
                for gpu in data['gpus']:
                    telemetry.append(dict(seed=seed,file=str(path.relative_to(RAW)),temperature=float(gpu['temperature.gpu']),utilization=float(gpu['utilization.gpu']),clock=float(gpu['clocks.sm'])))
    assert all_rows==read(RAW/'all_measurements.csv') and len(all_rows)==6144

    # Recompute every calibration threshold, selection, rank correlation, control,
    # oracle bound and cost projection from raw rows without importing executed code.
    groups=collections.defaultdict(list)
    for r in all_rows: groups[(int(r['seed']),int(r['epoch']),int(r['cut']),r['mode'])].append(r)
    comparisons=read(RAW/'signal_comparisons.csv'); costs=read(RAW/'projected_costs.csv')
    assert len(comparisons)==768 and len(costs)==192 and len(groups)==96
    detail=[]
    for r in comparisons:
        key=(int(r['seed']),int(r['epoch']),int(r['cut']),r['mode']); g=groups[key]
        cal=[a for a in g if a['role']=='calibration']; score=sorted([a for a in g if a['role']=='scoring'],key=lambda a:int(a['probe']))
        f=r['feature']; q=float(r['calibration_quantile']); threshold=quantile([float(a[f]) for a in cal],q)
        close(threshold,r['threshold']); chosen=[a for a in score if float(a[f])<=threshold]; k=len(chosen)
        assert int(r['calibration_n'])==len(cal)==32 and int(r['scoring_n'])==len(score)==32 and int(r['selected_n'])==k
        close(k/32,r['realized_eligibility_fraction'])
        close(rho([float(a[f]) for a in score],[float(a[METRICS[0]]) for a in score]),r['spearman_score_update_error'])
        periodic=[score[int((i+.5)*32/k)] for i in range(k)] if k else []
        for m in METRICS:
            for label,subset in [('selected',chosen),('blind_expected',score),('periodic',periodic),('oracle',sorted(score,key=lambda a:float(a[m]))[:k])]:
                if subset: close(st.mean(float(a[m]) for a in subset),r[label+'_'+m])
                else: assert r[label+'_'+m]==''
        if not k: continue
        selected_ids={int(a['probe']) for a in chosen}
        actual_cost=st.mean(float(a['split_partial_seconds'] if int(a['probe']) in selected_ids else a['split_full_seconds']) for a in score)
        normal=st.mean(float(a['normal_full_seconds']) for a in score)
        blind_cost=st.mean((1-k/32)*float(a['split_full_seconds'])+k/32*float(a['split_partial_seconds']) for a in score)
        detail.append(dict(seed=key[0],epoch=key[1],cut=key[2],mode=key[3],feature=f,quantile=q,selected_n=k,eligibility=k/32,
            error_ratio=float(r['selected_prefix_update_error_l2'])/float(r['blind_expected_prefix_update_error_l2']),
            rho=float(r['spearman_score_update_error']),
            selected_validation_delta=float(r['selected_validation_omit_minus_full']),
            blind_validation_delta=float(r['blind_expected_validation_omit_minus_full']),
            selected_minus_blind_validation=float(r['selected_validation_omit_minus_full'])-float(r['blind_expected_validation_omit_minus_full']),
            selected_minus_periodic_validation=float(r['selected_validation_omit_minus_full'])-float(r['periodic_validation_omit_minus_full']),
            eligibility_projected_ratio=actual_cost/normal,matched_blind_projected_ratio=blind_cost/normal))
    for r in costs:
        key=(int(r['seed']),int(r['epoch']),int(r['cut']),r['mode']); score=[a for a in groups[key] if a['role']=='scoring']; q=float(r['assumed_partial_fraction'])
        base=st.mean(float(a['normal_full_seconds']) for a in score)
        projected=st.mean((1-q)*float(a['split_full_seconds'])+q*float(a['split_partial_seconds']) for a in score)
        close(base,r['normal_full_seconds']); close(projected,r['projected_split_seconds']); close(projected/base,r['projected_ratio'])
        close(st.mean(float(a['features_seconds']) for a in score),r['mean_features_seconds'])
    summary=[]
    for cut in range(6):
        for mode in ['zero','freeze']:
            for f in FEATURES:
                for q in [.2,.4]:
                    a=[r for r in detail if r['cut']==cut and r['mode']==mode and r['feature']==f and r['quantile']==q]
                    summary.append(dict(cut=cut,mode=mode,feature=f,quantile=q,states=len(a),mean_rho=st.mean(r['rho'] for r in a),
                        mean_error_ratio=st.mean(r['error_ratio'] for r in a),mean_eligibility=st.mean(r['eligibility'] for r in a),
                        mean_selected_validation_delta=st.mean(r['selected_validation_delta'] for r in a),mean_blind_validation_delta=st.mean(r['blind_validation_delta'] for r in a),
                        mean_selected_minus_blind_validation=st.mean(r['selected_minus_blind_validation'] for r in a),
                        mean_selected_minus_periodic_validation=st.mean(r['selected_minus_periodic_validation'] for r in a),
                        states_better_than_blind=sum(r['selected_minus_blind_validation']<0 for r in a),
                        states_better_than_periodic=sum(r['selected_minus_periodic_validation']<0 for r in a),
                        mean_eligibility_projected_ratio=st.mean(r['eligibility_projected_ratio'] for r in a)))
    cost_summary=[]
    for cut in range(6):
        for mode in ['zero','freeze']:
            state_costs=[]; breaks=[]; fulls=[]
            for key,g in groups.items():
                if key[2:]!=(cut,mode): continue
                a=[r for r in g if r['role']=='scoring']; base=st.mean(float(r['normal_full_seconds']) for r in a)
                full=st.mean(float(r['split_full_seconds']) for r in a); partial=st.mean(float(r['split_partial_seconds']) for r in a)
                state_costs.append((.6*full+.4*partial)/base); fulls.append(full/base); breaks.append((full-base)/(full-partial))
            score=[r for r in all_rows if int(r['cut'])==cut and r['mode']==mode and r['role']=='scoring']
            cost_summary.append(dict(cut=cut,mode=mode,prefix_parameters=expected_prefix[cut],mean_ratio_40=st.mean(state_costs),min_ratio_40=min(state_costs),max_ratio_40=max(state_costs),
                mean_split_full_ratio=st.mean(fulls),mean_break_even_omission_fraction=st.mean(breaks),
                mean_omission_validation_delta=st.mean(float(r['validation_omit_minus_full']) for r in score)))
    for r in read(RAW/'boundary_cost_summary.csv'):
        a=next(a for a in cost_summary if a['cut']==int(r['cut']) and a['mode']==r['mode']); close(a['mean_ratio_40'],r['mean_projected_ratio'])

    checkpoint_result='not requested'
    if args.check_checkpoints:
        import torch
        for seed,epoch,path in checkpoints:
            cp=torch.load(path,map_location='cpu',weights_only=True)
            assert cp['seed']==seed and cp['epoch']==epoch and cp['updates']==epoch*176 and cp['fingerprint']==fp
            assert sum(t.numel() for t in cp['model_state'].values())==2693578
            assert len(cp['optimizer_state']['state'])==len(cp['model_state'])
            for t in cp['model_state'].values(): assert torch.isfinite(t).all()
            for state in cp['optimizer_state']['state'].values():
                assert int(state['step'])==epoch*176
                assert state['step'].device.type=='cpu'
                assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
                assert (state['exp_avg_sq']>=0).all()
        checkpoint_result='PASS: eight CPU-loaded checkpoints; all parameter/moment tensors finite; step counters and identities verified'

    prior=[]
    for name in ['protocol/stage7/frozen_files_before.csv','results/stage8A_review_v1/preserved_v1_sha256.csv','protocol/stage8/stage8_v2_preparation_sha256.csv','results/stage8A_review_v2/preserved_development_sha256.csv','results/stage8B_review_v2/preserved_evaluation_sha256.csv','protocol/stage9/boundary_preparation_sha256_v1.csv']:
        rows=read(ROOT/name)
        for r in rows:
            filename=r.get('path',r.get('Path')); expected=r.get('sha256',r.get('Hash'))
            protect(filename,expected)
        prior.append(dict(manifest=name,files=len(rows),status='PASS'))
    manuscript=json.loads((ROOT/'NeuralNetworks/rendered_v7/verification.json').read_text())['sha256']
    for path,h in manuscript.items(): protect(ROOT/path,h)
    prior.append(dict(manifest='NeuralNetworks/rendered_v7/verification.json',files=len(manuscript),status='PASS'))
    # Additional cheap-feature comparisons are explicitly post hoc. Their
    # direction was chosen after inspecting this diagnostic, not preregistered.
    cheap=[]
    for key,g in groups.items():
        cal=[a for a in g if a['role']=='calibration']
        score=sorted([a for a in g if a['role']=='scoring'],key=lambda a:int(a['probe']))
        functions={
            'negative_mixup_strength':lambda a:-min(float(a['mixup_lambda']),1-float(a['mixup_lambda'])),
            'negative_target_entropy':lambda a:float(a['excess_loss'])-float(a['raw_loss'])}
        for q in [.2,.4]:
            boundary_threshold=quantile([float(a['boundary_grad_rms']) for a in cal],q)
            boundary=[a for a in score if float(a['boundary_grad_rms'])<=boundary_threshold]
            for feature,fun in functions.items():
                threshold=quantile([fun(a) for a in cal],q)
                selected=[a for a in score if fun(a)<=threshold]
                if not selected or not boundary: continue
                # Top-k cheap-feature ranks only: equal count for information
                # comparison, but uses the score window and is NOT an online policy.
                topk=sorted(score,key=lambda a:(fun(a),int(a['probe'])))[:len(boundary)]
                cheap.append(dict(seed=key[0],epoch=key[1],cut=key[2],mode=key[3],feature=feature,quantile=q,
                    post_hoc=True,calibration_only_threshold=threshold,selected_n=len(selected),boundary_selected_n=len(boundary),
                    selected_error_ratio=st.mean(float(a['prefix_update_error_l2']) for a in selected)/st.mean(float(a['prefix_update_error_l2']) for a in score),
                    selected_validation_minus_blind=st.mean(float(a['validation_omit_minus_full']) for a in selected)-st.mean(float(a['validation_omit_minus_full']) for a in score),
                    boundary_minus_cheap_validation=st.mean(float(a['validation_omit_minus_full']) for a in boundary)-st.mean(float(a['validation_omit_minus_full']) for a in selected),
                    boundary_error_over_equal_count_cheap=st.mean(float(a['prefix_update_error_l2']) for a in boundary)/st.mean(float(a['prefix_update_error_l2']) for a in topk),
                    boundary_minus_equal_count_cheap_validation=st.mean(float(a['validation_omit_minus_full']) for a in boundary)-st.mean(float(a['validation_omit_minus_full']) for a in topk),
                    equal_count_overlap_fraction=len({a['probe'] for a in boundary}&{a['probe'] for a in topk})/len(boundary),
                    rho_boundary_vs_cheap=rho([float(a['boundary_grad_rms']) for a in score],[fun(a) for a in score])))
    write('posthoc_cheap_feature_comparisons.csv',cheap)
    # Save preservation baseline before any mutable project pointers are updated.
    baseline=OUT/'preserved_inputs_sha256.csv'
    if baseline.exists():
        for r in read(baseline): assert sha(r['path'])==updates.get(r['path'],r['sha256']),r['path']
    else:
        write('preserved_inputs_sha256.csv',[dict(path=p,sha256=h) for p,h in sorted(protected.items())])
    write('signal_state_details.csv',detail); write('signal_summary.csv',summary); write('cost_summary.csv',cost_summary); write('training_checkpoints.csv',training)
    audit=dict(status='PASS',fingerprint=fp,raw_rows=6144,distinct_probe_batches=512,scoring_probe_batches=256,
        model_seeds=2,states=8,markers=11,independently_recomputed_signal_rows=768,independently_recomputed_cost_rows=192,
        checkpoint_audit=checkpoint_result,prior_preservation=prior,
        telemetry_samples=len(telemetry),temperature_range=[min(r['temperature'] for r in telemetry),max(r['temperature'] for r in telemetry)],
        clock_range=[min(r['clock'] for r in telemetry),max(r['clock'] for r in telemetry)])
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    print('BOUNDARY GRADIENT, 40% CALIBRATION QUANTILE')
    for r in summary:
        if r['feature']=='boundary_grad_rms' and r['quantile']==.4: print(r)

if __name__=='__main__': main()
