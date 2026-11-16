"""Independent trace, quota, past-only controller, checkpoint and pilot-screen audit."""
from pathlib import Path
import argparse, collections, csv, hashlib, json, math, random, statistics as st

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-11),(a,b)
def linear_quantile(values,q):
    a=sorted(values);p=(len(a)-1)*q;lo=math.floor(p);hi=math.ceil(p)
    return a[lo]+(a[hi]-a[lo])*(p-lo)
def completion(folder):
    r=json.loads((folder/'COMPLETE.json').read_text())
    for p,h in r['sha256'].items():assert sha(p)==h,p
    return r
def target(rows,threshold,sustained):
    streak=0;first=None
    for r in rows:
        if float(r['validation_accuracy'])>=threshold:
            if first is None:first=r
            streak+=1
        else:streak=0
        if streak>=sustained:return first,r
    return first,None
def count(n,k,previous):
    if k<0 or n<0:return 0
    if k==0:return 1
    a=n-k+1-int(previous)
    return math.comb(a,k) if a>=k else 0

def audit(out,write=False):
    import numpy as np
    import torch
    out=Path(out);cfg=json.loads((out/'configuration.json').read_text());spec=cfg['spec']
    assert digest({k:v for k,v in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    freeze_path=out/'SYNTHETIC_FREEZE.json' if cfg['synthetic'] else ROOT/'protocol/stage9/FROZEN_QUALITY_PILOT_v1.json'
    frozen=json.loads(freeze_path.read_text())
    assert digest({k:v for k,v in frozen.items() if k!='freeze_hash'})==frozen['freeze_hash']==cfg['freeze_hash']
    assert frozen['spec']==spec and frozen['source_sha256']==cfg['source_sha256'] and frozen['expected_split']==cfg['split']
    for p,h in {**frozen['source_sha256'],**frozen['evidence_sha256']}.items():assert sha(p)==h,p
    if (out/'COMPLETE.json').exists():assert completion(out)['fingerprint']==cfg['fingerprint']
    if write:assert not (out/'COMPLETE.json').exists(),'Completed output is read only'
    base=json.loads((ROOT/'protocol/stage8/study_spec_v2.json').read_text())
    all_runs=[];targets=[];runs={};initials={};stream_states={};audited_steps=0;reason_counts=collections.Counter()
    for seed,order in zip(spec['development_seeds'],spec['orders']):
        for position,variant in enumerate(order):
            folder=out/f'seed_{seed}'/variant;marker=completion(folder);attempt=Path(marker['attempt'])
            rc=json.loads((folder/'run_config.json').read_text())
            assert digest({k:v for k,v in rc.items() if k!='fingerprint'})==rc['fingerprint']==marker['fingerprint']
            assert rc['root_fingerprint']==cfg['fingerprint'] and rc['seed']==seed and rc['variant']==variant and rc['order_position']==position
            summary=json.loads((attempt/'summary.json').read_text());epochs=read(attempt/'epochs.csv');steps=read(attempt/'steps.csv')
            assert summary['fingerprint']==rc['fingerprint'] and summary['root_fingerprint']==cfg['fingerprint']
            assert summary['status']=='complete' and summary['seed']==seed and summary['variant']==variant and summary['order_position']==position
            assert len(epochs)==spec['epochs'] and len(steps)==summary['updates']==spec['expected_total_updates_per_arm']
            initials.setdefault(seed,summary['initial_state_sha256']);assert initials[seed]==summary['initial_state_sha256']
            history=collections.deque(maxlen=spec['history_size']);rng=random.Random(seed+171001);mix=np.random.default_rng(seed+98001)
            previous=False;n=k=window_position=0;partials=omitted=0;training=validation=0.;prior_observed=0.
            for expected_epoch,epoch_row in enumerate(epochs,1):
                epoch=int(epoch_row['epoch']);assert epoch==expected_epoch
                active=epoch>spec['full_warmup_epochs'];regular_left=spec['expected_regular_batches_per_epoch'];n=k=0
                group=steps[(epoch-1)*spec['expected_batches_per_epoch']:epoch*spec['expected_batches_per_epoch']]
                ep_partial=ep_omitted=seen=0;loss_sum=0.
                for j,r in enumerate(group):
                    step=(epoch-1)*spec['expected_batches_per_epoch']+j
                    assert int(r['epoch'])==epoch and int(r['step'])==step+1
                    examples=int(r['batch_examples']);is_regular=j<spec['expected_regular_batches_per_epoch']
                    assert examples==spec['batch_size'] if is_regular else 0<examples<spec['batch_size']
                    lam=float(mix.beta(spec['mixup_alpha'],spec['mixup_alpha']));close(lam,r['mixup_lambda'])
                    warm=base['lr_warmup_epochs']*spec['expected_batches_per_epoch']
                    total=spec['schedule_horizon_epochs']*spec['expected_batches_per_epoch']
                    expected_lr=spec['learning_rate']*(step+1)/warm if step<warm else spec['learning_rate']*(base['min_lr_ratio']+(1-base['min_lr_ratio'])*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1))))
                    close(expected_lr,r['lr']);assert math.isfinite(float(r['loss']))
                    collect=is_regular and epoch>=spec['full_warmup_epochs'] and variant in ['mixup_only','boundary_rms']
                    assert bool(r['signal'])==collect
                    signal=float(r['signal']) if collect else None
                    if signal is not None:
                        assert math.isfinite(signal)
                        if variant=='mixup_only':close(signal,-min(lam,1-lam))
                        else:assert signal>=0
                    assert int(r['history_count'])==len(history)
                    if not active or not is_regular or variant=='full_adamw':
                        expected_partial=False;requested=0;threshold='';reason='full' if variant=='full_adamw' else ('warmup' if not active else 'short_batch')
                        assert int(r['remaining_n'])==int(r['remaining_k'])==-1 and r['random_draw']==''
                    else:
                        if not n:
                            n=min(regular_left,spec['quota_window']);k=round(n*spec['omission_fraction']);window_position=0
                            plan={int((i+.5)*n/k) for i in range(k)} if k else set()
                        assert int(r['remaining_n'])==n and int(r['remaining_k'])==k
                        threshold=linear_quantile(history,spec['calibration_quantile']) if variant in ['mixup_only','boundary_rms'] else ''
                        if threshold!='':assert len(history)==spec['history_size'];close(threshold,r['threshold'])
                        skip=count(n-1,k-1,True) if not previous else 0;full=count(n-1,k,False);assert skip+full>0
                        if variant=='periodic':requested=int(window_position in plan)
                        elif variant=='random':
                            draw=rng.random();close(draw,r['random_draw']);requested=int(draw<skip/(skip+full))
                        else:requested=int(signal<=threshold)
                        if variant!='random':assert r['random_draw']==''
                        if not skip:expected_partial=False;reason='refresh' if previous else 'quota_exhausted'
                        elif not full:expected_partial=True;reason='quota_deadline'
                        else:expected_partial=bool(requested);reason=variant
                        n-=1;k-=int(expected_partial);regular_left-=1;window_position+=1
                        assert count(n,k,expected_partial)>0
                    if threshold=='':assert r['threshold']==''
                    assert int(r['partial'])==int(expected_partial) and int(r['requested'])==requested and r['reason']==reason
                    assert not (previous and expected_partial);previous=expected_partial
                    if signal is not None:history.append(signal)
                    ep_partial+=int(expected_partial);ep_omitted+=examples*int(expected_partial);seen+=examples;loss_sum+=examples*float(r['loss'])
                    reason_counts[variant+':'+reason]+=1
                if active and variant!='full_adamw':assert regular_left==n==k==0
                expected=spec['expected_partial_updates_per_active_epoch'] if active and variant!='full_adamw' else 0
                assert ep_partial==int(epoch_row['partial_updates'])==expected and ep_omitted==int(epoch_row['omitted_prefix_examples'])==expected*spec['batch_size']
                assert int(epoch_row['updates'])==epoch*spec['expected_batches_per_epoch']
                close(loss_sum/seen,epoch_row['training_loss']);partials+=ep_partial;omitted+=ep_omitted
                training+=float(epoch_row['epoch_training_seconds']);validation+=float(epoch_row['validation_seconds'])
                close(training,epoch_row['cumulative_training_seconds']);close(validation,epoch_row['cumulative_validation_seconds'])
                observed=float(epoch_row['cumulative_observed_seconds'])
                assert observed>=training+validation-1e-6 and observed>prior_observed and 0<=float(epoch_row['validation_accuracy'])<=1
                prior_observed=observed
            assert partials==summary['partial_updates']==(0 if variant=='full_adamw' else spec['expected_total_partial_updates_per_arm'])
            assert omitted==summary['omitted_prefix_examples']==(0 if variant=='full_adamw' else spec['expected_omitted_examples_per_arm'])
            close(training,summary['training_seconds']);close(validation,summary['validation_seconds']);close(prior_observed,summary['observed_seconds'])
            close(epochs[-1]['validation_accuracy'],summary['final_validation_accuracy']);close(epochs[-1]['validation_loss'],summary['final_validation_loss'])
            cp=torch.load(attempt/'final_checkpoint.pt',map_location='cpu',weights_only=True)
            assert cp['seed']==seed and cp['variant']==variant and cp['updates']==len(steps) and cp['partial_updates']==partials
            assert cp['root_fingerprint']==cfg['fingerprint'] and cp['run_fingerprint']==rc['fingerprint'] and list(history)==cp['controller_history']
            names=list(cp['model_state']);ids=[i for g in cp['optimizer_state']['param_groups'] for i in g['params']]
            assert len(names)==len(ids)==len(cp['optimizer_state']['state'])
            prefix_parameters=suffix_parameters=0
            suffix_roots=tuple(f'encoder.layers.encoder_layer_{i}.' for i in range(spec['cut'],2 if cfg['synthetic'] else 6))+('encoder.ln.','heads.')
            for name,i in zip(names,ids):
                tensor=cp['model_state'][name];state=cp['optimizer_state']['state'][i];is_suffix=name.startswith(suffix_roots)
                assert torch.isfinite(tensor).all() and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
                assert state['exp_avg'].shape==tensor.shape and state['exp_avg_sq'].shape==tensor.shape and (state['exp_avg_sq']>=0).all()
                assert int(state['step'])==len(steps)-(0 if is_suffix else partials)
                if is_suffix:suffix_parameters+=tensor.numel()
                else:prefix_parameters+=tensor.numel()
            assert summary['prefix_parameters']==prefix_parameters and summary['suffix_parameters']==suffix_parameters
            for g in cp['optimizer_state']['param_groups']:close(g['lr'],steps[-1]['lr'])
            if seed not in stream_states:stream_states[seed]=cp
            else:
                baseline=stream_states[seed]
                for field in ['batch_rng','augmentation_rng','permutation_rng','torch_rng']:assert torch.equal(cp[field],baseline[field]),field
                assert cp['mixup_rng']==baseline['mixup_rng']
                assert len(cp['cuda_rng'])==len(baseline['cuda_rng'])
                for a,b in zip(cp['cuda_rng'],baseline['cuda_rng']):assert torch.equal(a,b),'dropout CUDA stream mismatch'
            audited_steps+=len(steps)
            flat={k:summary[k] for k in ['seed','variant','order_position','updates','partial_updates','omitted_prefix_examples','final_validation_accuracy','training_seconds','validation_seconds','observed_seconds','hardware_warmup_seconds','checkpoint_seconds','experiment_elapsed_seconds']}
            for label,threshold in [('primary',spec['primary_target']),('secondary',spec['secondary_target'])]:
                first,reached=target(epochs,threshold,spec['sustained_epochs'])
                t=dict(seed=seed,variant=variant,target_role=label,threshold=threshold,reached=reached is not None,
                    first_crossing_epoch=int(first['epoch']) if first else '',confirming_epoch=int(reached['epoch']) if reached else '',
                    observed_seconds=float(reached['cumulative_observed_seconds']) if reached else '',training_seconds=float(reached['cumulative_training_seconds']) if reached else '',
                    censoring_epoch=spec['epochs'] if not reached else '',censoring_observed_seconds=summary['observed_seconds'] if not reached else '')
                targets.append(t)
                if label=='primary':flat['primary_reached']=t['reached'];flat['primary_observed_seconds']=t['observed_seconds']
            all_runs.append(flat);runs[seed,variant]=flat
    screens=[]
    for candidate in ['mixup_only','boundary_rms']:
        for seed in spec['development_seeds']:
            a=runs[seed,candidate];full=runs[seed,'full_adamw']
            controls=['periodic','random']+(['mixup_only'] if candidate=='boundary_rms' else [])
            reached=[runs[seed,c] for c in controls if runs[seed,c]['primary_reached']]
            best=min(reached,key=lambda r:(r['primary_observed_seconds'],r['variant'])) if reached else None
            full_ratio=a['primary_observed_seconds']/full['primary_observed_seconds'] if a['primary_reached'] and full['primary_reached'] else None
            control_ratio=a['primary_observed_seconds']/best['primary_observed_seconds'] if a['primary_reached'] and best else None
            delta=100*(a['final_validation_accuracy']-full['final_validation_accuracy'])
            control_delta=100*(a['final_validation_accuracy']-best['final_validation_accuracy']) if best else None
            speed=full_ratio is not None and control_ratio is not None and max(full_ratio,control_ratio)<=1-spec['minimum_time_gain']
            quality=delta>=-spec['final_accuracy_margin_pp'] and control_delta is not None and control_delta>=-spec['final_accuracy_margin_pp']
            screens.append(dict(seed=seed,candidate=candidate,candidate_reached=a['primary_reached'],full_reached=full['primary_reached'],
                fastest_reaching_control=best['variant'] if best else '',time_ratio_to_full=full_ratio if full_ratio is not None else '',
                time_ratio_to_fastest_control=control_ratio if control_ratio is not None else '',final_delta_vs_full_pp=delta,
                final_delta_vs_control_pp=control_delta if control_delta is not None else '',speed_screen_pass=speed,quality_screen_pass=quality,seed_screen_pass=speed and quality))
    readiness=all(runs[seed,'full_adamw']['primary_reached'] for seed in spec['development_seeds'])
    candidate_pass={name:readiness and all(r['seed_screen_pass'] for r in screens if r['candidate']==name) for name in ['mixup_only','boundary_rms']}
    record=dict(status='PASS',synthetic=cfg['synthetic'],fingerprint=cfg['fingerprint'],runs=len(all_runs),audited_steps=audited_steps,
        matched_initialization_data_mixup_augmentation_and_dropout_streams=True,quota_masks_and_past_only_thresholds_reconstructed=True,
        finite_checkpoint_moments_and_counters_verified=True,full_baseline_target_readiness=readiness,candidate_both_seed_screen_pass=candidate_pass,
        decision_reason_counts=dict(sorted(reason_counts.items())),inference='DEVELOPMENT SCREEN ONLY; no confidence interval, confirmatory result or automatic extension')
    if write:
        def save(name,rows):
            with (out/name).open('w',encoding='utf8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        save('all_runs.csv',all_runs);save('targets.csv',targets);save('candidate_screen.csv',screens)
        (out/'audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
        lines=['# Stage9E five-arm quality pilot','',record['inference'],'',
            f'Synthetic: {cfg["synthetic"]}. All{len(all_runs)} runs audited; {audited_steps} updates. Full-target readiness: {readiness}.','',
            '| Seed | Arm | Final validation accuracy | Primary sustained target reached | Observed seconds to target | Training seconds total |',
            '|---:|---|---:|---|---:|---:|']
        for r in all_runs:
            time_label=f'{r["primary_observed_seconds"]:.2f}' if r['primary_reached'] else 'not reached (censored)'
            lines.append(f'| {r["seed"]} | {r["variant"]} | {100*r["final_validation_accuracy"]:.2f}% | {r["primary_reached"]} | {time_label} | {r["training_seconds"]:.2f} |')
        lines+=['',f'Both-seed development screen: {candidate_pass}.',
            'Primary target is confirmed only at the last of the required consecutive qualifying validations.',
            'Never-reached targets are retained in targets.csv with censoring; no target was lowered.',
            'Time to target includes validation, calibration, gate, trace logging and scheduled telemetry; common hardware warmup and setup are separate.',
            'All partial arms have identical prefix update/example budgets and freezing semantics.',
            'Review all controls, realized forced decisions, time/quality tradeoffs and both seeds before any next study.',
            'No automatic extension, reserved evaluation, publication or novelty claim follows.','']
        (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:record[k] for k in ['status','synthetic','runs','audited_steps','full_baseline_target_readiness','candidate_both_seed_screen_pass']},indent=2))
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'results/stage9E_quality_pilot_v1');args=p.parse_args()
    audit(args.root,write=False)
