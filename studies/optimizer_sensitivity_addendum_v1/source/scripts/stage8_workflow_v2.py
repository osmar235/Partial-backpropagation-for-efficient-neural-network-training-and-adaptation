"""Stage 8: development on training-only validation, then frozen ViT evaluation."""
from __future__ import annotations
import argparse, csv, json, math, statistics as st
from pathlib import Path
import torch
import stage8_vit_v2 as v

ROOT=v.ROOT

class BaselineReadinessError(RuntimeError):
    """Expected scientific stop; preserve the screen and do not open evaluation."""

def lr_label(lr):return ('lr_'+str(lr)).replace('.','p')

def completed(folder,phase,seed,variant,lr,spec,fake):
    config=json.loads((folder/'run_config.json').read_text())
    r=json.loads((folder/'summary.json').read_text())
    assert v.digest({k:x for k,x in config.items() if k!='fingerprint'})==config['fingerprint']
    assert r['status']=='complete' and r['fingerprint']==config['fingerprint']
    assert config['source_sha256']==v.source_hashes(),'Development code changed: use a new amended study/output directory'
    assert config['phase']==phase and config['seed']==seed and config['variant']==variant
    assert config['lr']==lr and config['spec']==spec and config['fake']==fake
    checkpoint=folder/'final_checkpoint.pt'
    assert v.hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
    assert r['epochs']==spec['epochs'] and r['updates']==(4 if fake else (176 if phase=='development' else 196))*spec['epochs']
    return r

def select_baseline(dev,spec,fake):
    rows=[]
    for lr in spec['learning_rates']:
        runs=[completed(dev/'screen'/lr_label(lr)/f'seed_{seed}'/'full_adamw',
            'development',seed,'full_adamw',lr,spec,fake) for seed in spec['development_seeds']]
        accuracies=[r['final_eval_accuracy'] for r in runs]
        rows.append(dict(lr=lr,mean_validation_accuracy=st.mean(accuracies),
                         min_validation_accuracy=min(accuracies)))
    selected=sorted(rows,key=lambda r:(-r['mean_validation_accuracy'],r['lr']))[0]
    ready=selected['mean_validation_accuracy']>=spec['baseline_min_mean_validation_accuracy'] and selected['min_validation_accuracy']>=spec['baseline_min_seed_validation_accuracy']
    record=dict(phase='development_only',selected_lr=selected['lr'],ready=ready,
                criterion='highest mean FINAL validation accuracy over development seeds; tie to lower LR',
                screen=rows,spec=spec,source_sha256=v.source_hashes(),fake=fake)
    v.atomic_json(dev/'baseline_selection.json',record)
    v.write_csv(dev/'baseline_validation_screen.csv',rows)
    if not ready:
        (dev/'DEVELOPMENT_REPORT.md').write_text('# Baseline readiness failed\n\nAll fixed LR-screen results are retained. Do not open evaluation seeds.\n'
            'Inspect baseline quality and prospectively document any amended recipe before more training.\n')
        raise BaselineReadinessError('Baseline did not meet the unchanged validation readiness thresholds. Send the v2 development folder for review; evaluation is blocked. No further automatic search is authorized.')
    return record

def development(args,spec,device):
    out=args.development_dir;out.mkdir(parents=True,exist_ok=True)
    preflight=[] if args.fake else v.idle_preflight()
    v.atomic_json(out/'preflight.json',preflight)
    data=v.load_data('development',device,fake=args.fake)
    for i,seed in enumerate(spec['development_seeds']):
        rates=spec['learning_rates'] if i%2==0 else list(reversed(spec['learning_rates']))
        for lr in rates:
            v.run_trajectory(out/'screen'/lr_label(lr)/f'seed_{seed}'/'full_adamw',
                'development',seed,'full_adamw',lr,device,data,spec,tiny=args.tiny,fake=args.fake,order=rates.index(lr))
    selection=select_baseline(out,spec,args.fake)
    for seed in spec['development_seeds']:
        v.run_trajectory(out/'calibration'/f'seed_{seed}'/'lgpb','development',seed,'lgpb',
            selection['selected_lr'],device,data,spec,tiny=args.tiny,fake=args.fake)
    (out/'DEVELOPMENT_REPORT.md').write_text('# Stage 8A development complete\n\n'
        f'Selected learning rate: {selection["selected_lr"]}. Baseline readiness: PASS.\n\n'
        'Only the training-derived validation split was used; CIFAR-10 test data were not loaded.\n'
        'The two LGPB calibration traces are retained for the prospective periodic profile.\n'
        'Send this complete folder for review before the freeze/evaluation stage.\n')
    print('DEVELOPMENT COMPLETE:',out,flush=True)

def freeze(args,spec):
    dev=args.development_dir
    selected=select_baseline(dev,spec,args.fake)
    traces=[];input_hashes={}
    for seed in spec['development_seeds']:
        d=dev/'calibration'/f'seed_{seed}'/'lgpb'
        completed(d,'development',seed,'lgpb',selected['selected_lr'],spec,args.fake)
        with (d/'epoch_trace.csv').open(newline='') as f:trace=list(csv.DictReader(f))
        assert len(trace)==spec['epochs']
        traces.append([float(r['partial_fraction']) for r in trace])
    for path in sorted(dev.rglob('*')):
        if path.is_file() and path.name not in ('preflight.json','baseline_selection.json','baseline_validation_screen.csv'):
            input_hashes[str(path)]=v.hashlib.sha256(path.read_bytes()).hexdigest()
    profile=[st.mean(values) for values in zip(*traces)]
    for q in profile:assert 0<=q<=.5
    record=dict(
        spec=spec,selected_lr=selected['selected_lr'],periodic_epoch_q=profile,source_sha256=v.source_hashes(),
        development_input_sha256=input_hashes,fake=args.fake,tiny=args.tiny,
        evaluation_seed_roles='reserved new training seeds; CIFAR-10 has prior project exposure')
    record['freeze_hash']=v.digest(record)
    path=args.freeze_file;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        existing=json.loads(path.read_text())
        if existing!=record:raise ValueError('A different freeze already exists. Do not overwrite it; document an amended study instead.')
    else:
        if args.evaluation_dir.exists() and any(args.evaluation_dir.glob('seed_*')):
            raise RuntimeError('Evaluation output exists without a freeze. Do not freeze retrospectively.')
        v.atomic_json(path,record)
    print('FREEZE:',path,'hash:',record['freeze_hash'],flush=True)
    return record

def evaluation(args,spec,device):
    if not args.freeze_file.exists():raise FileNotFoundError('Freeze file missing; complete development and freeze first')
    record=json.loads(args.freeze_file.read_text())
    assert record['freeze_hash']==v.digest({k:x for k,x in record.items() if k!='freeze_hash'})
    assert record['source_sha256']==v.source_hashes() and record['spec']==spec
    assert record['fake']==args.fake and record['tiny']==args.tiny
    for path,h in record['development_input_sha256'].items():assert v.hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    preflight=[] if args.fake else v.idle_preflight()
    args.evaluation_dir.mkdir(parents=True,exist_ok=True)
    v.atomic_json(args.evaluation_dir/'preflight.json',preflight)
    data=v.load_data('evaluation',device,fake=args.fake)
    for seed,order in zip(spec['evaluation_seeds'],spec['evaluation_orders']):
        for i,variant in enumerate(order):
            v.run_trajectory(args.evaluation_dir/f'seed_{seed}'/variant,'evaluation',seed,variant,
                record['selected_lr'],device,data,spec,record['periodic_epoch_q'],tiny=args.tiny,fake=args.fake,
                order=i,freeze_hash=record['freeze_hash'])
    summarize(args,spec)
    from stage8_audit_v2 import audit
    audit(args.evaluation_dir,args.freeze_file)

def interval(values,log=False):
    assert len(values)==6
    values=[math.log(x) for x in values] if log else values
    mu=st.mean(values);half=2.570581835636314*st.stdev(values)/math.sqrt(6)
    return [math.exp(x) for x in (mu,mu-half,mu+half)] if log else [mu,mu-half,mu+half]

def summarize(args,spec):
    record=json.loads(args.freeze_file.read_text());runs={}
    assert record['source_sha256']==v.source_hashes()
    for seed in spec['evaluation_seeds']:
        for variant in spec['variants']:
            d=args.evaluation_dir/f'seed_{seed}'/variant
            r=completed(d,'evaluation',seed,variant,record['selected_lr'],spec,args.fake)
            assert r['freeze_hash']==record['freeze_hash']
            runs[seed,variant]=r
    contrasts=[];target_rows=[]
    for variant,ref in [('lgpb','full_adamw'),('periodic_matched','full_adamw'),('replay_prefix','full_adamw'),
                        ('lgpb','periodic_matched'),('lgpb','replay_prefix')]:
        pairs=[(runs[seed,variant],runs[seed,ref]) for seed in spec['evaluation_seeds']]
        accuracy=interval([100*(a['final_eval_accuracy']-b['final_eval_accuracy']) for a,b in pairs])
        wall=interval([a['training_wall_seconds']/b['training_wall_seconds'] for a,b in pairs],True)
        contrasts.append(dict(variant=variant,reference=ref,n=6,accuracy_delta_pp=accuracy[0],
            accuracy_ci95_low=accuracy[1],accuracy_ci95_high=accuracy[2],wall_ratio=wall[0],
            wall_ci95_low=wall[1],wall_ci95_high=wall[2]))
        for threshold in spec['targets']:
            key=f'training_only_time_to_{round(threshold*100)}pct'
            paired=[(a,b) for a,b in pairs if a[key] is not None and b[key] is not None]
            ci=interval([a[key]/b[key] for a,b in paired],True) if len(paired)==6 else ['','','']
            target_rows.append(dict(variant=variant,reference=ref,target=threshold,
                variant_reached=sum(a[key] is not None for a,b in pairs),reference_reached=sum(b[key] is not None for a,b in pairs),
                paired_n=len(paired),training_only_time_ratio=ci[0],ci95_low=ci[1],ci95_high=ci[2]))
    primary=contrasts[0]
    gates=dict(accuracy_margin_pp=spec['accuracy_noninferiority_margin_pp'],
        accuracy_pass=primary['accuracy_ci95_low']>-spec['accuracy_noninferiority_margin_pp'],
        speed_pass=primary['wall_ci95_high']<1,
        context='SYNTHETIC DIAGNOSTIC ONLY' if args.fake else 'New Stage 8 primary criteria; separate from Stage 6')
    flat=[]
    for (seed,variant),r in runs.items():
        flat.append({k:r[k] for k in ['seed','variant','final_eval_accuracy','final_eval_loss','training_wall_seconds',
            'evaluation_seconds','warmup_seconds','checkpoint_seconds','total_experiment_elapsed_seconds','partial_fraction']})
    v.write_csv(args.evaluation_dir/'stage8_all_runs.csv',flat)
    v.write_csv(args.evaluation_dir/'stage8_paired_estimates.csv',contrasts)
    v.write_csv(args.evaluation_dir/'stage8_all_targets.csv',target_rows)
    v.atomic_json(args.evaluation_dir/'stage8_primary_criteria.json',gates)
    lines=['# Stage 8 ViT four-arm results','',gates['context'],'',
           '| Variant | Reference | Accuracy delta pp [95% CI] | Wall ratio [95% CI] |',
           '|---|---|---:|---:|']
    for r in contrasts:
        lines.append(f"| {r['variant']} | {r['reference']} | {r['accuracy_delta_pp']:+.3f} [{r['accuracy_ci95_low']:+.3f}, {r['accuracy_ci95_high']:+.3f}] | {r['wall_ratio']:.4f} [{r['wall_ci95_low']:.4f}, {r['wall_ci95_high']:.4f}] |")
    lines+=['','Primary criteria apply only to the new frozen Stage 8 LGPB/full-AdamW contrast.',
            'All mechanism/target intervals are descriptive and unadjusted. Inspect telemetry and all seeds.',
            'Training-only wall excludes warm-up, validation, telemetry queries and final checkpoint writes.',
            'Total experiment elapsed seconds separately include these per-run costs; initial data/model construction is excluded.']
    (args.evaluation_dir/'STAGE8_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['development','freeze','evaluate','freeze-evaluate','summarize','smoke'])
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--development-dir',type=Path,default=ROOT/'results/stage8A_vit_development_v2')
    p.add_argument('--evaluation-dir',type=Path,default=ROOT/'results/stage8B_vit_evaluation_v2')
    p.add_argument('--freeze-file',type=Path,default=ROOT/'protocol/stage8/FROZEN_EVALUATION_v2.json')
    p.add_argument('--fake',action='store_true');p.add_argument('--tiny',action='store_true')
    args=p.parse_args();spec=json.loads(v.SPEC_PATH.read_text())
    if args.action=='smoke':
        torch.set_num_threads(1)
        args.fake=True;args.tiny=True;args.device='cpu'
        args.development_dir=ROOT/'results/stage8_sanity_recipe_v2/development'
        args.evaluation_dir=ROOT/'results/stage8_sanity_recipe_v2/evaluation'
        args.freeze_file=ROOT/'results/stage8_sanity_recipe_v2/SYNTHETIC_FREEZE.json'
    if args.tiny and not args.fake:raise ValueError('Tiny model is artificial-data diagnostics only')
    if args.fake:
        if 'stage8_sanity' not in str(args.development_dir) or 'stage8_sanity' not in str(args.evaluation_dir):
            raise ValueError('Synthetic outputs must remain under a stage8_sanity directory')
        spec.update(epochs=2,batch_size=4,eval_batch_size=4,lr_warmup_epochs=0,algorithm_warmup_steps=2,
            development_seeds=[9001,9002],evaluation_seeds=list(range(9101,9107)),
            baseline_min_mean_validation_accuracy=0.,baseline_min_seed_validation_accuracy=0.)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if not args.fake and args.device!='cuda':raise ValueError('Scientific runs require the declared CUDA device')
    device=torch.device(args.device)
    if args.action in ('development','smoke'):development(args,spec,device)
    if args.action in ('freeze','freeze-evaluate','smoke'):freeze(args,spec)
    if args.action in ('evaluate','freeze-evaluate','smoke'):evaluation(args,spec,device)
    if args.action=='summarize':summarize(args,spec)

if __name__=='__main__':
    try:main()
    except BaselineReadinessError as error:
        print('DEVELOPMENT STOPPED:',error,flush=True)
        raise SystemExit(2)
