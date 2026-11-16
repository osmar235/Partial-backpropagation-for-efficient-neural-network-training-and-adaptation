"""Independent, read-only audit of Stage 7E CIFAR-10 AdamW robustness records."""
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/stage7E_adamw_optional'
OUT=ROOT/'results/stage7E_review_v1'
SEEDS=range(69,75)
T5=2.570581835636314

def records(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8),(a,b)

def estimate(vs,log=False):
    values=[math.log(v) for v in vs] if log else vs
    mu=st.mean(values);half=T5*st.stdev(values)/math.sqrt(len(values))
    return [math.exp(v) for v in (mu,mu-half,mu+half)] if log else [mu,mu-half,mu+half]

def main():
    assert {p.name for p in DATA.glob('seed_*')}=={f'seed_{s:03d}' for s in SEEDS}
    runs={}; input_hashes={}
    for path in sorted(DATA.rglob('*')):
        if path.is_file():input_hashes[str(path.relative_to(DATA))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for seed in SEEDS:
        folder=DATA/f'seed_{seed:03d}'
        assert {p.name for p in folder.iterdir() if p.is_dir()}=={'adam','lgpb'}
        config=json.loads((folder/'run_config.json').read_text())
        fingerprint=config.pop('config_fingerprint')
        hashed={k:v for k,v in config.items() if k not in ('force','out')}
        assert hashlib.sha256(json.dumps(hashed,sort_keys=True).encode()).hexdigest()==fingerprint
        for path,h in config['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
        for key,value in dict(suite='transfer',dataset='cifar10',train_subset=50000,eval_subset=10000,
            batch_size=256,epochs=30,schedule_epochs=30,seed=seed,device='cuda',optimizer='adamw',weight_decay=.0005,
            lr=.0003,rho=.95,warmup_steps=196,eval_batch_size=512,min_lr_ratio=.01,
            log_steps=True,tiny_model=False,rotate_order=True,rotation_start=69,num_classes=10).items():
            assert config[key]==value,(seed,key)
        order=['adam','lgpb'] if (seed-69)%2==0 else ['lgpb','adam']
        assert config['variants']==order
        for variant in order:
            d=folder/variant
            r=json.loads((d/'summary.json').read_text())
            epochs=records(d/'epoch_trace.csv');steps=records(d/'step_trace.csv')
            assert r['status']=='complete' and r['config_fingerprint']==fingerprint
            assert r['seed']==seed and r['variant']==variant and r['dataset']=='cifar10' and r['optimizer']=='adamw'
            assert r['updates']==5880 and len(epochs)==30 and len(steps)==5880
            assert r['run_order_position']==order.index(variant)
            assert r['prefix_parameter_count']==149824 and r['suffix_parameter_count']==11024138
            ema=None;previous=False
            for i,s in enumerate(steps):
                loss=float(s['loss']);partial=bool(int(s['partial']))
                assert int(s['step'])==i+1 and int(s['epoch'])==i//196+1
                assert math.isfinite(loss) and not(partial and previous)
                if ema is None:assert s['ema_before_update']==''
                else:close(s['ema_before_update'],ema)
                expected=variant=='lgpb' and ema is not None and not previous and loss<=ema
                assert partial==expected,(seed,variant,i)
                ema=loss if ema is None else .95*ema+.05*loss
                previous=partial
                close(s['lr'],.0003*(.01+.99*.5*(1+math.cos(math.pi*i/5879))))
                for key in ('aug_ms','forward_ms','backward_ms','optimizer_ms'):
                    assert math.isfinite(float(s[key])) and float(s[key])>=0
            for i,e in enumerate(epochs):
                batch=steps[i*196:(i+1)*196];partial=sum(int(s['partial']) for s in batch)
                assert int(e['epoch'])==i+1 and int(e['updates'])==(i+1)*196
                assert int(e['partial_updates'])==partial and int(e['full_updates'])==196-partial
                close(e['partial_fraction'],partial/196)
                close(e['cumulative_training_seconds'],sum(float(x['epoch_seconds']) for x in epochs[:i+1]))
                for flag,label in ((0,'full'),(1,'partial')):
                    mode=[s for s in batch if int(s['partial'])==flag]
                    close(e[f'mean_backward_{label}_seconds'],st.mean(float(s['backward_ms'])/1000 for s in mode) if mode else 0)
            assert sum(int(s['partial']) for s in steps)==r['partial_updates']
            assert r['full_updates']+r['partial_updates']==5880
            close(r['partial_fraction'],r['partial_updates']/5880)
            close(r['final_eval_accuracy'],epochs[-1]['eval_accuracy'])
            close(r['final_eval_loss'],epochs[-1]['eval_loss'])
            close(r['training_wall_seconds'],sum(float(e['epoch_seconds']) for e in epochs))
            for key,field in [('aug_ms','augmentation_seconds'),('forward_ms','forward_seconds'),
                              ('backward_ms','backward_seconds'),('optimizer_ms','optimizer_seconds')]:
                close(r[field],sum(float(s[key])/1000 for s in steps))
            for flag,label in ((0,'full'),(1,'partial')):
                close(r[f'backward_{label}_seconds'],sum(float(s['backward_ms'])/1000 for s in steps if int(s['partial'])==flag))
            close(r['timing_unattributed_seconds'],r['training_wall_seconds']-sum(r[k] for k in
                ('augmentation_seconds','forward_seconds','backward_seconds','optimizer_seconds')))
            for target in (60,70,80):
                first=next((e for e in epochs if float(e['eval_accuracy'])>=target/100),None)
                assert r[f'epoch_to_{target}pct']==(int(first['epoch']) if first else None)
                if first:close(r[f'time_to_{target}pct_seconds'],first['cumulative_training_seconds'])
                else:assert r[f'time_to_{target}pct_seconds'] is None
            runs[seed,variant]=r
    pairs=[(runs[s,'lgpb'],runs[s,'adam']) for s in SEEDS]
    delta=estimate([100*(a['final_eval_accuracy']-b['final_eval_accuracy']) for a,b in pairs])
    wall=estimate([a['training_wall_seconds']/b['training_wall_seconds'] for a,b in pairs],True)
    reported=records(DATA/'stage7_descriptive_paired_estimates.csv')
    assert len(reported)==1 and int(reported[0]['n_paired_seeds'])==6
    for key,value in zip(('mean_accuracy_delta_pp','ci95_accuracy_low_pp','ci95_accuracy_high_pp'),delta):close(reported[0][key],value)
    for key,value in zip(('geometric_wall_ratio','ci95_wall_ratio_low','ci95_wall_ratio_high'),wall):close(reported[0][key],value)
    frozen=records(ROOT/'protocol/stage7/frozen_files_before.csv')
    for row in frozen:assert hashlib.sha256(Path(row['Path']).read_bytes()).hexdigest().upper()==row['Hash'],row['Path']
    result=dict(accuracy_delta_pp_ci95=delta,wall_ratio_ci95=wall,
        wall_reduction_pct_ci95=[100*(1-wall[0]),100*(1-wall[2]),100*(1-wall[1])],
        seeds_lgpb_faster=sum(a['training_wall_seconds']<b['training_wall_seconds'] for a,b in pairs),
        seeds_lgpb_more_accurate=sum(a['final_eval_accuracy']>b['final_eval_accuracy'] for a,b in pairs))
    for variant in ('adam','lgpb'):
        rs=[runs[s,variant] for s in SEEDS]
        result[variant]=dict(mean_accuracy_pct=100*st.mean(r['final_eval_accuracy'] for r in rs),
            mean_wall_seconds=st.mean(r['training_wall_seconds'] for r in rs),
            mean_q_pct=100*st.mean(r['partial_fraction'] for r in rs),
            mean_full_backward_ms=st.mean(r['avg_backward_full_ms'] for r in rs),
            mean_partial_backward_ms=st.mean(r['avg_backward_partial_ms'] for r in rs),
            mean_unattributed_seconds=st.mean(r['timing_unattributed_seconds'] for r in rs))
    result['time_to_quality']={}
    for target in (60,70,80):
        key=f'time_to_{target}pct_seconds'
        eligible=[(a,b) for a,b in pairs if a[key] is not None and b[key] is not None]
        result['time_to_quality'][str(target)]=dict(lgpb_reached=sum(a[key] is not None for a,b in pairs),
            adam_reached=sum(b[key] is not None for a,b in pairs),paired_n=len(eligible),
            ratio_ci95=estimate([a[key]/b[key] for a,b in eligible],True) if len(eligible)==6 else None)
    OUT.mkdir(exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(dict(status='PASS',completed_runs=12,epochs=360,
        steps=70560,original_files_unchanged=len(frozen),input_sha256=input_hashes,
        checks=['configurations, code hashes, fingerprints and alternating order',
        'all loss gate decisions, EMA history and learning rates',
        'mode counts and timing reconciliation','final metrics and all target crossings',
        'paired effect and CI independent recomputation']),indent=2)+'\n')
    (OUT/'independent_results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'timing_by_seed.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['seed','variant','order','accuracy_pct','wall_seconds',
                                     'full_backward_ms','partial_backward_ms','partial_fraction'])
        w.writeheader()
        for (seed,variant),r in runs.items():
            w.writerow(dict(seed=seed,variant=variant,order=r['run_order_position'],
                accuracy_pct=100*r['final_eval_accuracy'],wall_seconds=r['training_wall_seconds'],
                full_backward_ms=r['avg_backward_full_ms'],partial_backward_ms=r['avg_backward_partial_ms'],
                partial_fraction=r['partial_fraction']))
    print(json.dumps(result,indent=2));print('PASS: 12 runs, 70,560 steps; frozen files unchanged')

if __name__=='__main__':main()

