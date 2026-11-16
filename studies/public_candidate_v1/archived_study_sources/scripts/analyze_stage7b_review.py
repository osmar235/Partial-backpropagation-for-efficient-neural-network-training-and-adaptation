"""Independent, read-only Stage 7B audit and descriptive analysis of returned ZIP."""
import csv
import hashlib
import io
import json
import math
import random
import statistics as st
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/stage7B_review_v1'
VARIANTS = ['adam','lgpb','random_matched','periodic_matched','freeze_prefix','replay_prefix']
T5 = 2.570581835636314

def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8),(a,b)

def estimate(values,log=False):
    vs=[math.log(v) for v in values] if log else values
    mu=st.mean(vs)
    half=T5*st.stdev(vs)/math.sqrt(len(vs))
    return [math.exp(x) for x in (mu,mu-half,mu+half)] if log else [mu,mu-half,mu+half]

def write(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    archive=ROOT/'results/stage7B_core_ablation.zip'
    runs={}; matched={}; checked=0
    with (ROOT/'results/stage7A_original_audit/archived_epoch_q_summary.csv').open(newline='') as f:
        q=[float(r['mean_q']) for r in csv.DictReader(f)]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for info in z.infolist():
            if info.is_dir(): continue
            path=Path(info.filename)
            assert not path.is_absolute() and '..' not in path.parts
            assert (ROOT/'results'/path).read_bytes()==z.read(info),info.filename
            checked+=1
        def read(name):return z.read('stage7B_core_ablation/'+name).decode('utf-8-sig')
        def records(name):return list(csv.DictReader(io.StringIO(read(name))))
        for seed in range(51,57):
            config=json.loads(read(f'seed_{seed:03d}/run_config.json'))
            fingerprint=config.pop('config_fingerprint')
            hashed={k:v for k,v in config.items() if k not in ('force','out')}
            assert hashlib.sha256(json.dumps(hashed,sort_keys=True).encode()).hexdigest()==fingerprint
            for path,h in config['source_sha256'].items():
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
            for key,value in dict(dataset='cifar10',train_subset=50000,eval_subset=10000,
                batch_size=256,epochs=30,schedule_epochs=30,seed=seed,device='cuda',
                optimizer='adam',lr=.0003,rho=.95,match_policy='epoch',warmup_steps=196,
                eval_batch_size=512,min_lr_ratio=.01,log_steps=True,tiny_model=False,
                rotate_order=True,rotation_start=51,num_classes=10).items():
                assert config[key]==value,(seed,key)
            offset=(seed-51)%6
            assert config['variants']==VARIANTS[offset:]+VARIANTS[:offset]
            for variant in VARIANTS:
                folder=f'seed_{seed:03d}/{variant}'
                r=json.loads(read(folder+'/summary.json'))
                epochs=records(folder+'/epoch_trace.csv');steps=records(folder+'/step_trace.csv')
                assert r['status']=='complete' and r['config_fingerprint']==fingerprint
                assert r['seed']==seed and r['variant']==variant and r['updates']==5880
                assert r['run_order_position']==config['variants'].index(variant)
                assert len(epochs)==30 and len(steps)==5880
                planned=set()
                if variant in ('random_matched','periodic_matched'):
                    for epoch,qe in enumerate(q):
                        k=round(qe*196)
                        if variant=='random_matched':
                            positions=sorted(random.Random(seed+101*epoch+95113).sample(range(196-k),k))
                            local={j+i+1 for i,j in enumerate(positions)}
                        else:
                            local={int((j+.5)*195/k)+1 for j in range(k)}
                        planned.update(epoch*196+j for j in local)
                    assert len(planned)==2116
                ema=None;previous=False
                for i,s in enumerate(steps):
                    loss=float(s['loss']); partial=bool(int(s['partial']))
                    assert int(s['step'])==i+1 and int(s['epoch'])==i//196+1
                    assert math.isfinite(loss) and not(partial and previous)
                    if ema is None: assert s['ema_before_update']==''
                    else:close(s['ema_before_update'],ema)
                    expected=(i in planned) if variant in ('random_matched','periodic_matched') else (
                        variant!='adam' and ema is not None and not previous and loss<=ema)
                    assert partial==expected,(seed,variant,i)
                    ema=loss if ema is None else .95*ema+.05*loss
                    previous=partial
                    for field in ('aug_ms','forward_ms','backward_ms','optimizer_ms'):
                        assert math.isfinite(float(s[field])) and float(s[field])>=0
                    expected_lr=.0003*(.01+.99*.5*(1+math.cos(math.pi*i/5879)))
                    close(s['lr'],expected_lr)
                for epoch,e in enumerate(epochs):
                    subset=steps[epoch*196:(epoch+1)*196]
                    n=sum(int(s['partial']) for s in subset)
                    assert int(e['partial_updates'])==n and int(e['full_updates'])==196-n
                    close(e['partial_fraction'],n/196)
                    close(e['cumulative_training_seconds'],sum(float(x['epoch_seconds']) for x in epochs[:epoch+1]))
                    for flag,label in ((0,'full'),(1,'partial')):
                        mode=[s for s in subset if int(s['partial'])==flag]
                        close(e[f'mean_backward_{label}_seconds'],st.mean(float(s['backward_ms'])/1000 for s in mode) if mode else 0)
                assert sum(int(s['partial']) for s in steps)==r['partial_updates']
                assert r['full_updates']+r['partial_updates']==5880
                close(r['partial_fraction'],r['partial_updates']/5880)
                close(r['final_eval_accuracy'],epochs[-1]['eval_accuracy'])
                close(r['final_eval_loss'],epochs[-1]['eval_loss'])
                close(r['training_wall_seconds'],sum(float(e['epoch_seconds']) for e in epochs))
                for field,summary in [('aug_ms','augmentation_seconds'),('forward_ms','forward_seconds'),
                                      ('backward_ms','backward_seconds'),('optimizer_ms','optimizer_seconds')]:
                    close(r[summary],sum(float(s[field])/1000 for s in steps))
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
        reported=records('stage7_descriptive_paired_estimates.csv')
        assert len(reported)==10
        for e in reported:
            pairs=[(runs[s,e['variant']],runs[s,e['reference']]) for s in range(51,57)]
            delta=estimate([100*(a['final_eval_accuracy']-b['final_eval_accuracy']) for a,b in pairs])
            wall=estimate([a['training_wall_seconds']/b['training_wall_seconds'] for a,b in pairs],True)
            for key,value in zip(('mean_accuracy_delta_pp','ci95_accuracy_low_pp','ci95_accuracy_high_pp'),delta):close(e[key],value)
            for key,value in zip(('geometric_wall_ratio','ci95_wall_ratio_low','ci95_wall_ratio_high'),wall):close(e[key],value)
            assert int(e['n_paired_seeds'])==6
    contrasts=[]
    for reference in VARIANTS:
        if reference=='lgpb':continue
        a=[runs[s,'lgpb'] for s in range(51,57)]; b=[runs[s,reference] for s in range(51,57)]
        accuracy=estimate([100*(x['final_eval_accuracy']-y['final_eval_accuracy']) for x,y in zip(a,b)])
        timing=estimate([x['training_wall_seconds']/y['training_wall_seconds'] for x,y in zip(a,b)],True)
        contrasts.append(dict(variant='lgpb',reference=reference,n=6,
            accuracy_delta_pp=accuracy[0],accuracy_low_pp=accuracy[1],accuracy_high_pp=accuracy[2],
            wall_ratio=timing[0],wall_ratio_low=timing[1],wall_ratio_high=timing[2],
            wall_reduction_pct=100*(1-timing[0]),
            seeds_lgpb_more_accurate=sum(x['final_eval_accuracy']>y['final_eval_accuracy'] for x,y in zip(a,b)),
            seeds_lgpb_faster=sum(x['training_wall_seconds']<y['training_wall_seconds'] for x,y in zip(a,b))))
    aggregate=[]
    for variant in VARIANTS:
        rs=[runs[s,variant] for s in range(51,57)]
        aggregate.append(dict(variant=variant,mean_accuracy_pct=100*st.mean(r['final_eval_accuracy'] for r in rs),
            mean_wall_seconds=st.mean(r['training_wall_seconds'] for r in rs),
            mean_q_pct=100*st.mean(r['partial_fraction'] for r in rs),
            mean_full_backward_ms=st.mean(r['avg_backward_full_ms'] for r in rs),
            mean_partial_backward_ms=st.mean(r['avg_backward_partial_ms'] for r in rs),
            mean_unattributed_seconds=st.mean(r['timing_unattributed_seconds'] for r in rs),
            targets_80pct_reached=sum(r['time_to_80pct_seconds'] is not None for r in rs)))
    frozen=list(csv.DictReader((ROOT/'protocol/stage7/frozen_files_before.csv').open(encoding='utf-8-sig')))
    for row in frozen:assert hashlib.sha256(Path(row['Path']).read_bytes()).hexdigest().upper()==row['Hash'],row['Path']
    write('independent_lgpb_contrasts.csv',contrasts);write('variant_aggregate.csv',aggregate)
    audit=dict(status='PASS',zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        zip_files_matching_local=checked,completed_runs=len(runs),epochs=1080,steps=211680,
        original_files_unchanged=len(frozen),checks=['ZIP CRC and local equality','configuration fingerprints and source hashes',
        'planned settings and rotated order','full step-level EMA/gate/matched-mask reconstruction',
        'cosine LR, counts, epoch and component timing reconciliation','final metrics and target crossings',
        'all ten reported paired estimates and confidence intervals independently reproduced'])
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    print(json.dumps(contrasts,indent=2));print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
