"""Independent stdlib-only integrity and arithmetic audit of Stage9D output."""
from pathlib import Path
import collections, csv, hashlib, json, math, statistics as st

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/stage9D_deployment_cost_v1'
OUT=ROOT/'results/stage9D_review_v1'
PATHS=['ordinary_full','predecided_freeze','boundary_full','boundary_freeze']
MUTABLE={str(ROOT/'README_STAGE9.md'),str(ROOT/'CURRENT_RELEASE.md')}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save(name,rows):
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)

def main():
    OUT.mkdir(exist_ok=True);protected={}
    def verify(p,h):
        assert sha(p)==h.lower(),str(p)
        if str(p) not in MUTABLE:protected[str(p)]=h.lower()
    cfg=json.loads((RAW/'configuration.json').read_text());fp=cfg.pop('fingerprint')
    assert hashlib.sha256(json.dumps(cfg,sort_keys=True,allow_nan=False).encode()).hexdigest()==fp
    assert not cfg['synthetic'] and cfg['gpu']=='NVIDIA GeForce RTX 5080'
    spec=cfg['spec'];assert spec['cut']==4 and spec['batch_size']==256 and spec['probe_batches']==32
    assert spec['development_seeds']==[1201,1202] and spec['checkpoint_epochs']==[15,30]
    for p,h in cfg['source_sha256'].items():verify(p,h)
    c=json.loads((RAW/'COMPLETE.json').read_text())
    assert c['fingerprint']==fp and c['status']=='complete' and c['rows']==512 and not c['synthetic']
    for p,h in c['sha256'].items():verify(p,h)
    for p in RAW.rglob('*'):
        if p.is_file():verify(p,sha(p))
    before_c=json.loads((ROOT/'results/stage9C_boundary_diagnostic_v1/configuration.json').read_text())
    assert cfg['split']==before_c['split']
    for key,info in cfg['checkpoints'].items():
        seed,epoch=map(int,key.split(':'));verify(info['path'],info['sha256'])
        marker=json.loads((ROOT/f'results/stage9C_boundary_diagnostic_v1/seed_{seed}/training/COMPLETE.json').read_text())
        assert marker['checkpoints'][str(epoch)]==info['path'] and marker['sha256'][info['path']]==info['sha256']
    latest_pointer_record=OUT/'documentation_updates_sha256.json'
    if not latest_pointer_record.exists():latest_pointer_record=ROOT/'results/stage9C_review_v1/documentation_updates_sha256.json'
    pointer_hashes=json.loads(latest_pointer_record.read_text());assert set(pointer_hashes)==MUTABLE
    for p,h in pointer_hashes.items():verify(p,h)
    previous=read(ROOT/'results/stage9C_review_v1/preserved_inputs_sha256.csv')
    for r in previous:
        if r['path'] not in MUTABLE:verify(r['path'],r['sha256'])
    for r in read(ROOT/'protocol/stage9/deployment_cost_preparation_sha256_v1.csv'):
        if r['path'] not in MUTABLE:verify(r['path'],r['sha256'])
    for p in (ROOT/'results/stage9C_review_v1').rglob('*'):
        if p.is_file():verify(p,sha(p))
    attempt=Path(c['attempt']);rows=read(attempt/'measurements.csv');summaries=read(attempt/'cost_summary.csv')
    assert len(rows)==512 and len(summaries)==8
    expected={(seed,epoch,probe,path) for seed in [1201,1202] for epoch in [15,30] for probe in range(32) for path in PATHS}
    assert {(int(r['seed']),int(r['epoch']),int(r['probe']),r['path']) for r in rows}==expected
    grouped=collections.defaultdict(list)
    for r in rows:
        assert int(r['cut'])==4 and int(r['batch_size'])==256
        for k in ['mixup_lambda','operation_seconds','augmentation_seconds','total_seconds','loss']:
            assert math.isfinite(float(r[k]))
        assert 0<=float(r['mixup_lambda'])<=1 and float(r['operation_seconds'])>0 and float(r['augmentation_seconds'])>0
        close(r['total_seconds'],float(r['operation_seconds'])+float(r['augmentation_seconds']))
        assert bool(r['boundary_rms'])==r['path'].startswith('boundary')
        if r['boundary_rms']:assert math.isfinite(float(r['boundary_rms'])) and float(r['boundary_rms'])>0
        grouped[(int(r['seed']),int(r['epoch']),int(r['probe']))].append(r)
    assert len(grouped)==128
    for (seed,epoch,probe),a in grouped.items():
        assert len(a)==4
        for field in ['mixup_lambda','augmentation_seconds']:assert len({r[field] for r in a})==1
        for r in a:assert math.isclose(float(r['loss']),float(a[0]['loss']),rel_tol=1e-5,abs_tol=1e-6)
        signal=[float(r['boundary_rms']) for r in a if r['boundary_rms']];close(signal[0],signal[1])
        order=PATHS[probe%4:]+PATHS[:probe%4]
        if (probe//4)%2:order=list(reversed(order))
        for r in a:assert int(r['order_position'])==order.index(r['path'])
    manifests=json.loads((attempt/'probe_indices.json').read_text());assert len(manifests)==4
    for block in manifests:
        assert len(block['probes'])==32
        indices=[]
        for i,r in enumerate(block['probes']):
            assert r['probe']==i and len(r['indices'])==256
            assert r['dropout_seed']==block['seed']+block['epoch']*10000+145001+i
            indices.extend(r['indices'])
        assert len(indices)==len(set(indices))==8192 and min(indices)>=0 and max(indices)<45000
    calculated=[];path_stats=[];sensitivity=[]
    for seed in [1201,1202]:
        for epoch in [15,30]:
            a=[r for r in rows if int(r['seed'])==seed and int(r['epoch'])==epoch]
            for variant in ['all_samples','exclude_first_probe_posthoc','operation_only_posthoc']:
                b=[r for r in a if variant!='exclude_first_probe_posthoc' or int(r['probe'])>0]
                metric='operation_seconds' if variant=='operation_only_posthoc' else 'total_seconds'
                means={path:st.mean(float(r[metric]) for r in b if r['path']==path) for path in PATHS}
                for q in [.2,.4]:
                    pre=(1-q)*means['ordinary_full']+q*means['predecided_freeze']
                    boundary=(1-q)*means['boundary_full']+q*means['boundary_freeze']
                    result=dict(seed=seed,epoch=epoch,variant=variant,omission_fraction=q,
                        predecided_ratio=pre/means['ordinary_full'],boundary_ratio=boundary/means['ordinary_full'],boundary_over_predecided=boundary/pre,
                        required_update_count_reduction_for_equal_time=1-pre/boundary)
                    sensitivity.append(result)
                    if variant=='all_samples':
                        calculated.append(result)
                        r=next(r for r in summaries if int(r['seed'])==seed and int(r['epoch'])==epoch and float(r['assumed_omission_fraction'])==q)
                        close(result['predecided_ratio'],r['predecided_projected_ratio']);close(result['boundary_ratio'],r['boundary_projected_ratio']);close(result['boundary_over_predecided'],r['boundary_over_predecided_ratio'])
                        for path in PATHS:close(means[path],r[path+'_seconds'])
                if variant=='all_samples':
                    for path in PATHS:
                        values=[float(r['operation_seconds']) for r in a if r['path']==path]
                        aug=[float(r['augmentation_seconds']) for r in a if r['path']==path]
                        path_stats.append(dict(seed=seed,epoch=epoch,path=path,n=len(values),mean_operation_seconds=st.mean(values),median_operation_seconds=st.median(values),min_operation_seconds=min(values),max_operation_seconds=max(values),mean_augmentation_seconds=st.mean(aug),max_augmentation_seconds=max(aug)))
    telemetry=read(attempt/'telemetry.csv');assert len(telemetry)==16
    samples=[]
    for r in telemetry:
        data=json.loads(r['sample']);assert data['available']
        samples.extend(data['gpus'])
    preflight=json.loads((attempt/'preflight.json').read_text());assert len(preflight)==3
    assert all(float(g['utilization.gpu'])<=20 for sample in preflight for g in sample['gpus'])
    aggregate=[]
    for variant in ['all_samples','exclude_first_probe_posthoc','operation_only_posthoc']:
        for q in [.2,.4]:
            a=[r for r in sensitivity if r['variant']==variant and r['omission_fraction']==q]
            aggregate.append(dict(variant=variant,omission_fraction=q,states=4,
                mean_predecided_ratio=st.mean(r['predecided_ratio'] for r in a),mean_boundary_ratio=st.mean(r['boundary_ratio'] for r in a),
                mean_boundary_over_predecided=st.mean(r['boundary_over_predecided'] for r in a),
                min_boundary_over_predecided=min(r['boundary_over_predecided'] for r in a),max_boundary_over_predecided=max(r['boundary_over_predecided'] for r in a),
                mean_required_update_count_reduction=st.mean(r['required_update_count_reduction_for_equal_time'] for r in a)))
    save('recomputed_state_mixtures.csv',calculated);save('timing_sensitivity.csv',sensitivity);save('aggregate_costs.csv',aggregate);save('path_statistics.csv',path_stats)
    audit=dict(status='PASS',fingerprint=fp,raw_rows=512,matched_probe_batches=128,model_seeds=2,states=4,
        executed_source_hashes=len(cfg['source_sha256']),checkpoint_hashes=4,independently_recomputed_mixtures=8,
        pairing_order_index_and_timing_checks='PASS',executed_paired_gradient_and_freeze_counter_assertions='enforced by executed source before samples; completion reached',
        independent_gradient_reexecution=False,previous_immutable_manifest_entries_checked=sum(r['path'] not in MUTABLE for r in previous),
        temperature_range=[min(float(g['temperature.gpu']) for g in samples),max(float(g['temperature.gpu']) for g in samples)],
        clock_range=[min(float(g['clocks.sm']) for g in samples),max(float(g['clocks.sm']) for g in samples)],
        sensitivity_note='First-probe exclusion and operation-only summaries are post hoc; primary samples retained.')
    baseline=OUT/'preserved_inputs_sha256.csv'
    if baseline.exists():
        for r in read(baseline):verify(r['path'],r['sha256'])
    else:save('preserved_inputs_sha256.csv',[dict(path=p,sha256=h) for p,h in sorted(protected.items())])
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    print(json.dumps(audit,indent=2));print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
