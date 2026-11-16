"""Independent standard-library review of completed evidence; never trains."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'results/attention_reference_v1'
OUT=ROOT/'results/attention_reference_review_v1'
MUTABLE={ROOT/p for p in ('CURRENT_RELEASE.md','README_STAGE9.md','PROJECT_INDEX.md','paper_support/README.md')}

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')
def verify(mapping):
    def one(item):
        p,h=item;assert sha(p)==h,p
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,mapping.items()))
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def main():
    assert not OUT.exists()
    prep=ROOT/'results/attention_reference_preparation_v2'
    seal=read(prep/'review_artifacts_sha256.json');verify(seal)
    protected={**read(prep/'preserved_inputs_sha256.json'),**{p:h for p,h in seal.items() if Path(p) not in MUTABLE}}
    protected[str(prep/'review_artifacts_sha256.json')]=sha(prep/'review_artifacts_sha256.json')
    protected.update({str(p):sha(p) for p in RAW.rglob('*') if p.is_file()})
    verify(protected);OUT.mkdir()
    save(OUT/'preserved_inputs_sha256.json',protected)
    save(OUT/'mutable_pointers_before.json',{str(p):sha(p) for p in MUTABLE})
    cfg=read(RAW/'configuration.json');spec=cfg['spec']
    assert cfg==read(ROOT/'protocol/attention_reference/FROZEN_REFERENCE_v1.json')
    assert digest({k:v for k,v in cfg.items() if k!='fingerprint'})==cfg['fingerprint']
    verify(cfg['sources'])
    for marker in RAW.rglob('COMPLETE.json'):verify(read(marker)['sha256'])
    allrows=[];pairs=[];gates=[];profiles=[];telemetry=[];pids=[];state_groups=defaultdict(list)
    for session in range(spec['sessions']):
        parent=RAW/f'session_{session:02d}';launches=list((parent/'launches').glob('attempt_*'));assert len(launches)==1
        process=read(launches[0]/'process.json');assert process['session']==session;pids.append(process['pid'])
        for name in ('preflight','telemetry_before','telemetry_after'):
            telemetry.append(dict(session=session,kind=name,record=read(launches[0]/(name+'.json'))))
        for ci,(seed,batch) in enumerate(itertools.product(spec['seeds'],spec['batch_sizes'])):
            folder=parent/f'seed_{seed}_batch_{batch}';attempts=list(folder.glob('attempt_*'));assert len(attempts)==1
            attempt=attempts[0];rc=read(folder/'run_config.json');marker=read(folder/'COMPLETE.json')
            assert (rc['root_fingerprint'],rc['session'],rc['seed'],rc['batch'],rc['condition_index'])==(cfg['fingerprint'],session,seed,batch,ci)
            assert rc['fingerprint']==digest({k:v for k,v in rc.items() if k!='fingerprint'})==marker['fingerprint']
            assert Path(marker['attempt']).resolve()==attempt.resolve()
            assert set(marker['sha256'])=={str(p.resolve()) for p in attempt.rglob('*') if p.is_file()}
            assert read(attempt/'process.json')['pid']==process['pid']
            checks=read(attempt/'correctness.json');mechanism=read(attempt/'mechanism.json')
            assert list(checks)==spec['arms'] and list(mechanism)==spec['arms']
            for arm,check in checks.items():
                assert check['rng_match'] and check['deterministic_repeat']
                metrics=check['metrics'];assert metrics
                assert {m['step'] for m in metrics}==set(range(3))
                assert {m['kind'] for m in metrics}==set(spec['tolerances'])
                assert all(m['pass_tolerance'] and math.isfinite(m['max_abs']) and math.isfinite(m['max_scaled_error']) and m['max_scaled_error']<=1 for m in metrics)
                assert check['status']=='PASS'
                for kind in spec['tolerances']:
                    subset=[m for m in metrics if m['kind']==kind]
                    gates.append(dict(session=session,seed=seed,batch=batch,arm=arm,kind=kind,
                        max_abs=max(m['max_abs'] for m in subset),max_scaled=max(m['max_scaled_error'] for m in subset)))
                probe=mechanism[arm];policy=probe['policy']
                assert policy['deterministic'] and policy['math_sdp'] and not policy['warn_only']
                assert not any(policy[k] for k in ('tf32_matmul','tf32_cudnn','flash_sdp','mem_efficient_sdp','cudnn_sdp'))
                assert probe['attention_operators']['aten::_scaled_dot_product_attention_math']==6
                assert probe['unbind_backward_nodes']==(0 if arm=='math' else 6)
                assert probe['full_finite_gradients']
                profiles.append(dict(session=session,seed=seed,batch=batch,arm=arm,**probe))
            rows=read(attempt/'trials.json');assert len(rows)==18
            permutations=list(itertools.permutations(spec['arms']))
            expected=[(ri,pos,arm) for ri in range(6) for pos,arm in enumerate(permutations[(ri+session+ci)%6])]
            assert [(r['round'],r['position'],r['arm']) for r in rows]==expected
            for key in ('initial_sha256','input_sha256','labels_sha256'):assert len({r[key] for r in rows})==1
            for r in rows:
                assert (r['seed'],r['batch'],r['optimizer_age'])==(seed,batch,33)
                assert len(r['steady_losses'])==32 and r['all_gradients_finite']
                assert all(math.isfinite(z) for z in [r['first_loss'],*r['steady_losses']])
                assert all(math.isfinite(r[k]) and r[k]>0 for k in ('setup_seconds','warmup_restore_seconds','first_seconds','steady_seconds'))
                assert math.isclose(r['steady_update_seconds'],r['steady_seconds']/32,abs_tol=1e-12)
                for mem in ('first_memory','steady_memory'):
                    assert r[mem]['peak']-r[mem]['baseline']==r[mem]['incremental']
                state_groups[(seed,batch,r['arm'])].append([r[k] for k in ('final_model_sha256','final_optimizer_sha256','final_rng','first_loss','steady_losses')])
                allrows.append(dict(session=session,**r))
            for baseline,candidate in [('math','qkv_unbind'),('math','timm_reference'),('timm_reference','qkv_unbind')]:
                for ri in range(6):
                    a=next(r for r in rows if r['round']==ri and r['arm']==baseline)
                    b=next(r for r in rows if r['round']==ri and r['arm']==candidate)
                    lossdiff=max(abs(x-y) for x,y in zip([a['first_loss'],*a['steady_losses']],[b['first_loss'],*b['steady_losses']]))
                    pairs.append(dict(session=session,seed=seed,batch=batch,round=ri,baseline=baseline,candidate=candidate,
                        saving_percent=100*(a['steady_seconds']-b['steady_seconds'])/a['steady_seconds'],
                        baseline_ms=1000*a['steady_seconds']/32,candidate_ms=1000*b['steady_seconds']/32,
                        max_recorded_loss_difference=lossdiff,exact_timed_learning_records=all(a[k]==b[k] for k in ('first_loss','steady_losses','final_model_sha256','final_optimizer_sha256','final_rng')),
                        peak_memory_delta=b['steady_memory']['peak']-a['steady_memory']['peak'],
                        incremental_memory_delta=b['steady_memory']['incremental']-a['steady_memory']['incremental']))
    assert len(set(pids))==4 and len(allrows)==288 and len(pairs)==288
    assert all(len(values)==24 and all(v==values[0] for v in values) for values in state_groups.values())
    saved=read(RAW/'paired_results.json')
    for actual,prior in zip(pairs,saved):
        for k in ('session','seed','batch','round','baseline','candidate','exact_timed_learning_records'):assert actual[k]==prior[k]
        for k in ('saving_percent','baseline_ms','candidate_ms'):assert math.isclose(actual[k],prior[k],abs_tol=1e-10)
    summaries=[]
    for seed,batch,base,cand in sorted({(r['seed'],r['batch'],r['baseline'],r['candidate']) for r in pairs}):
        group=[r for r in pairs if (r['seed'],r['batch'],r['baseline'],r['candidate'])==(seed,batch,base,cand)]
        sessions=[median(r['saving_percent'] for r in group if r['session']==s) for s in range(4)]
        summaries.append(dict(seed=seed,batch=batch,baseline=base,candidate=cand,median_saving_percent=median(sessions),
            session_medians=sessions,positive_sessions=sum(v>0 for v in sessions),positive_pairs=sum(r['saving_percent']>0 for r in group),
            exact_pairs=sum(r['exact_timed_learning_records'] for r in group),max_loss_difference=max(r['max_recorded_loss_difference'] for r in group),
            peak_memory_deltas=sorted({r['peak_memory_delta'] for r in group}),incremental_memory_deltas=sorted({r['incremental_memory_delta'] for r in group})))
    recorded=read(RAW/'audit.json')
    for a,b in zip(summaries,recorded['summary']):
        for k in ('seed','batch','baseline','candidate','positive_sessions'):assert a[k]==b[k]
        assert math.isclose(a['median_saving_percent'],b['median_saving_percent'],abs_tol=1e-10)
    positions=[]
    for seed,batch,arm in sorted(state_groups):
        for pos in range(3):
            group=[r for r in allrows if (r['seed'],r['batch'],r['arm'],r['position'])==(seed,batch,arm,pos)]
            assert len(group)==8
            positions.append(dict(seed=seed,batch=batch,arm=arm,position=pos,trials=len(group),median_ms=median(r['steady_update_seconds']*1000 for r in group)))
    result=dict(status='PASS',trials=288,measured_updates=9504,warmup_updates=1440,
        correctness_updates=16*7*3,mechanism_updates=16*3,sessions=4,distinct_process_ids=pids,
        numerical_failures=0,fragmented_sessions=0,all_within_arm_trajectories_repeat_exactly=True,
        no_experiment_rerun=True,summary=summaries)
    save(OUT/'independent_audit.json',result);save(OUT/'independent_pairs.json',pairs)
    save(OUT/'numerical_summary.json',gates);save(OUT/'mechanism_summary.json',profiles)
    save(OUT/'position_summary.json',positions);save(OUT/'telemetry.json',telemetry)
    save(OUT/'trials_export.json',allrows)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
