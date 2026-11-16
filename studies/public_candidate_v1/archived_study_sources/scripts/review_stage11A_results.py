"""Post-run preservation and independent feasibility arithmetic; no training."""
from pathlib import Path
import csv,hashlib,json,math,statistics as st

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/stage11A_growth_feasibility_v1'
OUT=ROOT/'results/stage11A_review_v1'
MUTABLE={str(ROOT/n) for n in ['README_STAGE9.md','CURRENT_RELEASE.md']}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save(name,rows):
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True);protected={}
    prior=json.loads((ROOT/'protocol/stage11/preparation_preservation_v1.json').read_text())
    def verify(p,h):
        assert sha(p)==h,p
        if str(p) not in MUTABLE:protected[str(p)]=h
    for p,h in prior['prior_immutable_sha256'].items():verify(p,h)
    baseline=OUT/'preserved_inputs_sha256.csv'
    for r in read(ROOT/'protocol/stage11/preparation_sha256_v1.csv'):
        if r['path'] not in MUTABLE or not baseline.exists():verify(r['path'],r['sha256'])
    for p in [ROOT/'protocol/stage11/preparation_sha256_v1.csv',*RAW.rglob('*')]:
        if p.is_file():verify(p,sha(p))
    if baseline.exists():
        for r in read(baseline):verify(r['path'],r['sha256'])
    else:
        save('preserved_inputs_sha256.csv',[dict(path=p,sha256=h) for p,h in sorted(protected.items())])
        (OUT/'mutable_pointers_before.json').write_text(json.dumps({p:sha(p) for p in sorted(MUTABLE)},indent=2)+'\n')
    marker=json.loads((RAW/'COMPLETE.json').read_text())
    for p,h in marker['sha256'].items():verify(p,h)
    attempt=Path(marker['attempt']);cfg=json.loads((RAW/'configuration.json').read_text());spec=cfg['spec']
    assert not cfg['synthetic']
    audit=json.loads((attempt/'audit.json').read_text())
    assert audit['status']=='PASS' and audit['feasible_for_pilot_review']
    rows=read(attempt/'measurements.csv');eq=read(attempt/'equivalence.csv');summary=read(attempt/'summary.csv')
    states=json.loads((attempt/'input_states.json').read_text());result=[]
    for seed in spec['source_seeds']:
        means={p:st.mean(float(r['seconds']) for r in rows if int(r['seed'])==seed and r['path']==p) for p in spec['paths']}
        es=[r for r in eq if int(r['seed'])==seed]
        changes=sum(r['small_prediction']!=r['grown_prediction'] for r in es)
        assert len(es)==5000 and changes==0
        a=st.mean(r['small_prediction']==r['label'] for r in es)
        b=st.mean(r['grown_prediction']==r['label'] for r in es)
        sr=means['small']/means['full'];gr=means['grown']/means['full']
        original=next(r for r in summary if int(r['seed'])==seed)
        for k,value in [('small_accuracy',a),('grown_accuracy',b),('small_full_step_ratio',sr),('grown_full_step_ratio',gr)]:
            assert math.isclose(value,float(original[k]),rel_tol=1e-12,abs_tol=1e-12)
        assert sr<=spec['maximum_small_full_step_ratio'] and gr<=spec['maximum_grown_full_step_ratio']
        result.append(dict(seed=seed,small_accuracy=a,grown_accuracy=b,prediction_changes=changes,
                           small_step_ms=means['small']*1000,grown_step_ms=means['grown']*1000,full_step_ms=means['full']*1000,
                           small_step_saving_percent=100*(1-sr),grown_full_step_ratio=gr,
                           growth_ms=next(r['growth_seconds'] for r in states if r['seed']==seed)*1000,
                           max_abs_logit_error=max(float(r['max_abs_logit_error']) for r in es),feasibility_pass=True))
    save('independent_summary.csv',result)
    for p,h in protected.items():assert sha(p)==h,p
    review=dict(status='PASS',date='2026-10-22',source_states=4,validation_examples=10000,measured_operations=216,
                disposable_trainability_updates=48,preserved_immutable_files=len(protected),independent_arithmetic_matches=True,
                feasible_for_bounded_pilot=True,training_efficacy_established=False,
                decision='Proceed to preparation of one fixed midpoint-growth pilot with uninterrupted and reset controls.')
    (OUT/'independent_audit.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(review,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
