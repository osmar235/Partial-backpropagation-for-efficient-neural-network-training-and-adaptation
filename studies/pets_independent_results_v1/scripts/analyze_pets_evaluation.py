"""Frozen paired-seed inference for the two prespecified efficiency claims."""
import math
import numpy as np
from scipy.stats import t


def interval(values,alpha=.025):
    x=np.asarray(values,dtype=float)
    assert len(x)>=2 and np.isfinite(x).all()
    mean=float(x.mean());sd=float(x.std(ddof=1));half=float(t.ppf(1-alpha,len(x)-1)*sd/math.sqrt(len(x)))
    return dict(n=len(x),mean=mean,sd=sd,lower=mean-half,upper=mean+half,half_width=half)


def analyze(rows,spec,fragmented=False,smoke=False):
    assert {(r['seed'],r['arm']) for r in rows}=={(s,a) for s in spec['seeds'] for a in spec['arms']}
    assert len(rows)==len(spec['seeds'])*len(spec['arms'])
    means={a:{k:float(np.mean([r[k] for r in rows if r['arm']==a])) for k in
        ('accuracy_percent','own_accuracy_percent','complete_seconds','training_seconds','test_assessment_seconds')} for a in spec['arms']}
    pairs=[];claims={}
    for arm in ('periodic_r4','frozen'):
        selected=[]
        for seed in spec['seeds']:
            full=next(r for r in rows if r['seed']==seed and r['arm']=='full')
            r=next(r for r in rows if r['seed']==seed and r['arm']==arm)
            assert r['complete_seconds']>0 and full['complete_seconds']>0
            selected.append(dict(seed=seed,arm=arm,accuracy_difference_pp=r['accuracy_percent']-full['accuracy_percent'],
                own_accuracy_difference_pp=r['own_accuracy_percent']-full['own_accuracy_percent'],
                log_time_ratio=math.log(r['complete_seconds']/full['complete_seconds']),
                complete_saving_percent=100*(1-r['complete_seconds']/full['complete_seconds']),
                training_saving_percent=100*(1-r['training_seconds']/full['training_seconds'])))
        accuracy=interval([r['accuracy_difference_pp'] for r in selected],spec['per_comparison_one_sided_alpha'])
        timing=interval([r['log_time_ratio'] for r in selected],spec['per_comparison_one_sided_alpha'])
        quality=accuracy['lower']>-spec['accuracy_margin_pp'];faster=timing['upper']<0
        claims[arm]=dict(role='primary' if arm=='periodic_r4' else 'secondary',accuracy=accuracy,log_time_ratio=timing,
            geometric_time_ratio=math.exp(timing['mean']),geometric_saving_percent=100*(1-math.exp(timing['mean'])),
            geometric_saving_interval_percent=[100*(1-math.exp(timing['upper'])),100*(1-math.exp(timing['lower']))],
            arithmetic_mean_paired_saving_percent=float(np.mean([r['complete_saving_percent'] for r in selected])),
            quality_bound_pass=quality,time_bound_pass=faster,combined_claim=bool(quality and faster and not fragmented and not smoke),
            own_accuracy_descriptive=interval([r['own_accuracy_difference_pp'] for r in selected]))
        pairs.extend(selected)
    return dict(means=means,pairs=pairs,claims=claims,fragmented=fragmented,smoke=smoke,
        inference='Two composite method claims, each intersection-union alpha .025; Bonferroni family bound .05. Conditional on the test sample, backbone and timing session.')
