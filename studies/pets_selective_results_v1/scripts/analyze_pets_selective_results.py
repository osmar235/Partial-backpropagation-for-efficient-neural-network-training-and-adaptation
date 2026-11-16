"""Portable saved-logit analysis and precision planning; development only."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2, t


def analyze(root):
    cfg=json.loads((root/'development_freeze.json').read_text(encoding='utf8'))
    rows=[]
    for seed in cfg['spec']['seeds']:
        for arm in cfg['plan']['arms']:
            parent=root/f'seed_{seed}'/arm
            marker=json.loads((parent/'COMPLETE.json').read_text(encoding='utf8'))
            attempt=parent/marker['attempt'];assert attempt.parent==parent
            s=json.loads((attempt/'summary.json').read_text(encoding='utf8'))
            with np.load(attempt/'logits.npz') as z:
                assert len(z['labels'])==736
                scores={mode:100*float(np.mean(z[mode].argmax(1)==z['labels'])) for mode in ('common','own')}
            for mode in scores:assert abs(scores[mode]-100*s[mode+'_accuracy'])<1e-5
            rows.append(dict(seed=seed,arm=arm,common_accuracy_percent=scores['common'],own_accuracy_percent=scores['own'],
                training_seconds=s['training_seconds'],complete_seconds=s['complete_seconds']))
    pairs=[]
    for seed in cfg['spec']['seeds']:
        full=next(r for r in rows if r['seed']==seed and r['arm']=='full')
        for arm in ('frozen','periodic_r4'):
            r=next(r for r in rows if r['seed']==seed and r['arm']==arm)
            pairs.append(dict(seed=seed,arm=arm,accuracy_difference_pp=r['common_accuracy_percent']-full['common_accuracy_percent'],
                own_accuracy_difference_pp=r['own_accuracy_percent']-full['own_accuracy_percent'],
                complete_saving_percent=100*(1-r['complete_seconds']/full['complete_seconds']),
                training_saving_percent=100*(1-r['training_seconds']/full['training_seconds'])))
    means={arm:{k:float(np.mean([r[k] for r in rows if r['arm']==arm])) for k in
        ('common_accuracy_percent','own_accuracy_percent','complete_seconds','training_seconds')} for arm in cfg['plan']['arms']}
    comparisons={}
    for arm in ('frozen','periodic_r4'):
        selected=[r for r in pairs if r['arm']==arm]
        d=np.array([r['accuracy_difference_pp'] for r in selected]);n=len(d);sd=float(d.std(ddof=1));mean=float(d.mean())
        half=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
        comparisons[arm]=dict(mean_accuracy_difference_pp=mean,paired_sd_pp=sd,exploratory_paired_t_95_ci_pp=[mean-half,mean+half],
            mean_complete_saving_percent=float(np.mean([r['complete_saving_percent'] for r in selected])),
            complete_saving_range_percent=[min(r['complete_saving_percent'] for r in selected),max(r['complete_saving_percent'] for r in selected)],
            mean_training_saving_percent=float(np.mean([r['training_saving_percent'] for r in selected])))
    sd=comparisons['periodic_r4']['paired_sd_pp'];df=len(cfg['spec']['seeds'])-1
    upper=float(sd*np.sqrt(df/chi2.ppf(.05,df)))
    def half_width(n,s):return float(t.ppf(.975,n-1)*s/np.sqrt(n))
    def required(s):return next(n for n in range(2,10001) if half_width(n,s)<=.20)
    planning=dict(target_half_width_pp=.20,cap=24,development_pairs=df+1,paired_sd_pp=sd,
        one_sided_95_percent_upper_sd_pp=upper,point_estimate_required_n=required(sd),upper_sd_required_n=required(upper),
        half_width_at_cap_point_sd_pp=half_width(24,sd),half_width_at_cap_upper_sd_pp=half_width(24,upper),
        recommended_evaluation_seeds=24,
        assumptions='Normal paired differences; t interval; one-sided chi-square upper SD. Trainval variance may not transfer to official test.',
        status='Planning recommendation; final evaluation protocol/code/independent seeds still to be frozen. No test access.')
    return dict(rows=rows,pairs=pairs,means=means,comparisons=comparisons,planning=planning,
        scope='Six development seeds on one validation split and pretrained backbone; not confirmatory evaluation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    print(json.dumps(analyze(parser.parse_args().input),indent=2))
