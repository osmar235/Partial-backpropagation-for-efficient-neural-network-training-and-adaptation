"""Recompute descriptive Pet baseline results from a compact evidence capsule."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(root):
    cfg=json.loads((root/'baseline_freeze.json').read_text(encoding='utf8'))
    rows=[]
    for seed in cfg['spec']['seeds']:
        for policy in cfg['spec']['policies']:
            parent=root/f'seed_{seed}'/policy
            marker=json.loads((parent/'COMPLETE.json').read_text(encoding='utf8'))
            attempt=parent/marker['attempt']
            assert attempt.parent==parent
            s=json.loads((attempt/'summary.json').read_text(encoding='utf8'))
            with np.load(attempt/'logits.npz') as z:
                accuracy={mode:100*float(np.mean(z[mode].argmax(1)==z['labels'])) for mode in ('common','own')}
            for mode in accuracy: assert abs(accuracy[mode]-100*s[mode+'_accuracy'])<1e-5
            rows.append(dict(seed=seed,policy=policy,common_accuracy_percent=accuracy['common'],
                own_accuracy_percent=accuracy['own'],training_seconds=s['training_seconds'],complete_seconds=s['complete_seconds']))
    pairs=[]
    for seed in cfg['spec']['seeds']:
        fp=next(r for r in rows if r['seed']==seed and r['policy']=='fp32_tf32')
        bf=next(r for r in rows if r['seed']==seed and r['policy']=='bf16')
        pairs.append(dict(seed=seed,deficit_common_pp=fp['common_accuracy_percent']-bf['common_accuracy_percent'],
            deficit_own_pp=fp['own_accuracy_percent']-bf['own_accuracy_percent'],
            complete_saving_percent=100*(1-bf['complete_seconds']/fp['complete_seconds'])))
    return dict(rows=rows,pairs=pairs,scope='Descriptive development baseline only; no selective-method or official-test result.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    print(json.dumps(analyze(parser.parse_args().input),indent=2))
