"""Reproduce the frozen final analysis from compact evidence (NumPy/SciPy only)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_pets_evaluation import analyze


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def attempt(parent,cfg,names):
    marker=read(parent/'COMPLETE.json');assert marker['fingerprint']==cfg['fingerprint']
    folder=parent/marker['attempt'];assert folder.parent==parent
    for name in names:assert sha(folder/name)==marker['sha256'][name]
    return folder


def reproduce(root):
    cfg=read(root/'evaluation_freeze.json');rows=[];reference_labels=None
    for seed in cfg['spec']['seeds']:
        for arm in cfg['plan']['arms']:
            trained=attempt(root/'training'/f'seed_{seed}'/arm,cfg,['summary.json'])
            scored=attempt(root/'test_scores'/f'seed_{seed}'/arm,cfg,['summary.json','logits.npz'])
            train=read(trained/'summary.json');score=read(scored/'summary.json');scores={}
            with np.load(scored/'logits.npz') as z:
                labels=z['labels'];assert len(labels)==cfg['evaluation_spec']['official_test_examples']
                if reference_labels is None:reference_labels=labels.copy()
                assert np.array_equal(labels,reference_labels)
                for mode in ('common','own'):
                    x=z[mode].astype(np.float64);assert np.isfinite(x).all()
                    acc=float(np.mean(x.argmax(1)==labels));loss=float(np.mean(np.logaddexp.reduce(x,axis=1)-x[np.arange(len(labels)),labels]))
                    assert abs(acc-score[mode+'_accuracy'])<1e-7 and abs(loss-score[mode+'_loss'])<1e-5
                    scores[mode]=100*acc
            rows.append(dict(seed=seed,arm=arm,accuracy_percent=scores['common'],own_accuracy_percent=scores['own'],
                complete_seconds=train['complete_seconds'],training_seconds=train['training_seconds'],test_assessment_seconds=score['test_assessment_seconds']))
    original=read(root/'analysis.json')
    result=analyze(rows,cfg['evaluation_spec'],fragmented=read(root/'training/audit.json')['fragmented'],smoke=cfg['smoke'])
    assert all(result[k]==original[k] for k in result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    result=reproduce(parser.parse_args().input)
    print(json.dumps(dict(status='PASS',means=result['means'],claims=result['claims']),indent=2))
