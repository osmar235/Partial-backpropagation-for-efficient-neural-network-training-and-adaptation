"""Descriptive ranges and paired controls, without inferential significance tests."""
import argparse
import json
from pathlib import Path
from summarize_results_v1 import span

def extract(data):
    findings=[]
    for subset in ('all_bf16_gradients','archived_bf16_gradients','native_fp32_control'):
        selected=[r for r in data['comparisons'] if ('bf16' in r['arm'] if subset!='native_fp32_control' else r['arm']=='native_fp32_math') and (subset!='archived_bf16_gradients' or r['context']=='archived')]
        for policy in sorted({r['policy'] for r in selected}):
            rows=[r for r in selected if r['policy']==policy]
            effects={}
            for effect in ('full','key_only','key_restored'):
                effects[effect]=dict(max_abs=span(r[effect]['logit_max_abs'] for r in rows),rms=span(r[effect]['logit_rms'] for r in rows),
                    abs_loss=span(abs(r[effect]['loss_change']) for r in rows),prediction_change_cases=sum(r[effect]['prediction_disagreements']>0 for r in rows),
                    max_predictions_changed=max(r[effect]['prediction_disagreements'] for r in rows),correct_change=span(r[effect]['correct_change'] for r in rows))
            findings.append(dict(subset=subset,policy=policy,n=len(rows),effects=effects,
                key_to_full=span(r['key_to_full_logit_l2'] for r in rows if r['key_to_full_logit_l2'] is not None),
                restored_to_full=span(r['key_restored']['logit_l2']/r['full']['logit_l2'] for r in rows if r['full']['logit_l2']),
                restoration_reduces_logit_l2=sum(r['key_restored']['logit_l2']<r['full']['logit_l2'] for r in rows),
                interaction_to_full=span(r['interaction_logit_l2']/r['full']['logit_l2'] for r in rows if r['full']['logit_l2'])))
    return findings

def main():
    ap=argparse.ArgumentParser();ap.add_argument('summary',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    findings=extract(json.loads(args.summary.read_text(encoding='utf8')))
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(findings,indent=2)+'\n',encoding='utf8')
    for f in findings:
        k=f['effects']['key_only'];r=f['restored_to_full']
        print(f['subset'],f['policy'],'n',f['n'],'keymax',k['max_abs']['max'],'keyRMS',k['rms'],
              'keyLossMax',k['abs_loss']['max'],'keyPredCases',k['prediction_change_cases'],'maxPred',k['max_predictions_changed'],
              'key/full',f['key_to_full'],'restored/full',r,'restorationLower',f['restoration_reduces_logit_l2'])

if __name__=='__main__':main()
