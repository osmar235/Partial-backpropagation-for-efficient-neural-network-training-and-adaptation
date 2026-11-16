"""Portable descriptive summaries of audited functional-comparison JSON."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

def span(values):
    values=list(values)
    return dict(min=min(values),median=median(values),max=max(values)) if values else None

def summarize(data):
    groups=defaultdict(list)
    for row in data['comparisons']:
        family='bf16_gradient' if '_bf16_' in row['arm'] else 'native_fp32_control'
        groups[(family,row['anchor'],row['context'],row['batch'],row['policy'])].append(row)
    summaries=[]
    for (family,anchor,context,batch,policy),rows in sorted(groups.items()):
        result=dict(family=family,anchor=anchor,context=context,batch=batch,policy=policy,n=len(rows))
        for effect in ('full','key_only','key_restored'):
            result[effect]={metric:span(r[effect][metric] for r in rows) for metric in rows[0][effect]}
            result[effect]['absolute_loss_change']=span(abs(r[effect]['loss_change']) for r in rows)
            result[effect]['prediction_change_cases']=sum(r[effect]['prediction_disagreements']>0 for r in rows)
        result['key_to_full_logit_l2']=span(r['key_to_full_logit_l2'] for r in rows if r['key_to_full_logit_l2'] is not None)
        result['restored_to_full_logit_l2']=span(r['key_restored']['logit_l2']/r['full']['logit_l2'] for r in rows if r['full']['logit_l2'])
        result['restoration_reduces_logit_l2_cases']=sum(r['key_restored']['logit_l2']<r['full']['logit_l2'] for r in rows)
        result['interaction_logit_l2']=span(r['interaction_logit_l2'] for r in rows)
        summaries.append(result)
    runtime=[]
    for policy in sorted({r['policy'] for r in data['runtime_reference_differences']}):
        rows=[r for r in data['runtime_reference_differences'] if r['policy']==policy]
        runtime.append(dict(policy=policy,n=len(rows),logit_max_abs=span(r['logit_max_abs'] for r in rows),logit_rms=span(r['logit_rms'] for r in rows),
            absolute_loss_change=span(abs(r['loss_change']) for r in rows),prediction_disagreements=span(r['prediction_disagreements'] for r in rows),
            prediction_change_cases=sum(r['prediction_disagreements']>0 for r in rows)))
    repeats=data['repeat_differences']
    return dict(groups=summaries,runtime=runtime,repeat_max_abs=max(r['logit_max_abs'] for r in repeats),repeat_max_absolute_loss_change=max(abs(r['loss_change']) for r in repeats))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('summary',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    result=summarize(json.loads(args.summary.read_text(encoding='utf8')))
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf8')
    else:print(text)
