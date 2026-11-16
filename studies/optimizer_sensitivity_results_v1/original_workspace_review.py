"""Independent CPU analysis of saved tensors. Never runs model gradients/updates."""
from collections import Counter,defaultdict
import math
from pathlib import Path
import sys
from statistics import median
from record_preparation import ROOT,read,save,sha,verify,MUTABLE
import torch

RAW=ROOT/'results/optimizer_sensitivity_v1';OUT=ROOT/'results/optimizer_sensitivity_review_v1'

def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12),(a,b)
def aggregate(a,b):
    aa=bb=dd=dot=0.;maximum=0.
    for n in a:
        x=a[n].double().flatten();y=b[n].double().flatten();z=x-y
        aa+=float((x*x).sum());bb+=float((y*y).sum());dd+=float((z*z).sum());dot+=float((x*y).sum());maximum=max(maximum,float(z.abs().max()))
    return dict(relative_l2=math.sqrt(dd)/max(math.sqrt(aa),1e-12),cosine=dot/math.sqrt(aa*bb) if aa and bb else 1. if not aa and not bb else 0.,max_abs=maximum,squared_difference=dd)

def main():
    torch.set_num_threads(1);assert not OUT.exists()
    prep=ROOT/'results/optimizer_sensitivity_preparation_v1';seal=read(prep/'review_artifacts_sha256.json');verify(seal)
    protected={p:h for p,h in {**read(prep/'preserved_inputs_sha256.json'),**seal}.items() if Path(p) not in MUTABLE}
    protected[str(prep/'review_artifacts_sha256.json')]=sha(prep/'review_artifacts_sha256.json')
    protected.update({str(p):sha(p) for p in RAW.rglob('*') if p.is_file()});verify(protected)
    OUT.mkdir();save(OUT/'preserved_inputs_sha256.json',protected);save(OUT/'mutable_pointers_before.json',{str(p):sha(p) for p in MUTABLE})
    cfg=read(RAW/'configuration.json');assert cfg==read(ROOT/'protocol/optimizer_sensitivity/FROZEN_DIAGNOSTIC_v1.json')
    verify(cfg['sources']);verify(cfg['evidence'])
    for marker in RAW.rglob('COMPLETE.json'):verify(read(marker)['sha256'])
    exports=read(RAW/'summary.json');indexed={(r['task'],r['anchor'],r['context'],r['arm']):r for r in exports};assert len(indexed)==132
    rows=[];coordinates=[];profiles=[];source_metadata=[];counts=Counter()
    for parent in sorted(p for p in RAW.iterdir() if p.is_dir()):
        attempts=list(parent.glob('attempt_*'));assert len(attempts)==1;attempt=attempts[0]
        source_metadata.append(dict(task=parent.name,source=read(attempt/'source.json')))
        for anchor in (0,2):
            raw=torch.load(attempt/f'anchor_{anchor}.pt',map_location='cpu',weights_only=False)
            summaries=read(attempt/f'anchor_{anchor}_summary.json');bykey={(r['context'],r['arm']):r for r in summaries}
            weights=raw['weights'];captures=raw['captures'];task=raw['task'];ref=captures['reference_fp32_math']['gradient']
            assert len(weights)==80 and sum(t.numel() for t in weights.values())==2693578
            assert raw['reference_repeat_exact']
            for arm,cap in captures.items():
                assert cap['status']=='FINITE';counts['captures']+=1
                profiles.append(dict(task=parent.name,anchor=anchor,arm=arm,dispatch=cap['dispatch']))
            for context,state in raw['contexts'].items():
                records=raw['replays'][context];reference_delta=records['reference_fp32_math']['delta']
                for arm,rec in records.items():
                    saved=bykey[context,arm];cand=captures[arm]['gradient'];delta=rec['delta']
                    grad=aggregate(ref,cand);update=aggregate(reference_delta,delta)
                    for k in ('relative_l2','cosine','max_abs'):close(grad[k],saved['gradient'][k]);close(update[k],saved['update'][k])
                    key_sq=0.;key_count=0;key_refmax=key_candmax=key_upmax=0.;nonkey_upmax=0.;flips=zeros=0
                    bins=[dict(coordinates=0,sign_flips=0,zero_disagreements=0,squared_difference=0.) for _ in range(6)]
                    for name in weights:
                        a=ref[name].double().flatten();b=cand[name].double().flatten();error=(reference_delta[name].double()-delta[name].double()).flatten()
                        sign=a*b<0;zero=(a==0)!=(b==0);flips+=int(sign.sum());zeros+=int(zero.sum())
                        bounds=[0,*cfg['spec']['gradient_bin_upper_bounds'],float('inf')]
                        for i in range(6):
                            mask=(a.abs()>=bounds[i])&(a.abs()<bounds[i+1]);entry=bins[i]
                            entry['coordinates']+=int(mask.sum());entry['sign_flips']+=int(sign[mask].sum());entry['zero_disagreements']+=int(zero[mask].sum());entry['squared_difference']+=float((error[mask]**2).sum())
                        if name.endswith('self_attention.in_proj_bias'):
                            assert len(a)==576
                            mask=torch.zeros(len(a),dtype=torch.bool);mask[192:384]=True;key_count+=192
                            key_sq+=float((error[mask]**2).sum());key_refmax=max(key_refmax,float(a[mask].abs().max()));key_candmax=max(key_candmax,float(b[mask].abs().max()))
                            key_upmax=max(key_upmax,float(error[mask].abs().max()));nonkey_upmax=max(nonkey_upmax,float(error[~mask].abs().max()))
                        else:nonkey_upmax=max(nonkey_upmax,float(error.abs().max()))
                    assert key_count==1152 and flips==saved['strict_sign_flips'] and zeros==saved['zero_disagreements']
                    for actual,expected in zip(bins,saved['bins']):
                        for k in ('coordinates','sign_flips','zero_disagreements'):assert actual[k]==expected[k]
                        close(actual['squared_difference'],expected['update_difference_squared'])
                    group=state['param_groups'][0];names=list(weights);max_analytic_error=0.;topflips=0;topkey=0
                    for c in saved['top_coordinates']:
                        name=c['name'];j=c['flat_index'];i=names.index(name);hist=state['state'].get(i)
                        theta=float(weights[name].flatten()[j]);a=float(ref[name].flatten()[j]);b=float(cand[name].flatten()[j])
                        m=float(hist['exp_avg'].flatten()[j]) if hist else 0.;v=float(hist['exp_avg_sq'].flatten()[j]) if hist else 0.;age=int(hist['step']) if hist else 0
                        assert (theta,a,b,m,v,age)==(c['parameter'],c['reference_gradient'],c['candidate_gradient'],c['initial_first_moment'],c['initial_second_moment'],c['initial_step'])
                        for g,key,measured in ((a,'reference_analytic_delta','reference_delta'),(b,'candidate_analytic_delta','candidate_delta')):
                            beta1,beta2=group['betas'];mm=(beta1*m+(1-beta1)*g)/(1-beta1**(age+1));vv=(beta2*v+(1-beta2)*g*g)/(1-beta2**(age+1))
                            estimate=-group['lr']*(group['weight_decay']*theta+mm/(math.sqrt(vv)+group['eps']))
                            close(estimate,c[key]);max_analytic_error=max(max_analytic_error,abs(estimate-c[measured]))
                        topflips+=a*b<0;topkey+=name.endswith('self_attention.in_proj_bias') and 192<=j<384
                        coordinates.append(dict(task=parent.name,anchor=anchor,context=context,arm=arm,**c))
                    denom=update['squared_difference']
                    row=dict(task=parent.name,kind=task['kind'],domain=task['domain'],seed=task['seed'],anchor=anchor,context=context,arm=arm,
                        gradient_relative_l2=grad['relative_l2'],gradient_cosine=grad['cosine'],update_relative_l2=update['relative_l2'],max_update_difference=update['max_abs'],
                        sign_flip_fraction=flips/2693578,small_gradient_mass_lt_1e6=sum(z['squared_difference'] for z in bins[:3])/denom if denom else 0.,
                        small_gradient_mass_lt_1e4=sum(z['squared_difference'] for z in bins[:4])/denom if denom else 0.,
                        exploratory_key_bias_mass=key_sq/denom if denom else 0.,key_bias_coordinates=key_count,key_bias_reference_gradient_max=key_refmax,
                        key_bias_candidate_gradient_max=key_candmax,key_bias_update_difference_max=key_upmax,other_update_difference_max=nonkey_upmax,
                        top32_sign_flips=int(topflips),top32_key_bias=int(topkey),top32_max_analytic_error=max_analytic_error)
                    original=indexed[parent.name,anchor,context,arm]
                    for k in ('gradient_relative_l2','gradient_cosine','update_relative_l2','max_update_difference','sign_flip_fraction'):close(row[k],original[k])
                    close(row['small_gradient_mass_lt_1e6'],original['small_gradient_update_mass']);rows.append(row)
            print('REVIEWED',parent.name,anchor,flush=True)
    assert len(rows)==132 and counts['captures']==72
    bf16=[r for r in rows if 'bf16' in r['arm']];assert len(bf16)==88
    grouped=[]
    for key in sorted({(r['kind'],r['domain'],r['anchor'],r['context']) for r in bf16}):
        group=[r for r in bf16 if (r['kind'],r['domain'],r['anchor'],r['context'])==key]
        metrics=['gradient_relative_l2','update_relative_l2','max_update_difference','sign_flip_fraction','small_gradient_mass_lt_1e6','small_gradient_mass_lt_1e4','exploratory_key_bias_mass','top32_max_analytic_error']
        grouped.append(dict(kind=key[0],domain=key[1],anchor=key[2],context=key[3],comparisons=len(group),
            ranges={k:dict(min=min(r[k] for r in group),max=max(r[k] for r in group),median=median(r[k] for r in group)) for k in metrics}))
    idx={(r['task'],r['anchor'],r['context'],r['arm']):r for r in rows};contrasts=[]
    for row in bf16:
        if row['context']=='fresh':continue
        fresh=idx[row['task'],row['anchor'],'fresh',row['arm']]
        contrasts.append(dict(task=row['task'],domain=row['domain'],kind=row['kind'],anchor=row['anchor'],context=row['context'],arm=row['arm'],
            relative_l2_ratio=row['update_relative_l2']/fresh['update_relative_l2'],maximum_difference_ratio=row['max_update_difference']/fresh['max_update_difference']))
    result=dict(status='PASS',trajectories=6,anchors=12,captures=72,comparisons=132,bf16_comparisons=88,
        fresh_bf16_comparisons=sum(r['context']=='fresh' for r in bf16),
        fresh_top32_all_flip=all(r['top32_sign_flips']==32 for r in bf16 if r['context']=='fresh'),
        retained_two_improves_l2=sum(r['relative_l2_ratio']<1 for r in contrasts if r['context']=='retained_two_updates'),
        retained_two_comparisons=sum(r['context']=='retained_two_updates' for r in contrasts),
        archived_training_worsens_l2=sum(r['relative_l2_ratio']>1 for r in contrasts if r['context']=='archived' and r['domain']=='training'),
        archived_training_comparisons=sum(r['context']=='archived' and r['domain']=='training' for r in contrasts),
        archived_maxima_all_key_bias=all(r['key_bias_update_difference_max']>r['other_update_difference_max'] for r in bf16 if r['context']=='archived'),
        max_top32_analytic_error=max(r['top32_max_analytic_error'] for r in bf16),
        exploratory_partition='Post hoc Q/K/V packed-bias slicing; no excluded coordinates or changed historical gate',
        model_forward_backward_rerun=False,optimizer_step_rerun=False,quality_established=False,performance_measured=False)
    for name,value in [('independent_audit.json',result),('comparison_rows.json',rows),('group_summary.json',grouped),('history_contrasts.json',contrasts),('coordinate_traces.json',coordinates),('dispatch.json',profiles),('source_metadata.json',source_metadata)]:save(OUT/name,value)
    save(OUT/'frozen_audit_recheck_receipt.json',dict(status='PASS',exit_code=0,command='python -B scripts/review_optimizer_sensitivity.py results/optimizer_sensitivity_v1',model_forward_backward_rerun=False,record_type='Observed successful tool execution'))
    verify(protected);print(result)

if __name__=='__main__':main()
