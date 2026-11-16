"""Read-only Stage15A review: independent arithmetic and source/final rescoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/stage15A_side_adaptation_v1'
OUT=ROOT/'results/stage15A_review_v1'
MUTABLE={str(ROOT/name) for name in ('CURRENT_RELEASE.md','README_STAGE9.md')}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')

def save(name,rows):
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)

def metrics(z,y):
    a=z.astype(np.float64);a-=a.max(1,keepdims=True)
    return dict(accuracy=float(np.mean(z.argmax(1)==y)),loss=float(np.mean(np.log(np.exp(a).sum(1))-a[np.arange(len(y)),y])))

def main():
    assert not OUT.exists();OUT.mkdir()
    prep=ROOT/'protocol/stage15a'
    protected=json.loads((prep/'preparation_preservation_v1.json').read_text())['prior_immutable_sha256']
    seal=prep/'preparation_sha256_v1.json';entries=json.loads(seal.read_text())
    for p,h in {**protected,**entries}.items():assert sha(p)==h,p
    protected.update({p:h for p,h in entries.items() if p not in MUTABLE});protected[str(seal)]=sha(seal)
    protected.update({str(p):sha(p) for p in RAW.rglob('*') if p.is_file()})
    dump('preserved_inputs_sha256.json',protected);dump('mutable_pointers_before.json',{p:sha(p) for p in sorted(MUTABLE)})
    import torch
    import stage15_side_adaptation as r
    from review_stage15A import audit
    r.m.configure();cfg=json.loads((RAW/'configuration.json').read_text());spec=cfg['spec']
    assert not cfg['synthetic'] and r.d.validate_complete(RAW,cfg['fingerprint'])
    frozen=audit(RAW,False);dump('frozen_audit_recheck.json',frozen)
    assert frozen['status']=='PASS' and frozen['training_updates']==8800
    summaries={(int(row['seed']),row['mode']):row for row in read(RAW/'summary.csv')}
    screens={int(row['seed']):row for row in read(RAW/'screen.csv')}
    predictions={};attempts={};curves=[];pairs=[];decisions=[];ordinary=[];labels=None
    single_launch=len({row['launch_record'] for row in summaries.values()})==1
    for seed in spec['seeds']:
        for arm in spec['modes']:
            parent=RAW/f'seed_{seed}'/arm;attempt=Path(json.loads((parent/'COMPLETE.json').read_text())['attempt']);attempts[seed,arm]=attempt
            with np.load(attempt/'predictions.npz',allow_pickle=False) as z:
                if labels is None:labels=z['labels'].copy()
                assert np.array_equal(labels,z['labels'])
                predictions[seed,arm]={epoch:z[f'epoch_{epoch}'].copy() for epoch in range(spec['epochs']+1)}
            saved=read(attempt/'curves.csv')
            for epoch,z in predictions[seed,arm].items():
                row=metrics(z,labels);close(row['accuracy'],saved[epoch]['accuracy']);close(row['loss'],saved[epoch]['loss'])
                curves.append(dict(seed=seed,arm=arm,epoch=epoch,**row))
            summary=summaries[seed,arm]
            close(summary['final_accuracy'],metrics(predictions[seed,arm][5],labels)['accuracy'])
            close(summary['complete_adaptation_seconds'],float(summary['adaptation_seconds'])+float(summary['shared_data_seconds']))
        def accuracy(arm):return metrics(predictions[seed,arm][5],labels)['accuracy']
        def seconds(arm):return float(summaries[seed,arm]['complete_adaptation_seconds'])
        source=metrics(predictions[seed,'full'][0],labels)['accuracy']
        controls=['full','head','lp_full','final_side'];anchor=max([source]+[accuracy(arm) for arm in controls])
        eligible=[arm for arm in controls if 100*(anchor-accuracy(arm))<=spec['accuracy_margin_pp']+1e-10]
        ready=accuracy('full')>=spec['baseline_min_accuracy'] and 100*(source-accuracy('full'))<=spec['accuracy_margin_pp']+1e-10 and bool(eligible)
        comparator=min(set(eligible+['full']),key=seconds);fastest=min(eligible,key=seconds) if eligible else ''
        ratio=seconds('multi_side')/seconds(comparator);deficit=100*(anchor-accuracy('multi_side'))
        quality=deficit<=spec['accuracy_margin_pp']+1e-10;time_pass=ratio<=1-spec['minimum_time_saving']+1e-12
        raw=screens[seed]
        for key,val in [('best_control_accuracy',anchor),('final_delta_vs_best_pp',-deficit),('complete_time_ratio',ratio),
            ('delta_vs_final_side_pp',100*(accuracy('multi_side')-accuracy('final_side')))]:close(raw[key],val)
        for key,val in [('reference_qualified',ready),('quality_pass',quality),('time_pass',time_pass),('timing_eligible',single_launch),
            ('final_side_quality_qualified','final_side' in eligible),('seed_screen_pass',ready and quality and time_pass and single_launch)]:assert raw[key]==str(val)
        assert raw['fastest_qualified_control']==fastest
        candidate=summaries[seed,'multi_side'];ref=summaries[seed,comparator]
        decisions.append(dict(seed=seed,source_accuracy=source,best_control_accuracy=anchor,candidate_accuracy=accuracy('multi_side'),
            deficit_vs_best_pp=deficit,quality_pass=quality,time_pass=time_pass,reference_qualified=ready,
            practical_seed_pass=ready and quality and time_pass and single_launch,qualifying_controls=';'.join(eligible),
            time_comparator=comparator,candidate_adaptation_seconds=seconds('multi_side'),comparator_adaptation_seconds=seconds(comparator),
            adaptation_saving_percent=100*(1-ratio),seconds_saved=seconds(comparator)-seconds('multi_side'),
            source_training_inclusive_saving_percent=100*(1-float(candidate['source_training_plus_adaptation_seconds'])/float(ref['source_training_plus_adaptation_seconds'])),
            incremental_memory_saving_percent=100*(1-float(candidate['max_training_incremental_allocated'])/float(ref['max_training_incremental_allocated'])),
            absolute_training_peak_saving_percent=100*(1-float(candidate['max_training_peak_allocated'])/float(ref['max_training_peak_allocated'])),
            inference_time_ratio_vs_full=float(candidate['mean_inference_seconds'])/float(summaries[seed,'full']['mean_inference_seconds']),
            candidate_minus_source_pp=100*(accuracy('multi_side')-source),candidate_minus_head_pp=100*(accuracy('multi_side')-accuracy('head')),
            candidate_minus_final_side_pp=100*(accuracy('multi_side')-accuracy('final_side'))))
        ordinary.append(dict(seed=seed,lp_full_accuracy=accuracy('lp_full'),full_accuracy=accuracy('full'),
            lp_full_minus_full_pp=100*(accuracy('lp_full')-accuracy('full')),quality_qualified='lp_full' in eligible,
            saving_vs_full_percent=100*(1-seconds('lp_full')/seconds('full')),
            meets_10_percent_time_rule=seconds('lp_full')<=.9*seconds('full')))
        for control in controls+['source']:
            base=predictions[seed,'full'][0] if control=='source' else predictions[seed,control][5]
            pred=predictions[seed,'multi_side'][5];a,b=base.argmax(1),pred.argmax(1)
            fixed=int(np.sum((a!=labels)&(b==labels)));regressed=int(np.sum((a==labels)&(b!=labels)))
            gain=100*(fixed-regressed)/len(labels);close(gain,100*(metrics(pred,labels)['accuracy']-metrics(base,labels)['accuracy']))
            pairs.append(dict(seed=seed,comparison='multi_side_minus_'+control,corrected=fixed,regressed=regressed,
                net_correct=fixed-regressed,delta_accuracy_pp=gain,changed_predictions=int(np.sum(a!=b)),
                loss_difference=metrics(pred,labels)['loss']-metrics(base,labels)['loss']))
    save('independent_curves.csv',curves);save('paired_prediction_changes.csv',pairs)
    save('independent_decisions.csv',decisions);save('ordinary_control_comparison.csv',ordinary)
    device=torch.device('cuda');assert torch.cuda.is_available();dump('rescore_preflight.json',r.v.idle_preflight())
    x,y,ev,ey,split=r.v.load_data('development',device);del x,y
    assert split==cfg['split'] and np.array_equal(ey.cpu().numpy(),labels)
    freeze=json.loads(r.FREEZE.read_text());rescored=[]
    for seed in spec['seeds']:
        for arm in spec['modes']:
            for epoch in (0,spec['epochs']):
                core,source_cp=r.source(seed,False,freeze)
                model,opt=r.make_condition(core,source_cp,seed,arm,spec,device);del core,source_cp,opt
                if epoch:
                    path=attempts[seed,arm]/'final_checkpoint.pt';cp=torch.load(path,map_location='cpu',weights_only=False)
                    model.load_state_dict(cp['model_state'],strict=True);del cp
                    if arm=='lp_full':model.body.requires_grad_(True)
                else:path=Path(freeze['checkpoints'][str(seed)]['path'])
                before=r.old.state_hash(model.state_dict());z=r.t.predictions(model,ev,spec['eval_batch_size'])
                error=float(np.max(np.abs(z-predictions[seed,arm][epoch])))
                assert error<=spec['source_logit_tolerance'] and r.old.state_hash(model.state_dict())==before
                calculated=metrics(z,labels);expected=metrics(predictions[seed,arm][epoch],labels)
                close(calculated['accuracy'],expected['accuracy']);assert abs(calculated['loss']-expected['loss'])<1e-6
                assert np.array_equal(z.argmax(1),predictions[seed,arm][epoch].argmax(1))
                rescored.append(dict(seed=seed,arm=arm,epoch=epoch,**calculated,max_abs_logit_error=error,
                    checkpoint=str(path),sha256=sha(path),weights_unchanged=True))
                print('RESCORED',seed,arm,epoch,'max_error',error,flush=True);del model
    save('saved_state_rescore.csv',rescored)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    colors=dict(full='#1769aa',head='#777777',lp_full='#278246',final_side='#9865ad',multi_side='#c46a13')
    for ax,seed in zip(axes,spec['seeds']):
        for arm in spec['modes']:
            rows=[row for row in curves if row['seed']==seed and row['arm']==arm]
            ax.plot([row['epoch'] for row in rows],[100*row['accuracy'] for row in rows],color=colors[arm],
                marker='o',markersize=3,label=arm.replace('_',' '))
        anchor=next(row['best_control_accuracy'] for row in decisions if row['seed']==seed)
        ax.axhline(100*anchor-spec['accuracy_margin_pp'],color='black',linestyle=':',linewidth=1,label='Final quality floor')
        ax.set_title(f'Seed {seed}');ax.set_xticks(range(6));ax.set_xlabel('Adaptation epoch');ax.grid(alpha=.2)
    axes[0].set_ylabel('Development accuracy (%)');axes[1].legend(fontsize=8,loc='upper left')
    fig.suptitle('Stage15A: lower adaptation cost, insufficient final quality')
    fig.text(.5,.01,'Only epoch 5 decides. Reused development states/data; no independent confirmation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'side_adaptation_trajectories.png',dpi=180)
    fig.savefig(OUT/'side_adaptation_trajectories.svg');plt.close(fig)
    for p,h in protected.items():assert sha(p)==h,p
    result=dict(status='PASS',training_updates_verified=8800,discarded_warmup_updates=30,
        independent_curve_points=len(curves),saved_model_scoring_conditions=len(rescored),
        max_abs_rescored_logit_error=max(row['max_abs_logit_error'] for row in rescored),
        both_seed_reference_qualified=all(row['reference_qualified'] for row in decisions),
        both_seed_practical_pass=all(row['practical_seed_pass'] for row in decisions),
        both_seed_quality_pass=all(row['quality_pass'] for row in decisions),both_seed_time_pass=all(row['time_pass'] for row in decisions),
        historical_full_trajectory_reproduced=frozen['historical_full_trajectory_reproduced'],retained_arms_share_one_launch=single_launch,
        decision=frozen['decision'],prior_and_raw_files_preserved=len(protected),parameter_updates_performed=False,
        test_scoring_performed=False,development_scoring_performed=True)
    dump('independent_audit.json',result);print(json.dumps(result,indent=2));print(json.dumps(decisions,indent=2));print(json.dumps(ordinary,indent=2))

if __name__=='__main__':main()
