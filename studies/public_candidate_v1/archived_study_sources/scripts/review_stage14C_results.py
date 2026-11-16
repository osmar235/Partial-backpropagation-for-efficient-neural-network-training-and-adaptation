"""Read-only post-run audit, independent endpoint arithmetic and checkpoint rescoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/stage14C_optimizer_history_v1'
OUT=ROOT/'results/stage14C_review_v1'
MUTABLE={str(ROOT/name) for name in ('CURRENT_RELEASE.md','README_STAGE9.md')}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def save(name,rows):
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)


def metrics(z,y):
    a=z.astype(np.float64);a-=a.max(1,keepdims=True)
    return dict(accuracy=float(np.mean(z.argmax(1)==y)),loss=float(np.mean(np.log(np.exp(a).sum(1))-a[np.arange(len(y)),y])))


def main():
    assert not OUT.exists();OUT.mkdir()
    prep=ROOT/'protocol/stage14c'
    protected=json.loads((prep/'preparation_preservation_v1.json').read_text())['prior_immutable_sha256']
    seal=prep/'preparation_sha256_v1.json';entries=json.loads(seal.read_text())
    for path,h in {**protected,**entries}.items():assert sha(path)==h,path
    protected.update({path:h for path,h in entries.items() if path not in MUTABLE});protected[str(seal)]=sha(seal)
    protected.update({str(p):sha(p) for p in RAW.rglob('*') if p.is_file()})
    dump('preserved_inputs_sha256.json',protected)
    dump('mutable_pointers_before.json',{path:sha(path) for path in sorted(MUTABLE)})
    import torch
    import stage14_optimizer_history as r
    from review_stage14C import audit
    r.m.configure();cfg=json.loads((RAW/'configuration.json').read_text());spec=cfg['spec']
    assert not cfg['synthetic'] and r.d.validate_complete(RAW,cfg['fingerprint'])
    frozen=audit(RAW,False);dump('frozen_audit_recheck.json',frozen)
    assert frozen['status']=='PASS' and frozen['training_updates']==10560
    summaries={(int(row['seed']),row['mode']):row for row in read(RAW/'summary.csv')}
    screens={int(row['seed']):row for row in read(RAW/'screen.csv')}
    attempts={};predictions={};curves=[];paired=[];decisions=[];labels=None
    for seed in spec['seeds']:
        for arm in spec['modes']:
            parent=RAW/f'seed_{seed}'/arm
            attempt=Path(json.loads((parent/'COMPLETE.json').read_text())['attempt']);attempts[seed,arm]=attempt
            with np.load(attempt/'predictions.npz',allow_pickle=False) as z:
                if labels is None:labels=z['labels'].copy()
                assert np.array_equal(labels,z['labels'])
                predictions[seed,arm]={epoch:z[f'epoch_{epoch}'].copy() for epoch in range(spec['epochs']+1)}
            saved_curves=read(attempt/'curves.csv')
            for epoch,z in predictions[seed,arm].items():
                row=metrics(z,labels)
                close(row['accuracy'],saved_curves[epoch]['accuracy']);close(row['loss'],saved_curves[epoch]['loss'])
                curves.append(dict(seed=seed,arm=arm,epoch=epoch,**row))
            summary=summaries[seed,arm]
            close(summary['final_accuracy'],metrics(predictions[seed,arm][5],labels)['accuracy'])
            close(summary['complete_adaptation_seconds'],float(summary['adaptation_seconds'])+float(summary['shared_data_seconds']))
        def accuracy(arm):return metrics(predictions[seed,arm][5],labels)['accuracy']
        def seconds(arm):return float(summaries[seed,arm]['complete_adaptation_seconds'])
        def loss(arm):return metrics(predictions[seed,arm][5],labels)['loss']
        gains={}
        for mode in ('full','spatial','similarity'):
            retained=predictions[seed,mode+'_restore'][5].argmax(1)
            fresh=predictions[seed,mode+'_fresh'][5].argmax(1)
            fixed=int(np.sum((retained!=labels)&(fresh==labels)))
            regressed=int(np.sum((retained==labels)&(fresh!=labels)))
            gain=100*(fixed-regressed)/len(labels);gains[mode]=gain
            close(gain,100*(accuracy(mode+'_fresh')-accuracy(mode+'_restore')))
            paired.append(dict(seed=seed,computation=mode,restore_accuracy=accuracy(mode+'_restore'),
                fresh_accuracy=accuracy(mode+'_fresh'),reset_gain_pp=gain,corrected=fixed,regressed=regressed,
                net_correct=fixed-regressed,changed_predictions=int(np.sum(retained!=fresh)),
                restore_loss=loss(mode+'_restore'),fresh_loss=loss(mode+'_fresh'),
                fresh_minus_restore_loss=loss(mode+'_fresh')-loss(mode+'_restore')))
        controls=['full_restore','full_fresh','spatial_restore','spatial_fresh']
        source_accuracy=metrics(predictions[seed,'full_restore'][0],labels)['accuracy']
        best=max([source_accuracy]+[accuracy(arm) for arm in controls])
        qualified=[arm for arm in controls if 100*(best-accuracy(arm))<=spec['accuracy_margin_pp']+1e-10]
        ready=accuracy('full_restore')>=.80 and 100*(source_accuracy-accuracy('full_restore'))<=.30+1e-10 and bool(qualified)
        comparison=min(set(qualified+['full_restore','full_fresh']),key=seconds)
        tr=seconds('similarity_fresh')/seconds(comparison)
        deficit=100*(best-accuracy('similarity_fresh'));quality=deficit<=.30+1e-10;time_pass=tr<=.90+1e-12
        interaction=gains['similarity']-gains['full']
        mechanism=gains['similarity']>1e-10 and interaction>1e-10
        raw=screens[seed]
        for key,val in [('interaction_pp',interaction),('similarity_reset_gain_pp',gains['similarity']),
                        ('full_reset_gain_pp',gains['full']),('spatial_interaction_pp',gains['spatial']-gains['full']),
                        ('final_delta_vs_best_pp',-deficit),('complete_time_ratio',tr)]:close(raw[key],val)
        for key,val in [('reference_qualified',ready),('quality_pass',quality),('time_pass',time_pass),
                        ('mechanism_seed_support',mechanism),('seed_screen_pass',ready and quality and time_pass)]:assert raw[key]==str(val)
        candidate=summaries[seed,'similarity_fresh'];full=summaries[seed,comparison]
        decisions.append(dict(seed=seed,source_accuracy=source_accuracy,best_control_accuracy=best,
            candidate_accuracy=accuracy('similarity_fresh'),deficit_vs_best_pp=deficit,
            quality_pass=quality,time_pass=time_pass,practical_seed_pass=ready and quality and time_pass,
            reference_qualified=ready,qualifying_controls=';'.join(qualified),time_comparator=comparison,
            candidate_adaptation_seconds=seconds('similarity_fresh'),comparator_adaptation_seconds=seconds(comparison),
            adaptation_saving_percent=100*(1-tr),seconds_saved=seconds(comparison)-seconds('similarity_fresh'),
            source_training_inclusive_saving_percent=100*(1-float(candidate['source_training_plus_recovery_seconds'])/float(full['source_training_plus_recovery_seconds'])),
            incremental_memory_saving_percent=100*(1-float(candidate['max_training_incremental_allocated'])/float(full['max_training_incremental_allocated'])),
            absolute_training_peak_saving_percent=100*(1-float(candidate['max_training_peak_allocated'])/float(full['max_training_peak_allocated'])),
            similarity_reset_gain_pp=gains['similarity'],full_reset_gain_pp=gains['full'],interaction_pp=interaction,
            spatial_interaction_pp=gains['spatial']-gains['full'],mechanism_seed_support=mechanism))
    save('independent_curves.csv',curves);save('paired_reset_effects.csv',paired);save('independent_decisions.csv',decisions)
    # Source logits have two identical optimizer conditions: rescore six unique
    # source computations and all twelve distinct final models, with no updates.
    device=torch.device('cuda');assert torch.cuda.is_available()
    dump('rescore_preflight.json',r.v.idle_preflight())
    x,y,ev,ey,split=r.v.load_data('development',device);del x,y
    assert split==cfg['split'] and np.array_equal(ey.cpu().numpy(),labels)
    freeze=json.loads(r.FREEZE.read_text());rescored=[]
    for seed in spec['seeds']:
        conditions=[(mode+'_restore',0) for mode in ('full','spatial','similarity')]+[(arm,5) for arm in spec['modes']]
        for arm,epoch in conditions:
            mode=arm.rsplit('_',1)[0]
            if epoch==0:
                path=Path(freeze['checkpoints'][str(seed)]['path']);core=r.t.make_core(seed,False,freeze,device)
            else:
                path=attempts[seed,arm]/'final_checkpoint.pt'
                cp=torch.load(path,map_location='cpu',weights_only=False)
                core=r.m.make_model('full_rgb32',seed,False);core.load_state_dict(cp['model_state'],strict=True)
                core.to(device);del cp
            before=r.old.state_hash(core.state_dict());model=r.t.TokenModel(core,mode,spec['after_blocks'])
            z=r.t.predictions(model,ev,spec['eval_batch_size']);error=float(np.max(np.abs(z-predictions[seed,arm][epoch])))
            assert error<=1e-5 and r.old.state_hash(core.state_dict())==before
            calculated=metrics(z,labels);expected=metrics(predictions[seed,arm][epoch],labels)
            close(calculated['accuracy'],expected['accuracy']);assert abs(calculated['loss']-expected['loss'])<1e-6
            rescored.append(dict(seed=seed,arm=arm,epoch=epoch,**calculated,max_abs_logit_error=error,checkpoint=str(path),sha256=sha(path)))
            del core,model
    save('saved_state_rescore.csv',rescored)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    colors=dict(full='#1769aa',spatial='#888888',similarity='#c46a13')
    for ax,seed in zip(axes,spec['seeds']):
        for arm in spec['modes']:
            mode,history=arm.rsplit('_',1);rows=[row for row in curves if row['seed']==seed and row['arm']==arm]
            ax.plot([row['epoch'] for row in rows],[100*row['accuracy'] for row in rows],
                linestyle='-' if history=='fresh' else '--',marker='o' if history=='fresh' else None,
                markersize=3,color=colors[mode],label=arm.replace('_',' '))
        ax.set_title(f'Seed {seed}');ax.set_xticks(range(6));ax.set_xlabel('Adaptation epoch');ax.grid(alpha=.2)
    axes[0].set_ylabel('Development accuracy (%)');axes[1].legend(fontsize=8,loc='lower right')
    fig.suptitle('Stage14C: small reset gains, remaining quality deficit')
    fig.text(.5,.01,'Only epoch 5 decides. Reused development states/data; no independent confirmation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'optimizer_history_trajectories.png',dpi=180)
    fig.savefig(OUT/'optimizer_history_trajectories.svg');plt.close(fig)
    for path,h in protected.items():assert sha(path)==h,path
    result=dict(status='PASS',training_updates_verified=10560,discarded_warmup_updates=36,
        independent_curve_points=len(curves),saved_model_scoring_conditions=len(rescored),
        max_abs_rescored_logit_error=max(row['max_abs_logit_error'] for row in rescored),
        both_seed_reference_qualified=all(row['reference_qualified'] for row in decisions),
        both_seed_mechanism_direction=all(row['mechanism_seed_support'] for row in decisions),
        both_seed_practical_pass=all(row['practical_seed_pass'] for row in decisions),
        historical_restored_trajectory_reproduced=frozen['historical_restored_trajectory_reproduced'],
        retained_arms_share_one_launch=frozen['retained_arms_share_one_launch'],
        decision='mechanism_only_no_efficiency_claim',prior_and_raw_files_preserved=len(protected),
        parameter_updates_performed=False,test_scoring_performed=False,development_scoring_performed=True)
    assert result['both_seed_mechanism_direction'] and not result['both_seed_practical_pass']
    dump('independent_audit.json',result);print(json.dumps(result,indent=2))
    print(json.dumps(decisions,indent=2));print(json.dumps(paired,indent=2))


if __name__=='__main__':main()
