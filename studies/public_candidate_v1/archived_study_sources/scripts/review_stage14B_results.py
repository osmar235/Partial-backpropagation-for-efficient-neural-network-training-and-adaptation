"""Independent post-run review: arithmetic and saved-state inference, no training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/stage14B_token_recovery_v1'
OUT=ROOT/'results/stage14B_review_v1'
MUTABLE={str(ROOT/name) for name in ('CURRENT_RELEASE.md','README_STAGE9.md')}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def save(name,rows):
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def metrics(z,labels):
    zz=z.astype(np.float64);zz-=zz.max(1,keepdims=True)
    return float(np.mean(np.log(np.exp(zz).sum(1))-zz[np.arange(len(labels)),labels])),float(np.mean(z.argmax(1)==labels))


def close(x,y):assert math.isclose(float(x),float(y),rel_tol=1e-9,abs_tol=1e-9),(x,y)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'review_artifacts_sha256.json').exists()
    prep=ROOT/'protocol/stage14b'
    protected=json.loads((prep/'preparation_preservation_v1.json').read_text())['prior_immutable_sha256']
    seal=prep/'preparation_sha256_v1.json';entries=json.loads(seal.read_text())
    for path,h in {**protected,**entries}.items():assert sha(path)==h,path
    protected.update({path:h for path,h in entries.items() if path not in MUTABLE});protected[str(seal)]=sha(seal)
    protected.update({str(path):sha(path) for path in RAW.rglob('*') if path.is_file()})
    save('preserved_inputs_sha256.csv',[dict(path=path,sha256=h) for path,h in sorted(protected.items())])
    dump('mutable_pointers_before.json',{path:sha(path) for path in sorted(MUTABLE)})
    import torch
    import stage14_recovery as r
    from review_stage14B import audit
    r.m.configure();cfg=json.loads((RAW/'configuration.json').read_text())
    assert not cfg['synthetic'];assert r.d.validate_complete(RAW,cfg['fingerprint'])
    checked=audit(RAW,False);dump('frozen_audit_recheck.json',checked)
    assert checked['training_updates']==5280 and checked['both_seed_reference_qualified']
    assert not any(checked['both_seed_recovery_screen'].values())
    freeze=json.loads(r.FREEZE.read_text());spec=cfg['spec']
    summaries={(int(row['seed']),row['mode']):row for row in read(RAW/'summary.csv')}
    screens={(int(row['seed']),row['mode']):row for row in read(RAW/'screen.csv')}
    curves=[];derived=[];attempts={};predictions={};labels=None
    for seed in spec['seeds']:
        for mode in spec['modes']:
            key=(seed,mode);parent=RAW/f'seed_{seed}'/mode
            attempts[key]=attempt=Path(json.loads((parent/'COMPLETE.json').read_text())['attempt'])
            with np.load(attempt/'predictions.npz',allow_pickle=False) as z:
                if labels is None:labels=z['labels'].copy()
                assert np.array_equal(labels,z['labels'])
                predictions[key]={epoch:z[f'epoch_{epoch}'].copy() for epoch in range(spec['epochs']+1)}
            rawcurves=read(attempt/'curves.csv')
            for epoch,z in predictions[key].items():
                loss,acc=metrics(z,labels);close(loss,rawcurves[epoch]['loss']);close(acc,rawcurves[epoch]['accuracy'])
                curves.append(dict(seed=seed,mode=mode,epoch=epoch,accuracy=acc,loss_float64=loss))
            steps=read(attempt/'steps.csv');restoration=json.loads((attempt/'restoration.json').read_text())
            train=sum(float(row['transform_seconds'])+float(row['update_seconds']) for row in steps)
            charge=train+restoration['setup_seconds']+restoration['warmup_restore_seconds']
            src=freeze['checkpoints'][str(seed)]
            close(charge,summaries[key]['charged_recovery_seconds']);close(train,summaries[key]['training_seconds'])
            close(src['source_training_seconds']+charge,summaries[key]['source_training_plus_recovery_seconds'])
            close(src['source_total_run_seconds']+float(summaries[key]['observed_arm_seconds']),summaries[key]['source_run_plus_observed_recovery_seconds'])
            assert int(summaries[key]['max_training_incremental_allocated'])==max(int(row['incremental_allocated']) for row in steps)
        full=summaries[seed,'full'];initial_full=float(full['initial_accuracy']);final_full=float(full['final_accuracy'])
        best=max(initial_full,final_full,float(summaries[seed,'spatial']['final_accuracy']))
        for mode in spec['modes']:
            row=summaries[seed,mode];initial=float(row['initial_accuracy']);final=float(row['final_accuracy'])
            time_ratio=float(row['charged_recovery_seconds'])/float(full['charged_recovery_seconds'])
            mem_ratio=float(row['max_training_incremental_allocated'])/float(full['max_training_incremental_allocated'])
            quality=100*(best-final)<=spec['accuracy_margin_pp']+1e-10
            resource=time_ratio<=1-spec['minimum_time_saving']+1e-12 or mem_ratio<=1-spec['minimum_incremental_memory_saving']+1e-12
            if mode!='full':
                screen=screens[seed,mode];close(time_ratio,screen['charged_time_ratio']);close(mem_ratio,screen['incremental_memory_ratio'])
                assert str(quality)==screen['quality_pass'] and str(quality and resource)==screen['seed_screen_pass']
            f=predictions[seed,'full'][5].argmax(1);c=predictions[seed,mode][5].argmax(1)
            full_correct_candidate_wrong=int(np.sum((f==labels)&(c!=labels)))
            candidate_correct_full_wrong=int(np.sum((f!=labels)&(c==labels)))
            close(100*(candidate_correct_full_wrong-full_correct_candidate_wrong)/len(labels),100*(final-final_full))
            derived.append(dict(seed=seed,mode=mode,initial_accuracy=initial,final_accuracy=final,
                improvement_from_own_start_pp=100*(final-initial),delta_vs_original_full_pp=100*(final-initial_full),
                delta_vs_adapted_full_pp=100*(final-final_full),
                gap_change_from_stage14A_pp=100*((final-final_full)-(initial-initial_full)),
                quality_pass=quality,seed_screen_pass=quality and resource if mode!='full' else False,
                charged_recovery_seconds=float(row['charged_recovery_seconds']),
                charged_recovery_time_saving_percent=100*(1-time_ratio),
                incremental_training_memory_saving_percent=100*(1-mem_ratio),
                absolute_training_peak_saving_percent=100*(1-float(row['max_training_peak_allocated'])/float(full['max_training_peak_allocated'])),
                source_training_plus_recovery_seconds=float(row['source_training_plus_recovery_seconds']),
                source_inclusive_training_saving_percent=100*(1-float(row['source_training_plus_recovery_seconds'])/float(full['source_training_plus_recovery_seconds'])),
                source_run_plus_observed_recovery_seconds=float(row['source_run_plus_observed_recovery_seconds']),
                source_inclusive_observed_saving_percent=100*(1-float(row['source_run_plus_observed_recovery_seconds'])/float(full['source_run_plus_observed_recovery_seconds'])),
                inference_time_saving_percent=100*(1-float(row['mean_inference_seconds'])/float(full['mean_inference_seconds'])),
                inference_incremental_memory_saving_percent=100*(1-float(row['mean_inference_incremental_allocated'])/float(full['mean_inference_incremental_allocated'])),
                changed_predictions=int(np.sum(f!=c)),full_correct_candidate_wrong=full_correct_candidate_wrong,
                candidate_correct_full_wrong=candidate_correct_full_wrong))
    save('independent_curves.csv',curves);save('independent_summary.csv',derived)
    # Read-only rescore of all source-mode and final-model conditions on CUDA.
    device=torch.device('cuda');assert torch.cuda.is_available()
    x,y,ev,ey,split=r.v.load_data('development',device);del x,y
    assert split==cfg['split'] and np.array_equal(ey.cpu().numpy(),labels)
    rescored=[]
    for seed in spec['seeds']:
        for mode in spec['modes']:
            for epoch in (0,5):
                if epoch==0:
                    path=Path(freeze['checkpoints'][str(seed)]['path'])
                    core=r.t.make_core(seed,False,freeze,device)
                else:
                    path=attempts[seed,mode]/'final_checkpoint.pt'
                    cp=torch.load(path,map_location='cpu',weights_only=False)
                    core=r.m.make_model('full_rgb32',seed,False)
                    core.load_state_dict(cp['model_state'],strict=True);core.to(device);del cp
                before=r.old.state_hash(core.state_dict())
                model=r.t.TokenModel(core,mode,spec['after_blocks'])
                z=r.t.predictions(model,ev,spec['eval_batch_size'])
                error=float(np.max(np.abs(z-predictions[seed,mode][epoch])));assert error<=1e-5
                assert r.old.state_hash(core.state_dict())==before
                loss,acc=metrics(z,labels);expected_loss,expected_acc=metrics(predictions[seed,mode][epoch],labels)
                close(acc,expected_acc);assert abs(loss-expected_loss)<1e-6
                rescored.append(dict(seed=seed,mode=mode,epoch=epoch,accuracy=acc,loss_float64=loss,
                    max_abs_logit_error=error,checkpoint=str(path),sha256=sha(path)))
                del core,model
    save('saved_state_rescore.csv',rescored)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    colors=dict(full='#1769aa',spatial='#888888',similarity='#c46a13')
    for ax,seed in zip(axes,spec['seeds']):
        for mode in spec['modes']:
            rows=[row for row in curves if row['seed']==seed and row['mode']==mode]
            ax.plot([row['epoch'] for row in rows],[100*row['accuracy'] for row in rows],'o-',label=mode,color=colors[mode],markersize=4)
        ax.set_title(f'Seed {seed}');ax.set_xticks(range(6));ax.set_xlabel('Recovery epoch');ax.grid(alpha=.2)
    axes[0].set_ylabel('Development accuracy (%)');axes[1].legend(fontsize=9)
    fig.suptitle('Stage14B: adaptation improves compressed and full models')
    fig.text(.5,.01,'Only epoch 5 decides; similarity fails the two-seed quality rule. Reused development data.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'recovery_trajectories.png',dpi=180);fig.savefig(OUT/'recovery_trajectories.svg');plt.close(fig)
    for path,h in protected.items():assert sha(path)==h,path
    result=dict(status='PASS',training_updates_verified=5280,discarded_warmup_updates=18,
        independent_curve_points=36,saved_model_scoring_conditions=12,
        max_abs_rescored_logit_error=max(row['max_abs_logit_error'] for row in rescored),
        both_seed_reference_qualified=True,both_seed_spatial_pass=False,both_seed_similarity_pass=False,
        decision='park_fixed_recipe',prior_and_raw_files_preserved=len(protected),
        parameter_updates_performed=False,test_scoring_performed=False,development_scoring_performed=True)
    dump('independent_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
