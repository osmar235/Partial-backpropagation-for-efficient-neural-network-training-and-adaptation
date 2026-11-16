"""Independent combined review; saved-state scoring only, no optimizer updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from review_stage13B_results import metrics

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'results/stage13C_local_trajectory_v1'
T=ROOT/'results/stage14A_token_diagnostic_v1'
OUT=ROOT/'results/stage13C_stage14A_review_v1'
MUTABLE={str(ROOT/n) for n in ('CURRENT_RELEASE.md','README_STAGE9.md')}


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    with Path(path).open(encoding='utf8',newline='') as f: return list(csv.DictReader(f))


def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def save(name,rows):
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def close(x,y): assert math.isclose(float(x),float(y),rel_tol=1e-8,abs_tol=1e-8),(x,y)


def main():
    OUT.mkdir(exist_ok=True); assert not (OUT/'review_artifacts_sha256.json').exists()
    protected=json.loads((ROOT/'protocol/stage13c/preparation_preservation_v1.json').read_text())['prior_immutable_sha256']
    seal=ROOT/'protocol/stage13c/preparation_sha256_v1.json'; entries=json.loads(seal.read_text())
    for path,h in {**protected,**entries}.items(): assert sha(path)==h,path
    protected.update({path:h for path,h in entries.items() if path not in MUTABLE}); protected[str(seal)]=sha(seal)
    for folder in (C,T): protected.update({str(path):sha(path) for path in folder.rglob('*') if path.is_file()})
    save('preserved_inputs_sha256.csv',[dict(path=path,sha256=h) for path,h in sorted(protected.items())])
    dump('mutable_pointers_before.json',{path:sha(path) for path in sorted(MUTABLE)})
    import torch
    import stage13c_trajectory as c
    import stage14_diagnostic as t
    from review_stage13C import audit as audit_c
    from review_stage14A import audit as audit_t
    c.p.execution.configure()
    cfgc=json.loads((C/'configuration.json').read_text()); cfgt=json.loads((T/'configuration.json').read_text())
    assert not cfgc['synthetic'] and not cfgt['synthetic']
    c.d.validate_complete(C,cfgc['fingerprint']); t.d.validate_complete(T,cfgt['fingerprint'])
    ac=audit_c(C,False); at=audit_t(T,False)
    dump('stage13C_frozen_audit_recheck.json',ac); dump('stage14A_frozen_audit_recheck.json',at)
    assert ac['both_seed_reference_qualified'] and not ac['both_seed_local_screen_pass']
    assert not any(at['both_seed_review_flags'].values())
    pc=Path(json.loads((C/'RAW_COMPLETE.json').read_text())['attempt'])
    pt=Path(json.loads((T/'RAW_COMPLETE.json').read_text())['attempt'])
    fc=json.loads(c.FREEZE.read_text()); ft=json.loads(t.FREEZE.read_text())
    cc={(int(r['seed']),r['arm'],int(r['step'])):r for r in read(C/'curves.csv')}
    cs={(int(r['seed']),r['arm']):r for r in read(C/'summary.csv')}
    ts={(int(r['seed']),r['mode']):r for r in read(T/'summary.csv')}
    curves,local_summary,token_summary=[],[],[]
    for seed in cfgc['spec']['seeds']:
        states={}
        for arm in cfgc['spec']['arms']:
            with np.load(pc/f'scores_{seed}_{arm}.npz',allow_pickle=False) as z:
                for step in cfgc['spec']['score_steps']:
                    loss,accuracy=metrics(z[f'step_{step}'],z['labels'])
                    assert abs(loss-float(cc[seed,arm,step]['loss']))<1e-6
                    close(accuracy,cc[seed,arm,step]['accuracy'])
                    curves.append(dict(seed=seed,arm=arm,step=step,loss_float64=loss,accuracy=accuracy))
                initial_loss,initial_acc=metrics(z['step_0'],z['labels'])
                loss,accuracy=metrics(z['step_352'],z['labels'])
            rows=read(pc/f'steps_{seed}_{arm}.csv')
            calibration=json.loads((pc/f'calibration_{seed}.json').read_text())['seconds'] if arm in ('local','full_aux') else 0
            training=sum(float(r['training_seconds']) for r in rows); charged=training+calibration
            memory=max(int(r['incremental_allocated']) for r in rows)
            close(charged,cs[seed,arm]['charged_training_seconds']); close(memory,cs[seed,arm]['max_incremental_allocated'])
            states[arm]=dict(seed=seed,arm=arm,initial_accuracy=initial_acc,accuracy=accuracy,initial_loss=initial_loss,loss_float64=loss,
                training_seconds=training,calibration_seconds=calibration,charged_training_seconds=charged,max_incremental_allocated=memory)
        full=states['full']; local=states['local']
        best=max(states[arm]['accuracy'] for arm in ('full','full_aux','suffix'))
        qualified=[arm for arm in ('full','full_aux','suffix') if 100*(best-states[arm]['accuracy'])<=cfgc['spec']['accuracy_margin_pp']+1e-10]
        assert qualified==['full']
        for arm,row in states.items():
            row.update(delta_vs_full_pp=100*(row['accuracy']-full['accuracy']),delta_vs_start_pp=100*(row['accuracy']-row['initial_accuracy']),
                charged_time_change_percent=100*(row['charged_training_seconds']/full['charged_training_seconds']-1),
                incremental_memory_saving_percent=100*(1-row['max_incremental_allocated']/full['max_incremental_allocated']))
            local_summary.append(row)
        assert local['accuracy']<local['initial_accuracy'] and local['loss_float64']>local['initial_loss']
        assert local['charged_training_seconds']>full['charged_training_seconds']
        assert local['incremental_memory_saving_percent']<100*cfgc['spec']['minimum_memory_saving']
    save('stage13C_independent_curves.csv',curves); save('stage13C_independent_summary.csv',local_summary)
    for seed in cfgt['spec']['seeds']:
        rows=read(pt/f'measurements_{seed}.csv'); means={}
        for kind in cfgt['spec']['operations']:
            for mode in cfgt['spec']['modes']:
                selected=[r for r in rows if r['operation']==kind and r['mode']==mode]
                assert len(selected)==12
                means[kind,mode]={key:float(np.mean([float(r[key]) for r in selected])) for key in ('seconds','incremental_allocated','peak_allocated')}
        with np.load(pt/f'predictions_{seed}.npz',allow_pickle=False) as z:
            full_loss,full_acc=metrics(z['full'],z['labels']); full_correct=z['full'].argmax(1)==z['labels']
            for mode in cfgt['spec']['modes']:
                loss,accuracy=metrics(z[mode],z['labels']); assert abs(loss-float(ts[seed,mode]['loss']))<1e-6
                close(accuracy,ts[seed,mode]['accuracy'])
                tr=means['forward_backward',mode]['seconds']/means['forward_backward','full']['seconds']
                mr=means['forward_backward',mode]['incremental_allocated']/means['forward_backward','full']['incremental_allocated']
                close(tr,ts[seed,mode]['forward_backward_time_ratio']); close(mr,ts[seed,mode]['forward_backward_incremental_memory_ratio'])
                quality=100*(full_acc-accuracy)<=cfgt['spec']['accuracy_margin_pp']+1e-10
                resource=tr<=1-cfgt['spec']['minimum_time_saving'] or mr<=1-cfgt['spec']['minimum_incremental_memory_saving']
                assert str(quality)==ts[seed,mode]['quality_pass'] and str(resource)==ts[seed,mode]['resource_pass']
                correct=z[mode].argmax(1)==z['labels']
                token_summary.append(dict(seed=seed,mode=mode,accuracy=accuracy,loss_float64=loss,delta_vs_full_pp=100*(accuracy-full_acc),
                    changed_predictions=int(np.sum(z[mode].argmax(1)!=z['full'].argmax(1))),
                    full_correct_candidate_wrong=int(np.sum(full_correct & ~correct)),candidate_correct_full_wrong=int(np.sum(correct & ~full_correct)),
                    forward_backward_ms=1000*means['forward_backward',mode]['seconds'],forward_backward_time_saving_percent=100*(1-tr),
                    incremental_memory_saving_percent=100*(1-mr),absolute_peak_saving_percent=100*(1-means['forward_backward',mode]['peak_allocated']/means['forward_backward','full']['peak_allocated']),
                    inference_time_saving_percent=100*(1-means['inference',mode]['seconds']/means['inference','full']['seconds']),
                    quality_pass=quality,resource_pass=resource,seed_pass=quality and resource))
    save('stage14A_independent_summary.csv',token_summary)

    device=torch.device('cuda'); assert torch.cuda.is_available()
    x,y,ev,ey,split=c.p.load_data(device,False); del x,y
    assert split==cfgc['split']; rescored=[]
    for seed in cfgc['spec']['seeds']:
        for arm in ['source']+cfgc['spec']['arms']:
            path=Path(fc['checkpoints'][str(seed)]['path']) if arm=='source' else pc/f'final_{seed}_{arm}.pt'
            cp=torch.load(path,map_location='cpu',weights_only=True)
            state=cp['model_state'] if arm=='source' else {n[4:]:value for n,value in cp['model_state'].items() if n.startswith('net.')}
            model=c.p.make_model(seed,False).to(device); model.load_state_dict(state,strict=True)
            before=c.old.state_hash(model.state_dict()); logits=c.b.predict(model,ev,cfgc['recipe'])
            loss,accuracy=metrics(logits,ey.cpu().numpy()); target_arm='full' if arm=='source' else arm; step=0 if arm=='source' else 352
            with np.load(pc/f'scores_{seed}_{target_arm}.npz',allow_pickle=False) as z:
                assert np.array_equal(z['labels'],ey.cpu().numpy()); error=float(np.max(np.abs(logits-z[f'step_{step}'])))
                assert error<=1e-5
            assert c.old.state_hash(model.state_dict())==before
            close(accuracy,cc[seed,target_arm,step]['accuracy']); assert abs(loss-float(cc[seed,target_arm,step]['loss']))<1e-6
            rescored.append(dict(stage='13C',seed=seed,arm=arm,accuracy=accuracy,loss_float64=loss,max_abs_logit_error=error,checkpoint=str(path),sha256=sha(path)))
            del model,cp,state
    del ev,ey
    x,y,ev,ey,split=t.v.load_data('development',device); del x,y
    assert split==cfgt['split']
    for seed in cfgt['spec']['seeds']:
        core=t.make_core(seed,False,ft,device); before=t.old.state_hash(core.state_dict())
        model=t.TokenModel(core,'full',cfgt['spec']['after_blocks'])
        for mode in cfgt['spec']['modes']:
            model.mode=mode; logits=t.predictions(model,ev,cfgt['spec']['eval_batch_size'])
            loss,accuracy=metrics(logits,ey.cpu().numpy())
            with np.load(pt/f'predictions_{seed}.npz',allow_pickle=False) as z:
                assert np.array_equal(z['labels'],ey.cpu().numpy()); error=float(np.max(np.abs(logits-z[mode])))
                assert error<=1e-5
            close(accuracy,ts[seed,mode]['accuracy']); assert abs(loss-float(ts[seed,mode]['loss']))<1e-6
            rescored.append(dict(stage='14A',seed=seed,arm=mode,accuracy=accuracy,loss_float64=loss,max_abs_logit_error=error,
                checkpoint=ft['checkpoints'][str(seed)]['path'],sha256=ft['checkpoints'][str(seed)]['sha256']))
        assert t.old.state_hash(core.state_dict())==before; del model,core
    del ev,ey
    save('saved_state_rescore.csv',rescored)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    colors=dict(full='#1769aa',full_aux='#6a4c93',local='#c43c39',suffix='#24865a')
    for ax,seed in zip(axes,cfgc['spec']['seeds']):
        for arm in cfgc['spec']['arms']:
            rows=[row for row in curves if row['seed']==seed and row['arm']==arm]
            ax.plot([row['step'] for row in rows],[100*row['accuracy'] for row in rows],'o-',label=arm,color=colors[arm],markersize=4)
        initial=next(row['accuracy'] for row in curves if row['seed']==seed and row['step']==0)
        ax.axhline(100*initial,linestyle=':',color='gray'); ax.set_xscale('symlog',linthresh=1)
        ax.set_xticks(cfgc['spec']['score_steps']); ax.set_xticklabels(cfgc['spec']['score_steps'])
        ax.set_xlim(-.12,440); ax.tick_params(axis='x',labelrotation=30)
        ax.set_title(f'Seed {seed}'); ax.grid(alpha=.2); ax.set_xlabel('Updates (nonlinear spacing)')
    axes[0].set_ylabel('Development accuracy (%)'); axes[1].legend(fontsize=8)
    fig.suptitle('Stage13C: local objectives fail the fixed one-epoch comparison')
    fig.tight_layout(); fig.savefig(OUT/'local_trajectory.png',dpi=180); fig.savefig(OUT/'local_trajectory.svg'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for seed,color in [(1501,'#1769aa'),(1502,'#d97813')]:
        for row in [z for z in token_summary if z['seed']==seed]:
            ax.scatter(row['forward_backward_time_saving_percent'],-row['delta_vs_full_pp'],
                color=color,marker=dict(full='o',spatial='s',similarity='^')[row['mode']],s=80)
            if row['mode']!='full':
                ax.annotate(f"{row['mode']} / {seed}",(row['forward_backward_time_saving_percent'],-row['delta_vs_full_pp']),
                    xytext=(-9 if seed==1501 else 7,8 if seed==1501 else -14),textcoords='offset points',
                    ha='right' if seed==1501 else 'left',fontsize=9,color=color)
    ax.axhline(.3,color='#555555',linestyle='--',label='Frozen accuracy-loss limit: 0.30 pp')
    ax.axvline(10,color='#aaaaaa',linestyle=':',label='Time-saving threshold: 10%')
    ax.set_xlim(-2,34); ax.set_ylim(-.1,2); ax.set_xlabel('Forward/backward time saved (%)'); ax.set_ylabel('Accuracy loss versus full (percentage points)')
    ax.set_title('Stage14A: similarity merging preserves more accuracy than spatial averaging')
    ax.legend(loc='upper left',fontsize=8); ax.grid(alpha=.15)
    fig.text(.5,.01,'No optimizer updates; previously used development data. Similarity fails the two-seed quality rule.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(OUT/'token_tradeoff.png',dpi=180); fig.savefig(OUT/'token_tradeoff.svg'); plt.close(fig)
    for path,h in protected.items(): assert sha(path)==h,path
    result=dict(status='PASS',stage13C_updates_verified=2816,stage13C_calibration_updates=256,stage14A_operations=144,
        stage13C_reference_qualified=True,stage13C_local_screen_pass=False,stage14A_both_seed_screen_pass=False,
        source_final_or_mode_scores_rescored=16,max_abs_rescored_logit_error=max(z['max_abs_logit_error'] for z in rescored),
        original_files_preserved=len(protected),parameter_updates_performed=False,test_scoring_performed=False,development_scoring_performed=True)
    dump('independent_audit.json',result); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
