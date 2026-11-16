"""Independent post-run arithmetic and saved-model validation for Stage12A."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/stage12A_adaptation_v1';OUT=ROOT/'results/stage12A_review_v1'
MUTABLE={str(ROOT/n) for n in ['CURRENT_RELEASE.md','README_STAGE9.md']}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
def save(name,rows):
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10),(a,b)
def crossing(rows,threshold,length):
    first=None;streak=0
    for er in rows:
        if float(er['validation_accuracy'])>=threshold:first=first or er;streak+=1
        else:streak=0
        if streak==length:return first,er
    return first,None
def compare(actual,expected):
    for k,value in actual.items():
        if isinstance(value,(bool,str)):assert str(value)==expected[k],(k,value,expected[k])
        else:close(value,expected[k])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rescore-cuda',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True);protected={};baseline=OUT/'preserved_inputs_sha256.csv'
    def verify(path,digest):
        assert sha(path)==digest,path
        if str(path) not in MUTABLE:protected[str(path)]=digest
    prior=json.loads((ROOT/'protocol/stage12/preparation_preservation_v1.json').read_text())
    for path,h in prior['prior_immutable_sha256'].items():verify(path,h)
    manifest=ROOT/'protocol/stage12/preparation_sha256_v1.json'
    for path,h in json.loads(manifest.read_text()).items():
        if path not in MUTABLE or not baseline.exists():verify(path,h)
    verify(manifest,sha(manifest))
    for path in RAW.rglob('*'):
        if path.is_file():verify(path,sha(path))
    if baseline.exists():
        for row in read(baseline):verify(row['path'],row['sha256'])
    else:
        save('preserved_inputs_sha256.csv',[dict(path=path,sha256=h) for path,h in sorted(protected.items())])
        dump('mutable_pointers_before.json',{path:sha(path) for path in sorted(MUTABLE)})
    import torch
    import stage12_pretrained as p
    from review_stage12A import audit
    p.execution.configure();audited=audit(RAW,False);dump('frozen_audit_recheck.json',audited)
    cfg=json.loads((RAW/'configuration.json').read_text());spec=cfg['spec'];assert not cfg['synthetic']
    saved_targets={(int(row['seed']),row['arm'],row['role']):row for row in read(RAW/'targets.csv')}
    saved_screens={int(row['seed']):row for row in read(RAW/'screen.csv')}
    records={};paths={};targets=[];comparisons=[];screens=[];contrasts=[]
    for seed in spec['seeds']:
        ready={};target_times={}
        for arm in spec['arms']:
            marker=json.loads((RAW/f'seed_{seed}/{arm}/COMPLETE.json').read_text());attempt=Path(marker['attempt'])
            summary=json.loads((attempt/'summary.json').read_text());records[seed,arm]=summary
            paths[seed,arm]=attempt/'final_checkpoint.pt';epochs=read(attempt/'epochs.csv')
            for target_role in ['readiness','secondary']:
                first,last=crossing(epochs,spec[target_role+'_target'],spec['sustained_epochs'])
                row=dict(seed=seed,arm=arm,role=target_role,threshold=spec[target_role+'_target'],reached=last is not None,
                    first_crossing_epoch=int(first['epoch']) if first else '',confirming_epoch=int(last['epoch']) if last else '',
                    observed_seconds=float(last['observed_seconds']) if last else '',censoring_epoch='' if last else spec['epochs'])
                compare(row,saved_targets[seed,arm,target_role]);targets.append(row)
                target_times[arm,target_role]=row['observed_seconds']
                if target_role=='readiness':ready[arm]=last is not None
        controls=['full','head','lp_full','lp_suffix'];full=records[seed,'full'];candidate=records[seed,'lp_periodic']
        best=max(records[seed,c]['final_accuracy'] for c in controls)
        qualified=[c for c in controls if (best-records[seed,c]['final_accuracy'])*100<=spec['final_accuracy_margin_pp']+1e-12]
        fastest=min(qualified,key=lambda c:(records[seed,c]['total_run_seconds'],c))
        delta=100*(candidate['final_accuracy']-best)
        rf=candidate['total_run_seconds']/full['total_run_seconds'];rb=candidate['total_run_seconds']/records[seed,fastest]['total_run_seconds']
        quality=delta>=-spec['final_accuracy_margin_pp']-1e-12;speed=max(rf,rb)<=1-spec['minimum_time_gain']
        baseline_ready=ready['full'] or ready['lp_full']
        screen=dict(seed=seed,best_control_accuracy=best,qualified_controls=';'.join(qualified),fastest_qualified_control=fastest,
            delta_vs_best_control_pp=delta,delta_vs_fixed_suffix_pp=100*(candidate['final_accuracy']-records[seed,'lp_suffix']['final_accuracy']),
            time_ratio_to_fastest_qualified=rb,time_ratio_to_full=rf,baseline_ready=baseline_ready,candidate_ready=ready['lp_periodic'],
            quality_pass=quality,speed_pass=speed,opportunity_pass=quality and speed and baseline_ready and ready['lp_periodic'])
        compare(screen,saved_screens[seed]);screens.append(screen)
        for arm in spec['arms']:
            a=records[seed,arm]
            comparisons.append(dict(seed=seed,arm=arm,final_accuracy=a['final_accuracy'],
                delta_vs_full_pp=100*(a['final_accuracy']-full['final_accuracy']),training_seconds=a['training_seconds'],total_run_seconds=a['total_run_seconds'],
                total_time_saving_vs_full_percent=100*(1-a['total_run_seconds']/full['total_run_seconds']),
                prefix_updates=a['group_updates']['prefix'],suffix_updates=a['group_updates']['suffix'],head_updates=a['group_updates']['head'],
                sustained90_seconds=target_times[arm,'readiness'],sustained92_seconds=target_times[arm,'secondary']))
        lp=records[seed,'lp_full'];suffix=records[seed,'lp_suffix']
        contrasts.append(dict(seed=seed,lp_full_delta_vs_full_pp=100*(lp['final_accuracy']-full['final_accuracy']),
            lp_full_time_saving_vs_full_percent=100*(1-lp['total_run_seconds']/full['total_run_seconds']),
            periodic_delta_vs_lp_full_pp=100*(candidate['final_accuracy']-lp['final_accuracy']),
            periodic_time_saving_vs_lp_full_percent=100*(1-candidate['total_run_seconds']/lp['total_run_seconds']),
            periodic_delta_vs_suffix_pp=100*(candidate['final_accuracy']-suffix['final_accuracy']),
            periodic_extra_time_vs_suffix_percent=100*(candidate['total_run_seconds']/suffix['total_run_seconds']-1),
            periodic_sustained92_saving_vs_lp_full_percent=100*(1-target_times['lp_periodic','secondary']/target_times['lp_full','secondary'])
                if target_times['lp_periodic','secondary']!='' and target_times['lp_full','secondary']!='' else ''))
    means=[]
    for arm in spec['arms']:
        rr=[row for row in comparisons if row['arm']==arm]
        means.append(dict(arm=arm,n=len(rr),mean_final_accuracy=statistics.mean(row['final_accuracy'] for row in rr),
            mean_total_run_seconds=statistics.mean(row['total_run_seconds'] for row in rr),mean_delta_vs_full_pp=statistics.mean(row['delta_vs_full_pp'] for row in rr)))
    for name,rows in [('independent_targets.csv',targets),('independent_screen.csv',screens),('independent_seed_results.csv',comparisons),
                      ('independent_mean_results.csv',means),('descriptive_control_contrasts.csv',contrasts)]:save(name,rows)
    if args.rescore_cuda:
        assert torch.cuda.is_available();preflight=p.v.idle_preflight()
        x,y,ev,ey,split=p.load_data(torch.device('cuda'),False);assert split==cfg['split'];del x,y
        rescored=[]
        for (seed,arm),path in paths.items():
            cp=torch.load(path,map_location='cpu',weights_only=True);model=p.make_model(seed,False).to('cuda')
            model.load_state_dict(cp['model_state'],strict=True);loss,accuracy=p.evaluate(model,ev,ey,spec)
            expected=records[seed,arm];close(accuracy,expected['final_accuracy']);assert abs(loss-expected['final_loss'])<1e-6
            rescored.append(dict(seed=seed,arm=arm,checkpoint_sha256=sha(path),validation_accuracy=accuracy,validation_loss=loss,
                recorded_accuracy=expected['final_accuracy'],recorded_loss=expected['final_loss']))
            del model,cp
        save('checkpoint_validation_rescore.csv',rescored)
        dump('rescore_metadata.json',dict(status='PASS',device=torch.cuda.get_device_name(0),test_dataset_loaded=False,training_performed=False,
            preflight=preflight,execution=p.execution.execution_settings()))
    for path,h in protected.items():assert sha(path)==h,path
    result=dict(status='PASS',runs=10,updates=42240,preserved_immutable_files=len(protected),independent_targets_and_screens_match=True,
        final_checkpoints_rescored_this_invocation=args.rescore_cuda,baselines_ready_both_seeds=all(s['baseline_ready'] for s in screens),
        both_seed_speed_pass=all(s['speed_pass'] for s in screens),both_seed_quality_pass=all(s['quality_pass'] for s in screens),
        seeds_passing=sum(s['opportunity_pass'] for s in screens),
        decision='Periodic pretrained-prefix updates fail final-quality criterion in both seeds. Preserve useful tradeoffs and strong LP-FT control; no automatic tuning. Publication on hold.')
    dump('independent_audit.json',result);print(json.dumps(result,indent=2));print(json.dumps(means,indent=2));print(json.dumps(contrasts,indent=2))

if __name__=='__main__':main()
