"""Post-run Stage11B arithmetic and checkpoint rescore; never trains models."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/stage11B_growth_pilot_v1'
OUT=ROOT/'results/stage11B_review_v1'
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
    for r in rows:
        if float(r['validation_accuracy'])>=threshold:first=first or r;streak+=1
        else:streak=0
        if streak==length:return first,r
    return first,None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rescore-cuda',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True);protected={};baseline=OUT/'preserved_inputs_sha256.csv'
    def verify(p,h):
        assert sha(p)==h,p
        if str(p) not in MUTABLE:protected[str(p)]=h
    prior=json.loads((ROOT/'protocol/stage11/pilot_preparation_preservation_v1.json').read_text())
    for p,h in prior['prior_immutable_sha256'].items():verify(p,h)
    manifest=ROOT/'protocol/stage11/pilot_preparation_sha256_v1.json'
    for p,h in json.loads(manifest.read_text()).items():
        if p not in MUTABLE or not baseline.exists():verify(p,h)
    verify(manifest,sha(manifest))
    for p in RAW.rglob('*'):
        if p.is_file():verify(p,sha(p))
    if baseline.exists():
        for r in read(baseline):verify(r['path'],r['sha256'])
    else:
        save('preserved_inputs_sha256.csv',[dict(path=p,sha256=h) for p,h in sorted(protected.items())])
        dump('mutable_pointers_before.json',{p:sha(p) for p in sorted(MUTABLE)})
    import torch
    import stage8_vit_v2 as v
    import stage10_models as m
    from review_stage11B import audit
    m.configure();audited=audit(RAW,False);dump('frozen_audit_recheck.json',audited)
    cfg=json.loads((RAW/'configuration.json').read_text());spec=cfg['spec'];assert not cfg['synthetic']
    records={};paths={};targets=[];comparisons=[];screens=[];curves=[]
    saved_targets={(int(r['seed']),r['variant'],r['role']):r for r in read(RAW/'targets.csv')}
    saved_screens={int(r['seed']):r for r in read(RAW/'screen.csv')}
    for seed in spec['development_seeds']:
        ready={};times={}
        for arm in spec['variants']:
            marker=json.loads((RAW/f'seed_{seed}/{arm}/COMPLETE.json').read_text());attempt=Path(marker['attempt'])
            summary=json.loads((attempt/'summary.json').read_text());records[seed,arm]=summary
            paths[seed,arm]=attempt/'final_checkpoint.pt';epochs=read(attempt/'epochs.csv')
            for r in epochs:
                if int(r['epoch']) in [50,51,55,60,75,100]:
                    curves.append(dict(seed=seed,variant=arm,epoch=int(r['epoch']),validation_accuracy=float(r['validation_accuracy']),validation_loss=float(r['validation_loss'])))
            for role in ['readiness','secondary']:
                first,last=crossing(epochs,spec[role+'_target'],spec['sustained_epochs'])
                row=dict(seed=seed,variant=arm,role=role,threshold=spec[role+'_target'],reached=last is not None,
                         first_crossing_epoch=int(first['epoch']) if first else '',confirming_epoch=int(last['epoch']) if last else '',
                         observed_seconds=float(last['cumulative_observed_seconds']) if last else '',censoring_epoch=spec['epochs'] if not last else '')
                for k,value in row.items():
                    expected=saved_targets[seed,arm,role][k]
                    if isinstance(value,(bool,str)):assert str(value)==expected
                    else:close(value,expected)
                targets.append(row)
                if role=='readiness':ready[arm]=last is not None;times[arm]=row['observed_seconds']
        control_arms=['full','small','full_reset','small_reset']
        best=max(records[seed,c]['final_validation_accuracy'] for c in control_arms)
        qualified=[c for c in control_arms if (best-records[seed,c]['final_validation_accuracy'])*100<=spec['final_accuracy_margin_pp']+1e-12]
        fastest=min(qualified,key=lambda c:(records[seed,c]['total_run_seconds'],c))
        grow=records[seed,'grow'];full=records[seed,'full']
        delta=100*(grow['final_validation_accuracy']-best)
        rf=grow['total_run_seconds']/full['total_run_seconds'];rb=grow['total_run_seconds']/records[seed,fastest]['total_run_seconds']
        quality=delta>=-spec['final_accuracy_margin_pp']-1e-12;speed=max(rf,rb)<=1-spec['minimum_time_gain']
        screen=dict(seed=seed,best_control_accuracy=best,qualified_controls=';'.join(qualified),fastest_qualified_control=fastest,
                    final_delta_vs_best_control_pp=delta,total_time_ratio_to_full=rf,total_time_ratio_to_best_qualified=rb,
                    baseline_ready=ready['full'],candidate_ready=ready['grow'],quality_pass=quality,speed_pass=speed,
                    seed_screen_pass=quality and speed and ready['full'] and ready['grow'])
        for k,value in screen.items():
            expected=saved_screens[seed][k]
            if isinstance(value,(bool,str)):assert str(value)==expected
            else:close(value,expected)
        screens.append(screen)
        for arm in spec['variants']:
            a=records[seed,arm]
            comparisons.append(dict(seed=seed,variant=arm,final_accuracy=a['final_validation_accuracy'],
                final_delta_vs_full_pp=100*(a['final_validation_accuracy']-full['final_validation_accuracy']),
                final_delta_vs_small_pp=100*(a['final_validation_accuracy']-records[seed,'small']['final_validation_accuracy']),
                final_delta_vs_small_reset_pp=100*(a['final_validation_accuracy']-records[seed,'small_reset']['final_validation_accuracy']),
                total_run_seconds=a['total_run_seconds'],total_time_saving_vs_full_percent=100*(1-a['total_run_seconds']/full['total_run_seconds']),
                sustained80_observed_seconds=times[arm],sustained80_time_ratio_to_full=times[arm]/times['full'] if ready[arm] and ready['full'] else ''))
    means=[]
    for arm in spec['variants']:
        rr=[r for r in comparisons if r['variant']==arm]
        means.append(dict(variant=arm,n=len(rr),mean_final_accuracy=statistics.mean(r['final_accuracy'] for r in rr),
                          mean_total_run_seconds=statistics.mean(r['total_run_seconds'] for r in rr),
                          mean_final_delta_vs_full_pp=statistics.mean(r['final_delta_vs_full_pp'] for r in rr)))
    for name,rows in [('independent_targets.csv',targets),('independent_screen.csv',screens),('independent_seed_results.csv',comparisons),
                      ('independent_mean_results.csv',means),('transition_curve_samples.csv',curves)]:save(name,rows)
    if args.rescore_cuda:
        assert torch.cuda.is_available();preflight=v.idle_preflight()
        x,y,ev,ey,split=v.load_data('development',torch.device('cuda'));assert split==cfg['split'];del x,y
        rescored=[]
        for (seed,arm),path in paths.items():
            cp=torch.load(path,map_location='cpu',weights_only=True)
            variant='small_rgb32' if arm in ['small','small_reset'] else 'full_rgb32'
            model=m.make_model(variant,seed).to('cuda');model.load_state_dict(cp['model_state'],strict=True)
            loss,accuracy=v.evaluate(model,ev,ey,spec['eval_batch_size']);expected=records[seed,arm]
            close(accuracy,expected['final_validation_accuracy']);assert abs(loss-expected['final_validation_loss'])<1e-6
            rescored.append(dict(seed=seed,variant=arm,checkpoint_sha256=sha(path),validation_accuracy=accuracy,validation_loss=loss,
                                 recorded_accuracy=expected['final_validation_accuracy'],recorded_loss=expected['final_validation_loss']))
            del model,cp
        save('checkpoint_validation_rescore.csv',rescored)
        dump('rescore_metadata.json',dict(status='PASS',device=torch.cuda.get_device_name(0),test_dataset_loaded=False,training_performed=False,
                                         preflight=preflight,execution=m.execution_settings()))
    for p,h in protected.items():assert sha(p)==h,p
    result=dict(status='PASS',runs=10,updates=176000,preserved_immutable_files=len(protected),independent_targets_and_screens_match=True,
                final_checkpoints_rescored_this_invocation=args.rescore_cuda,
                both_seed_speed_pass=all(r['speed_pass'] for r in screens),both_seed_quality_pass=all(r['quality_pass'] for r in screens),
                seeds_passing=sum(r['seed_screen_pass'] for r in screens),
                decision='Fixed midpoint growth fails the frozen both-seed criterion. Preserve evidence; no automatic extension or transition search. Publication on hold.')
    dump('independent_audit.json',result);print(json.dumps(result,indent=2));print(json.dumps(means,indent=2))

if __name__=='__main__':main()
