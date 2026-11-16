"""Independent diagnostic arithmetic and saved-state scoring; no updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,math,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'results/stage13A_local_diagnostic_v1';OUT=ROOT/'results/stage13A_review_v1'
MUTABLE={str(ROOT/n) for n in ['CURRENT_RELEASE.md','README_STAGE9.md']}
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
def save(name,rows):
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-10),(a,b)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rescore-cuda',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True)
    protected={};baseline=OUT/'preserved_inputs_sha256.csv'
    def verify(path,h):
        assert sha(path)==h,path
        if str(path) not in MUTABLE:protected[str(path)]=h
    prior=json.loads((ROOT/'protocol/stage13/preparation_preservation_v1.json').read_text())
    for path,h in prior['prior_immutable_sha256'].items():verify(path,h)
    manifest=ROOT/'protocol/stage13/preparation_sha256_v1.json'
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
    import torch.nn.functional as F
    import stage13_diagnostic as r
    from review_stage13A import audit
    r.p.execution.configure();audited=audit(RAW,False);dump('frozen_audit_recheck.json',audited)
    cfg=json.loads((RAW/'configuration.json').read_text());spec=cfg['spec'];assert not cfg['synthetic']
    marker=json.loads((RAW/'RAW_COMPLETE.json').read_text());attempt=Path(marker['attempt'])
    saved={(int(row['seed']),row['mode']):row for row in read(RAW/'summary.csv')}
    means=[];alignments=[];decisions=[];raw_by_seed={}
    for seed in spec['source_seeds']:
        rows=read(attempt/f'measurements_{seed}.csv');raw_by_seed[seed]=rows
        metadata=json.loads((attempt/f'seed_{seed}.json').read_text());aggregates={}
        for mode in spec['modes']:
            rr=[row for row in rows if row['mode']==mode]
            for row in rr:close(row['score_improvement'],float(row['score_loss_before'])-float(row['score_loss_after']))
            a=dict(seed=seed,mode=mode,mean_step_seconds=st.mean(float(row['seconds']) for row in rr),
                mean_score_improvement=st.mean(float(row['score_improvement']) for row in rr),
                positive_score_fraction=st.mean(float(row['score_improvement'])>0 for row in rr),
                mean_incremental_peak_bytes=st.mean(int(row['incremental_peak_bytes']) for row in rr),
                mean_absolute_peak_bytes=st.mean(int(row['peak_allocated_bytes']) for row in rr))
            for key,value in a.items():
                if isinstance(value,str):assert value==saved[seed,mode][key]
                else:close(value,saved[seed,mode][key])
            a.update(mean_score_loss_before=st.mean(float(row['score_loss_before']) for row in rr),
                     mean_score_loss_after=st.mean(float(row['score_loss_after']) for row in rr),negative_updates=sum(float(row['score_improvement'])<0 for row in rr))
            aggregates[mode]=a;means.append(a)
        full=aggregates['full'];local=aggregates['local']
        ratio=local['mean_step_seconds']/full['mean_step_seconds'];memory=local['mean_incremental_peak_bytes']/full['mean_incremental_peak_bytes']
        absolute=local['mean_absolute_peak_bytes']/full['mean_absolute_peak_bytes']
        full_positive=full['mean_score_improvement']>0
        utility=full_positive and local['mean_score_improvement']>0 and local['mean_score_improvement']>=spec['minimum_local_full_utility_ratio']*full['mean_score_improvement']
        resources=ratio<=1-spec['minimum_time_saving'] or memory<=1-spec['minimum_incremental_memory_saving']
        original=saved[seed,'local'];close(ratio,original['local_full_time_ratio']);close(memory,original['local_full_incremental_memory_ratio'])
        assert original['local_utility_review_flag']==str(utility) and original['local_resource_review_flag']==str(resources)
        assert original['qualifies_for_pilot_review']==str(utility and resources)
        saving=full['mean_step_seconds']-local['mean_step_seconds']
        breakeven=math.ceil(metadata['calibration_overhead_seconds']/saving) if saving>0 else ''
        if breakeven!='':close(breakeven,original['optimistic_calibration_break_even_steps'])
        else:assert original['optimistic_calibration_break_even_steps']==''
        decisions.append(dict(seed=seed,full_reference_positive=full_positive,utility_interpretable=full_positive,
            local_time_saving_percent=100*(1-ratio),local_incremental_memory_saving_percent=100*(1-memory),
            local_absolute_peak_saving_percent=100*(1-absolute),local_utility_flag=utility,local_resource_flag=resources,
            qualifies_for_pilot_review=utility and resources,calibration_overhead_seconds=metadata['calibration_overhead_seconds'],
            optimistic_calibration_break_even_steps=breakeven))
        alignment=read(attempt/f'alignment_{seed}.csv')
        for mode in spec['modes']:
            for block in ['block0','block1','block2']:
                rr=[row for row in alignment if row['mode']==mode and row['block']==block];defined=[row for row in rr if row['defined']=='True']
                for row in defined:close(row['cosine'],float(row['dot'])/math.sqrt(float(row['norm_squared'])*float(row['full_norm_squared'])))
                alignments.append(dict(seed=seed,mode=mode,block=block,defined=len(defined),
                    mean_cosine=st.mean(float(row['cosine']) for row in defined) if defined else '',
                    min_cosine=min(float(row['cosine']) for row in defined) if defined else '',
                    max_cosine=max(float(row['cosine']) for row in defined) if defined else ''))
    save('independent_measurement_summary.csv',means);save('independent_review_flags.csv',decisions);save('gradient_alignment_summary.csv',alignments)
    if args.rescore_cuda:
        assert torch.cuda.is_available();preflight=r.v.idle_preflight();x,y,split=r.data(torch.device('cuda'),False);assert split==cfg['split']
        rescored=[]
        for seed in spec['source_seeds']:
            stream=json.loads((attempt/f'streams_{seed}.json').read_text());xx,yy=r.batch(x,y,stream['score_batches'][0],spec['image_size'])
            calibrated=torch.load(attempt/f'calibrated_{seed}.pt',map_location='cpu',weights_only=True)
            model=r.l.LocalModel(r.p.make_model(seed,False),seed,False).to('cuda');model.load_state_dict(calibrated['model_state'],strict=True);model.eval()
            with torch.no_grad():before=F.cross_entropy(model(xx),yy).item()
            for mode in spec['modes']:
                path=attempt/f'one_step_{seed}_{mode}.pt';cp=torch.load(path,map_location='cpu',weights_only=True)
                model.load_state_dict(cp['model_state'],strict=True)
                with torch.no_grad():after=F.cross_entropy(model(xx),yy).item()
                original=next(row for row in raw_by_seed[seed] if row['mode']==mode and int(row['trial'])==0)
                close(before,original['score_loss_before']);close(after,original['score_loss_after'])
                rescored.append(dict(seed=seed,mode=mode,trial=0,checkpoint_sha256=sha(path),score_loss_before=before,
                    score_loss_after=after,score_improvement=before-after,recorded_score_improvement=float(original['score_improvement'])))
                del cp
            del model,calibrated
        save('saved_one_step_rescore.csv',rescored)
        dump('rescore_metadata.json',dict(status='PASS',source_states=2,saved_updated_states=8,device=torch.cuda.get_device_name(0),
            parameter_updates_performed=False,test_or_validation_scoring=False,training_pool_score_examples_only=True,preflight=preflight))
    for path,h in protected.items():assert sha(path)==h,path
    result=dict(status='PASS',date='2026-10-23',measured_operations=96,preserved_immutable_files=len(protected),independent_summary_and_flags_match=True,
        saved_first_trial_states_rescored_this_invocation=args.rescore_cuda,negative_updates=sum(a['negative_updates'] for a in means),
        full_reference_positive_both_states=all(a['full_reference_positive'] for a in decisions),
        resource_flags_pass_both_states=all(a['local_resource_flag'] for a in decisions),
        decision='One-step utility inconclusive because the full reference worsens scoring loss; this local configuration also misses resource thresholds. No long pilot or automatic retuning.')
    dump('independent_audit.json',result);print(json.dumps(result,indent=2));print(json.dumps(decisions,indent=2))
    print(json.dumps([a for a in alignments if a['mode']=='local'],indent=2))

if __name__=='__main__':main()
