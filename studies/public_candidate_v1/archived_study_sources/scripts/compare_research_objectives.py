"""Read-only synthesis of reviewed studies; no training, rescoring or new selection rule."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research_objective_comparison_v1'
MUTABLE={str(ROOT/name) for name in ('CURRENT_RELEASE.md','README_STAGE9.md')}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'comparison_artifacts_sha256.json').exists()
    previous=ROOT/'results/stage14B_review_v1'
    protected={row['path']:row['sha256'] for row in read(previous/'preserved_inputs_sha256.csv')}
    seal=previous/'review_artifacts_sha256.json';entries=json.loads(seal.read_text())
    for path,h in {**protected,**entries}.items():assert sha(path)==h,path
    protected.update({path:h for path,h in entries.items() if path not in MUTABLE});protected[str(seal)]=sha(seal)
    dump('preserved_inputs_sha256.json',protected)
    dump('mutable_pointers_before.json',{path:sha(path) for path in sorted(MUTABLE)})
    growth=read(ROOT/'results/stage11B_review_v1/independent_seed_results.csv')
    transfer=read(ROOT/'results/stage12A_review_v1/independent_seed_results.csv')
    local=read(ROOT/'results/stage13C_stage14A_review_v1/stage13C_independent_summary.csv')
    tokens=read(ROOT/'results/stage14B_review_v1/independent_summary.csv')
    token_raw=read(ROOT/'results/stage14B_token_recovery_v1/summary.csv')
    result=[]
    for seed in (1501,1502):
        arms={r['variant']:r for r in growth if int(r['seed'])==seed}
        g,f=arms['grow'],arms['full']
        saving=100*(1-float(g['total_run_seconds'])/float(f['total_run_seconds']))
        deficit=100*(float(f['final_accuracy'])-float(g['final_accuracy']))
        target_penalty=100*(float(g['sustained80_observed_seconds'])/float(f['sustained80_observed_seconds'])-1)
        close(saving,g['total_time_saving_vs_full_percent']);close(-deficit,g['final_delta_vs_full_pp'])
        close(target_penalty,100*(float(g['sustained80_time_ratio_to_full'])-1))
        result.append(dict(stage='11B',strategy='growth',seed=seed,objective='training_from_scratch',
            clock='complete fixed-100-epoch run, common loading and final audit excluded',
            saving_percent=saving,accuracy_deficit_pp=deficit,original_quality_margin_pp=.2,
            within_original_margin=deficit<=.2+1e-10,
            sustained80_observed_time_increase_percent=target_penalty,
            limitation='Fixed-budget saving; sustained target reached later; two-seed quality rule fails.'))
    for seed in (1601,1602):
        arms={r['arm']:r for r in transfer if int(r['seed'])==seed};f=arms['full'];lp=arms['lp_full']
        for name in ('lp_full','lp_periodic'):
            a=arms[name];saving=100*(1-float(a['total_run_seconds'])/float(f['total_run_seconds']))
            deficit=100*(float(f['final_accuracy'])-float(a['final_accuracy']))
            close(saving,a['total_time_saving_vs_full_percent']);close(-deficit,a['delta_vs_full_pp'])
            result.append(dict(stage='12A',strategy=name,seed=seed,objective='existing_checkpoint_adaptation',
                clock='complete fixed-12-epoch adaptation run; external pretraining excluded',saving_percent=saving,
                accuracy_deficit_pp=deficit,original_quality_margin_pp=.3,within_original_margin=deficit<=.3+1e-10,
                saving_vs_lp_full_percent=100*(1-float(a['total_run_seconds'])/float(lp['total_run_seconds'])),
                sustained92_saving_vs_lp_full_percent=100*(1-float(a['sustained92_seconds'])/float(lp['sustained92_seconds'])),
                limitation='lp_full is an ordinary control, not a promoted primary candidate; periodic fails primary quality.'))
        a=next(r for r in local if int(r['seed'])==seed and r['arm']=='local')
        f=next(r for r in local if int(r['seed'])==seed and r['arm']=='full')
        saving=100*(1-float(a['charged_training_seconds'])/float(f['charged_training_seconds']))
        close(saving,-float(a['charged_time_change_percent']))
        result.append(dict(stage='13C',strategy='local_loss',seed=seed,objective='existing_checkpoint_adaptation',
            clock='one-epoch training + auxiliary calibration, evaluation/logging excluded',saving_percent=saving,
            accuracy_deficit_pp=-float(a['delta_vs_full_pp']),original_quality_margin_pp=.3,within_original_margin=False,
            limitation='Short trajectory from an intermediate state; slower and below source quality.'))
    for seed in (1501,1502):
        a=next(r for r in tokens if int(r['seed'])==seed and r['mode']=='similarity')
        full=next(r for r in token_raw if int(r['seed'])==seed and r['mode']=='full')
        sim=next(r for r in token_raw if int(r['seed'])==seed and r['mode']=='similarity')
        p=float(full['source_training_seconds']);f=float(full['charged_recovery_seconds']);c=float(sim['charged_recovery_seconds'])
        marginal=100*(1-c/f);inclusive=100*(1-(p+c)/(p+f))
        fraction=f/(p+f);close(inclusive,marginal*fraction)
        close(marginal,a['charged_recovery_time_saving_percent']);close(inclusive,a['source_inclusive_training_saving_percent'])
        inference=100*(1-float(sim['mean_inference_seconds'])/float(full['mean_inference_seconds']))
        close(inference,a['inference_time_saving_percent'])
        deficit=-float(a['delta_vs_adapted_full_pp'])
        result.append(dict(stage='14B',strategy='similarity_merging',seed=seed,
            objective='existing_checkpoint_adaptation',clock='charged five-epoch recovery including setup/warmup; excludes evaluation/logging',
            saving_percent=marginal,accuracy_deficit_pp=deficit,original_quality_margin_pp=.3,
            within_original_margin=deficit<=.3+1e-10,
            source_inclusive_training_saving_percent=inclusive,recovery_share_of_full_training_cost=fraction,
            observed_source_plus_arm_saving_percent=float(a['source_inclusive_observed_saving_percent']),
            inference_operation_time_saving_percent=inference,
            incremental_training_memory_saving_percent=float(a['incremental_training_memory_saving_percent']),
            absolute_training_peak_saving_percent=float(a['absolute_training_peak_saving_percent']),
            limitation='Two cost scopes of the same run; no across-architecture head-to-head or matched-quality confirmation.'))
    dump('verified_comparison.json',result)
    for path,h in protected.items():assert sha(path)==h,path
    dump('comparison_verification.json',dict(status='PASS',protected_files=len(protected),
        per_seed_strategy_records=len(result),parameter_updates_performed=False,prediction_scoring_performed=False,
        thresholds_changed=False,selection_protocol_prepared=False,
        interpretation='Descriptive objective comparison; primary objective recommended, not yet author-selected.'))
    print('Verified comparison:',len(result),'per-seed strategy records;',len(protected),'prior files unchanged.')


if __name__=='__main__':main()
