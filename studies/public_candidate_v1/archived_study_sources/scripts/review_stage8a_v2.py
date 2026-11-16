"""Audit completed Stage8 v2 development; never train or construct test data."""
import csv, hashlib, json, math, statistics
from pathlib import Path
import torch
import stage8_vit_v2 as v
import stage8_workflow_v2 as w
from stage8_audit_v2 import close, records


def main():
    torch.set_num_threads(4)
    dev=v.ROOT/'results/stage8A_vit_development_v2'
    out=v.ROOT/'results/stage8A_review_v2'
    assert not (v.ROOT/'protocol/stage8/FROZEN_EVALUATION_v2.json').exists(), 'Review must precede freeze'
    assert not (v.ROOT/'results/stage8B_vit_evaluation_v2').exists()
    spec=v.SPEC;total=176*spec['epochs'];assert total==52800
    selection=json.loads((dev/'baseline_selection.json').read_text())
    assert selection['ready'] and selection['source_sha256']==v.source_hashes()
    assert selection['spec']==spec and not selection['fake']
    # Independent validation rescoring loads only the CIFAR training dataset.
    dataset=v.CIFAR10(str(v.ROOT/'data'),train=True,download=False)
    tr,va=v.split_indices(dataset.targets,spec)
    x=v.frozen6.normalize_cifar(torch.from_numpy(dataset.data[va.numpy()]).permute(0,3,1,2).contiguous().float().div_(255))
    y=torch.as_tensor(dataset.targets)[va]
    split=dict(kind='train_only_stratified_45000_5000',
        train_index_sha256=hashlib.sha256(tr.numpy().tobytes()).hexdigest(),
        validation_index_sha256=hashlib.sha256(va.numpy().tobytes()).hexdigest())
    summaries=[];curves=[];mix_stats=[];telemetry=[];seed_states={};rescore=[]
    assert len(list(dev.rglob('summary.json')))==4
    for seed in spec['development_seeds']:
        paired={}
        for variant in ('full_adamw','lgpb'):
            folder=(dev/'screen'/'lr_0p001'/f'seed_{seed}'/variant if variant=='full_adamw'
                    else dev/'calibration'/f'seed_{seed}'/variant)
            r=w.completed(folder,'development',seed,variant,.001,spec,False)
            config=json.loads((folder/'run_config.json').read_text())
            assert config['split']==split and not config['tiny'] and config['device']=='cuda'
            assert config['freeze_hash'] is None and config['profile'] is None
            seed_states.setdefault(seed,set()).add(r['initial_model_sha256'])
            steps=records(folder/'step_trace.csv');epochs=records(folder/'epoch_trace.csv')
            assert len(steps)==total and len(epochs)==300
            ema=None;previous=False;rng=v.np.random.default_rng(seed+98001)
            for i,s in enumerate(steps):
                loss=float(s['loss']);partial=bool(int(s['partial']))
                assert math.isfinite(loss) and not(partial and previous)
                assert int(s['step'])==i+1 and int(s['epoch'])==i//176+1
                if ema is None:assert s['ema_before']==''
                else:close(s['ema_before'],ema)
                chosen=variant!='full_adamw' and ema is not None and not previous and loss<=ema
                assert partial==chosen,(seed,variant,i)
                close(s['lr'],v.learning_rate(i,total,176,.001,spec))
                close(s['mixup_lambda'],float(rng.beta(.2,.2)))
                for key in ('augmentation','forward','backward','optimizer'):
                    assert float(s[key+'_seconds'])>=0
                ema=loss if ema is None else .95*ema+.05*loss;previous=partial
            assert sum(int(s['partial']) for s in steps)==r['partial_updates']
            assert r['full_updates']+r['partial_updates']==total
            close(r['partial_fraction'],r['partial_updates']/total)
            for e,epoch in enumerate(epochs):
                chosen=steps[e*176:(e+1)*176]
                assert int(epoch['epoch'])==e+1 and int(epoch['updates'])==(e+1)*176
                assert int(epoch['partial_updates'])==sum(int(s['partial']) for s in chosen)
                assert int(epoch['full_updates'])+int(epoch['partial_updates'])==176
                close(epoch['partial_fraction'],int(epoch['partial_updates'])/176)
                close(epoch['cumulative_training_seconds'],sum(float(x['epoch_training_seconds']) for x in epochs[:e+1]))
                curves.append(dict(seed=seed,variant=variant,**epoch))
            for key in ('augmentation','forward','backward','optimizer'):
                close(r['component_seconds'][key],sum(float(s[key+'_seconds']) for s in steps))
            for flag,label in ((0,'full'),(1,'partial')):
                close(r['component_seconds']['backward_'+label],sum(float(s['backward_seconds']) for s in steps if int(s['partial'])==flag))
            close(r['training_wall_seconds'],epochs[-1]['cumulative_training_seconds'])
            close(r['final_eval_accuracy'],epochs[-1]['eval_accuracy']);close(r['final_eval_loss'],epochs[-1]['eval_loss'])
            close(r['timing_unattributed_seconds'],r['training_wall_seconds']-sum(r['component_seconds'][k] for k in ('augmentation','forward','backward','optimizer')))
            assert r['scalar_loss_item_calls']==total and r['timed_explicit_cuda_sync_calls']==8*total+600
            for target in spec['targets']:
                label=round(100*target);first=next((e for e in epochs if float(e['eval_accuracy'])>=target),None)
                assert r[f'epoch_to_{label}pct']==(int(first['epoch']) if first else None)
                if first:close(r[f'training_only_time_to_{label}pct'],first['cumulative_training_seconds'])
                else:assert r[f'training_only_time_to_{label}pct'] is None
            ck=torch.load(folder/'final_checkpoint.pt',map_location='cpu',weights_only=True)
            assert ck['fingerprint']==r['fingerprint'] and ck['seed']==seed and ck['variant']==variant
            close(ck['ema'],ema)
            model=v.make_model(spec);model.load_state_dict(ck['model_state'],strict=True)
            prefix,suffix,names=v.partition(model,2);assert names==r['prefix_parameter_names']
            assert r['prefix_parameter_count']==sum(p.numel() for p in prefix)
            assert r['suffix_parameter_count']==sum(p.numel() for p in suffix)
            ids=[i for group in ck['optimizer_state']['param_groups'] for i in group['params']]
            assert len(ids)==len(list(model.parameters()))
            for p,param_id in zip(model.parameters(),ids):
                assert bool(torch.isfinite(p).all())
                state=ck['optimizer_state']['state'][param_id]
                assert int(state['step'])==total
                for value in state.values():assert bool(torch.isfinite(value).all())
            loss,accuracy=v.evaluate(model,x,y,64)
            # CPU vs GPU arithmetic can differ; allow at most one validation decision.
            assert abs(accuracy-r['final_eval_accuracy'])<=1/5000+1e-12,(seed,variant,accuracy)
            assert abs(loss-r['final_eval_loss'])<1e-4,(seed,variant,loss)
            rescore.append(dict(seed=seed,variant=variant,saved_accuracy=r['final_eval_accuracy'],
                cpu_accuracy=accuracy,saved_loss=r['final_eval_loss'],cpu_loss=loss))
            # At the actual final checkpoint, verify fresh suffix gradients for mixed targets.
            model.eval();xx,yy,_=v.mix_batch(x[:4],y[:4],v.np.random.default_rng(123),torch.Generator().manual_seed(456),.2,10)
            objective=v.F.cross_entropy(model(xx),yy,label_smoothing=.1)
            full=torch.autograd.grad(objective,list(model.parameters()),retain_graph=True)
            fresh=torch.autograd.grad(objective,suffix)
            expected={id(p):g for p,g in zip(model.parameters(),full)}
            for p,g in zip(suffix,fresh):assert torch.equal(g,expected[id(p)])
            for flag in (0,1):
                chosen=[s for s in steps if int(s['partial'])==flag]
                if chosen:mix_stats.append(dict(seed=seed,variant=variant,partial=flag,n=len(chosen),
                    mean_mixup_strength=statistics.mean(min(float(s['mixup_lambda']),1-float(s['mixup_lambda'])) for s in chosen),
                    mean_loss=statistics.mean(float(s['loss']) for s in chosen)))
            gpu_rows=records(folder/'gpu_telemetry.csv');assert len(gpu_rows)==601
            gpu=[g for sample in gpu_rows for g in json.loads(sample['sample']).get('gpus',[])]
            assert len(gpu)==601
            telemetry.append(dict(seed=seed,variant=variant,samples=len(gpu),
                min_clock=min(float(g['clocks.sm']) for g in gpu),max_clock=max(float(g['clocks.sm']) for g in gpu),
                min_temp=min(float(g['temperature.gpu']) for g in gpu),max_temp=max(float(g['temperature.gpu']) for g in gpu),
                first50_mean_epoch_seconds=statistics.mean(float(e['epoch_training_seconds']) for e in epochs[:50]),
                last50_mean_epoch_seconds=statistics.mean(float(e['epoch_training_seconds']) for e in epochs[-50:])))
            best=max(epochs,key=lambda e:float(e['eval_accuracy']))
            paired[variant]=r
            summaries.append(dict(seed=seed,variant=variant,final_validation_accuracy=r['final_eval_accuracy'],
                best_validation_accuracy=float(best['eval_accuracy']),best_epoch=int(best['epoch']),
                training_seconds=r['training_wall_seconds'],partial_fraction=r['partial_fraction'],
                updates=total,checkpoint_sha256=r['checkpoint_sha256']))
            print('VERIFIED',seed,variant,'validation',accuracy,flush=True)
            del model,ck,steps
        assert len(seed_states[seed])==1
    baseline=[r['final_validation_accuracy'] for r in summaries if r['variant']=='full_adamw']
    lgpb=[r['final_validation_accuracy'] for r in summaries if r['variant']=='lgpb']
    assert statistics.mean(baseline)>=.85 and min(baseline)>=.83
    close(selection['screen'][0]['mean_validation_accuracy'],statistics.mean(baseline))
    close(selection['screen'][0]['min_validation_accuracy'],min(baseline))
    out.mkdir(parents=True,exist_ok=True)
    v.write_csv(out/'per_seed.csv',summaries);v.write_csv(out/'validation_rescore_cpu.csv',rescore)
    v.write_csv(out/'learning_curves.csv',curves);v.write_csv(out/'mixup_gate_descriptive.csv',mix_stats)
    v.write_csv(out/'telemetry_summary.csv',telemetry)
    paths=sorted(p for p in dev.rglob('*') if p.is_file())+[Path(p) for p in v.source_hashes()]
    v.write_csv(out/'preserved_development_sha256.csv',[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    result=dict(status='PASS',runs=4,epochs=1200,steps=4*total,baseline_readiness='PASS',
        baseline_mean_validation_accuracy=statistics.mean(baseline),lgpb_mean_validation_accuracy=statistics.mean(lgpb),
        lgpb_minus_baseline_pp=100*(statistics.mean(lgpb)-statistics.mean(baseline)),
        checkpoint_hashes_counters_and_finiteness=True,identical_paired_initialization=True,
        validation_rescored_from_checkpoints_on_cpu=True,exact_suffix_gradients_at_saved_checkpoints=True,
        evaluation_outputs_absent_at_review=True,
        interpretation='Severe LGPB development degradation; no integrity failure detected. Development is not reserved-seed evaluation.')
    v.atomic_json(out/'audit.json',result)
    lines=['# Stage8 v2 development review','',
        'Integrity review PASS: four runs,1200 epochs,211,200 updates. Baseline readiness PASS.',
        'Source/configuration hashes, fixed split, gate/EMA, nonadjacency, LR, MixUp lambdas,',
        'counts, timings, target crossings, checkpoint hashes/counters/finite states verified.',
        'Saved checkpoints independently rescored on the training-derived validation split',
        'using CPU. Exact fresh suffix gradients also checked at all four saved checkpoints.',
        'No test images were loaded; no training was performed.','',
        '| Seed | Arm | Final validation | Partial fraction | Training seconds |',
        '|---:|---|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['variant']} | {100*r['final_validation_accuracy']:.2f}% | {100*r['partial_fraction']:.2f}% | {r['training_seconds']:.1f} |" for r in summaries]
    lines += ['',f"Full AdamW mean {100*statistics.mean(baseline):.2f}%; LGPB mean {100*statistics.mean(lgpb):.2f}%; difference {result['lgpb_minus_baseline_pp']:+.2f}pp.",
        'This is a large adverse development result. It must be retained and disclosed.',
        'Two development seeds do not support a six-seed evaluation inference or a causal',
        'claim that MixUp, loss selection, prefix size or AdamW continuation caused the gap.',
        'Sequential baseline-then-LGPB ordering also limits interpretation of wall ratios.',
        'See all learning curves and telemetry; no best epochs replace final outcomes.','',
        'Recommendation: keep the prospectively declared four-arm evaluation unchanged,',
        'including unfavorable results, to characterize this architecture/recipe boundary.',
        'Baseline readiness was the declared quality gate; no LGPB efficacy gate was declared.',
        'Do not invent one retrospectively or tune a third recipe to erase this warning.',
        'Reserved-seed evaluation will compare periodic and replay controls as planned.',
        'This does not guarantee positive transfer; no implementation integrity defect',
        'was detected by these checks, which cannot exclude every possible defect.','',
        'Freeze must follow this review and precede any evaluation. All new results remain',
        'separate from Stage6 confirmation and the completed Stage7 manuscript evidence.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
