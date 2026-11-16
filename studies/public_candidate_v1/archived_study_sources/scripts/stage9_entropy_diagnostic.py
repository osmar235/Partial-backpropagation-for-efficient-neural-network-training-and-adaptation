"""Reconstruct target entropy on existing development traces; never train/test-load."""
from pathlib import Path
import argparse,csv,hashlib,json,math,statistics as st,unittest
import numpy as np
import torch
import torch.nn.functional as F
import stage8_vit_v2 as v


def entropy_floor(labels,permutation,lam,smoothing,classes):
    """Mean entropy of the EXACT MixUp+label-smoothed target distribution."""
    def plogp(p):return p*math.log(p) if p else 0.
    c=smoothing/classes
    hard=-(plogp(1-smoothing+c)+(classes-1)*plogp(c))
    a=(1-smoothing)*lam+c;b=(1-smoothing)*(1-lam)+c
    different=-(plogp(a)+plogp(b)+(classes-2)*plogp(c))
    same=float((labels==labels[permutation]).double().mean())
    return same*hard+(1-same)*different


class Checks(unittest.TestCase):
    def test_exact_target_entropy_and_unchanged_gradient(self):
        y=torch.tensor([0,1,1,2]);perm=torch.tensor([1,0,3,2])
        for labels in [y,torch.zeros_like(y)]:
            for lam in [0.,.01,.3,.5,1.]:
                for smoothing in [0.,.1]:
                    target=lam*F.one_hot(labels,10).double()+(1-lam)*F.one_hot(labels[perm],10).double()
                    target=(1-smoothing)*target+smoothing/10
                    expected=float(-(torch.where(target>0,target*target.clamp_min(1e-300).log(),0.)).sum(1).mean())
                    actual=entropy_floor(labels,perm,lam,smoothing,10)
                    self.assertAlmostEqual(expected,actual,places=12)
                    logits=torch.randn(4,10,dtype=torch.double,requires_grad=True)
                    loss=F.cross_entropy(logits,target)
                    raw=torch.autograd.grad(loss,logits,retain_graph=True)[0]
                    corrected=torch.autograd.grad(loss-actual,logits)[0]
                    self.assertTrue(torch.equal(raw,corrected))


def write(path,rows):v.write_csv(path,rows)
def corr(centered):
    xx,yy,xy=centered
    return xy/math.sqrt(xx*yy) if xx>0 and yy>0 else None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checks-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    checked=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not checked.wasSuccessful():raise SystemExit(1)
    if args.checks_only:return
    dev=v.ROOT/'results/stage8A_vit_development_v2';out=v.ROOT/'results/stage9A_entropy_diagnostic_v1'
    out.mkdir(parents=True,exist_ok=True)
    dataset=v.CIFAR10(str(v.ROOT/'data'),train=True,download=False)
    tr,_=v.split_indices(dataset.targets);labels=torch.as_tensor(dataset.targets)[tr]
    rows=[];epoch_rows=[];inputs={};per_seed=[]
    for seed in [1001,1002]:
        bg=torch.Generator().manual_seed(seed+91001);pg=torch.Generator().manual_seed(seed+99001)
        rng=np.random.default_rng(seed+98001)
        paths={name:(dev/'screen/lr_0p001'/f'seed_{seed}'/name if name=='full_adamw'
                     else dev/'calibration'/f'seed_{seed}'/name) for name in ['full_adamw','lgpb']}
        configs={name:json.loads((p/'run_config.json').read_text()) for name,p in paths.items()}
        assert all(c['spec']==v.SPEC and c['source_sha256']==v.source_hashes() for c in configs.values())
        handles={name:(p/'step_trace.csv').open(newline='') for name,p in paths.items()}
        readers={name:csv.DictReader(f) for name,f in handles.items()}
        ema={(name,policy):None for name in paths for policy in ['raw','excess']}
        previous={key:False for key in ema}
        groups={};centered={name:{score:[0.,0.,0.] for score in ['raw','excess']} for name in paths}
        step=0
        for epoch in range(1,301):
            sample={name:[] for name in paths}
            for ix in v.frozen6.epoch_batches(len(labels),256,bg):
                y=labels[ix];perm=torch.randperm(len(y),generator=pg);lam=float(rng.beta(.2,.2))
                floor=entropy_floor(y,perm,lam,.1,10);strength=min(lam,1-lam);step+=1
                for name in paths:
                    r=next(readers[name]);loss=float(r['loss']);excess=loss-floor
                    assert int(r['step'])==step and int(r['epoch'])==epoch
                    assert math.isclose(float(r['mixup_lambda']),lam,rel_tol=1e-10,abs_tol=1e-10)
                    assert excess>=-1e-5,excess
                    choices={}
                    for policy,score in [('raw',loss),('excess',excess)]:
                        key=(name,policy);old=ema[key]
                        selected=old is not None and not previous[key] and score<=old
                        choices[policy]=selected
                        ema[key]=score if old is None else .95*old+.05*score
                        previous[key]=selected
                    if name=='lgpb':assert choices['raw']==bool(int(r['partial']))
                    sample[name].append((floor,loss,excess))
                    bucket='weak_le_0p05' if strength<=.05 else ('strong_ge_0p20' if strength>=.20 else 'middle')
                    for policy,partial in [('actual',bool(int(r['partial']))),*choices.items()]:
                        key=(name,policy,int(partial),'all')
                        val=groups.setdefault(key,[0,0.,0.,0.,0.]);val[0]+=1
                        for i,x in enumerate([strength,floor,loss,excess],1):val[i]+=x
                        key=(name,policy,int(partial),bucket)
                        val=groups.setdefault(key,[0,0.,0.,0.,0.]);val[0]+=1
                        for i,x in enumerate([strength,floor,loss,excess],1):val[i]+=x
            for name,data in sample.items():
                mean=np.mean(data,axis=0)
                for policy,index in [('raw',1),('excess',2)]:
                    de=[(x[0]-mean[0],x[index]-mean[index]) for x in data]
                    centered[name][policy][0]+=sum(x*x for x,y in de)
                    centered[name][policy][1]+=sum(y*y for x,y in de)
                    centered[name][policy][2]+=sum(x*y for x,y in de)
                epoch_rows.append(dict(seed=seed,variant=name,epoch=epoch,
                    mean_entropy_floor=mean[0],mean_raw_cross_entropy=mean[1],mean_excess_cross_entropy=mean[2]))
        assert step==52800
        for name,reader in readers.items():assert next(reader,None) is None;handles[name].close()
        for (name,policy,partial,bucket),val in groups.items():
            n=val[0];rows.append(dict(seed=seed,variant=name,policy=policy,partial=partial,mixing_bin=bucket,
                updates=n,mean_mixing_strength=val[1]/n,mean_target_entropy=val[2]/n,
                mean_raw_cross_entropy=val[3]/n,mean_excess_cross_entropy=val[4]/n))
        for name in paths:
            per_seed.append(dict(seed=seed,variant=name,updates=step,
                epoch_centered_corr_entropy_raw=corr(centered[name]['raw']),
                epoch_centered_corr_entropy_excess=corr(centered[name]['excess'])))
        for p in paths.values():
            for name in ['step_trace.csv','run_config.json','summary.json']:
                path=p/name;inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        print('RECONSTRUCTED DEVELOPMENT SEED',seed,flush=True)
    write(out/'policy_groups.csv',rows);write(out/'epochs.csv',epoch_rows);write(out/'correlations.csv',per_seed)
    comparison=[]
    for seed in [1001,1002]:
        for name in paths:
            for policy in ['raw','excess']:
                chosen=[r for r in rows if r['seed']==seed and r['variant']==name and r['policy']==policy]
                def probability(bucket):
                    group=[r for r in chosen if r['mixing_bin']==bucket]
                    return sum(r['updates'] for r in group if r['partial'])/sum(r['updates'] for r in group)
                overall=probability('all');weak=probability('weak_le_0p05');strong=probability('strong_ge_0p20')
                comparison.append(dict(seed=seed,variant=name,shadow_policy=policy,partial_fraction=overall,
                    p_partial_weak_mixing=weak,p_partial_strong_mixing=strong,weak_minus_strong=weak-strong))
    write(out/'shadow_policy_comparison.csv',comparison)
    record=dict(status='PASS',development_seeds=[1001,1002],trajectories=4,updates_reconstructed=211200,
        images_used='CIFAR10 training-label metadata only; no test dataset constructed',
        new_training=False,closed_loop_efficacy_established=False,correlations=per_seed,
        shadow_policies=comparison,input_sha256=inputs,
        limitation='Shadow masks use observed original trajectories; future weights/losses would change after actual intervention.')
    v.atomic_json(out/'diagnostic.json',record)
    lines=['# Stage9A: target-entropy diagnostic on existing development data','',
        'No new training. Target entropy reconstructed with exact batch/permutation RNG and label smoothing.',
        'Cross-entropy = target entropy + KL(target || prediction). Only the gate score is reconsidered;',
        'subtracting the target-dependent constant does not change training gradients.',
        'No threshold, architecture or new model has been selected from test-seed outcomes.','',
        '| Seed | Observed trajectory | Shadow score | Partial fraction | P(partial), weak mixing | P(partial), strong mixing |',
        '|---:|---|---|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['variant']} | {r['shadow_policy']} | {r['partial_fraction']:.3f} | {r['p_partial_weak_mixing']:.3f} | {r['p_partial_strong_mixing']:.3f} |" for r in comparison]
    lines += ['', 'Weak=min(lambda,1-lambda)<=0.05; strong>=0.20. Bins are diagnostic, not selected gates.',
        'These are offline SHADOW decisions on already observed trajectories. They cannot estimate',
        'accuracy/time of a newly trained controller or prove the cause of Stage8 failure.',
        'Entropy subtraction is a diagnostic repair hypothesis, not established algorithmic novelty.',
        'A redesigned method must beat periodic/random skipping at matched resources, counting all overhead.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',correlations=per_seed,comparison=comparison),indent=2),flush=True)


if __name__=='__main__':main()
