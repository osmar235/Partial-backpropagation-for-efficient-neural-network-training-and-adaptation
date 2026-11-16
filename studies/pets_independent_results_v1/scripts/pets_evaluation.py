"""Frozen 24-seed evaluation; all training and audit precede official-test access."""
import argparse
from contextlib import contextmanager
import gc
from pathlib import Path
import re
import time
import uuid
import scipy
import numpy as np
import torch
from PIL import Image
import pets_selective as p
from analyze_pets_evaluation import analyze

b=p.b
SPEC=b.ROOT/'protocol/pets_evaluation/spec_v1.json'
PREPARED=b.ROOT/'protocol/pets_evaluation/PREPARED_v1.json'
REAL=b.ROOT/'results/pets_independent_evaluation_v1'
FILES=p.FILES+['scripts/pets_evaluation.py','scripts/analyze_pets_evaluation.py','scripts/test_pets_evaluation.py',
    'scripts/run_pets_evaluation.cmd','protocol/pets_evaluation/spec_v1.json',
    'protocol/pets_evaluation/PROTOCOL_v1.md','README_PETS_EVALUATION.md']


def source_hashes():return {rel:b.sha(b.ROOT/rel) for rel in FILES}


def configuration(smoke,tiny,device):
    cfg=b.configuration(smoke,tiny,device);spec=b.read(SPEC)
    assert len(spec['seeds'])==24 and len(set(spec['seeds']))==24
    assert not set(spec['seeds']) & set(range(2601,2604)) and not set(spec['seeds']) & set(range(2701,2707))
    assert spec['accuracy_margin_pp']==.30 and spec['per_comparison_one_sided_alpha']==.025
    if smoke:spec['seeds']=[1,2]
    else:
        assert device.type=='cuda'
        prepared=b.read(PREPARED);assert prepared['validation_status']=='PASS'
        for rel,h in {**prepared['sources'],**prepared['evidence']}.items():assert b.sha(b.ROOT/rel)==h,rel
        assert cfg['runtime']==prepared['runtime'] and scipy.__version__==prepared['scipy_version']
        assert cfg['inputs']==prepared['inputs']
        assert b.read(b.ROOT/'results/pets_selective_review_v1/verification.json')['status']=='PASS'
    cfg['evaluation_spec']=spec;cfg['spec']['seeds']=spec['seeds']
    cfg['plan']=dict(arms=spec['arms'],orders=[spec['order_cycle'][i%6] for i in range(len(spec['seeds']))])
    cfg['source_hashes']=source_hashes();cfg['scipy_version']=scipy.__version__
    cfg.pop('fingerprint');cfg['fingerprint']=b.digest(cfg)
    return cfg


@contextmanager
def audit_sources():
    previous=p.FILES;p.FILES=FILES
    try:yield
    finally:p.FILES=previous


def train_all(root,cfg,device):
    folder=root/'training';folder.mkdir(exist_ok=True)
    with b.lock(folder):
        freeze=folder/'development_freeze.json'
        if freeze.exists():assert b.read(freeze)==cfg
        else:b.write(freeze,cfg)
        complete=all(b.verify_complete(folder/f'seed_{s}'/a,cfg) for s in cfg['spec']['seeds'] for a in cfg['plan']['arms'])
        if not complete:
            launch=folder/'launches.json';history=b.read(launch) if launch.exists() else []
            history.append(dict(id=uuid.uuid4().hex,unix_time=time.time()));b.write(launch,history)
            for i,seed in enumerate(cfg['spec']['seeds']):
                for position,arm in enumerate(cfg['plan']['orders'][i]):p.run_one(folder,cfg,seed,arm,position,device)
        with audit_sources():result=p.audit(folder,device)
        assert len(result['rows'])==len(cfg['spec']['seeds'])*3
        marker_hashes={f'seed_{s}/{a}/COMPLETE.json':b.sha(folder/f'seed_{s}'/a/'COMPLETE.json') for s in cfg['spec']['seeds'] for a in cfg['plan']['arms']}
        gate=dict(fingerprint=cfg['fingerprint'],audit_sha256=b.sha(folder/'audit.json'),markers=marker_hashes)
        path=root/'TRAINING_COMPLETE.json'
        if path.exists():assert b.read(path)==gate
        else:b.write(path,gate)


def check_gate(root,cfg):
    assert source_hashes()==cfg['source_hashes']
    assert b.read(root/'evaluation_freeze.json')==cfg
    gate=b.read(root/'TRAINING_COMPLETE.json');assert gate['fingerprint']==cfg['fingerprint']
    folder=root/'training';assert b.read(folder/'development_freeze.json')==cfg
    assert gate['audit_sha256']==b.sha(folder/'audit.json')
    audit=b.read(folder/'audit.json');assert audit['integrity']=='PASS' and not audit['test_access']
    expected={f'seed_{s}/{a}/COMPLETE.json' for s in cfg['spec']['seeds'] for a in cfg['plan']['arms']}
    assert set(gate['markers'])==expected and len(audit['rows'])==len(expected)
    for rel,h in gate['markers'].items():assert b.sha(folder/rel)==h


def smoke_test_data(cfg):
    gen=torch.Generator().manual_seed(291099);n=5;size=cfg['spec']['cache_size']
    return dict(images=torch.randint(0,256,(n,3,size,size),dtype=torch.uint8,generator=gen),
        labels=torch.arange(n)%cfg['spec']['classes'],ids=[f'artificial_test_{i}' for i in range(n)],validation=list(range(n)))


def prepare_test(root,cfg):
    check_gate(root,cfg)  # No test annotation/image read may precede this gate.
    if cfg['smoke']:
        cache=smoke_test_data(cfg)
        meta=dict(smoke=True,examples=len(cache['labels']),cache_sha256=b.tensor_hash(cache['images']),
            labels_sha256=b.tensor_hash(cache['labels']),preparation_seconds=0)
        folder=None
    else:
        folder=b.DATA/'official_test_evaluation_v1'
        with b.lock(folder):
            metadata=folder/'metadata.json';cachefile=folder/'cache.pt'
            if metadata.exists():
                meta=b.read(metadata);assert meta['fingerprint']==cfg['fingerprint']
                assert b.sha(cachefile)==meta['cache_sha256']
                assert b.sha(b.DATA/'oxford-iiit-pet/annotations/test.txt')==meta['annotation_sha256']
            else:
                assert not cachefile.exists(),'Preserve uncommitted test cache for review.'
                start=time.perf_counter();annotation=b.DATA/'oxford-iiit-pet/annotations/test.txt'
                records=[]
                for line in annotation.read_text(encoding='utf8').splitlines():
                    if not line.strip() or line.startswith('#'):continue
                    parts=line.split();assert len(parts)==4 and re.fullmatch(r'[A-Za-z0-9_]+',parts[0])
                    label=int(parts[1])-1;assert 0<=label<37
                    records.append((parts[0],label))
                assert len(records)==cfg['evaluation_spec']['official_test_examples']
                assert len({name for name,_ in records})==len(records)
                training=b.data(cfg);assert not {name for name,_ in records}&set(training['ids']);del training
                images=[];image_hashes={}
                for i,(name,label) in enumerate(records):
                    path=b.DATA/'oxford-iiit-pet/images'/f'{name}.jpg'
                    with Image.open(path) as raw:
                        im=b.TVF.center_crop(b.TVF.resize(raw.convert('RGB'),[cfg['spec']['cache_size']],antialias=True),[cfg['spec']['cache_size']]*2)
                        images.append(b.TVF.pil_to_tensor(im))
                    image_hashes[path.name]=b.sha(path)
                    if (i+1)%500==0:print(f'TEST CACHE {i+1}/{len(records)}',flush=True)
                payload=dict(images=torch.stack(images),labels=torch.tensor([y for _,y in records]),
                    ids=[name for name,_ in records],validation=list(range(len(records))))
                pending=folder/f'cache.pending_{uuid.uuid4().hex}.pt';torch.save(payload,pending);pending.replace(cachefile)
                del payload,images
                meta=dict(smoke=False,fingerprint=cfg['fingerprint'],examples=len(records),cache_sha256=b.sha(cachefile),
                    annotation_sha256=b.sha(annotation),image_hashes=image_hashes,preparation_seconds=time.perf_counter()-start)
                b.write(metadata,meta)
    path=root/'test_inputs.json'
    if path.exists():assert b.read(path)==meta
    else:b.write(path,meta)
    return meta


def test_data(cfg,meta):
    cache=smoke_test_data(cfg) if cfg['smoke'] else torch.load(b.DATA/'official_test_evaluation_v1/cache.pt',map_location='cpu',weights_only=True)
    assert len(cache['labels'])==meta['examples']
    return cache


def score_one(root,cfg,meta,seed,arm,device):
    parent=root/'test_scores'/f'seed_{seed}'/arm;parent.mkdir(parents=True,exist_ok=True)
    if b.verify_complete(parent,cfg):print('SKIP verified final scoring',seed,arm,flush=True);return
    training=b.verify_complete(root/'training'/f'seed_{seed}'/arm,cfg);assert training
    checkpoint_hash=b.sha(training/'final.pt')
    attempt=parent/f'attempt_{uuid.uuid4().hex}';attempt.mkdir()
    b.sync(device);start=time.perf_counter()
    cache=test_data(cfg,meta);model=b.make_model(seed,cfg,device)
    cp=torch.load(training/'final.pt',map_location='cpu',weights_only=True)
    assert cp['fingerprint']==cfg['fingerprint'] and cp['seed']==seed and cp['arm']==arm
    model.load_state_dict(cp['model'])
    common=b.evaluate(model,cache,'common_fp32',cfg,device);own=b.evaluate(model,cache,'bf16',cfg,device)
    assert torch.equal(common['labels'],own['labels'])
    np.savez_compressed(attempt/'logits.npz',common=common['logits'].numpy(),own=own['logits'].numpy(),labels=common['labels'].numpy())
    b.sync(device);elapsed=time.perf_counter()-start
    b.write(attempt/'summary.json',dict(seed=seed,arm=arm,fingerprint=cfg['fingerprint'],checkpoint_sha256=checkpoint_hash,
        cache_sha256=meta['cache_sha256'],test_assessment_seconds=elapsed,examples=len(cache['labels']),
        common_accuracy=common['accuracy'],own_accuracy=own['accuracy'],common_loss=common['loss'],own_loss=own['loss'],smoke=cfg['smoke']))
    b.write(parent/'COMPLETE.json',dict(fingerprint=cfg['fingerprint'],attempt=attempt.name,
        sha256={f.name:b.sha(f) for f in attempt.iterdir() if f.is_file()}))
    print('Final scoring saved',seed,arm,flush=True)  # No interim test accuracy.
    del cache,model,cp,common,own;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()


def audit_scores(root,cfg,meta):
    check_gate(root,cfg)
    for seed in cfg['spec']['seeds']:
        for arm in cfg['plan']['arms']:assert b.verify_complete(root/'test_scores'/f'seed_{seed}'/arm,cfg),'All scoring must complete before analysis'
    cache=test_data(cfg,meta);labels=cache['labels'].numpy().copy();del cache
    rows=[]
    for seed in cfg['spec']['seeds']:
        for arm in cfg['plan']['arms']:
            training=b.verify_complete(root/'training'/f'seed_{seed}'/arm,cfg);train=b.read(training/'summary.json')
            attempt=b.verify_complete(root/'test_scores'/f'seed_{seed}'/arm,cfg);score=b.read(attempt/'summary.json')
            assert score['seed']==seed and score['arm']==arm and score['fingerprint']==cfg['fingerprint']
            assert score['checkpoint_sha256']==b.sha(training/'final.pt') and score['cache_sha256']==meta['cache_sha256']
            assert score['examples']==len(labels) and np.isfinite(score['test_assessment_seconds']) and score['test_assessment_seconds']>0
            scores={}
            with np.load(attempt/'logits.npz') as z:
                assert np.array_equal(z['labels'],labels)
                for mode in ('common','own'):
                    x=z[mode].astype(np.float64);assert x.shape==(len(labels),cfg['spec']['classes']) and np.isfinite(x).all()
                    acc=float(np.mean(x.argmax(1)==labels));loss=float(np.mean(np.logaddexp.reduce(x,axis=1)-x[np.arange(len(labels)),labels]))
                    assert abs(acc-score[mode+'_accuracy'])<1e-7 and abs(loss-score[mode+'_loss'])<1e-5
                    scores[mode]=100*acc
            rows.append(dict(seed=seed,arm=arm,accuracy_percent=scores['common'],own_accuracy_percent=scores['own'],
                complete_seconds=train['complete_seconds'],training_seconds=train['training_seconds'],test_assessment_seconds=score['test_assessment_seconds']))
    training_audit=b.read(root/'training/audit.json')
    result=analyze(rows,cfg['evaluation_spec'],fragmented=training_audit['fragmented'],smoke=cfg['smoke'])
    result.update(integrity='PASS',rows=rows,test_examples=len(labels),source_fingerprint=cfg['fingerprint'],test_cache_preparation_seconds=meta['preparation_seconds'])
    b.write(root/'analysis.json',result)
    b.write(root/'COMPLETE.json',dict(fingerprint=cfg['fingerprint'],analysis_sha256=b.sha(root/'analysis.json'),
        training_gate_sha256=b.sha(root/'TRAINING_COMPLETE.json'),test_inputs_sha256=b.sha(root/'test_inputs.json'),
        scores={f'seed_{s}/{a}/COMPLETE.json':b.sha(root/'test_scores'/f'seed_{s}'/a/'COMPLETE.json') for s in cfg['spec']['seeds'] for a in cfg['plan']['arms']}))
    print('EVALUATION AUDIT PASS; all trajectories scored and verified.',flush=True)
    return result


def run(args):
    device=torch.device(args.device);b.configure(device);root=args.output or REAL
    if args.smoke:assert args.output and root.resolve()!=REAL.resolve()
    cfg=configuration(args.smoke,args.smoke and not args.native,device)
    with b.lock(root):
        freeze=root/'evaluation_freeze.json'
        if freeze.exists():assert b.read(freeze)==cfg,'Evaluation freeze differs; preserve and review.'
        else:b.write(freeze,cfg)
        train_all(root,cfg,device)
        meta=prepare_test(root,cfg)
        for i,seed in enumerate(cfg['spec']['seeds']):
            for arm in cfg['plan']['orders'][i]:score_one(root,cfg,meta,seed,arm,device)
        result=audit_scores(root,cfg,meta)
        print(b.json.dumps(dict(means=result['means'],claims=result['claims']),indent=2),flush=True)
        print('COMPLETE. Send '+str(root.resolve())+' for review.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['run'])
    parser.add_argument('--device',default='cuda');parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--native',action='store_true')
    run(parser.parse_args())
