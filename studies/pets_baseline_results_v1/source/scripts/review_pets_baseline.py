"""Read-only review of completed Pet baseline trajectories; no training/test access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import pets_transfer as m


def review(root, output):
    assert not output.exists(), 'Use a new review directory; preserve previous reviews.'
    before = {str(p): m.sha(p) for p in root.rglob('*') if p.is_file() and p.name != 'RUN.lock'}
    cfg = m.read(root/'baseline_freeze.json')
    assert not cfg['smoke'] and not cfg['tiny']
    assert cfg['spec'] == m.read(m.SPEC)
    assert cfg['fingerprint'] == m.digest({k:v for k,v in cfg.items() if k != 'fingerprint'})
    assert m.sources() == cfg['source_hashes']
    meta = m.read(m.DATA/'cache_metadata.json')
    assert m.sha(m.DATA/'cache_metadata.json') == cfg['cache_metadata_sha256']
    for path, key in ((m.DATA/'trainval_cache.pt','cache_sha256'),(m.WEIGHTS,'weights_sha256')):
        assert m.sha(path) == cfg['inputs'][key] == meta[key]
    assert not meta['test_read'] and meta['examples'] == 3680
    archives = {}
    for name, expected in [('images','5c4f3ee8e5d25df40f4fd59a7f44e54c'),('annotations','95a8c909bbe2e81eed6a22bccdf3f68f')]:
        path=m.DATA/'oxford-iiit-pet'/f'{name}.tar.gz'
        with path.open('rb') as stream: actual=hashlib.file_digest(stream,'md5').hexdigest()
        assert actual == expected
        archives[name] = dict(bytes=path.stat().st_size, md5=actual)
    assert m.sha(m.DATA/'oxford-iiit-pet/annotations/trainval.txt') == meta['trainval_annotation_sha256']
    for name, h in meta['trainval_image_hashes'].items():
        assert m.sha(m.DATA/'oxford-iiit-pet/images'/name) == h
    cache=m.data(cfg)
    train,val=m.split_indices(cache['labels'].tolist(),cfg['spec'])
    assert train == cache['train'] and val == cache['validation']
    assert m.digest(dict(train=train,validation=val)) == cfg['inputs']['split_hash']
    assert set(train).isdisjoint(val) and sorted(train+val)==list(range(3680))
    del cache
    captured={}
    original_write=m.write
    def capture(path, value):
        assert Path(path)==root/'audit.json'
        captured.update(value)
    m.write=capture
    try:
        with m.lock(root): m.audit(root,torch.device('cpu'),rescore=False)
    finally:
        m.write=original_write
    previous=m.read(root/'audit.json')
    assert {k:v for k,v in previous.items() if k!='rescores'} == {k:v for k,v in captured.items() if k!='rescores'}
    assert len(previous['rescores'])==12 and all(x['max_error']==0 for x in previous['rescores'])
    assert captured['qualified'] and not captured['fragmented']
    rows=[]
    for seed in cfg['spec']['seeds']:
        initial=m.make_model(seed,cfg,torch.device('cpu')).state_dict()
        for policy in cfg['spec']['policies']:
            parent=root/f'seed_{seed}'/policy
            attempt=m.verify_complete(parent,cfg)
            summary=m.read(attempt/'summary.json')
            cp=torch.load(attempt/'final.pt',map_location='cpu',weights_only=True)
            assert m.state_hash(initial)==cp['initial_hash']==summary['initial_hash']
            for name,value in initial.items():
                if 'running_' in name or 'num_batches_tracked' in name:
                    assert torch.equal(value,cp['model'][name]), name
            with np.load(attempt/'logits.npz') as logits:
                counts={name:int((logits[name].argmax(1)==logits['labels']).sum()) for name in ('common','own')}
                n=len(logits['labels'])
            assert n==736 and summary['updates']==690 and not summary['test_access']
            rows.append(dict(seed=seed,policy=policy,common_correct=counts['common'],own_correct=counts['own'],
                validation_n=n,common_accuracy_percent=100*counts['common']/n,own_accuracy_percent=100*counts['own']/n,
                training_seconds=summary['training_seconds'],complete_seconds=summary['complete_seconds']))
    means={policy:{key:float(np.mean([r[key] for r in rows if r['policy']==policy]))
        for key in ('common_accuracy_percent','own_accuracy_percent','training_seconds','complete_seconds')}
        for policy in cfg['spec']['policies']}
    after={str(p):m.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='RUN.lock'}
    assert before==after
    output.mkdir(parents=True)
    result=dict(status='PASS',qualified=True,rows=rows,means=means,pairs=captured['pairs'],archives=archives,
        baseline_files_unchanged=len(before),audit_repeated_without_training=True,
        saved_checkpoint_rescores=dict(count=12,max_error=0,origin='original completed-run audit; not repeated in this review'),
        test_access=False,training_runs_this_review=0,cache_preparation_seconds=meta['cache_and_weights_preparation_seconds'],
        download_seconds_note='Metadata download_seconds excludes the separate recovery downloader and its failed attempts.',
        next_step='Prepare fixed BF16 full/frozen/periodic-r4 trainval development comparison; official test remains disabled.')
    m.write(output/'verification.json',result)
    m.write(output/'baseline_inputs_sha256.json',before)
    m.write(output/'recomputed_audit.json',captured)
    print(json.dumps(dict(status='PASS',means=means,files_unchanged=len(before)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=m.REAL)
    parser.add_argument('--output',type=Path,default=m.ROOT/'results/pets_baseline_review_v1')
    args=parser.parse_args()
    m.configure(torch.device('cpu'))
    review(args.input.resolve(),args.output.resolve())
