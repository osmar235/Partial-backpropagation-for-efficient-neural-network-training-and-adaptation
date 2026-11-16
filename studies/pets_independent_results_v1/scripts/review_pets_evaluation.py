"""Read-only final evaluation audit with complete checkpoint rescoring."""
from pathlib import Path
import gc
import numpy as np
import torch
import pets_evaluation as e
from reproduce_pets_evaluation import reproduce


def snapshot(root):return {str(p):e.b.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='RUN.lock'}


def main():
    root=e.REAL;out=e.b.ROOT/'results/pets_evaluation_review_v1'
    assert not out.exists(),'Preserve completed reviews.'
    device=torch.device('cuda');e.b.configure(device)
    with e.b.lock(root):
        before=snapshot(root);cfg=e.configuration(False,False,device)
        assert e.b.read(root/'evaluation_freeze.json')==cfg
        marker=e.b.read(root/'COMPLETE.json');assert marker['fingerprint']==cfg['fingerprint']
        for key,path in [('analysis_sha256','analysis.json'),('training_gate_sha256','TRAINING_COMPLETE.json'),('test_inputs_sha256','test_inputs.json')]:assert e.b.sha(root/path)==marker[key]
        expected={f'seed_{s}/{a}/COMPLETE.json' for s in cfg['spec']['seeds'] for a in cfg['plan']['arms']}
        assert set(marker['scores'])==expected
        for rel,h in marker['scores'].items():assert e.b.sha(root/'test_scores'/rel)==h
        meta=e.b.read(root/'test_inputs.json')
        datafolder=e.b.DATA/'official_test_evaluation_v1'
        assert meta==e.b.read(datafolder/'metadata.json')
        assert e.b.sha(datafolder/'cache.pt')==meta['cache_sha256']
        annotation=e.b.DATA/'oxford-iiit-pet/annotations/test.txt'
        assert e.b.sha(annotation)==meta['annotation_sha256']
        for name,h in meta['image_hashes'].items():assert e.b.sha(e.b.DATA/'oxford-iiit-pet/images'/name)==h
        cache=e.test_data(cfg,meta);training=e.b.data(cfg)
        assert not set(cache['ids']) & set(training['ids']);del training
        records=[line.split() for line in annotation.read_text(encoding='utf8').splitlines() if line.strip() and not line.startswith('#')]
        assert cache['ids']==[r[0] for r in records] and cache['labels'].tolist()==[int(r[1])-1 for r in records]
        assert len(cache['ids'])==len(set(cache['ids']))==3669
        captured={};original_write=e.b.write
        allowed={root/'training/audit.json',root/'analysis.json',root/'COMPLETE.json'}
        def capture(path,value):
            path=Path(path);assert path in allowed
            assert value==e.b.read(path),path
            captured[str(path)]=True
        e.b.write=capture
        try:
            print('Rechecking all training checkpoints and optimizer/data streams.',flush=True)
            with e.audit_sources():training_audit=e.p.audit(root/'training',device)
            print('Recomputing official-test saved-logit analysis.',flush=True)
            analysis=e.audit_scores(root,cfg,meta)
        finally:e.b.write=original_write
        assert len(captured)==3 and len(training_audit['rescores'])==144
        assert len(analysis['rows'])==72 and not analysis['fragmented'] and not analysis['smoke']
        rescores=[]
        for i,seed in enumerate(cfg['spec']['seeds']):
            for arm in cfg['plan']['arms']:
                training_parent=root/'training'/f'seed_{seed}'/arm
                trained=training_parent/e.b.read(training_parent/'COMPLETE.json')['attempt']
                scoring_parent=root/'test_scores'/f'seed_{seed}'/arm
                scored=scoring_parent/e.b.read(scoring_parent/'COMPLETE.json')['attempt']
                cp=torch.load(trained/'final.pt',map_location='cpu',weights_only=True)
                model=e.b.make_model(seed,cfg,device);model.load_state_dict(cp['model'])
                with np.load(scored/'logits.npz') as z:
                    for mode,name in [('common_fp32','common'),('bf16','own')]:
                        pred=e.b.evaluate(model,cache,mode,cfg,device)['logits'].numpy()
                        error=float(np.max(np.abs(pred-z[name])))
                        assert error<=1e-5 and np.array_equal(pred.argmax(1),z[name].argmax(1)),(seed,arm,name,error)
                        rescores.append(dict(seed=seed,arm=arm,mode=mode,max_error=error))
                del model,cp;gc.collect()
            print(f'Official-test checkpoint rescores: {i+1}/24 seeds',flush=True)
        portable=reproduce(root)
        assert portable['claims']==analysis['claims']
        after=snapshot(root);assert before==after
    out.mkdir()
    e.b.write(out/'evaluation_inputs_sha256.json',before)
    e.b.write(out/'test_checkpoint_rescores.json',rescores)
    e.b.write(out/'analysis.json',analysis)
    e.b.write(out/'verification.json',dict(status='PASS',seeds=24,trajectories=72,test_examples=3669,
        training_checkpoint_rescores=144,test_checkpoint_rescores=144,
        max_training_logit_error=max(r['max_error'] for r in training_audit['rescores']),max_test_logit_error=max(r['max_error'] for r in rescores),
        unchanged_files=len(before),fragmented=False,training_updates_this_review=0,portable_analysis='PASS'))
    print('PASS: full final audit; all scientific records unchanged.',flush=True)


if __name__=='__main__':main()
