"""Access gate, test-cache integrity and prespecified inference checks."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import pets_evaluation as e
from analyze_pets_evaluation import analyze,interval

DEVICE=torch.device('cpu')


def fixture_rows(spec,periodic_accuracy=89.8,periodic_time=8):
    return [dict(seed=s,arm=a,accuracy_percent=90 if a=='full' else (89.9 if a=='frozen' else periodic_accuracy),
        own_accuracy_percent=90,complete_seconds=10 if a=='full' else (7 if a=='frozen' else periodic_time),
        training_seconds=5,test_assessment_seconds=1) for s in spec['seeds'] for a in spec['arms']]


class Tests(unittest.TestCase):
    def test_gate_blocks_before_test_data(self):
        cfg=e.configuration(True,True,DEVICE);cfg['smoke']=False
        with tempfile.TemporaryDirectory() as td,patch.object(e.b,'data') as data,patch.object(e.Image,'open') as image:
            with self.assertRaises(FileNotFoundError):e.prepare_test(Path(td),cfg)
            data.assert_not_called();image.assert_not_called()

    def test_real_parser_on_artificial_files_and_hash_guard(self):
        cfg=e.configuration(True,True,DEVICE);cfg['smoke']=False
        cfg['evaluation_spec']['official_test_examples']=4
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fake=root/'data';images=fake/'oxford-iiit-pet/images';annotations=fake/'oxford-iiit-pet/annotations'
            images.mkdir(parents=True);annotations.mkdir()
            (annotations/'test.txt').write_text('\n'.join(f'fake_{i} {i%3+1} 1 1' for i in range(4)),encoding='utf8')
            for i in range(4):Image.new('RGB',(31,25),(i*30,50,100)).save(images/f'fake_{i}.jpg')
            with patch.object(e.b,'DATA',fake),patch.object(e,'check_gate'),patch.object(e.b,'data',return_value={'ids':['training_only']}):
                meta=e.prepare_test(root,cfg);self.assertEqual(meta['examples'],4)
                cache=e.test_data(cfg,meta);self.assertEqual(tuple(cache['images'].shape),(4,3,20,20))
                self.assertEqual(cache['labels'].tolist(),[0,1,2,0])
                self.assertEqual(e.prepare_test(root,cfg),meta)
                path=fake/'official_test_evaluation_v1/cache.pt'
                with path.open('ab') as f:f.write(b'tampered')
                with self.assertRaises(AssertionError):e.prepare_test(root,cfg)

    def test_test_ids_cannot_overlap_training(self):
        cfg=e.configuration(True,True,DEVICE);cfg['smoke']=False;cfg['evaluation_spec']['official_test_examples']=1
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fake=root/'data';a=fake/'oxford-iiit-pet/annotations';a.mkdir(parents=True)
            (a/'test.txt').write_text('shared_1 1 1 1\n',encoding='utf8')
            with patch.object(e.b,'DATA',fake),patch.object(e,'check_gate'),patch.object(e.b,'data',return_value={'ids':['shared_1']}),patch.object(e.Image,'open') as image:
                with self.assertRaises(AssertionError):e.prepare_test(root,cfg)
                image.assert_not_called()

    def test_claim_needs_both_endpoints(self):
        spec=e.b.read(e.SPEC)
        self.assertTrue(analyze(fixture_rows(spec),spec)['claims']['periodic_r4']['combined_claim'])
        self.assertFalse(analyze(fixture_rows(spec,periodic_accuracy=89.5),spec)['claims']['periodic_r4']['combined_claim'])
        self.assertFalse(analyze(fixture_rows(spec,periodic_time=11),spec)['claims']['periodic_r4']['combined_claim'])

    def test_fragmented_and_smoke_never_claim(self):
        spec=e.b.read(e.SPEC);rows=fixture_rows(spec)
        for result in (analyze(rows,spec,fragmented=True),analyze(rows,spec,smoke=True)):
            self.assertFalse(any(c['combined_claim'] for c in result['claims'].values()))

    def test_uncertainty_not_just_mean_and_complete_pairs(self):
        spec=e.b.read(e.SPEC);rows=fixture_rows(spec)
        for r in rows:
            if r['arm']=='periodic_r4':r['accuracy_percent']=89.75+(.5 if r['seed']%2 else -.5)
        result=analyze(rows,spec)
        self.assertGreater(result['claims']['periodic_r4']['accuracy']['mean'],-.3)
        self.assertFalse(result['claims']['periodic_r4']['quality_bound_pass'])
        with self.assertRaises(AssertionError):analyze(rows[:-1],spec)
        bounds=interval([1,2,3,4,5,6])
        self.assertAlmostEqual(bounds['mean'],3.5)
        self.assertAlmostEqual(bounds['half_width'],1.963314306980323,places=6)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path)
    args=parser.parse_args();DEVICE=torch.device(args.device);e.b.configure(DEVICE)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:e.b.write(args.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,device=str(DEVICE),real_test_access=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
