"""Test donor isolation, null effects, logit decomposition and saved-step recovery."""
import argparse
from pathlib import Path
import unittest
import torch
import key_bias_function as f
from review_key_bias_function import verify_swaps

DEVICE=torch.device('cpu')

class Tests(unittest.TestCase):
    def test_exact_donors_and_immutability(self):
        a={'x.self_attention.in_proj_bias':torch.arange(9,dtype=torch.float32),'other':torch.tensor([3.,7.])}
        b={n:t+10 for n,t in a.items()};before=f.cpu([a,b]);parts,count=f.swapped(a,b)
        self.assertEqual(count,3);self.assertEqual(verify_swaps(a,b,parts),3)
        for old,new in zip(before,[a,b]):
            self.assertTrue(all(torch.equal(t,new[n]) for n,t in old.items()))
        parts['key_only']['x.self_attention.in_proj_bias'][0]+=1
        with self.assertRaises(AssertionError):verify_swaps(a,b,parts)

    def output(self,x,y,spec):
        x=torch.tensor(x,dtype=torch.float32)
        loss=float(f.F.cross_entropy(x,y,label_smoothing=spec['label_smoothing']))
        return dict(status='FINITE',logits=x,execution_loss=loss,scoring_loss=loss)

    def packet(self,ref,full,key,rest,y):
        block=dict(reference=ref,repeat=ref,candidates=dict(test=dict(candidate=full,key_only=key,key_restored=rest)))
        return dict(batches=dict(original=dict(y=y)),unchanged=dict(original=dict(reference_fp32_math=ref)),
                    contexts=dict(fresh=dict(outputs=dict(original=dict(reference_fp32_math=block)))))

    def test_null_effect_and_invalid_loss(self):
        spec=f.read(f.SPEC);y=torch.tensor([0,1]);ref=self.output([[1,2],[3,1]],y,spec)
        result=f.summarize(self.packet(ref,ref,ref,ref,y),dict(spec=spec))['comparisons'][0]
        self.assertEqual(result['full']['logit_l2'],0);self.assertIsNone(result['key_to_full_logit_l2'])
        self.assertEqual(result['interaction_loss_change'],0)
        broken=dict(ref,scoring_loss=ref['scoring_loss']+1)
        with self.assertRaises(AssertionError):f.score(broken,y,spec)
        native_gap=dict(ref,execution_loss=ref['execution_loss']+.1)
        self.assertAlmostEqual(f.score(native_gap,y,spec)['execution_minus_uniform_loss'],.1,places=6)

    def test_known_decomposition_and_prediction_change(self):
        spec=f.read(f.SPEC);y=torch.tensor([0,1])
        ref=self.output([[3,1],[1,3]],y,spec);key=self.output([[4,1],[1,3]],y,spec)
        rest=self.output([[3,1],[4,3]],y,spec);full=self.output([[4,1],[4,3]],y,spec)
        r=f.summarize(self.packet(ref,full,key,rest,y),dict(spec=spec))['comparisons'][0]
        self.assertAlmostEqual(r['full']['logit_l2'],10**.5);self.assertEqual(r['interaction_logit_l2'],0)
        self.assertEqual(r['full']['prediction_disagreements'],1);self.assertEqual(r['full']['correct_change'],-1)
        self.assertAlmostEqual(r['interaction_loss_change'],0,places=12)

    def test_saved_step_recovery_and_forward_only(self):
        cfg=f.configuration(True,DEVICE);record=cfg['records'][0];raw=torch.load(record['raw_path'],map_location='cpu',weights_only=False)
        states,proof=f.recovered(raw,'archived',cfg,DEVICE);batch=f.batches(record,cfg)['second']
        policy='reference_bf16_math' if DEVICE.type=='cuda' else 'reference_fp32_math'
        output=f.evaluate(states['reference_fp32_math'],record,policy,cfg,batch,DEVICE)
        self.assertEqual(output['status'],'FINITE');self.assertEqual(output['dispatch']['math'],2)
        self.assertEqual(output['model_sha256'],proof['reference_fp32_math']['model_sha256'])
        f.score(output,batch['y'],cfg['spec'])
        corrupt=f.cpu(raw);corrupt['replays']['archived']['reference_fp32_math']['end_optimizer_sha256']='corrupt'
        with self.assertRaises(AssertionError):f.recovered(corrupt,'archived',cfg,DEVICE)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path)
    args=ap.parse_args();DEVICE=torch.device(args.device);f.e.q.m.configure()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        assert not args.output.exists();f.save(args.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,device=str(DEVICE)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
