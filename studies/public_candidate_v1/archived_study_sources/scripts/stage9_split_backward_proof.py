"""CPU correctness proof for deciding on prefix backward AFTER exact suffix backward."""
import copy,json,unittest
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import stage8_vit_v2 as v


def split_forward(model,x,blocks=2):
    tokens=model._process_input(x)
    tokens=torch.cat([model.class_token.expand(len(x),-1,-1),tokens],dim=1)
    h=model.encoder.dropout(tokens+model.encoder.pos_embedding)
    layers=list(model.encoder.layers)
    for layer in layers[:blocks]:h=layer(h)
    bridge=h.detach().requires_grad_(True)
    z=bridge
    for layer in layers[blocks:]:z=layer(z)
    return model.heads(model.encoder.ln(z)[:,0]),h,bridge


class Proof(unittest.TestCase):
    def test_train_mode_exactness_and_delayed_prefix_backward(self):
        torch.set_num_threads(1);v.configure();v.seed_all(12031)
        reference=v.make_model();reference.train()
        nn.init.normal_(reference.heads.head.weight,std=.02)
        model=copy.deepcopy(reference)
        x=torch.rand(3,3,32,32);labels=torch.tensor([0,1,2])
        x,targets,_=v.mix_batch(x,labels,v.np.random.default_rng(5),torch.Generator().manual_seed(6),.2,10)
        torch.manual_seed(12032)
        logits=reference(x);loss=F.cross_entropy(logits,targets,label_smoothing=.1);loss.backward()
        torch.manual_seed(12032)
        actual,h,bridge=split_forward(model,x)
        self.assertTrue(torch.equal(actual,logits))
        F.cross_entropy(actual,targets,label_smoothing=.1).backward()
        prefix,suffix,names=v.partition(model,2)
        reference_by_name=dict(reference.named_parameters())
        self.assertTrue(all(p.grad is None for p in prefix))
        self.assertIsNotNone(bridge.grad)
        self.assertTrue(bool(torch.isfinite(bridge.grad).all()))
        self.assertGreater(float(bridge.grad.norm()),0.)
        # Gate could inspect bridge.grad here. No prefix Jacobian has been traversed yet.
        for name,p in model.named_parameters():
            if name not in names:self.assertTrue(torch.equal(p.grad,reference_by_name[name].grad),name)
        h.backward(bridge.grad)
        self.assertTrue(any(float(p.grad.norm())>0 for p in prefix))
        for name,p in model.named_parameters():
            self.assertTrue(torch.equal(p.grad,reference_by_name[name].grad),name)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Proof))
    if not result.wasSuccessful():raise SystemExit(1)
    root=Path(__file__).resolve().parents[1];out=root/'results/stage9B_split_backward_proof_v1'
    out.mkdir(parents=True,exist_ok=True)
    record=dict(status='PASS',device='cpu',data='artificial',native_vit_six_blocks=True,
        matched_train_mode_dropout=True,exact_logits=True,exact_suffix_gradients=True,
        prefix_gradient_absent_until_explicit_backward=True,exact_full_gradients_after_prefix_backward=True,
        new_controller_tested=False,net_gpu_savings_established=False,
        note='Correctness/plumbing proof only; no gradient-importance proxy, training improvement or novelty is established.')
    (out/'proof.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record,indent=2))
