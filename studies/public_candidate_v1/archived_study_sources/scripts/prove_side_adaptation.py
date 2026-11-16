"""Structural proof on artificial weights/data only; no scientific efficacy screen."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import json
from pathlib import Path
import torch
import stage10_models as m
import stage10_representation as old
from side_adaptation_prototype import ResidualSideNet

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');args=ap.parse_args()
    m.configure();device=torch.device('cuda' if args.cuda else 'cpu')
    out=ROOT/f'results/side_adaptation_design_v1/artificial_{device.type}_v2.json'
    assert not out.exists()
    core=m.make_model('full_rgb32',1901,False)
    # Artificial nonzero head prevents the zero classifier from hiding side gradients.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(291001)
        with torch.no_grad():core.head.weight.normal_(std=.02)
    construction_rng=torch.get_rng_state().clone()
    candidate=ResidualSideNet(core)
    control=ResidualSideNet(core,final_features_only=True)
    frozen_reference=copy.deepcopy(core);frozen_reference.body.requires_grad_(False)
    assert torch.equal(construction_rng,torch.get_rng_state())
    assert old.state_hash(candidate.state_dict())==old.state_hash(control.state_dict())
    core.to(device);candidate.to(device);control.to(device);frozen_reference.to(device)
    gen=torch.Generator().manual_seed(291002)
    x=torch.randn(3,3,32,32,generator=gen).to(device);y=torch.tensor([1,2,3],device=device)
    initial_error={}
    assert old.state_hash(core.body.state_dict())==old.state_hash(candidate.body.state_dict())
    assert old.state_hash(core.head.state_dict())==old.state_hash(candidate.head.state_dict())
    for training in (False,True):
        outputs=[]
        for model in (core,frozen_reference,candidate,control):
            model.train(training);torch.manual_seed(291003)
            with torch.no_grad():outputs.append(model(x))
        # Freezing alone can select a numerically different execution path.
        # Require bitwise identity to native frozen control; separately compare
        # with the unfrozen source using the project's existing 1e-5 logit limit.
        assert torch.equal(outputs[1],outputs[2]) and torch.equal(outputs[1],outputs[3])
        error=float((outputs[0]-outputs[2]).abs().max())
        assert error<=1e-5 and torch.equal(outputs[0].argmax(1),outputs[2].argmax(1))
        initial_error['train' if training else 'eval']=error
    # Frozen source has no graph even if inputs request gradients.
    candidate.train();base,features=candidate.extract(x.detach().requires_grad_(True))
    assert len(features)==3 and all(z.shape==(3,65,192) and z.grad_fn is None for z in features)
    assert base.grad_fn is None
    results=[]
    for arm_name,model in [('multi_depth',candidate),('final_depth_control',control)]:
        backbone_before=old.state_hash(model.body.state_dict())
        head_before=old.state_hash(model.head.state_dict())
        side_before={name:p.detach().clone() for name,p in model.named_parameters() if not name.startswith(('body.','head.'))}
        opt=torch.optim.AdamW(model.trainable_parameters(),lr=1e-3,weight_decay=.05)
        assert not any(id(p) in {id(z) for z in model.body.parameters()} for group in opt.param_groups for p in group['params'])
        old.update(model,opt,x,y,291004,.1)
        assert model.output.weight.grad is not None and float(model.output.weight.grad.norm())>0
        # Zero output projection initially blocks upstream side gradients by design.
        upstream=[p for name,p in model.named_parameters() if name.startswith(('projections.','blocks.','side_norm.'))]
        assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in upstream)
        old.update(model,opt,x,y,291005,.1)
        assert all(p.grad is None for p in model.body.parameters())
        assert old.state_hash(model.body.state_dict())==backbone_before
        assert old.state_hash(model.head.state_dict())!=head_before
        active=[]
        for name,p in model.named_parameters():
            if name.startswith('body.'):continue
            assert p.grad is not None and bool(torch.isfinite(p.grad).all())
            if name.startswith(('projections.','blocks.')):active.append(float(p.grad.norm()))
        assert all(value>0 for value in active)
        assert any(not torch.equal(side_before[name],p) for name,p in model.named_parameters() if name in side_before)
        assert len(opt.state)==len(model.trainable_parameters())
        assert all(int(opt.state[p]['step'])==2 for p in model.trainable_parameters())
        # Optimizer/checkpoint round-trip; same next stochastic update reproduces.
        twin=copy.deepcopy(model);other=torch.optim.AdamW(twin.trainable_parameters(),lr=1e-3,weight_decay=.05)
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        assert old.update(model,opt,x,y,291006,.1)==old.update(twin,other,x,y,291006,.1)
        assert old.state_hash(model.state_dict())==old.state_hash(twin.state_dict())
        assert old.state_hash(model.body.state_dict())==backbone_before
        results.append(dict(arm=arm_name,trainable_parameters=sum(p.numel() for p in model.trainable_parameters()),
            frozen_parameters=sum(p.numel() for p in model.body.parameters()),
            backbone_unchanged=True,backbone_gradients_absent=True,upstream_side_gradients_active_on_second_step=True,
            optimizer_roundtrip_identical=True))
    assert results[0]['trainable_parameters']==results[1]['trainable_parameters']
    result=dict(status='PASS',date='2026-10-25',synthetic=True,device=str(device),
        architecture='six-block width-192 ViT; three width-48 side blocks; 65 tokens everywhere',
        source_initial_logits_max_abs_error=initial_error,native_frozen_initial_logits_bitwise_identical=True,
        source_initial_predictions_identical=True,side_variants_initial_states_identical=True,
        artificial_updates_per_arm=3,results=results,real_checkpoints_loaded=False,real_data_loaded=False,
        efficacy_measured=False,speed_measured=False,real_study_frozen=False)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
