"""Artificial-proof prototype of a frozen ViT with a trainable residual side net.

Inspired by side/ladder tuning; NOT an implementation of the published LAST/LST
algorithms and NOT a frozen real-data training recipe.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F


class ResidualSideNet(nn.Module):
    def __init__(self, core, side_width=48, taps=(2,4,6), final_features_only=False, seed=28001):
        super().__init__()
        assert core.variant=='full_rgb32' and not core.dual
        assert tuple(sorted(set(taps)))==tuple(taps) and taps[-1]==len(core.body.encoder.layers)
        assert side_width>0 and side_width%3==0
        self.body=copy.deepcopy(core.body)
        self.body.requires_grad_(False)
        self.head=copy.deepcopy(core.head)
        self.taps=tuple(taps);self.final_features_only=final_features_only
        width=core.head.in_features
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.projections=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,side_width)) for _ in taps])
            self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(side_width,3,4*side_width,
                dropout=0.,batch_first=True,norm_first=True) for _ in taps])
            self.side_norm=nn.LayerNorm(side_width)
            self.output=nn.Linear(side_width,width,bias=False)
            nn.init.zeros_(self.output.weight)

    def extract(self,images):
        # Dropout follows caller train/eval mode. Frozen parameters need not imply
        # eval mode. This preserves the source's stochastic-forward convention.
        features=[]
        def capture(module,inputs,output):features.append(output)
        handles=[self.body.encoder.layers[index-1].register_forward_hook(capture) for index in self.taps]
        try:
            with torch.no_grad():base=self.body(images)
        finally:
            for handle in handles:handle.remove()
        assert not base.requires_grad and all(not feature.requires_grad for feature in features)
        return base,features

    def forward(self,images):
        base,features=self.extract(images)
        # Same-depth control has exactly the same side parameters/operations,
        # but each ladder sees the final feature map instead of different depths.
        if self.final_features_only:features=[features[-1]]*len(features)
        side=None
        for feature,projection,block in zip(features,self.projections,self.blocks):
            z=projection(feature)
            side=block(z if side is None else side+z)
        correction=self.output(self.side_norm(side[:,0]))
        # Keep the source head's input layout unchanged at initialization.
        # Algebraically head(base + correction), with its bias counted once.
        return self.head(base)+F.linear(correction,self.head.weight,bias=None)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
