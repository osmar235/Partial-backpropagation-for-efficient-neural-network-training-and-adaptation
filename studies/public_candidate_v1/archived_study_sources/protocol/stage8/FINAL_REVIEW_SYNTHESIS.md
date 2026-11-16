# Stage8 final synthesis — 2026-10-20

All24 frozen evaluation runs completed and passed an independent audit of
1,411,200 updates. Sources, settings and completed development inputs match
the freeze. All five planned effects and15 target contrasts were independently
recomputed. Initial states, gate/EMA/LR/MixUp traces, masks, timings, finite
checkpoint states and continuation/replay counters were verified.

| Arm | Mean final test accuracy | Mean training wall | Partial updates |
|---|---:|---:|---:|
|Full AdamW|86.402%|1661.74s|0%|
|LGPB-MC|71.697%|1517.68s|39.753%|
|Periodic|85.175%|1517.82s|39.784%|
|Replay|72.058%|1544.11s|39.735%|

LGPB/full accuracy difference: **-14.705pp**,95% CI[-15.865,-13.545].
Training-wall ratio:0.9133[0.9065,0.9203], or8.67% less wall. The new joint
primary criterion **failed**: accuracy failed, speed passed. All six LGPB
runs were faster and less accurate. Original Stage6 inference is unchanged.

LGPB/periodic accuracy difference:-13.478pp[-14.608,-12.349]; wall
ratio0.9999[0.9985,1.0013]. Periodic itself lost1.227pp against full; no new
confirmatory equivalence/non-inferiority designation is assigned to it.
Replay retained a14.343pp deficit; LGPB/replay accuracy difference was uncertain.

LGPB required7.61x baseline training wall to70%[6.45,8.99]. LGPB/replay
reached neither80% nor85% in any seed. Periodic reached80% in6/6 and85%
at some epoch in5/6, but took1.22x baseline wall to80%[1.06,1.40]. Target
crossings are not final accuracy; unreached targets remain unestimated.

The periodic comparison is adverse evidence for the loss-informed policy in
this ViT/prefix/MixUp recipe. It does not isolate one cause or prove failure
of all transformers, prefixes or augmentation schemes. Weak mixing is
associated with loss-gated partial decisions; the association is descriptive.
Control/target intervals are unadjusted. There is no no-MixUp/common-mask arm.

All telemetry/seeds remain. Full baseline seed1101 took1686.58s versus
1655.56--1660.33s in the other baselines; its first/last50-epoch means were
5.823/5.520s. Drift and startup idle clocks are not exclusion criteria.
Training-only and per-run elapsed costs remain distinct and unpooled with
older phases. No energy-saving inference is made.

## Decision

All planned scientific training is complete. Retain the failed v1 screen,
prospective amendment, reused development validation, adverse calibration
and all24 evaluation trajectories. No third recipe, replacement seeds or
hidden tuning. The second architecture broadens the tested scope and
identifies a failure boundary, not another successful workload.

Current manuscript v7 incorporates this result in abstract, methods, results,
discussion, limitations and conclusion while retaining all prior numerical
evidence. Next: author review, current editorial files and a combined dated
code/data archive with verified public identifiers. No public release,
DOI, submission event or acceptance is implied.
