# Stage12A preparation validation — 2026-10-22

Five CPU and five CUDA tests passed. Coverage:

- Autograd cuts preserve logits and active-group gradients while inactive
  gradients remain None.
- After prior full updates, suffix-only steps leave prefix parameters, Adam
  moments/counters and weight decay unchanged; the next full step resumes the
  correct optimizer age. BatchNorm buffers remain fixed.
- Crop/flip streams replay exactly; ImageNet normalization checked.
- Periodic schedule, strong competing fixed-suffix control, quality margin and
  baseline-readiness logic checked.
- Native ResNet-18 backbone equals the downloaded checkpoint; torchvision's
  loader supplies missing legacy BatchNorm counters as zero. Native model and
  split-forward logits agree exactly. One CUDA full update with128artificial
  images at96px completes with finite weights and unchanged BN buffers.

The initial CPU test exposed missing legacy num_batches_tracked keys in the
published checkpoint. The test now checks torchvision's zero initialization
for those bookkeeping buffers and exact equality for all supplied backbone
weights. The native checkpoint and frozen old project code were not modified.

Complete artificial CPU/CUDA workflows passed: ten runs /240updates each, exact
head-warmup state/loss/metric agreement, matched data RNGs, frozen BN and inactive
parameters, all group optimizer counts, schedules, clocks and result arithmetic.
Artificial accuracy/timings do not establish efficacy or readiness. Completed
CPU resume re-audits without writes; all saved-file hashes are checked by the
preparation verifier. Rehearsals live in results/stage12_sanity_adaptation_v1_final_cpu
and results/stage12_sanity_adaptation_v1_final_gpu.

Official ImageNet checkpoint:46,830,571bytes, upstream SHA prefix f37072fd,
full SHA256 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec.
Source https://download.pytorch.org/models/resnet18-f37072fd.pth,
verified against torchvision ResNet18_Weights.IMAGENET1K_V1 URL. Full hash and
metadata are bound to the prospective freeze.

Freeze hash:4448fbf8945ec4f468aec644f513e63007f5c73515cf009e86ceddfc41275626.
Sources/spec/tests/protocol/launcher and preceding evidence are hash-bound.
960 prior immutable files verified. Only CURRENT_RELEASE.md and README_STAGE9.md
are intentionally updated existing pointers, with before/after hashes recorded.
Master7.4 is new; prior masters, all raw results and manuscript remain intact.

No real Stage12A training was executed. The real output directory must be absent
when preparation verification completes. The freeze and source manifest enable
rechecking this preparation against the user's eventual real run.

This96px/frozen-BN/12epoch experiment is a bounded opportunity screen, not a
fully tuned224px baseline or a novel method. Failed baseline readiness means
inconclusive; a passed screen still requires independent confirmation and
stronger competing recipes before publication claims.
