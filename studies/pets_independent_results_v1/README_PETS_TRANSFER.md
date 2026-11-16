# Oxford Pet independent validation

The next run is **baseline qualification**, not the final publication experiment.
It evaluates full ResNet-18 adaptation under FP32/TF32 and BF16 using only an
internal split of Oxford-IIIT Pet trainval. No official test evaluation occurs.

From Windows CMD in the project:

```
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_pets_baseline.cmd
```

The launcher downloads the official dataset if absent (roughly 800 MB), builds
the trainval image cache, verifies the public pretrained weights, runs six
15-epoch baseline trajectories and audits the saved results. Allow several GB
of free space for the data, cache and checkpoints. It prints progress and the
final result folder. Close other GPU jobs while measuring. No dependencies are
installed automatically. Required: Python 3.11+, torch, torchvision, numpy, Pillow.

Send `results\pets_baseline_v1` when complete. If interrupted, use the same command:
complete verified runs are skipped, incomplete attempts are preserved, and the
timing record is flagged as fragmented for review. Do not delete failed outputs
or change the spec to make a readiness check pass.

This source package has no dependency on earlier project scripts. In another
checkout, the same relative folder structure works; the PYTHON variable may point
to any compatible installed environment. Downloaded weights and data are separate
from source distribution. Cache construction time is reported separately from
per-adaptation complete time. See protocol/pets_transfer/PROTOCOL_v1.md for the
fixed recipe, clock, prospective quality checks and subsequent evaluation plan.

The official test phase and selective candidate phase are not enabled yet. The
baseline review determines whether we have a suitable execution reference before
those phases are prepared. This preserves a genuinely separate evaluation stage.
