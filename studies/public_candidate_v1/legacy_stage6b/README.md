# Loss-Gated Partial Backpropagation with Adam Moment Continuation

This repository contains the code, frozen protocol, experiment summaries, 
confirmatory analysis, figures, and manuscript-support materials associated with:

> **Loss-Gated Partial Backpropagation with Adam Moment Continuation for
> Efficient Neural Network Training**  
> Osmar Pinto Neto

LGPB-MC uses the ordinary mini-batch forward loss to decide whether to truncate
the backward pass through a fixed early prefix. On a partial update, the suffix
receives exact fresh gradients. The omitted prefix gradient is represented as
zero to Adam, so its moments and bias-correction step continue. Every partial
update forces the next update to be full.

## Confirmatory result

The frozen nine-seed Stage 6B experiment used augmented CIFAR-10,
ResNet-18-GroupNorm, 30 epochs, and an NVIDIA GeForce RTX 5080.

- Final accuracy versus Adam: **-0.548 percentage point**
  (95% CI -0.759 to -0.337).
- The pre-specified non-inferiority margin was -1.0 percentage point: **pass**.
- Candidate/Adam training-wall-time ratio: **0.8976**
  (95% CI 0.8743 to 0.9215), corresponding to a **10.24% reduction**.
- All nine held-out seeds were faster and within one accuracy point of Adam.
- Candidate/tuned-Adan wall-time ratio: **0.8791**; mean accuracy difference:
  **+2.424 percentage points**. This was a confirmatory paired comparison on the
  same held-out seed block; no separate non-inferiority margin was pre-specified
  for Adan.

These results support a workload-specific training-efficiency claim. They do
not establish universal optimizer superiority across datasets, architectures,
hardware, or training systems.

## Repository contents

| Path | Contents |
|---|---|
| `src/` | Final training harness and the local dependency modules it imports |
| `src/third_party/Adan/` | Adan implementation and its Apache-2.0 license |
| `scripts/` | Portable Windows launchers, confirmatory analysis, and checksums |
| `protocol/` | Stage 6 preregistration, development decisions, and project record |
| `results/stage5f3/` | Frozen short-horizon held-out failure evidence, seeds 31-39 |
| `results/stage6_development/` | Seed-40 development, timing qualification, and Adan tuning records |
| `results/stage6b/` | Raw confirmatory outputs for seeds 41-49 and aggregate statistics |
| `manuscript/` | Bibliography, publication figures, and reproducible figure source |

CIFAR-10 itself is not redistributed. The training script downloads it through
`torchvision` when needed.

## Verify the reported aggregate

From the repository root:

```powershell
python -B scripts/analyze_stage6b.py
```

This reads only `results/stage6b/stage6b_heldout_seed41` through
`stage6b_heldout_seed49` and rewrites:

- `results/stage6b/stage6b_heldout_aggregate/paired_results.csv`
- `results/stage6b/stage6b_heldout_aggregate/summary.json`
- `results/stage6b/stage6b_heldout_aggregate/REPORT.md`

The analysis uses a paired two-sided t interval with eight degrees of freedom.
Timing ratios are analyzed on the log scale and exponentiated.

## Environment

The recorded experiment environment was:

- Python 3.14.3
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- NumPy 2.4.3
- Matplotlib 3.10.8
- NVIDIA GeForce RTX 5080

See [`ENVIRONMENT.md`](ENVIRONMENT.md) for installation and hardware notes.

## Smoke test

The following command checks imports, model construction, all three optimizer
paths, reporting, and a tiny fake-data training run without downloading
CIFAR-10:

```powershell
python -B src/agrls_stage6a_cifar10_augmented_schedule.py `
  --dataset fake --train-subset 16 --eval-subset 8 --batch-size 8 `
  --epochs 1 --schedule-epochs 1 --seed 0 --device cpu `
  --adam-lr 0.0003 --adan-lr 0.001 --min-lr-ratio 0.01 `
  --ema-decay 0.95 --warmup-steps 0 `
  --algorithms adam,loss_gated_moment_continuation_adam,adan `
  --eval-batch-size 8 --out smoke_test
```

## Re-run Stage 6B

The PowerShell launcher reproduces the pre-specified Latin-square execution
order. Existing completed seed directories are skipped unless `-Force` is
given.

```powershell
$env:PYTHON = "C:\path\to\venv\Scripts\python.exe"
powershell -ExecutionPolicy Bypass -File scripts/run_stage6b_heldout.ps1
```

Re-running all nine seeds is computationally expensive and requires a CUDA GPU.
Wall-time results are hardware and software-stack dependent.

## Evidence chronology

The negative short-horizon Stage 5F3 result is retained in
`results/stage5f3/`. The Stage 6 controller was frozen before the confirmatory
seeds were opened. Seed 40 was used only for protocol development, timing
qualification, and Adan learning-rate selection. Seeds 41-49 were the final
held-out confirmatory block; all seeds 0-49 are now considered seen.

Read `protocol/STAGE6_PREREGISTRATION.md` before interpreting or extending the
study.

## Citation and archival DOI

Citation metadata, including the author's email and ORCID, are supplied in
`CITATION.cff`. After the first GitHub release is archived by Zenodo, add the
repository URL, Zenodo concept DOI, and version DOI to the citation metadata
and this README. See `UPLOAD_CHECKLIST.md`.

## Licenses

Original software in `src/`, `scripts/`, and the figure-generation script is
licensed under Apache-2.0. Original protocols, result records, figures, and
documentation are licensed under CC BY 4.0. The bundled Adan implementation
retains its upstream Apache-2.0 license and notices. See `LICENSE.md` for the
complete path-based mapping and the accompanying full license texts.

The first public release intentionally omits the manuscript DOCX and PDF. They
will be added after the repository has been archived by Zenodo and the final
manuscript has been updated with the repository URL and persistent DOI. Author
metadata, funding, and competing-interest declarations are already complete.
