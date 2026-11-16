# Competitive attention reference: independent review — 2026-10-26

The completed screen passes both the frozen read-only audit and an independent
standard-library reconstruction. The established attention implementation is a
stronger cost baseline than native MultiheadAttention in this configuration.
Our unbind adapter has not demonstrated a consistent advantage over it.

## Main result

Percentages below are time reductions: median across four session medians,
each computed from six round-paired ratios. Positive favors the named candidate.

| Source seed | Batch | Our unbind vs native | Upstream reference vs native | Our unbind vs reference |
|---|---:|---:|---:|---:|
| 99171 | 256 | 7.31% | 9.68% | -2.47% |
| 99171 | 200 | -0.04% | 6.25% | -5.64% |
| 99172 | 256 | 6.82% | 9.19% | -2.80% |
| 99172 | 200 | 6.39% | 6.07% | 0.67% |

The reference beats native in all 16 session-condition medians and 85/96
individual paired rounds. Our adapter beats native in 14/16 session-condition
medians and 78/96 rounds. Against the reference, our adapter is faster in only
3/16 session-condition medians and 26/96 rounds. Three of four condition-level
medians favor the reference; the fourth gives our adapter a small 0.67% edge.
These are descriptive comparisons, without a confirmatory significance claim.

The seed99171/batch200 unbind-versus-native session medians are +2.04%, -2.12%,
-4.74% and +5.48%. This condition provides no stable saving in the current
screen. It must not be dropped or described as another universal 5–7% win.
The previously confirmed real-data 6.34–6.40% result remains valid under its
own protocol; the new variability limits extrapolation across runs/conditions.

## Numerical qualification and scope

All 48 arm-condition-session gates pass at the full shapes; no numerical
threshold or parameter exception changed. The preparation's tiny-model
reference failure did not recur here. Across three checked optimizer updates,
our unbind arm has zero recorded difference from native for logits, losses,
all gradients, post-update parameters and both Adam moments. Its 96 timed
native comparisons also match all 33 losses, final model/optimizer hashes and
RNG states exactly.

The upstream reference passes the same declared tolerances but is not bitwise
identical. Its worst normalized error is 0.858514 of the allowed limit, in
the third-update first-block packed projection bias (absolute difference
1.717034e-5). Maximum recorded timed loss difference from native is
4.768372e-7. Every arm repeats its own timed learning records exactly across
rounds and sessions. Final state hashes differ for the upstream reference;
this audit does not quantify final parameter differences after all 33 updates.
Three-update numerical qualification and tiny artificial loss differences do
not establish real-data accuracy equivalence for that reference.

The comparison uses the exact upstream timm v0.9.2 Attention class at pinned
commit 3d05c0e86f2f4c57bf495468aa3f8a7d3487c986 inside our otherwise unchanged
six-block ViT. Its adapter preserves the native parameter names/objects and
restores native output strides so downstream dropout masks remain aligned;
the compatibility copy is charged to its timing. This is not a whole-timm
model or latest optimized-stack benchmark. The upstream batch-first projection
and packing differ from our sequence-first adapter. Multiple implementation
differences are involved, so the extra benefit cannot be attributed solely
to unbind or removal of one copy.

## Mechanism, memory and measurement

Each arm dispatches six math-SDPA calls under strict deterministic FP32 with
TF32 disabled. Native has 19 SelectBackward0 / zero UnbindBackward0 nodes;
both other arms have one SelectBackward0 / six UnbindBackward0 nodes. The
reference therefore already implements the shared-split mechanism.

Our adapter's paired allocated peak and incremental memory equal native in
every trial. The reference's steady-update allocated peak and incremental
memory are lower in every comparison with native: approximately 112.1–114.6 MB
at batch200 and 153.2–154.1 MB at batch256 (decimal MB). This is observed
allocated memory under this artificial workload, not a general memory claim.
Full recorded memory components are retained in the trial export.

All six arm permutations occur once in every session/condition, giving each
arm eight trials in each position across four sessions. Position summaries
are retained; no timing outlier was removed. Preflight samples show low GPU
utilization; post-session snapshots show 63–70 C and approximately 2767–2775
MHz SM clock. Sparse telemetry cannot exclude within-session drift or every
external influence. Four sequential processes share one machine and repeat
two fixed artificial learning trajectories, not four independent model seeds.

The clock measures artificial full-model AdamW update execution. Setup,
warmup/restore and first-update times are separate; data generation, profiling,
correctness snapshots, hashes and serialization are outside the steady clock.
This is not complete adaptation time, original pretraining time or inference.

## Integrity and decision

288 trials, 9,504 measured updates, 1,440 discarded warmup updates, 336 separate
correctness updates and 48 mechanism probes are present. Four distinct
processes, no interrupted/replacement attempts, all planned orders, provenance,
source/configuration/completion hashes and arithmetic checks pass. All 5,534
preserved prior/raw files were verified. No experiment or training was rerun.

Retain our original improvement as a precisely scoped implementation/control
finding. This screen does not support making unbind the paper's novel main
method or claiming an advantage over established attention implementations.
The original failed 10% discovery gate remains unchanged. Its separate
confirmation and real-data results are not revoked or relabeled.

Recommended next step: a bounded optimized-execution feasibility study with
the upstream reference included as a required baseline. Specify precision,
numerical/quality rules, supported SDPA dispatch and compilation/startup
accounting prospectively; record unsupported configurations. Follow any useful
winner with matched real-data quality and complete-workflow timing before a
practical efficiency claim. The remaining-copy hypothesis must also compare
against this reference, which already avoids the old packed representation.
No new run is prepared in this review; publication remains on hold.
