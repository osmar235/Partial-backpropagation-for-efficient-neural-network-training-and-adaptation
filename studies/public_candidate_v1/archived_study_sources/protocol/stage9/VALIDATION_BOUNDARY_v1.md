# Stage9C preparation validation — 2026-10-20

Real Stage9C seed1201/1202 training has NOT been started during preparation.
The user will launch the bounded diagnostic in CMD.

Five CPU tests passed in the existing project Python environment:

- Exact train-mode logits and full/suffix gradients for ALL six cuts of the
  native six-block ViT, with matched dropout and a nonzero head. The old cut2
  retains911,808 prefix parameters; cut0 includes token/position projection.
- Zero-gradient AdamW prefix updates match the analytical moment/decay step;
  freezing preserves weights/state/counter. Suffix updates match full AdamW.
- Calibration/scoring train and validation pools are disjoint and stratified.
- Thresholds fit calibration data only; scoring values do not change them.
  Rank/tie behavior is checked.
- Completed-file hashes/configuration are enforced; interrupted attempts
  remain separate instead of overwriting completed evidence.

Full artificial-data workflow passed on CPU AND the idle RTX5080 CUDA device.
These rehearsals use a tiny two-block model, two fake seeds/two snapshots,
two cut points and four probes per state. They are not scientific accuracy
or speed evidence. Native ALL-six-cut gradient correctness is CPU-tested;
the actual CUDA probe checks normal/split gradients at each real cut/state.

Outputs: `results/stage9_sanity_boundary_v1_final/` (CPU) and
`results/stage9_sanity_boundary_v1_gpu_ready/` (CUDA). Each contains64 branch
measurements and32 cost projections. Independent checks confirmed projection
arithmetic, finite/nonnegative costs/error norms and freeze displacement ratios.

A completed CPU rehearsal was rerun: both training trajectories and all four
probe blocks were skipped. All19 protected checkpoint/measurement/index/marker
files retained their hashes. No incomplete/completed scientific result was deleted.

During preparation, checkpoint loading was corrected to keep non-capturable
AdamW step counters on CPU rather than introducing CUDA synchronization into
the optimizer benchmark. This invariant is enforced when restoring state.
The startup utilization guard runs BEFORE GPU dataset preparation. Probes use
the next scheduled learning rate and verify the checkpoint update counters.

Execution-time fields in training/probe completion records include the work
inside those blocks (including warm-up/restoration/oracle scoring for probes),
but exclude initial dataset/model/probe-input preparation. They are not a
total process-duration or deployment-speed claim. The cost estimates are
explicitly microstep projections; no closed-loop adaptive policy has been trained.

Sources/spec/protocol are hashed into the real run configuration. Frozen
older studies and manuscript v7 are verified unchanged separately in
`boundary_preservation_check_v1.json`. Prepared files are listed in
`boundary_preparation_sha256_v1.csv`.
