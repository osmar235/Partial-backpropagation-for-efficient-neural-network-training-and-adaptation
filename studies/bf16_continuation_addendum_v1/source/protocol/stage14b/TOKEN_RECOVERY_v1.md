# Stage14B: fixed-budget recovery after token merging

Prospective protocol, 2026-10-24. Author approved the bounded follow-up after
reviewing Stage13C and Stage14A. The old experiments and failed rules remain
unchanged. This study asks whether short adaptation recovers quality at the
same token compression, and whether similarity is useful beyond spatial merging.

## Design and fixed budget

Use Stage11B full checkpoints for seeds 1501/1502 (100 epochs, 17,600 updates),
already scored by Stage14A. Three arms per seed: full, spatial, similarity.
Compressed arms retain CLS and reduce 65 to 33 tokens after block 3 of 6,
using exactly the frozen Stage14A implementation, without a compression search.
All parameters receive full backpropagation through each arm's computation.

Train each arm for exactly five epochs: 880 updates per arm, 5,280 total.
Identical per-seed batches, crop/flip, MixUp draws/permutations, rates and dropout
seeds across arms; changed token shapes need not produce identical dropout masks.
Use the existing 45,000/5,000 training-only development split. No test data.
Reverse arm order between seeds to reduce simple order bias; hardware variation
and only two seeds still limit inference.

Restore every parameter and its AdamW moments from the source, with explicit
ordered parameter/shape checks and age 17,600. There is no reset control here:
this is a matched adaptation recipe, not an optimizer-history causal study.
All arms use peak LR 1e-4, one epoch linear warmup, then cosine to 1e-5 at the
last update; weight decay .05, MixUp alpha .2, label smoothing .1, batch 256.
These choices are prospective, not selected using Stage14B results. Three
hardware-warmup updates are discarded, then exact source weights/moments
restored. Warmup and restoration costs are charged to the arm.

Reproduce all three unadapted Stage14A mode scores/logits before training.
Score epoch 0 and every completed epoch; only epoch 5 is primary. Do not
select a best checkpoint, extend training, change rates, or tune compression
after inspecting intermediate scores. Save logits and final weights/moments.
Nonfinite values, state/provenance mismatches or failed source reproduction
stop execution. A degraded final full control invalidates candidate promotion,
but still retain all outputs; transient early deterioration alone is not a gate.

## Acceptance and interpretation

Both seeds must satisfy all applicable rules. The full reference must finish
at least 80% accurate and within 0.30 points of the original full source.
Each compressed candidate must finish within 0.30 points of the best of the
original full source, adapted full, and adapted spatial control. This prevents
a weak full continuation or a stronger simple control from inflating success.

The candidate must also save at least 10% charged recovery time OR at least
20% maximum incremental training-operation allocation versus adapted full.
Report the time and memory routes separately. Memory-only success is not a
speedup. This screen is a recovery/resource feasibility result, not proof of
faster training from scratch or statistical equivalence. Record accuracy and
CE changes from each unadapted compressed source as descriptive recovery.

Similarity warrants priority over spatial only if it passes the above screen
in both seeds AND either spatial fails that screen, or similarity exceeds
spatial accuracy by at least 0.30 points in both seeds while costing no more
than 5% extra charged recovery time and incremental training memory in either.
Otherwise prefer the simpler passing spatial control. If neither passes, park
this fixed recipe; do not automatically expand the search. No publication or
novelty claim follows from a pass. This remains ToMe-inspired established
machinery on reused development data, not an exact prior-paper reproduction.

## Cost accounting

Record per-step input transformation and actual forward/backward/AdamW update
time with CUDA synchronization. Charge the sum plus source loading/model and
optimizer setup, hardware warmup, and exact restoration. Initial/final scores,
epoch evaluation, telemetry, log writing, final checkpoint saving and inference
benchmarking are excluded from this scoped recovery charge and reported as
separate/overall observed costs. Common data loading is reported at launch.
There is no hidden calibration or search in this study.

Capture incremental and absolute allocated peaks and reserved peaks for every
training step, with gradients cleared before the operation baseline. Restored
optimizer moments remain in the baseline. Report setup peaks separately and
the maximum allocated peak across measured phases, including evaluation, so a low operation increment cannot be
presented as a low total VRAM requirement. CPU artificial memory is unavailable.

Benchmark inference after adaptation on twelve fixed training batches, with
three discarded forward warmups, no gradients or updates. This is secondary
and does not select checkpoints or affect the training-resource screen.

Carry each original full source's recorded training and complete-run cost into
the summaries. Show source training + recovery charge, and source complete run
+ observed recovery arm time; include launch loading separately. The latter
is a sum of recorded costs from separate executions, not a new end-to-end
wall-time measurement. An already trained source cannot be treated as free
for a training-efficiency claim. No claim about time to matched quality is
established by this equal-update-budget screen; that needs a later design.

## Integrity and restart

Freeze spec, code, imported scientific dependencies, source checkpoints,
Stage14A logits and review evidence before real execution. Audits reconstruct
streams and optimizer ages, verify labels, metrics, arithmetic and manifests.
Completed arms are revalidated and skipped. An interrupted arm restarts from
the original source in a new attempt; earlier attempts are retained. Changed
code/environment/configuration blocks continuation. Final completion requires
the audit; do not rerun or alter a completed scientific study.

Artificial CPU/CUDA tests and end-to-end smoke workflows precede the freeze.
Synthetic tests use tiny networks, two epochs and two seeds; no scientific
success is allowed on artificial data. Preparation does not train or score
real checkpoints. The user runs the real launcher after preparation.
