# Stage9D cost-only deployment check — 2026-10-20

Prepared after reviewing Stage9C. Its larger boundaries offer computational
headroom, but the boundary signal's incremental utility over MixUp strength is
unproven. Before another training budget, compare efficient executable paths
at the actual batch256. One DEVELOPMENT cut4, freezing the prefix on partial
updates. No new controller or threshold selection.

Use the existing seeds1201/1202 full-AdamW checkpoints15/30. Load only the
development CIFAR10 training-derived split. Restore identical model/Adam state
before every disposable measurement. No update is carried into the next probe.
No baseline retraining, test data, accuracy selection or efficacy experiment.

Measure four paths on matched inputs, MixUp, dropout and next-update LR:

1. Ordinary native full forward/backward/AdamW step.
2. A predecided partial update: prefix forward under no_grad, suffix backward,
   prefix grad=None. A blind or augmentation-only schedule can know its choice
   before forward and avoid constructing the prefix graph.
3. Boundary-decision full update: split forward, suffix backward, boundary RMS
   reduction and scalar transfer, then continue prefix backward and step.
4. Boundary-decision partial: same forward/suffix/RMS, omit prefix backward,
   prefix grad=None on the optimizer step.

The boundary choice is FORCED, so both paths are measured independently of a
fitted threshold. This isolates cost and cannot show that either policy learns.
There is no free future-window policy. The predecided kernel is a lower-overhead
implementation available to blind/cheap schedules; it omits their actual
scheduling/calibration logic, which still belongs in any training measurement.

Four excluded warmup batches; paired gradients, updated parameters, prefix
freezing and optimizer counters verified outside measured samples.32 batches
per state, rotated/reversed path order,512 real timing rows. One synchronization
before and after each whole operation, not each backward component. The actual
training loss.item is included for every path; boundary RMS/item is additional.
Matching augmentation is measured once per batch and added to each path.
State restore, integrity checks, cloning and model creation are excluded.
Preparation, cache conditions and fixed-state microbenchmarks can differ from
long-run deployment; report the measurements with those limits.

Report hypothetical20% /40% mixtures relative to ordinary full, AND the boundary
pipeline relative to efficient predecided scheduling. No automatic success
threshold, scientific claim, cut search or training launch follows. A small
adaptive benefit cannot justify a larger added cost. Existing Stage9C evidence
and sources must pass hashes before scientific execution.

Output: results/stage9D_deployment_cost_v1. Completed matching output skips;
interrupted attempts restart separately with all previous attempts retained.
CPU artificial rehearsal uses a separate output and has no scientific meaning.
