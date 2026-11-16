# QKV unbind confirmation v1 — 2026-10-26

The author approved confirmation after reviewing the original 5.2–7.1% savings.
This is a prospective follow-up selected after seeing that result. The original
>=10% screen remains failed. No old outcomes, thresholds or implementations change.

## Question and scope

Does the unchanged QKV-unbind implementation reproducibly reduce synchronized
steady full-update time against native math attention on this RTX 5080, while
preserving checked numerical behavior? This first phase measures operations on
artificial data, not real training, accuracy, economic usefulness or novelty.
If confirmed, prepare a separately frozen, matched real-data adaptation comparison
including setup, data handling, validation and checkpoint costs. Do not launch it
automatically. Broader models/hardware and optimized execution remain later work.

## Fixed design

- Eight fresh Python processes, sequential on the same machine. These are session
  repetitions, not eight independent machines or training datasets. Avoid other
  GPU work; record idle preflight and before/after telemetry in every session.
- Two previously unused initialization/input seeds, 99101 and 99102, each at
  batches 256 and 200. Same six-block width-192 ViT and 65 tokens as discovery.
- Four paired rounds per condition per session. Each arm is initialized from
  identical weights/input and restored after 20 discarded warmup updates; one
  separately reported first update precedes 128 synchronized-block steady updates.
  The 128 updates form one timing observation, never 128 statistical repetitions.
- Alternate native/candidate first within rounds and across sessions. Rotate the
  four condition positions across sessions; reverse rotation in the second half.
- The original candidate, math SDPA, FP32, deterministic policy, TF32-off,
  AdamW, dropout, loss, numerical tolerances and model are unchanged. No profiler
  inside measured timing; original mechanism/correctness gates precede timings.
- 256 trials, 33,024 measured updates (including first updates), 5,120 discarded
  warmup updates, 320 numerical-check updates and 64 mechanism-check updates.
  Setup/first time and allocated memory are secondary and reported separately.
- Preserve every attempt. Resume skips hash-verified completed conditions/sessions;
  if a session was interrupted, discard none of its results but mark it fragmented.
  A fragmented session cannot support the primary confirmation conclusion. Do not
  replace it after looking at timings; review the incomplete study instead.

## Numerical and timing analysis

For each seed/batch/session, take the median of four paired candidate/native
steady-time ratios. Saving = 100*(1-ratio). Analyze the eight session summaries,
not the 32 individual pairs, as the repetitions for that condition. No pooling
with discovery. Report all values, median saving, and order-specific summaries.

The prespecified directional test is a one-sided exact sign test against no
positive median saving, counting zero as a non-win. Bonferroni-adjust the four
condition p-values; require adjusted p <= .05 in every condition. With eight
sessions this requires eight positive session medians in every condition (adjusted
p = .015625). This tests reproducibility of the direction, not a new minimum useful
effect and not equivalence to the originally observed 5–7% magnitude.

Report [minimum, maximum] of the eight session savings as a conservative exact
interval for the population median under independent identically distributed
continuous sessions: marginal coverage 1-2/2^8 = 99.21875%, simultaneous coverage
at least 96.875% across four conditions by union bound. State the assumptions;
fresh processes do not remove common hardware/environment drift. Also report
order-specific effects and telemetry for independent review. No outlier removal,
optional stopping on speed, extra sessions or threshold retuning.

Confirmation also requires all original numerical gates, deterministic same-arm
repeats, identical full timed loss sequences, final model/Adam hashes and RNG
states between paired arms over 129 updates. If exact trajectory identity fails,
report it and withhold confirmation, even if short checks pass. Do not relax gates.
Only eligible arms are timed. Numerical failure is reported, not a speed win.

An integrity PASS means records validate; it is distinct from a confirmed saving.
Smoke execution uses two reduced CPU/CUDA sessions, one seed, batches 3/4, two
rounds, one warmup, three steady updates; it can never confirm the scientific result.

Source/spec/environment fingerprints are frozen before real execution. Completed
audits are read-only, accept absolute/relative paths and reject changed artifacts.
No full benchmark or real-data training is run during preparation.

