# Stage 5D — full optimizer paper program

## Claim boundary

Stage 5C is a proof of concept, not a successful optimizer claim. The full paper
must compare against both Adam and the official Adan implementation, report
time-to-target and end-to-end wall time, transfer across at least two workloads,
and pass a fresh held-out seed block.

Seeds 0–9 are now seen for CIFAR-10. Seed 10 is the next development seed. Seeds
11–19 are reserved and must not be opened until the method, optimizer settings,
and pass criteria are frozen.

## Proposed mechanism

The next mechanism keeps gradients exact and makes the full-versus-suffix
decision after the current loss is observed. On an eligible batch, a loss below
its pre-update exponential moving average receives an exact
`layers2–4 + classifier` backward. A surprising high-loss batch receives a full
backward. A full backward is forced after every suffix-only backward, limiting
the prefix staleness horizon to one update. There are no probe forwards.

This rule is useful only if differentiating the suffix from a full autograd
graph remains materially cheaper than full backward. Stage 5D0 measures that
before any development training run.

## Baselines

- PyTorch Adam, with a development learning-rate screen.
- Official Adan from `third_party/Adan`, with its learning rate screened on the
  same development protocol. Adan is a distinct baseline, not a spelling of
  Adam.
- Fixed alternating exact blockwise Adam from Stage 5C.
- The loss-gated exact blockwise candidate.

The official Adan repository is Apache-2.0 licensed and retained with its
license and original history.

## Stage 5D0 result

Keeping a full autograd graph and requesting only `layers2–4 + classifier`
gradients reduced forward-plus-backward time by 30.13% relative to full
backward. Freezing the prefix before the forward saved 32.46%, so making the
decision after observing loss preserves most of the available margin. Computing
suffix gradients first and prefix gradients later was 16.83% slower than one
full backward and is excluded.

## Stage 5D1 baseline selection

On CIFAR-10 development seed 10 at 500 updates, the selected learning rate was
`3e-4` for both Adam and official Adan. Adam reached 62.07% evaluation accuracy;
Adan reached 58.83%. Adan's first grid improved toward its lower boundary, so a
one-time lower-rate expansion (`1e-4`, `3e-4`, `5e-4`) was completed before
selection. No further baseline learning-rate tuning is planned for this
protocol.

## Stage 5D2 fixed-update result

With no controller tuning, loss-gated blockwise Adam used 287 full and 213
suffix-only updates. At 500 updates it reached 61.25% evaluation accuracy versus
62.07% for tuned Adam and 58.83% for tuned official Adan. Its wall time was
12.52% below Adam. The frozen development criterion (within 1 accuracy point
and at least 5% wall saving) passed.

The measured wall ratio permits approximately 572 candidate updates inside the
500-update Adam budget. Stage 5D3 conservatively freezes the equal-wall horizon
at 570 without changing the controller.

## Stage 5D3 equal-wall result and held-out freeze

At essentially equal wall time, 570 loss-gated updates reached 63.76%
evaluation accuracy versus 62.07% for 500 Adam updates and 58.83% for 500 Adan
updates. Candidate wall time was 0.41% below Adam. Stage 5D3 passed.

Freeze seeds 11–19 as Stage 5D4 held-out validation. Each seed runs tuned Adam,
tuned official Adan, the 500-update loss-gated method, and its frozen 570-update
equal-wall mode. Execution order is counterbalanced. The fixed mode requires a
median accuracy delta of at least -1 point, median wall saving of at least 5%,
and 6/9 seeds meeting each condition. Equal-wall mode requires nonnegative
median accuracy delta and wall reduction, with at least 6/9 seeds individually
within 1 point and no slower. Both modes must pass for the aggregate gate.

## Stage 5D4 held-out result

The frozen seeds 11–19 batch completed. Fixed-update wall savings were robust:
the median was 12.92%, and all 9/9 seeds exceeded the 5% threshold. Optimization
quality did not pass: median accuracy delta was -1.95 points and only 1/9 seeds
was within 1 point of the stronger baseline.

Equal-wall mode had a positive median accuracy delta of +0.73 point and a +0.15%
median wall reduction, but only 5/9 seeds were within 1 point and 6/9 were no
slower. Both frozen modes therefore failed their aggregate gates. The method
must not be presented as held-out success.

## Stage 5E revision boundary

Seeds 11–19 are now seen and cannot validate a revision. The next development
seed is 20; seeds 21–29 are reserved. Stage 5E adds deterministic cadence
compensation only: after the prefix skips one gradient, its next exact Adam step
uses twice the base learning rate. Consecutive full updates use the base rate.
This reflects the maximum one-step staleness horizon, adds no neural-network
work, and introduces no fitted threshold.

Stage 5E1 failed on development seed 20. Cadence compensation reached 59.38%
versus Adam's 61.38%, a -2.00-point deficit, with 5.80% wall saving. It was also
slightly worse than the uncompensated loss-gated controller at 59.78%.

Stage 5E2 therefore tests prefix-only continuation: after each exact full Adam
step, store the displacement of the stem and layer 1 (1.34% of parameters). On
the next accepted suffix-only update, apply that stored prefix displacement
while computing fresh exact gradients for layers 2–4 and the classifier. The
reuse horizon remains one, and the added tensor work touches only the small
prefix.

Stage 5E2 passed its fixed-update development gate on seed 20. At 500 updates,
prefix continuation reached 60.44% versus Adam's 61.38%, a -0.94-point delta,
while reducing wall time by 9.31%. It improved by 0.66 point over the same
loss-gated controller without continuation (59.78%). The run used 277 full and
223 suffix-only updates. The measured wall ratio permits approximately 551
candidate updates inside the 500-update Adam budget, so Stage 5E3 freezes a
conservative equal-wall horizon of 550 updates without changing the controller.

Stage 5E3 reached 62.90% at 550 updates, +1.52 points over 500-update Adam, but
its measured wall time was 0.96% longer. It therefore failed the strict
equal-wall criterion despite the optimization gain. Because the miss is within
timing noise, Stage 5E4 applies one predeclared timing safety adjustment to 540
updates (a 2% reduction) while leaving the optimizer and controller unchanged.
This is the final development run before either freezing seeds 21–29 or retiring
the revision.

Stage 5E4 passed. At 540 updates, prefix continuation reached 62.42%, +1.04
points over 500-update Adam, while using 3.93% less wall time. The mechanism is
now frozen for held-out seeds 21–29: Adam and official Adan run for 500 updates;
prefix continuation runs in both 500-update fixed mode and 540-update equal-wall
mode; odd seeds use baseline-first order and even seeds candidate-first. The
aggregate criteria are unchanged from Stage 5D4: fixed mode requires median
accuracy delta at least -1 point, median wall saving at least 5%, and 6/9 seeds
meeting each condition; equal-wall mode requires nonnegative median accuracy
delta and wall reduction, with at least 6/9 seeds individually within 1 point
and no slower. Both modes must pass.

## Stage 5E5 held-out result

Prefix continuation did not pass the frozen seeds 21–29 gate. Fixed-update wall
savings were consistent (12.33% median; 9/9 seeds above 5%), but the median
accuracy delta was -1.70 points and only 2/9 seeds were within 1 point. At the
540-update budget, the method had a positive +0.75-point median accuracy delta
and 5.65% median wall saving; however, only 5/9 seeds were within 1 point even
though all 9/9 were faster. Both preregistered modes therefore failed. These
seeds are now seen and will not validate another revision.

The failure pattern isolates optimizer-state handling as the next target. Replay
continuation moves the prefix but leaves its Adam step count and moments frozen.
Stage 5F replaces replay with zero-gradient Adam continuation on skipped prefix
steps: first and second moments decay exactly as Adam specifies for a zero
gradient, bias correction advances, and the resulting momentum-predicted update
is applied. This touches only the 1.34% prefix, introduces no scalar tuning, and
keeps the full-versus-suffix controller unchanged. Seed 30 is development;
seeds 31–39 remain reserved.

Stage 5F1 passed on seed 30. At 500 updates, moment continuation reached 64.08%
versus Adam's 62.94% (+1.14 points) while reducing wall time by 9.72%. It also
improved over replay continuation at 63.41%. The measured ratio would permit
about 554 candidate updates in Adam's wall budget; Stage 5F2 conservatively
freezes 540 updates to retain a timing margin.

Stage 5F2 passed: 540 moment-continuation updates reached 63.53% versus 62.94%
for 500 Adam updates (+0.59 point) while using 6.96% less wall time. Seeds 31–39
are now frozen for held-out validation using the same 500-update fixed mode,
540-update equal-wall mode, counterbalanced order, baselines, and Stage 5D4
aggregate criteria.

## Stage 5F3 held-out result

Moment continuation also failed the full frozen aggregate gate on seeds 31–39.
The fixed 500-update mode saved 12.46% median wall time and passed speed on 9/9
seeds, but its median accuracy delta was -1.47 points and only 2/9 seeds were
within 1 point. The 540-update equal-wall mode was much closer: median accuracy
delta -0.21 point, median wall saving 5.59%, 6/9 accuracy passes, and 9/9 speed
passes. It failed only the preregistered requirement that median accuracy delta
be nonnegative. The overall gate is therefore false, and this result must not be
reported as a held-out win.

Stage 5F establishes a useful systems result rather than a competitive optimizer
result: loss-gated suffix backpropagation plus Adam-state continuation reliably
reduces end-to-end time, and an added 8% update budget recovers most accuracy,
but seed-to-seed optimization variance remains too high for an outperforming
claim. Mechanism tuning stops here. The paper draft will state this boundary and
pre-register the remaining requirements: longer schedules, at least one second
architecture/dataset family, repeated time-to-target measurements, and a fresh
untouched seed block for any future revision.

## Publication gate

The paper-facing method must show a useful time-to-target advantage over the
stronger of tuned Adam and tuned Adan, a wall-time margin large enough to exceed
run-to-run timing noise, and consistency on the fresh seed block. A promising
single development seed is insufficient.
