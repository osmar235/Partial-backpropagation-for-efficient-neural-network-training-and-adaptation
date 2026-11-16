# Post-confirmation decision — 2026-10-26

## Current decision: QKV operation saving confirmed — 2026-10-26

The independent review confirms the prospectively specified positive-direction
result in every condition: median session savings 7.14% /5.23% for seed99101
at batches256/200, and 6.17% /5.62% for seed99102. All eight fresh sessions
have positive condition medians (32/32 cells); 123/128 individual paired rounds
are faster. Five slower pairs are retained. Bonferroni-adjusted sign-test p=.015625
in each condition. No discovery data were pooled; the original >=10% screen stays
failed. A confirmed modest gain can justify follow-up below that practical target.

Conservative median intervals are [3.61,8.63], [2.26,7.60], [4.16,9.79] and
[1.53,8.09] percent respectively. Their exact coverage and sign-test interpretation
assume independent identically distributed session effects; sequential sessions
share one GPU, so do not generalize them to independent tasks or hardware. Both
run-order strata have positive median session savings in all four conditions.

All short numerical differences are zero. Every paired 129-update loss sequence,
final parameters, Adam state and RNG hashes match exactly, including cross-session
repetitions. The intended 19-to-1 select-backward reduction plus six unbind-backward
nodes is verified. Allocated peak and incremental steady memory are unchanged.
The candidate, full gradients, FP32, math attention and determinism are unchanged.
This is checked identity on these trajectories, not proof for arbitrary training.

Frozen read-only audit and independent raw reconstruction agree on 256 trials,
33,024 measured updates, 5,120 discarded warmups, 320 numerical-check updates and
64 mechanism updates. Eight distinct processes, no fragmented or replacement
attempts. Idle preflights pass; sparse active snapshots show 61–70 C and SM clocks
2760–2782 MHz, not continuous interference monitoring. No experiment was rerun.

Next: separately prepare a matched real-data native-versus-unbind adaptation study
that includes setup, data handling/augmentation, validation and checkpoint time,
with identical source states, schedules and data order and explicit quality checks.
Measure transfer to complete training before adding further layout changes. No
next run command or experiment is prepared/launched in this review. Accuracy,
end-to-end savings, broader workloads/hardware and novelty remain unestablished;
publication stays on hold. All prior/raw evidence remains unchanged.

Review: results/qkv_confirmation_review_v1/REPORT.md. Master v7.32 and paper memory
v21 are current; earlier statuses and commands are historical.

