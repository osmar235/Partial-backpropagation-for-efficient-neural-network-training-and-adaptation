# Stage13B decision — 2026-10-23

Both full-continuation references qualify. All704 original training losses
reproduce exactly; fixed-endpoint development loss/accuracy improve in both
seeds. Independent saved-state rescoring confirms sources and final states.

Retained and reset optimizer histories BOTH produce initial deterioration and
later recovery. Reset worsens the first-step drop in both states, but its
endpoint effect is mixed (-0.74 /+0.56pp relative to continuation). A harmful
first update is therefore insufficient to reject this useful training recipe.
Reset alone cannot explain the transient here; its mechanism remains unresolved.

Preserve Stage13A's operation-cost findings and inconclusive utility label.
Do not reinterpret it as a rejection of local learning or a published method.
Stage13B is known-outcome reference verification on reused development data.

Proceed conceptually to a separately specified, matched multi-step local/full/
connected-auxiliary/suffix comparison. Explicitly specify restored optimizer
history, local-head calibration, fixed endpoints, cost accounting and independent
assessment before execution. No new local-learning code or run is frozen here.
Stage14 remains queued; publication stays on hold.
