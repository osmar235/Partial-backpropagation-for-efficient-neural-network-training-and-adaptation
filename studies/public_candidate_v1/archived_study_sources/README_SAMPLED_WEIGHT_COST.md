# Exact forward and input gradients; sampled weight gradients

Stage15A side adaptation is closed. The next distinct feasibility test retains
the original model and learns all its layers. Only the MLP weight-gradient sum
uses sampled patch positions; forward and input-gradient propagation stay full.

This is a short artificial cost benchmark, not another real training study.
Six structural/mathematical tests on CPU and CUDA precede it. Compare native,
exact-custom and sampled-quarter implementations, 72 counterbalanced trials.
Require at least 10% median paired update-time saving against native in all four
seed/batch conditions before considering a real quality diagnostic. Full sampling
and index overhead is timed. There is no sampling-fraction sweep.

Stop other GPU work and run in CMD:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_sampled_weight_cost.cmd
```

Send `results\sampled_weight_cost_v1` after COMPLETE. Rerunning the command verifies
completed outputs; an incomplete benchmark restarts in a preserved new attempt.
No real checkpoint, CIFAR data, validation score or test set is used.

This belongs to established approximate-gradient literature; no novelty claim.
See `protocol/strategy/SAMPLED_WEIGHT_DESIGN_v1.md` and
`protocol/sampled_weight_cost/COST_PROTOCOL_v1.md`. A cost pass alone establishes
neither useful accuracy nor lower complete adaptation cost.
