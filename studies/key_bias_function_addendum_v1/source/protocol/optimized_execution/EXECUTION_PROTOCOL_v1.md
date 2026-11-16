# Optimized-execution feasibility v1 — 2026-10-27

Artificial full-model update feasibility, with the pinned upstream attention
reference required as the baseline. This is not a new optimization algorithm,
real-data quality study or complete adaptation benchmark. Previous results,
source files and thresholds remain frozen. No retrospective gate relaxation.

## Arms and environment

Six eager arms: native/reference attention crossed with FP32 forced math SDPA,
BF16 autocast forced math SDPA, and BF16 autocast automatic SDPA dispatch.
All retain FP32 parameters and AdamW state, TF32 off, identical model weights,
data, batch/seed schedule, and full gradients. No FP16/GradScaler, weight
quantization, optimizer compilation/fusion, or environment installation.
Native uses the existing MultiheadAttention; reference uses the exact pinned
timm Attention class and already validated native-stride compatibility copy.

Math arms require strict determinism; automatic BF16 arms explicitly use
deterministic_algorithms=False and cudnn_deterministic=False, allowing actual
optimized dispatch. This is a separately labeled practical execution policy,
not a pure precision-only contrast. cudnn_benchmark=False throughout. FP16/BF16
reduced-precision GEMM reductions are disabled explicitly; math-SDPA reduced
precision reductions stay disabled. Autocast covers forward and loss only;
backward/optimizer operate on FP32 parameter gradients/state. Record all flags.
Profile actual SDPA forward/backward operators and output dtypes; no claimed
fused backend without observed dispatch. Math fallback in automatic dispatch
is an explicit observed outcome, not proof of optimized-kernel execution.

Before freezing: a bounded fresh-process capability probe attempts a tiny CUDA
Inductor forward/backward with fullgraph=True and suppress_errors=False. Record
installed compiler/Triton availability, exception and first-call elapsed time.
Compilation is capability-only in this version; never substitute eager timing
and call it compiled. Missing dependencies are reported without installing or
changing the environment. A successful probe would require a separately
specified full-model compiled comparison, startup/cache accounting and quality
checks. The present eager screen remains informative if compilation is blocked.

## Numerical qualification, declared before measurements

Use the same reference FP32/math initialized model as the comparison anchor.
Disable nn.Dropout only for all numerical qualification trajectories, including
the anchor. Different dtypes/compilation can map RNG streams differently, so
stochastic-mask differences must not be mistaken for arithmetic error. All
timed/profiling updates retain the original training dropout. The resulting
qualification is a dropout-free numerical diagnostic, not a matched stochastic
learning-trajectory claim for mixed precision.

Three AdamW updates; all logits, loss, parameter gradients, post-update
parameters and both moments are checked. FP32 arms retain the previous
elementwise QKV tolerances and strict repeated-state determinism. No excluded
parameter, bias exemption or widened FP32 tolerance. BF16 arms use explicitly
prospective *sanity* budgets: logits relative L2 <=2% and max abs <=0.02; loss
abs <=0.02; concatenated all-parameter gradient relative L2 <=10% and cosine
>=0.99; post-update parameter max abs <=3e-4; first/second moment relative L2
<=10%/20%. Every tensor must be finite. Record per-tensor and aggregate errors;
near-zero reference tensors remain included in the aggregate, not dropped.

These BF16 budgets allow small low-precision arithmetic differences while
detecting gross implementation errors. They are engineering screening limits,
not statistically calibrated accuracy margins. The parameter bound is explicitly
loose relative to three 1e-4 learning-rate updates; it cannot establish task
quality. Passing authorizes only descriptive operation timing. All real-data
accuracy, convergence and noninferiority claims require a new prospective
study. Record repeat behavior for all arms; automatic-policy repeats need not
be bitwise equal. Numerical failures, unsupported backends and nonfinite runs
remain visible and ineligible for timing. Unexpected programming errors abort.

## Trial design and clocks

Full original six-block/width192/65-token model; two new initialized seeds,
batches256/200; four fresh processes. Six cyclic rotations of the six-arm
order per condition/session, reversing the base order by session+condition
parity. Each planned arm occupies every planned position once per condition.
Preserve excluded-arm slots and report actual eligible sequence. Required
reference FP32/math must pass or that condition is blocked from timing.

Each trial measures model/optimizer setup and one first invocation before
trial-local warmup. That invocation is discarded from learning. Then five
discarded warmup updates; restore initial model and fresh AdamW; separately
measure first post-restore update and32 steady updates. Primary: synchronized
steady seconds/update. Secondary: setup, first invocation, warmup/restore,
first post-restore update and GPU allocated/reserved memory. First invocation
is not a cold-machine/cache measurement: qualification and earlier arms have
already executed. Compilation startup belongs solely to the capability probe.
Hashes, data generation, profiling, validation and serialization are outside
these operation clocks. This is not a complete-workflow timing result.

At full eligibility:576 trials,19,008 post-restore measured updates,576
discarded first-invocation updates and2,880 discarded warmups, plus separately
counted numerical/profiling updates. Bound the cost by excluding ineligible
arms; no automatic extensions, retry tuning or extra seeds.

For every eligible candidate, report per-session median paired savings versus
the required reference FP32/math; then descriptive medians/ranges across
sessions. Also compare native/reference within each execution policy. Do not
choose a fastest result while hiding failed arms or ineligible sessions.
No confirmatory p-values or new practical percent gate. Shared hardware and
four repeated sessions do not create extra independent trained-model seeds.

Freeze source/spec/environment/capability evidence before full execution.
Record preflight, process IDs, before/after telemetry, all attempts and exact
planned/actual order. Completed output is verified and skipped; restarts are
labeled. A candidate passing this feasibility screen still needs matched
real-data quality and complete-workflow confirmation before adoption.

## Primary technical references (checked 2026-10-26)

- PyTorch AMP: https://docs.pytorch.org/docs/2.14/amp.html
- SDPA dispatch and precision: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
- Numerical precision notes: https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html
- torch.compile: https://docs.pytorch.org/docs/2.14/generated/torch.compile.html
- Windows CPU/XPU scope: https://docs.pytorch.org/tutorials/unstable/inductor_windows.html

Documentation describes current APIs, not a guarantee for this installed
torch2.11.0+cu128 environment. Local observed capability takes precedence.
