# AGRLS-v0.4 nonlinear diagnostic

## Why v0.4 exists

The user-machine AGRLS-v0.3 Rosenbrock run showed an important failure mode:
`rho = actual_improvement / predicted_improvement` could become much larger than 1.
The candidate step still reduced the objective, but the surrogate was badly
underpredicting the magnitude of improvement. v0.3 treated every `rho >= 0.75`
as a good prediction, so severe underprediction could incorrectly reset the
model-miss counter and enlarge the trust radius.

v0.4 treats prediction agreement symmetrically.

For positive `rho`:

```text
agreement = min(rho, 1/rho)
```

Therefore:

- rho = 1 -> agreement = 1 (perfect)
- rho = 2 or 0.5 -> agreement = 0.5
- rho = 10 or 0.1 -> agreement = 0.1

It also records a bounded symmetric prediction error:

```text
|actual - predicted| / (|actual| + |predicted|)
```

A beneficial step with poor agreement is **accepted** (we do not waste a real
objective improvement), but is labeled `accept_stale_model` and increments the
model-miscalibration counter. Repeated miscalibration triggers a new exact
gradient/map refresh.

## Files

- `agrls_stage0.py` - original Stage-0 utilities/baselines
- `agrls_stage0_v0_3.py` - previous persistent-surrogate implementation
- `agrls_stage0_v0_4.py` - new symmetric-calibration implementation
- `run_v04_rosenbrock_d20_seed0.cmd` - next user-machine diagnostic

The v0.4 runner includes v0.3 in the same benchmark output, so the new run gives
a direct same-seed comparison among GD, momentum, gradient reuse, AGRLS-v0.1,
AGRLS-v0.3, and AGRLS-v0.4.

## Run

Extract/copy these files into:

```text
C:\Users\osmar\projects\Projeto_AI_Fronteira
```

Then from CMD:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_v04_rosenbrock_d20_seed0.cmd
```

Equivalent direct command:

```cmd
python agrls_stage0_v0_4.py --objectives rosenbrock --dims 20 --seeds 0 --max-updates 120 --out stage0_v04_rosenbrock_d20_seed0
```

Upload the resulting `stage0_v04_rosenbrock_d20_seed0` folder as a ZIP.

## Important status

This remains a one-seed mechanistic diagnostic. Do not interpret it as a final
benchmark or publication result.
