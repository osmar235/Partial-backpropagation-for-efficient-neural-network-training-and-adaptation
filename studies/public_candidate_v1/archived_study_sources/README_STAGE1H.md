# AGRLS Stage 1H — frozen v0.9 incremental held-out MNIST validation

This package freezes the Stage 1G development choice `agrls_db_incremental` and evaluates it on untouched seeds 19–27 against Adam.

Primary median criteria declared before viewing seeds 19–27:
1. evaluation-accuracy deficit versus Adam <= 1.0 percentage point;
2. backward-pass reduction versus Adam >= 10%;
3. common calibrated neural-network compute reduction > 0%.

The batch alternates Adam/AGRLS execution order by seed. It reports both direct forward+backward timing and a common calibrated compute estimate using the same per-seed median graph-forward and cached-backward times for both algorithms.

Run from CMD after extracting into `C:\Users\osmar\projects\Projeto_AI_Fronteira`:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage1h_v09_heldout_seeds_19_to_27.cmd
```

Or directly:

```cmd
python agrls_stage1h_v09_heldout_batch.py --dataset mnist --train-subset 20000 --eval-subset 5000 --updates 60 --seeds 19 20 21 22 23 24 25 26 27 --device cuda --out stage1h_mnist_fullbatch_v09_heldout_seeds_19_to_27
```

When complete, ZIP the entire `stage1h_mnist_fullbatch_v09_heldout_seeds_19_to_27` folder and return it for analysis.

## Timing note
The Stage 1H runner uses `agrls_stage1_mnist_fullbatch_v09h.py`, which adds a reporting-only pure NN forward/backward calibration. It does not change the v0.9 incremental controller or its MVR decisions. The primary compute comparison uses pure calibrated forward/backward cost; implementation-path direct timing and wall time are secondary.
