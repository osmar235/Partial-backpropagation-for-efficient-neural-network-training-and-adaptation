# AGRLS-v0.5 Held-out Validation — Rosenbrock d=20

## Why this run is different

Seed 0 has been used repeatedly to develop AGRLS-v0.1 through v0.5. It is now permanently designated the **development seed**.

This package freezes the current policies and evaluates them on **seeds 1–9 together**. Do not inspect or tune on one held-out seed before the complete batch finishes.

## Frozen policies

- `fixed1`
- `fixed3`
- `phase3to1_30` — development-derived schedule; reference only
- `mvr_r2`
- `mvr_r3`
- `mvr_r4`

No thresholds or schedules should be changed during the batch.

## Run

From CMD:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_v05_mvr_heldout_seeds_1_to_9.cmd
```

The script will create:

```text
stage0_v05_mvr_heldout_rosenbrock_d20
```

with one folder per seed plus aggregate files.

## Upload back to ChatGPT

When the whole batch is complete, ZIP **the entire** folder:

```text
stage0_v05_mvr_heldout_rosenbrock_d20
```

and upload the ZIP.

Do not rerun or selectively omit a seed because its result looks unusual. Failed executions may be rerun only for technical failure and should be documented.
