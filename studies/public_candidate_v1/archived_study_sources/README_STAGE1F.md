# AGRLS Stage 1F — Frozen v0.8.1 held-out MNIST validation

Purpose: validate the anchor-backtrack bug-fixed AGRLS-v0.8.1 on completely unseen seeds 10–18.
Seeds 1–7 were part of the interrupted v0.8 validation, seed 8 exposed the bug and was used only as a diagnostic, and seed 9 is therefore not used as fresh validation evidence. No tuning should be performed using seeds 10–18 before the full batch completes.

Primary comparison: Adam vs frozen AGRLS-v0.8.1 (`agrls_adam_1d_pf_mvr`). Momentum is contextual only.

Pre-specified supportive criterion, carried forward unchanged from Stage 1E:
- median evaluation-accuracy deficit vs Adam <= 1.0 percentage point;
- median exact-gradient reduction >= 10%;
- median measured oracle time lower than Adam.

Run from Windows CMD:

    C:\Users\osmar\projects\.venv\Scripts\activate
    cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
    run_stage1f_v081_heldout_seeds_10_to_18.cmd

Direct command:

    python agrls_stage1f_v081_heldout_batch.py --dataset mnist --train-subset 20000 --eval-subset 5000 --updates 60 --seeds 10 11 12 13 14 15 16 17 18 --device cuda --out stage1f_mnist_fullbatch_v081_heldout_seeds_10_to_18

When complete, zip the entire `stage1f_mnist_fullbatch_v081_heldout_seeds_10_to_18` folder and return it for analysis.
