# Stage 2A v0.9.1 — Fashion-MNIST download-only fix

## Why this package exists
The original Stage 2A run stopped before seed 28 began because torchvision's default Fashion-MNIST HTTP S3 mirror timed out (`WinError 10060`). No training result from seed 28 was produced.

## What changed
Only `agrls_stage1_mnist_fullbatch_v081.py` was changed. For `dataset_name == "fashionmnist"`, the upstream Zalando Research GitHub raw-data mirror is placed first in `datasets.FashionMNIST.mirrors`, while torchvision's original mirror remains as fallback.

No scientific/optimization behavior changed: same data files, same subset sizes, seeds 28–36, model, Adam baseline, frozen AGRLS-v0.9 incremental controller, update count, calibration, metrics, and preregistered criteria.

## Run
From the project directory with the venv activated:

    run_stage2a_v09_fashion_seeds_28_to_36.cmd

or

    python agrls_stage2a_v09_fashion_heldout_batch.py --dataset fashionmnist --train-subset 20000 --eval-subset 5000 --updates 60 --seeds 28 29 30 31 32 33 34 35 36 --device cuda --out stage2a_fashionmnist_fullbatch_v09_seeds_28_to_36

The first run will download Fashion-MNIST into `data/FashionMNIST`.
