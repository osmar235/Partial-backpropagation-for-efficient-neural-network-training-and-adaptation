# AGRLS Stage 1A — deterministic full-batch MNIST gate

**Project folder**

`C:\Users\osmar\projects\Projeto_AI_Fronteira`

**Existing virtual environment**

`C:\Users\osmar\projects\.venv`

## Purpose

This is the first real neural-network test of AGRLS-MVR.

The objective remains deterministic by using one fixed 20,000-image MNIST training subset. This isolates whether the surrogate/refresh mechanism survives a neural-network loss landscape before mini-batch noise is introduced.

The script automatically measures the actual gradient-query / forward-query time ratio on the current hardware and uses that measured ratio in the MVR controller.

## Before running

Activate the existing environment:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
```

Check that PyTorch sees CUDA:

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Do **not** reinstall PyTorch if CUDA is already working.

## Run

```cmd
python agrls_stage1_mnist_fullbatch.py --dataset mnist --train-subset 20000 --eval-subset 5000 --updates 60 --seed 0 --device cuda --out stage1_mnist_fullbatch_seed0
```

Or double-click / run:

`run_stage1_mnist_fullbatch_seed0.cmd`

MNIST will be downloaded automatically into the local `data` folder if needed.

## Algorithms

- SGD, lr=0.5
- Momentum SGD, lr=0.1, momentum=0.9
- Adam, lr=0.01
- AGRLS fixed-1
- AGRLS MVR using the hardware-measured cost ratio

These standard-optimizer learning rates are diagnostic defaults and are **not** yet final tuned publication baselines.

## Outputs

- `REPORT.txt`
- `summary.csv`
- one trace CSV per algorithm
- `train_loss_vs_measured_oracle_time.png`
- `train_loss_vs_gradient_queries.png`

After the run, ZIP the entire output folder `stage1_mnist_fullbatch_seed0` and upload it to ChatGPT.
