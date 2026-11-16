# Recorded environment and installation notes

## Experiment environment

The confirmatory runs were executed on Windows with an NVIDIA GeForce RTX 5080.
The timed region included data augmentation on the device, forward computation,
backward computation, and optimizer updates. Algorithm-specific warm-up and
validation were excluded as specified in the preregistration.

Recorded package versions:

```text
Python       3.14.3
PyTorch      2.11.0+cu128
torchvision  0.26.0+cu128
NumPy        2.4.3
Matplotlib   3.10.8
CUDA build   12.8
GPU          NVIDIA GeForce RTX 5080
```

## Installation

Create and activate a virtual environment, then install the dependencies. For
an NVIDIA system, install the PyTorch build appropriate for the locally
supported CUDA version using the official PyTorch installation instructions.
The exact recorded Python packages are listed in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The code also runs on CPU for smoke testing, but the reported wall-time results
must not be interpreted as portable to different hardware or software stacks.

## Data

The final harness downloads CIFAR-10 through `torchvision.datasets.CIFAR10`.
The dataset is stored under `data/`, which is excluded from version control.

