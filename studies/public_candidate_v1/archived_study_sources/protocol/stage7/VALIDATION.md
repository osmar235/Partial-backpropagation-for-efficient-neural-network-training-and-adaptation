# Stage 7 local v1.1 validation — 2026-10-14

Environment: project Python 3.14.3, PyTorch 2.11.0+cu128, torchvision
0.26.0+cu128, CUDA available, NVIDIA RTX 5080. All checks below passed.
No substantive CIFAR-10/100 Stage 7 seed experiment was run.

1. Mechanics: deterministic random/periodic nonadjacent masks, exact per-epoch
   counts and 2,116 planned partial steps in 5,880; exact suffix gradients;
   Adam zero-gradient continuation versus `None` freezing; gate/forced-full.
2. Frozen implementation comparison: two epochs, tiny artificial model/data,
   four partial steps in sixteen updates. Final weights matched bit-for-bit;
   final evaluation loss and partial counts matched unchanged Stage 6.
3. CPU artificial smoke: all eleven variants, including all sensitivity paths,
   with per-algorithm warm-up and per-step logs.
4. GPU artificial integration: actual ResNet-18-GN, all eleven variants,
   two epochs/four updates per variant with warm-up. Partial steps were
   exercised. These durations/accuracies are diagnostics, not study evidence.
5. Transfer wiring: 100-class ResNet output shape, finite CIFAR-100
   normalization/augmentation, and exact prefix displacement replay.
6. Analysis: synthetic results produced both Adam- and LGPB-reference paired
   tables plus full/partial operation timing distributions.
7. Audit: nine original seeds 41–49; grand-mean q = 0.35990173847316703.
   All 30 epoch-summary rows agree numerically with the supplied audit ZIP.
8. Integrity: all 218 files in `frozen_files_before.csv` retained their hashes.
   This includes root frozen source/protocol/results, the original upload ZIP,
   and the expanded `github/` release tree.
9. Lifecycle: identical resume skipped all eleven completed variants; a changed
   learning rate failed without overwriting configuration; an incomplete
   expected seed matrix failed explicitly. Completed records' mode counts
   and split backward timers reconciled.

Reproduce mechanics and integrity/lifecycle checks from project root:

```cmd
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
"%PYTHON%" -B scripts\test_stage7_mechanics.py
"%PYTHON%" -B scripts\validate_stage7_outputs.py
```

The second command expects the retained synthetic integration and archived audit
outputs. It does not run or read new substantive CIFAR experiments. Hash records
use absolute local paths. For a copied project, rebase those paths explicitly
rather than treating the moved files as a source change.

## Material limits

Synthetic tests establish implementation consistency, not accuracy, speedup,
six-seed statistical power, numerical stability over 30 epochs, or dataset
transfer. No scientific conclusion about Stage 7 follows until the fixed runs
are complete. Matching archived rates does not match each new gate trace;
freeze/replay can also change realized rates. See the v1.1 prospective protocol.
