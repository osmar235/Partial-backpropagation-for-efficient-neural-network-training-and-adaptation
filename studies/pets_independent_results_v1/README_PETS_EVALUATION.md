# Oxford Pet independent evaluation

Run in Windows CMD, with other GPU jobs closed:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_pets_evaluation.cmd
```

This runs 24 fresh seeds × three BF16 arms (72 trajectories), all at 15 epochs,
then evaluates their final checkpoints on the official Pet test split. It uses
the existing dataset and pretrained weights. No download is needed.

Allow approximately 45–60 minutes for training, audits, test caching and scoring;
actual duration depends on disk/GPU load. Keep the computer awake. Allow roughly
15 GB of free space for checkpoints, the test cache and records. The full training
phase finishes before test access; no parameter tuning follows test results.

When COMPLETE appears, send results/pets_independent_evaluation_v1 for review
or tell us it is done. Run the same command after an interruption: completed
verified runs/scorings skip, partial attempts remain, and interrupted adaptation
timings are flagged for review. Do not edit the freeze or delete failed outputs.

Primary: periodic-r4 versus full. Secondary: frozen prefix versus full. Accuracy
margin stays 0.30 pp; all three arms use the same qualified BF16 recipe. See
protocol/pets_evaluation/PROTOCOL_v1.md for inference, precision limitations and
clock boundaries. Final-test assessment time is separately recorded and excluded
from the adaptation-time endpoint. Dependencies include the existing PyTorch,
torchvision, NumPy, SciPy and Pillow environment.
