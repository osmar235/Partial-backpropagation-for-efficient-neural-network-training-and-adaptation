# AGRLS-v0.4 Fixed-Refresh K Sweep

Extract into:

`C:\Users\osmar\projects\Projeto_AI_Fronteira`

Activate the existing environment:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
```

Run:

```cmd
run_v04_k_sweep_rosenbrock_d20_seed0.cmd
```

or directly:

```cmd
python agrls_stage0_v0_4_k_sweep.py --objective rosenbrock --dim 20 --seed 0 --max-updates 120 --ks 1 2 3 4 6 8 10 15 20 --out stage0_v04_k_sweep_rosenbrock_d20_seed0
```

Upload the resulting `stage0_v04_k_sweep_rosenbrock_d20_seed0` folder as a ZIP.
