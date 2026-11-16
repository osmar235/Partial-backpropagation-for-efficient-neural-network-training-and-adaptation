# AGRLS-v0.4 Refresh-Policy Ablation

Copy these files into:

`C:\Users\osmar\projects\Projeto_AI_Fronteira`

The project folder must already contain the existing AGRLS files from v0.4, especially:

- `agrls_stage0.py`
- `agrls_stage0_v0_3.py`
- `agrls_stage0_v0_4.py`

Activate the environment:

```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
```

Then run:

```cmd
run_v04_refresh_ablation_rosenbrock_d20_seed0.cmd
```

or directly:

```cmd
python agrls_stage0_v0_4_refresh_ablation.py --objective rosenbrock --dim 20 --seed 0 --max-updates 120 --out stage0_v04_refresh_ablation_rosenbrock_d20_seed0
```

Upload the complete output folder when finished.
