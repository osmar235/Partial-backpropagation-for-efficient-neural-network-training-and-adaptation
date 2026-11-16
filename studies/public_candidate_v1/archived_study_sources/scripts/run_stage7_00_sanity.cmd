@echo off
setlocal
cd /d "%~dp0.."
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -B scripts\test_stage7_mechanics.py
if errorlevel 1 exit /b 1
"%PYTHON%" -B scripts\stage7_review_experiments.py --variants adam lgpb random_matched periodic_matched freeze_prefix replay_prefix rho_090 rho_099 no_forced_full prefix_stem prefix_stem_layer12 --suite smoke --dataset fake --tiny-model --train-subset 64 --eval-subset 8 --batch-size 8 --epochs 1 --schedule-epochs 1 --warmup-steps 2 --seed 50 --device cpu --log-steps --out results\stage7_smoke_v1_1
if errorlevel 1 exit /b 1
echo SANITY COMPLETE. Results are artificial diagnostics, NOT scientific evidence.

