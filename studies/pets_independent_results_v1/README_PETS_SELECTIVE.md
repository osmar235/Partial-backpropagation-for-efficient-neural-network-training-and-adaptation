# Oxford Pet selective development

Run from Windows CMD:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_pets_selective.cmd
```

Close other GPU jobs first. This uses the existing verified cache and weights;
no download is needed. It runs six fresh development seeds with three arms each:
full backpropagation, permanently frozen prefix and periodic prefix refresh every
fourth global update. All use BF16 and the same 15-epoch recipe. The prefix is
stem + layer1 + layer2. There is no tuning sweep or official-test evaluation.

Allow roughly 10–15 minutes including checks; duration depends on the machine.
Results: `results/pets_selective_development_v1`. Send that folder for review
when COMPLETE appears. The comparison measures any additional benefit over
optimized BF16 full backpropagation; historical baseline results are preserved.

If interrupted, rerun this same command. Verified complete arms are skipped;
incomplete attempts remain and interrupted timings are flagged for review.
Do not delete results or edit the frozen recipe. A different runtime, source or
data hash stops the workflow. Official-test evaluation remains a separate phase.

Details: protocol/pets_selective/PROTOCOL_v1.md. Required dependencies are the
existing Python, torch, torchvision, NumPy and Pillow environment. Source package
includes the original baseline helper without importing older project runners.
