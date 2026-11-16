# Stage8 completed: current results and next step

**Later author decision,2026-10-20:** return to method development. Publication
preparation is on hold; `README_STAGE9.md` and master v6.4 now define the next
step. Stage8 results below are retained unchanged. Its completed training
program does not prohibit the newly authorized research reset.

2026-10-20. All24 evaluation runs completed and passed an independent audit
of1,411,200 updates. All planned scientific training is complete.

Full AdamW averaged86.402% test accuracy, LGPB-MC71.697%, periodic85.175%,
replay72.058%. LGPB used8.67% less training wall but lost14.705pp and failed
the new accuracy criterion. Periodic had nearly identical time with13.478pp
higher accuracy than LGPB. Replay did not recover the large loss. Neither
loss-gated method reached80% or85% in any seed.

Interpretation: `protocol/stage8/FINAL_REVIEW_SYNTHESIS.md`. Audit/results:
`results/stage8B_review_v2/REPORT.md` and `independent_results.json`. Retain
all sources, raw records, checkpoints, development history and freeze.

Current manuscript: `NeuralNetworks/LGPB_Neural_Networks_revised_v7_Stage8.docx`
and matching PDF. The20-page rendering has10 tables/5 figures, numerically
and visually verified. See `NeuralNetworks/REVISION_V7_NOTES.md`.

Current master: `adaptive_gradient_refresh_project_master_v6.2.md`.
Next: author review, current cover letter/highlights and combined dated
code/data archiving with verified public identifiers. Older editorial files
should not be reused unchanged. No public release or submission has occurred.
Older run guides are historical reproduction instructions, not new training.

Optional audit-only command in Windows CMD:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
"%PYTHON%" -B scripts\review_stage8b_v2.py
```

It loads no dataset and performs no training or optimizer update.
