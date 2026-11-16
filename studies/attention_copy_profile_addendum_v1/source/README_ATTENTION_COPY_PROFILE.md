# Run the bounded optimized attention profile

Prepared2026-10-28. Close other GPU workloads. In Windows CMD:

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_attention_copy_profile.cmd
```

When COMPLETE appears, return `results\attention_copy_profile_v1`, including its
trace subfolders. The launcher audits automatically; a completed rerun checks
saved evidence without training. Interrupted sessions are retained and retried
in full. No package installation is needed.

Compare the qualified compatibility layout with a diagnostic native-output-layout
control, using original and disabled dropout, two sources and batches256/200.
All gradients are retained. Plain timings are separate from events/profiler
overhead. This measures operation costs, not accuracy or complete adaptation.
Removing the adapter copy may change dropout behavior and does not establish
an equivalent or novel method. The full scientific profile remains author-run.

Protocol: `protocol/attention_copy_profile/PROFILE_PROTOCOL_v1.md`.
