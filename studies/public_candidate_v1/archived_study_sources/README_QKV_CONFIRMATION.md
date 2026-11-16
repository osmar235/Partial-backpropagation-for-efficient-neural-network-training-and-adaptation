# Confirm the QKV update-time saving

This is a new confirmation study; the original 10% screen stays failed.
Eight fresh sessions use two new seeds, batches 256/200, balanced order and
128-update timing blocks. Both arms retain FP32, math attention and full gradients.
No real-data training is launched. Read the frozen protocol in
`protocol/qkv_confirmation/CONFIRMATION_PROTOCOL_v1.md`.

In CMD, stop other GPU jobs and run:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_qkv_confirmation.cmd
```

Send `results\qkv_confirmation_v1` when complete. The run has about 33,000
measured updates plus warmups/checks; allow tens of minutes, depending on this
machine. Keep the computer awake. The launcher runs the eight sessions sequentially
and performs a read-only final audit. Completed results are never rerun.

If interrupted, use the same command to preserve and resume recorded work. A
session resumed across processes is marked fragmented and cannot establish the
primary confirmation outcome; return the folder for review rather than replacing
unfavorable timings. Audit alone (no experiments):

```bat
"%PYTHON%" -B scripts\review_qkv_confirmation.py results\qkv_confirmation_v1
```

Confirmation concerns repeatable positive update-time savings, with uncertainty
reported from sessions. It does not automatically establish practical usefulness,
the original 5–7% magnitude, complete training speed or publication readiness.
