# Oxford download recovery

The original run stopped during the image archive download, before caching or
training. Windows reported an interrupted connection followed by DNS failure.
No scientific run or baseline freeze was created. This repair changes download
transport only; the frozen experiment runner, specification and thresholds stay
byte-identical.

Run from Windows CMD:

```
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_pets_baseline_retry.cmd
```

The new launcher uses native curl with bounded retries, TLS verification and the
official Oxford archive checksums. It first tries to resume an existing file.
Both official endpoints ignored bounded Range probes on 2026-10-29; when resume
is unsupported, the partial file is renamed and retained before a fresh attempt.
It never appends an unrequested full archive to a partial file. If a transfer fails
again, rerun this same command later; verified archives are reused. No insecure
TLS option or antivirus change is requested.

Archives are checksum-verified before extraction. Extraction uses a separate
staging directory and rejects path traversal, links and special devices. A complete
directory replaces any old partial extraction only after the old directory has
been renamed for preservation. The unchanged baseline launcher then builds its
cache and starts the six planned trajectories. No official test scoring occurs.

Keep other GPU jobs closed once training begins. Send results/pets_baseline_v1
after completion. If retries are exhausted, send the new error without deleting
partial files. Native transport improves recovery but cannot guarantee that the
network connection will remain available.
