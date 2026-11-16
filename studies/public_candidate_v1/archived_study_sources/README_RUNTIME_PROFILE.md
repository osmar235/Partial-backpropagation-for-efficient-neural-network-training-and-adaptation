# Runtime diagnostic — next action

The sampled-gradient implementation remains parked after failing its native
time gate. Measure the costs before choosing another design. This diagnostic
uses artificial inputs and weights only; no new real training is authorized.

Close other GPU workloads. In CMD:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_runtime_profile.cmd
```

Send `results\runtime_profile_v1` when complete (include subfolders and traces).
The command performs correctness checks, runs the frozen diagnostic and audits
its output. Rerun the same command after an interruption: completed work is
validated and skipped, while incomplete attempts are preserved.

There are 12 conditions: two seeds, two batch sizes, three implementations
(native, exact custom and sampled custom). The experiment separates forward,
loss/readback, backward and optimizer costs, with paired controls for
instrumentation overhead. It includes 336 measured artificial updates and 36
burn-in updates. Profiler times are diagnostic and must not be treated as speed
claims. Missing GPU operator attribution is explicitly reported as unavailable.

Preparation validation uses small artificial CPU/CUDA models; it does not run
this full-shape diagnostic. The frozen protocol is
`protocol/runtime_profile/PROFILE_PROTOCOL_v1.md`. Review the full result before
choosing the next method. No accuracy claim or automatic follow-up follows.
