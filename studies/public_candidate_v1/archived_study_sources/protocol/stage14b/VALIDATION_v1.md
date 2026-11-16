# Stage14B preparation validation — 2026-10-24

No real Stage14B training or prediction scoring was performed during preparation.
All training tests used artificial inputs and weights. Real checkpoint checks
were limited to hashes, parameter mappings, saved optimizer moments and ages.

## Tests passed

`scripts/test_stage14_recovery.py`: six CPU tests and the same six CUDA tests:

- Exact restoration of mature Adam state and isolation of discarded warmup.
- Identical native/full-wrapper updates and optimizer states.
- Rejection of incorrect optimizer ages and parameter ordering.
- Full-size six-block architecture updates in all three modes, including
  finite gradients and nonzero gradients to every positional token.
- Matched streams and exact warmup/cosine endpoints.
- Quality boundaries, source-quality anchor, simpler-control preference,
  memory-only route and prohibition of artificial scientific passes.

Both complete artificial workflows pass:
`results/stage14B_sanity_recovery_v1_final_cpu` and
`results/stage14B_sanity_recovery_v1_final_gpu`. Each executes six trajectories,
72 retained updates, six discarded warmup updates and 18 curve points.
Audits independently reconstruct streams, labels, metrics, optimizer ages,
cost arithmetic, parameter states and acceptance decisions. Random artificial
scores do not qualify the reference or either scientific candidate; this is
expected and does not block the plumbing audit.

A completed CPU rerun audits/skips without modifying any output. The before
manifest is `cpu_before_resume_sha256_v1.json`, verified when preparation seals.
A separate artificial restart test retains `attempt_001`, executes in
`attempt_002`, skips its completed arm with all 11 files unchanged, rejects an
intentionally corrupted step file, then restores/verifies the original bytes.
See `restart_validation_v1.json` and `results/stage14B_sanity_restart_v1`.

## Real source mapping, read-only

`validate_sources.py` checks both Stage11B full checkpoints against Stage14A's
recorded state hashes. Each has 2,693,578 parameters in 80 parameter tensors;
all AdamW moments have matching shapes and finite values, with age 17,600.
The ordered mapping, weight decay, betas and epsilon match the recipe.
No real forward pass or optimizer update was used. Details and hashes are in
`source_mapping_validation_v1.json`.

## Freeze and preservation

The prospective freeze is `FROZEN_TOKEN_RECOVERY_v1.json`. Code, spec, prior
scientific dependencies, evidence, real source checkpoints and previous logits
must match before real execution. The final synthetic freezes must contain the
same scientific source hashes as the real freeze.

`preparation_preservation_v1.json` protects 1,396 prior immutable files. The
finish step verifies them before and after updating only CURRENT_RELEASE.md
and README_STAGE9.md. It creates master v7.13, retains all previous masters
and paper memory v4, and verifies that the real output directory is absent.
`preparation_verification_v1.json` records the resulting counts; the final
`preparation_sha256_v1.json` seals preparation and final artificial artifacts.

## Real execution

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage14B_token_recovery.cmd
```

The user runs this after preparation. Review the fixed endpoint after all six
runs. No best-epoch selection, automatic extension, test scoring or publication
claim. The earlier Stage14A failed outcome and Stage13C negative result remain
unchanged regardless of this follow-up's outcome.
