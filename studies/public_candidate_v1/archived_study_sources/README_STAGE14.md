# Idea 4: compress intermediate tokens

This short diagnostic reuses two trained ViTs. After three of six blocks,
compare all 64 patches against 32 merged patches, with spatial and similarity
merging. CLS stays separate; later attention accounts for merged group sizes.
Check development accuracy plus inference and forward/backward GPU costs.
No optimizer updates, new training, learning-rate tuning or test-set access.

Run in Windows Command Prompt, with other GPU work stopped:

```bat
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_stage14A_token_diagnostic.cmd
```

Output: `results/stage14A_token_diagnostic_v1`. Say "done" when complete.
Rerunning completed output verifies hashes and arithmetic without rerunning
measurements. Interrupted attempts remain preserved; a fresh attempt is used.
Do not change the frozen recipe after seeing results.

Both seeds must lose no more than 0.30 accuracy percentage points and save at
least 10% forward/backward time OR 20% incremental allocated memory. Spatial
averaging is a practical control for the similarity method. Review any pass
before designing a training pilot. These costs exclude an optimizer step and
cannot establish faster learning. Previously used development checkpoints and
validation examples make this exploratory.

See `protocol/stage14/TOKEN_DIAGNOSTIC_v1.md` and its validation record. The
similarity arm is ToMe-inspired, not a reproduction or generic novelty claim.
Earlier input-compression failures and the bounded local-learning findings are
retained in `paper/RESEARCH_DISCUSSION_MEMORY_v1.md`. Publication stays on hold.
