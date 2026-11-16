# Stage10A post-screen decision - 2026-10-21

Post-run decision record; does not amend the prospective specification or freeze.

The author-approved coarse/detail representation screen is complete. Ten
100-epoch runs and 176,000 updates passed independent review. All final
checkpoint accuracies were reproduced on the original training-only validation
split. No test dataset or additional training was used during review.

Both full baselines sustained the 80% primary target. Haar, pixel and coarse-only
arms never sustained it. Haar mean endpoint accuracy was 74.84%, versus full
82.15%, narrow full-resolution 81.43%, coarse 74.96% and pixel 74.73%.
Haar failed practical, detail and structure flags in each seed. This candidate
is rejected under the frozen development criteria.

The narrow full-resolution control reached the target 40.10% / 21.62% sooner
than full but lost 0.12 / 1.32 percentage points at the final endpoint. It fails
the desired two-seed final-quality tolerance and is not a novel-method win.
Its relative success is a local design clue about preserving usable spatial
information. It does not prove the cause of the coarse/detail failures.

Close this tested design. No automatic epoch extension, threshold change,
alternative transform, stronger detail network, optimizer mixture or seed grid.
Further method development remains possible, but requires a distinct mechanism
and a bounded controlled test, rather than modifying this completed screen.
No new training command is needed now. Publication stays on hold.

Do not generalize to all frequency methods: two development seeds, reused
validation, 32x32 images, one shared recipe, a fixed budget and one branch/fusion
allocation limit the inference. Information-preserving transforms alone do not
establish learnability or useful compute savings. Exact cause is unresolved.

Full review: `results/stage10A_review_v1/REPORT.md`. All raw data, unfavorable
outcomes, executed frozen code/protocol, masters and manuscript v7 are retained.
Only mutable current-state pointers are updated; their hashes are recorded.
