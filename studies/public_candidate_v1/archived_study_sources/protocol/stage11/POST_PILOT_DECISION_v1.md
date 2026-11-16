# Stage11B decision — 2026-10-22

Frozen midpoint growth fails the both-seed acceptance rule. Full readiness and
the time-saving condition pass in both seeds; final quality passes only seed1502.
Seed1501: -1.40pp /23.54% time saving. Seed1502: -0.12pp /21.99% saving.
The allowed accuracy deficit was0.20pp, with >=5% saving in both seeds.

Integrity audit passes176,000 updates; all ten final checkpoint accuracies were
independently reproduced. All prior evidence and real outputs are preserved.
Details and limitations: results/stage11B_review_v1/REPORT.md.

The result narrows the issue: feasible mapping and cost headroom did not ensure
reliable training recovery. Growth beats small-reset accuracy by0.36 /1.12pp,
but has an observed post-expansion training dip. These are descriptive contrasts,
not proof of a recovery mechanism or of general usefulness.

Close this tested recipe without changing criteria or extending training. Do not
promote its passing seed as confirmation. Preserve growth as an unresolved future
mechanism question; do not claim all possible growth approaches are ruled out.
The next queued idea in the agreed sequence is pretrained adaptation/selective
updates, requiring a separate prospective protocol. No new training command in
this review. Publication and manuscript revision remain on hold.
