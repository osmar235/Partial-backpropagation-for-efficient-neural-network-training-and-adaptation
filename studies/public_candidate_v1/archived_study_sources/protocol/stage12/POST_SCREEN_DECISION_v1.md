# Stage12A decision — 2026-10-22

The predefined lp_periodic candidate fails final quality in both seeds despite
valid baseline readiness and substantial time saving. Relative to full:
seed1601 -1.52pp /39.91% faster; seed1602 -0.96pp /38.81% faster. Allowed loss
was0.30pp. Relative to the fastest quality-qualified control lp_full, it loses
1.26 /0.76pp while saving31.98% /30.95% time. No acceptance rule is changed.

Preserve these useful descriptive findings:

- Periodic prefix updates recover1.70 /2.20pp over fixed suffix at about18%
  additional time, with matched head warmups and data streams.
- Head warmup then full fine-tuning retains accuracy within0.26 /0.20pp of full
  and saves11.66% /11.38% complete time. It is a conventional control, not a
  newly substituted primary candidate or novelty claim.
- Periodic reaches the predeclared secondary92% target about29–30% sooner than
  lp_full, but this does not erase its final-quality failure or license retargeting.

All42,240updates audited; all ten checkpoint validation accuracies independently
reproduced. No training/test evaluation in review. Prior evidence is preserved.
Details: results/stage12A_review_v1/REPORT.md.

Close this fixed schedule without an automatic tuning grid. The next queued
direction is local objectives / decomposed credit assignment, requiring its own
prospective mechanics-and-cost diagnostic before an efficacy claim. No new
experiment command is prepared in this review. Publication remains on hold.
