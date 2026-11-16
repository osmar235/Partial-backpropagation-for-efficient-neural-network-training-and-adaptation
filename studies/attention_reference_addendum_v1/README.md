# Competitive attention reference — paper-support addendum v1

Prepared 2026-10-26; full experiment pending. This addendum preserves the
validated source closure and prospective freeze for the next experiment.
The previous public_candidate_v1 package remains unchanged.

`source/scripts/attention_reference.py` is the runner;
`source/scripts/review_attention_reference.py` is its read-only audit;
`source/scripts/test_attention_reference.py` contains small checks;
`source/scripts/attention_reference_adapter.py` maps the pinned upstream
Attention class to native parameter/state names. The unchanged QKV adapter
and historical utility dependencies are included with original hashes.

Upstream timm v0.9.2 commit 3d05c0e86f2f4c57bf495468aa3f8a7d3487c986,
complete source and Apache-2.0 license are preserved under source/third_party/.
The exact class forward is used; the wrapper includes a timed output-stride
copy to preserve downstream dropout mask mapping. This is not a whole-timm
model benchmark, a latest-stack comparison, or a novel-unbind claim.

Four CPU and four CUDA tests pass. Corrected tiny workflows audit successfully.
The upstream reference fails the original post-Adam projection-bias gate on
the tiny model, and is excluded from those timings. This is not evidence of
task-quality failure. Full-shape eligibility and relative cost are unmeasured.
The original tolerances remain fixed. Full numerical failures will be recorded.

Use README_ATTENTION_REFERENCE.md in the original workspace to execute the
frozen study. The source capsule retains original provenance paths and is not
claimed to be a clean-directory portable training release. Nothing has been
published. SOURCE_PROVENANCE.json and SHA256SUMS.json identify the copied files.
No dataset or checkpoints are needed by this artificial screen. Mixed precision
and compilation remain separately scoped future work.
