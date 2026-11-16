# Optimized attention/copy profile — source addendum v1

Prepared and frozen 2026-10-28; full scientific run pending. Use
source/README_ATTENTION_COPY_PROFILE.md in the original project workspace.

Bounded operation-cost attribution with full gradients under BF16/automatic.
Two source checkpoints, batches 256/200, original/disabled dropout, compatible
attention layout and native-layout diagnostic control. Plain wall timing is
separate from events and profiler overhead. Native layout can change dropout
mapping; removing our compatibility copy is not equivalent-method qualification
or a novel improvement over upstream practice. No accuracy campaign is launched.

Six CPU and six CUDA correctness tests and both artificial workflows passed.
GPU operator attribution is available in the tiny CUDA smoke; instrumented
regimes preserve their matching plain trajectories. Full-size behavior remains
to be measured. No real forward/backward/optimizer update during preparation.

The source dependency closure, pinned upstream attention/license and protocol
are byte-identical copies listed in SOURCE_PROVENANCE. Installed-source hashes
and local data/checkpoint requirements are separate. Absolute provenance remains;
this capsule needs separate portability work before external release.

Raw data/checkpoints, internal preparation helpers and smoke/trace evidence are
kept at original project locations. Internal history: protocol/attention_copy_profile
and results/attention_copy_profile_preparation_v1. No external publishing.
