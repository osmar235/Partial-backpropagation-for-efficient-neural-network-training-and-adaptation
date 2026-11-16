# Reproduction and environment

## What works independently

`verify_package.py` checks all packaged files. `qkv/tools/analyze_training.py`
reads the included audited pair table, recomputes per-pair percentages, medians,
order strata and the two-family adjusted one-sided sign tests. Percent savings
are medians of paired ratios, not ratios of separate medians. This checks the
reported arithmetic; it does not independently establish the raw trajectory
or checkpoint claims.

`qkv/tools/check_adapter.py` imports only the packaged adapter and PyTorch. It
checks exact small attention outputs, all parameter and input gradients within
1e-12 in float64, parameter/optimizer identity, checkpoint keys and rejection
of unsupported inputs. This is packaging validation, not a replacement for the
original model-level tests, finite differences or performance experiment.

## Measured environment and scope

The completed study used Python 3.14.3, torch 2.11.0+cu128, torchvision
0.26.0+cu128, NumPy 2.4.3, Windows, and an NVIDIA GeForce RTX 5080.
Attention used the math SDPA backend, FP32, TF32 disabled and strict
determinism. These are recorded tested versions, not a claim that all other
versions behave identically. Installing torch for a different OS/GPU requires
the corresponding official distribution; no automatic dependency installer
is included here.

Model: torchvision-based ViT, six blocks, width 192, three heads, MLP width
768, 65 tokens. CIFAR-10 training-only split: 45,000 training / 5,000 validation.
Two existing full source checkpoints (1501/1502), fresh AdamW, five epochs,
batch 256 with final batch 200; eight paired sessions per source. The primary
clock includes per-arm data preparation/device transfer, setup, warmup/restore,
training/augmentation, initial and epoch validation, and ordinary final
serialization. Interpreter/import startup, preflight, provenance hashing and
post-run auditing are excluded. Source pretraining is a shared sunk cost;
OS cache was not purged and file writes were not explicitly fsynced.

Adapter restrictions: self-attention only, batch-first, equal Q/K/V embed
dimensions, biased projections, attention dropout zero, no masks, no causal
mode and `need_weights=False`. It is not a general replacement for every
MultiheadAttention call. `install()` expects the original model's
`model.body.encoder.layers` structure. The existing packed contiguous copy
remains; only the split/autograd construction changed.

## Original workflow source map

Paths below are under `archived_study_sources/`:

| Study | Runner | Audit | Specification/provenance |
|---|---|---|---|
| QKV discovery | `scripts/qkv_layout.py` | `scripts/review_qkv_layout.py` | `protocol/qkv_layout/` |
| Operation confirmation | `scripts/qkv_confirmation.py` | `scripts/review_qkv_confirmation.py` | `protocol/qkv_confirmation/` |
| Real-data confirmation | `scripts/qkv_training.py` | `scripts/review_qkv_training.py` | `protocol/qkv_training/` |
| Original four ideas and later controls | Stage11–15, sampled-weight, runtime-profile and efficient-backprop scripts | Corresponding `review_*.py` scripts | Corresponding `protocol/` directories |

The copies preserve their original hashes. Some historical utilities are
dependencies of newer studies; inclusion does not revive their failed
scientific hypotheses. Original tests and CMD launchers are retained next
to each runner. Frozen sources include installed torchvision fingerprints;
external dependency source is identified, not silently vendored.

For a full independent rerun, the missing distribution work is: export or
reconstruct the two source checkpoints, data split/augmentation streams,
Stage15A reference trajectories and dependent completion manifests; provide
a portable resolver for embedded original paths; then validate from a clean
directory. Dataset redistribution is not included. Original frozen manifests
must remain intact alongside any portable translation and its provenance map.
Until that work is complete, run historical audits only in their original
workspace, against original inputs. Do not interpret a copied launcher as a
self-contained reproduction command.
