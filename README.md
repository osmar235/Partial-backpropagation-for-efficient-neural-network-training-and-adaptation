# Partial backpropagation for efficient neural network training and adaptation

Code, frozen experimental protocols and processed evidence supporting the manuscript by **Osmar Pinto Neto**, Department of Kinesiology, California State University San Marcos. This collection corresponds to manuscript v14 and its subsequent availability-only revision.

The study examines when omitting prefix gradients reduces measured training or adaptation time, how Adam states behave during omitted updates, and whether savings persist against optimized execution. It includes positive results, failed accuracy screens and comparisons with conventional alternatives.

## Main findings

| Study | Evidence and interpretation |
| --- | --- |
| CNN confirmation | Nine independent training seeds; LGPB-MC reduced training time by 10.24% relative to Adam, with an accuracy difference of −0.548 percentage points (95% CI −0.759 to −0.337), within the prespecified 1-point margin. |
| CNN mechanism controls | Matched periodic and random schedules achieved similar savings; optimizer-state behavior contributed to the observed accuracy differences. |
| ViT training from scratch | The original controller did not transfer successfully. Its accuracy deficit was substantially larger than that of the periodic control. |
| Optimized ViT adaptation | Conventional BF16 automatic attention execution saved about 43% relative to the conventional FP32 reference. Periodic prefix updates saved a further 19.69–20.84% relative to optimized full backpropagation, but exceeded the 0.30-point accuracy screen for one of two source checkpoints. Timing repetitions are not independent accuracy seeds. |
| Independent Oxford-IIIT Pet evaluation | Across 24 paired training seeds, frozen-prefix adaptation saved 34.96% and met the prespecified accuracy criterion. Periodic updates saved 25.83% but failed that criterion. These are repeated adaptations of one pretrained backbone, not 24 independently pretrained models. |
| Attention layout | Local QKV layout improvements were measurable, but the competitive reference already used the relevant layout. This is not presented as a new attention algorithm. |

See [the evidence map](EVIDENCE_INDEX.json) and individual reports for estimands, uncertainty, timing definitions and limits. Percentages from different comparisons do not share a common denominator and should not be added.

## Start here

Use Python 3.11 or later. The analysis replay was checked with Python 3.14.3, NumPy 2.4.3 and SciPy 1.17.1.

```console
python -m pip install -r requirements-analysis.txt
python -B tools/verify_package.py
python -B tools/reproduce_analyses.py
```

These commands verify distributed file hashes and replay selected principal analyses from included evidence. They do not retrain networks or require a GPU. Outputs go into a new folder under `reproduced/`.

See [REPRODUCING.md](REPRODUCING.md) for the scope of each replay and the additional requirements for training. Passing the replay checks validates the included calculations; it does not independently regenerate every training run.

## Organization

- `studies/public_candidate_v1/`: original CNN code, historical experiment sources, selected positive and negative evidence, and the QKV adapter.
- `studies/*_addendum_v1/`: frozen experiment source and protocol capsules for subsequent ViT execution, optimizer and adaptation investigations.
- `studies/*_results_v1/`: corresponding compact results, analyses and verification records, including the independent Pet evaluation.
- `studies/pets_download_recovery_v2/`: archive transport repair; experimental definitions are unchanged.
- `evidence/`: consolidated study reports, processed tables and compact per-run CNN/ViT records.
- `SOURCE_PROVENANCE.json`: source-relative paths and hashes for copied materials; manuscript source identity.
- `SHA256SUMS.json`: authoritative manifest for this publication tree.

The root documents describe the current collection. Nested README files are historical snapshots and may mention earlier pending work or local publication status. Original machine paths in frozen provenance identify the archived execution environment; they are not installation paths for a fresh checkout. Internal project masters, reviewer correspondence, research planning files and manuscript drafts are excluded.

## Availability and citation

Repository: https://github.com/osmar235/Partial-backpropagation-for-efficient-neural-network-training-and-adaptation

Release **v1.0.1** is archived in Zenodo: **[10.5281/zenodo.24268905](https://doi.org/10.5281/zenodo.24268905)**. Use this version-specific DOI to cite the exact code and processed-evidence package supporting the manuscript. The archived release corresponds to GitHub commit `be13a61d9ffe9ba657a9a549dfaa278aa8e4c622`; subsequent citation-documentation updates on `main` do not modify that archive.

Suggested citation: Pinto Neto, O. (2026). *Partial backpropagation for efficient neural network training and adaptation* (Version 1.0.1) [Computer software and processed evidence]. Zenodo. https://doi.org/10.5281/zenodo.24268905

The concept DOI, [10.5281/zenodo.24268904](https://doi.org/10.5281/zenodo.24268904), refers to the version collection. This is a pre-submission research deposit; no journal publication or acceptance is implied.

Original software is licensed under Apache 2.0; original research artifacts under CC BY 4.0. Third-party licenses and notices remain in place. See [LICENSE.md](LICENSE.md).
