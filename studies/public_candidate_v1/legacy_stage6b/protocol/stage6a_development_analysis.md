# Stage 6A development analysis

All results below use seed 40 and are development evidence only. Seeds 41-49
remain unopened.

## Thirty-epoch augmented protocol

| Run order | Method | Test accuracy | Training time | Partial updates |
|---|---|---:|---:|---:|
| Adam, Adan, candidate | Adam | 84.87% | 279.88 s | 0% |
| Adam, Adan, candidate | Adan, lr 0.0003 | 78.41% | 331.00 s | 0% |
| Adam, Adan, candidate | Frozen candidate | 84.81% | 294.73 s | 36.41% |
| Candidate, Adam | Frozen candidate | 84.52% | 246.86 s | 36.29% |
| Candidate, Adam | Adam | 85.11% | 271.00 s | 0% |

The candidate-minus-Adam final accuracy differences were -0.06 and -0.59
percentage points, averaging -0.325 points. Raw candidate/Adam wall-time ratios
were 1.053 and 0.911. Their geometric order-balanced ratio was 0.979, equivalent
to a 2.1% reduction, but the order interaction was too large for a speed claim.

## Timing qualification

One algorithm-specific warm-up epoch was added before every timed run. Warm-up
uses a separate random stream, is excluded from timing, and is followed by full
restoration of model and optimizer state.

| Order | Adam | Candidate | Candidate reduction |
|---|---:|---:|---:|
| Adam first | 43.65 s | 38.53 s | 11.72% |
| Candidate first | 44.16 s | 40.21 s | 8.96% |

The geometric order-balanced reduction was 10.35%. This qualifies the warm-up
protocol for subsequent timing, but the five-epoch accuracy gap is not a final
quality estimate because the learning-rate schedule and horizon differ from the
primary 30-epoch endpoint.

## Current interpretation

The fixed controller preserves most of Adam's 30-epoch accuracy and executes
about 36% of updates with a partial backward pass. A stable timing protocol can
show roughly 9-12% throughput savings in the early schedule. The full 30-epoch
speed result still needs confirmation with standardized warm-up and balanced
order across held-out seeds. Adan also requires a larger, development-selected
learning rate before it can serve as a strong baseline.

## Adan learning-rate screen

The official Adan guidance recommends a peak learning rate five to ten times
the Adam/AdamW rate. Three values were screened for 10 epochs while retaining
the 30-epoch cosine horizon.

| Adan learning rate | Epoch-10 test accuracy |
|---:|---:|
| 0.001 | 69.80% |
| 0.0015 | 66.84% |
| 0.003 | 64.04% |

The prespecified rule selects 0.001 for the full development curve and later
held-out comparison.

The full 30-epoch tuned-Adan run reached 82.12% in 263.90 seconds. It improved
the original 0.0003 result by 3.71 percentage points, but remained 2.40-2.69
points below the two frozen-candidate development endpoints.

## First held-out seed

Seed 41 was opened only after the confirmatory protocol was frozen. Adam reached
84.79% in 250.15 seconds, the candidate reached 84.59% in 218.58 seconds with
36.28% partial updates, and tuned Adan reached 83.31% in 245.99 seconds. Thus
the candidate was 0.20 percentage points below Adam and 12.62% faster on this
seed; it was 1.28 points more accurate and 11.14% faster than tuned Adan. This is
one of nine confirmatory seeds and is not sufficient by itself for inference.
