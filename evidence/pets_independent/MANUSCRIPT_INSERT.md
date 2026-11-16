# Manuscript-ready result paragraph (for integration; not yet in v10)

In a prospectively specified Oxford-IIIT Pet evaluation using an ImageNet-pretrained
ResNet-18 and 24 new adaptation seeds, a fixed frozen-prefix policy reduced
complete adaptation time by 34.96% (paired log-ratio 95% interval, 34.66–35.25%)
relative to optimized BF16 full backpropagation. The mean final-test accuracy
difference was -0.114 percentage points (paired 95% interval, -0.262 to +0.035),
satisfying the prespecified 0.30-point noninferiority margin and the combined
quality/time criterion for this secondary comparison. The primary periodic-r4
comparison reduced time by 25.83% (25.44–26.21%) but incurred a 0.558-point mean
accuracy deficit (difference interval, -0.721 to -0.394), failing its accuracy
criterion. Thus, on this workload, periodic refresh was not required to obtain
useful savings and did not improve the observed tradeoff over freezing. Claims
follow the frozen two-comparison procedure; uncertainty is conditional on one
pretrained source, the shared official test sample and a single GPU timing session.
Complete adaptation clocks exclude separately recorded official-test assessment,
pretraining and one-time dataset preparation.

Keep the failed primary comparison visible in the abstract/discussion alongside
the successful secondary control. This paragraph reports an empirical comparison;
it does not introduce freezing as a novel optimization method.
