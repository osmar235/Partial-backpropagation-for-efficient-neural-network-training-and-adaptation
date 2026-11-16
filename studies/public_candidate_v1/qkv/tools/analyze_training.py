"""Recompute the published session-level timing summaries; standard library only.

This analyzes exported audited pairs. It does not rescore checkpoints or repeat
the provenance/trajectory audit performed in the original workspace.
"""
import csv
import json
import math
from pathlib import Path
from statistics import median


def analyze(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert len({(r['session'], r['seed']) for r in rows}) == 16
    summary = []
    for seed in ('1501', '1502'):
        group = [r for r in rows if r['seed'] == seed]
        assert {int(r['session']) for r in group} == set(range(8))
        assert sum(r['native_first'] == 'True' for r in group) == 4
        savings = []
        for row in group:
            native, candidate = float(row['native_seconds']), float(row['candidate_seconds'])
            assert math.isfinite(native) and math.isfinite(candidate) and min(native, candidate) > 0
            saving = 100 * (native - candidate) / native
            assert math.isclose(saving, float(row['saving_percent']), abs_tol=1e-10)
            assert float(row['final_accuracy']) == float(row['candidate_accuracy'])
            assert int(row['peak_delta_bytes']) == 0
            savings.append(saving)
        positive = sum(s > 0 for s in savings)
        nonzero = sum(s != 0 for s in savings)
        p = sum(math.comb(nonzero, k) for k in range(positive, nonzero + 1)) / 2**nonzero
        summary.append(dict(
            source_seed=int(seed), pairs=len(group), positive_pairs=positive,
            median_paired_time_reduction_percent=median(savings),
            median_paired_seconds_saved=median(float(r['native_seconds']) - float(r['candidate_seconds']) for r in group),
            adjusted_one_sided_sign_p=min(1, 2*p),
            final_accuracy_both_arms=float(group[0]['final_accuracy']),
            native_first_median_percent=median(s for r, s in zip(group, savings) if r['native_first'] == 'True'),
            candidate_first_median_percent=median(s for r, s in zip(group, savings) if r['native_first'] == 'False'),
        ))
    return summary


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'evidence/training_pairs.csv'
    print(json.dumps(analyze(path), indent=2))
