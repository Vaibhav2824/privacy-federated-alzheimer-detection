# Fix 1: private (eps = 1) vs non-private, same subjects

## Reproduction gate

| Task | Non-private (paper) | Non-private (rerun) | eps=1 (paper) | eps=1 (rerun) |
|---|---|---|---|---|
| CN vs AD | 88.5 | 88.5 | 89.5 | 89.5 |
| CN/MCI/AD | 61.4 | 61.4 | 61.8 | 61.8 |

Gate passed (within ±0.3 points): **yes**

## Table A

| Task | Seed | Acc non-private | Acc ε=1 | Acc diff [95% CI] | McNemar b / c | McNemar p | AUROC non-private | AUROC ε=1 | AUROC test p |
|---|---|---|---|---|---|---|---|---|---|
| CN vs AD | 42 | 88.6 | 89.4 | +0.8 [-0.9, +2.5] | 25 / 32 | 0.427 | 0.938 | 0.945 | DeLong: 0.030 (bootstrap 0.035) |
| CN vs AD | 123 | 88.0 | 89.5 | +1.5 [-0.2, +3.2] | 22 / 35 | 0.112 | 0.936 | 0.945 | DeLong: 0.006 |
| CN vs AD | 2024 | 89.0 | 89.7 | +0.7 [-0.8, +2.2] | 20 / 26 | 0.461 | 0.945 | 0.949 | DeLong: 0.131 |
| CN/MCI/AD | 42 | 60.0 | 61.9 | +1.9 [-0.1, +3.9] | 70 / 94 | 0.072 | 0.774 | 0.779 | paired bootstrap: 0.440 |
| CN/MCI/AD | 123 | 61.0 | 61.1 | +0.1 [-2.0, +2.0] | 85 / 86 | 1.000 | 0.770 | 0.775 | paired bootstrap: 0.458 |
| CN/MCI/AD | 2024 | 63.3 | 62.3 | -1.0 [-2.9, +0.9] | 80 / 67 | 0.322 | 0.777 | 0.784 | paired bootstrap: 0.187 |

b: subjects only the non-private model got right; c: only the ε=1 model.

## Table B: accuracy by sex, ε = 1 model (mean over seeds)

| Task | Sex | n | Accuracy | Wilson 95% CI |
|---|---|---|---|---|
| CN vs AD | female | 439 | 91.8 | [88.9, 94.0] |
| CN vs AD | male | 353 | 89.0 | [85.3, 91.8] |
| CN/MCI/AD | female | 600 | 66.6 | [62.8, 70.3] |
| CN/MCI/AD | male | 531 | 59.3 | [55.1, 63.4] |

Subjects with sex not recorded (excluded from Table B): CN vs AD 82, CN/MCI/AD 131.
FedAvg-vs-eps1 comparison: skipped (no FedAvg out-of-fold files)
