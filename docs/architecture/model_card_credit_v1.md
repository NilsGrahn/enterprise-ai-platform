# Model Card — credit v1

- **Evaluated on:** `test` split
- **Generated:** 2026-08-11T15:59:07.582703+00:00
- **Trained at:** 2026-08-11T15:57:26.706218+00:00
- **Algorithm:** XGBClassifier
- **Training rows:** 104,790
- **Features:** 17

## Metrics

| Metric | Value |
|---|---|
| AUC | 0.8659 |
| KS | 0.5795 |
| Precision | 0.2158 |
| Recall | 0.7781 |
| F1 | 0.3379 |
| Threshold | 0.50 |

## Confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 16,608 | 4,245 |
| **Actual 1** | 333 | 1,168 |

## Decile lift

Decile 0 is the highest predicted risk.

| Decile | n | Actual positives | Actual rate | Mean score | Lift |
|---|---|---|---|---|---|
| 0 | 2,236 | 818 | 0.3658 | 0.8521 | 5.45x |
| 1 | 2,235 | 270 | 0.1208 | 0.6384 | 1.80x |
| 2 | 2,235 | 157 | 0.0702 | 0.4888 | 1.05x |
| 3 | 2,236 | 89 | 0.0398 | 0.3620 | 0.59x |
| 4 | 2,235 | 60 | 0.0268 | 0.2648 | 0.40x |
| 5 | 2,235 | 43 | 0.0192 | 0.1907 | 0.29x |
| 6 | 2,236 | 25 | 0.0112 | 0.1449 | 0.17x |
| 7 | 2,235 | 20 | 0.0089 | 0.1130 | 0.13x |
| 8 | 2,235 | 9 | 0.0040 | 0.0858 | 0.06x |
| 9 | 2,236 | 10 | 0.0045 | 0.0567 | 0.07x |
