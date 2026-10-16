# Fairlearn cross-check of the subgroup panel

Fairlearn 0.15.0. Cohort: medical_shunt_cohort_large.csv (simulated), Option A split (seed 42): 336 training and 84 test patients; positive class "Responder"; threshold 0.5.

Every subgroup figure of the app (src/subgroups.js) was recomputed with Fairlearn MetricFrame and scikit-learn.

| Model | Subgroup column | Groups | Figures compared | Largest difference (AUC) | Largest difference (all other figures) | Result |
|---|---|---|---|---|---|---|
| glm | Sex | 2 | 20 | 1.4e-05 | 0.0e+00 | PASS |
| glm | Site | 3 | 30 | 2.2e-05 | 1.4e-17 | PASS |
| glm | Age | 3 | 30 | 4.3e-05 | 1.1e-16 | PASS |
| rf | Sex | 2 | 20 | 2.6e-05 | 1.1e-16 | PASS |
| rf | Site | 3 | 30 | 4.9e-05 | 1.1e-16 | PASS |
| rf | Age | 3 | 30 | 3.0e-05 | 4.4e-16 | PASS |
| gbm | Sex | 2 | 20 | 2.9e-05 | 1.1e-16 | PASS |
| gbm | Site | 3 | 30 | 5.0e-05 | 2.2e-16 | PASS |
| gbm | Age | 3 | 30 | 4.5e-05 | 2.2e-16 | PASS |

**All figures agree** (tolerance 1e-09; AUC 5e-5, because the app rounds it to 4 decimal places).

## Fairlearn disparity measures (test patients, threshold 0.5)

Demographic parity difference: the largest difference between groups in the share predicted as responders. Equalised odds difference: the larger of the largest between-group differences in sensitivity and in false-positive rate. 0 means no difference; the groups are small, so these figures are imprecise.

| Model | Column | Demographic parity difference | Equalised odds difference |
|---|---|---|---|
| glm | Sex | 0.095 | 0.103 |
| glm | Site | 0.252 | 0.618 |
| glm | Age | 0.344 | 0.478 |
| rf | Sex | 0.024 | 0.077 |
| rf | Site | 0.093 | 0.600 |
| rf | Age | 0.308 | 0.467 |
| gbm | Sex | 0.095 | 0.308 |
| gbm | Site | 0.055 | 0.800 |
| gbm | Age | 0.337 | 0.467 |
