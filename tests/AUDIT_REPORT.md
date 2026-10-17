# Independent checks with open-source audit tools

## Code audit (October 2026)

- **Static analysis:** ESLint 9 (recommended rules) over every source file: no undefined names, unreachable code
  or other logic errors; unused imports and variables left over from refactoring were removed.
- **Mathematics against Python libraries:** 66 reference checks against SciPy, scikit-learn, statsmodels,
  scikit-survival and pmsampsize, run on every push. Added in this audit: the time-dependent AUC and
  censoring-weighted Brier score (scikit-survival; the app now follows its conventions for tied times and
  matches to 1e-16) and the Riley minimum sample size (pmsampsize; every criterion identical).
- **Behaviour across browsers:** 19 browser tests in Chromium, Firefox and WebKit on every push, including
  combinations most likely to break (settings round-trip with the newer controls; export and re-import of a
  multiply imputed, recalibrated model with identical predictions; batch prediction with blanks and an
  out-of-range value; Option B with multiple imputation; the pipeline with recalibration and its report) and
  an axe-core accessibility scan (WCAG 2.1 AA) of every main screen in both themes.
- **Engine tests:** 25 end-to-end checks (leakage on pure noise, .632+, patient-level bootstrap, Shapley values
  against brute force, multiple imputation, reproducibility with fixed seeds, worker = page results), 9
  edge-case checks and 9 neural-network forward-pass checks.

## Open-source audit tools (Fairlearn and Deepchecks)

October 2026. The app's own engine (preprocessing, models, subgroup code) was run on the simulated iNPH cohort
(`medical_shunt_cohort_large.csv`, Option A split with seed 42: 336 training and 84 test patients) by
`tests/audit_app_runner.js`, and its outputs were checked with two widely used open-source tools. The raw
results are in [`AUDIT_FAIRLEARN.md`](AUDIT_FAIRLEARN.md) and [`AUDIT_DEEPCHECKS.md`](AUDIT_DEEPCHECKS.md).

## Summary

| Tool | What was checked | Result |
|---|---|---|
| Fairlearn 0.15 (MetricFrame) | The app's subgroup panel: patients, events, observed and mean predicted risk, O:E, AUC, Brier score, sensitivity, specificity and net benefit by sex, hospital and age band, for logistic regression, Random Forest and boosted trees (240 figures) | **All agree** (largest difference 4×10⁻¹⁶; AUC within its 4-decimal rounding) |
| Deepchecks 0.19 – data integrity | Duplicates, conflicting labels, mixed types and nulls, single-value columns, string problems, highly correlated predictors | **All pass** |
| Deepchecks – train/test validation | The app's split: test records found in the training set (leakage), new categories, feature, label and multivariate drift | **All pass** (no samples mix; domain-classifier AUC 0.53, i.e. training and test patients are indistinguishable) |
| Deepchecks – repeated patients | Every patient twice (two visits); random split by row vs the app's "keep each patient's records together" | Split by row: **FAIL**, 75% of test records also in training. The app's patient-level split: **PASS**, no mixing |
| Deepchecks – model evaluation | The app's three trained models, called through the app's own preprocessing | ROC, inference and prediction checks pass; the warnings below are expected and explained |

## The Deepchecks warnings, and what they mean

- **Train Test Performance (FAIL for all three models).** Performance is lower on test than on training
  patients (e.g. Random Forest AUC 0.98 training vs 0.78 test). This is overfitting, which every flexible model
  shows on 336 patients; it is why the app reports test-set (Options A/B) or bootstrap-corrected (Option C,
  .632+ for flexible models) performance rather than training performance. Not a software defect.
- **Simple Model Comparison (FAIL).** At the 0.5 threshold, the F1 score for "Responder" is slightly lower than
  that of a model that calls everyone a responder (69% of patients are). F1 for the majority class rewards
  predicting the majority; the models do discriminate (test AUC 0.75–0.80, against 0.5 for the simple model).
  It is a reminder that the decision threshold should be chosen clinically (the app's threshold table, decision
  curve and Youden's J help).
- **Prediction Drift (FAIL for the Random Forest and boosted trees).** Their predictions for training patients are
  more extreme than for test patients, another sign of overfitting rather than of a difference between the
  training and test patients (the data drift checks all pass).
- **Weak Segments Performance (WARN).** Deepchecks searches many combinations of one or two predictors for
  the subgroup with the worst accuracy; on the test set the worst segments hold only 5–8 patients
  (e.g. Baseline ICP 7.5–8.3 mmHg), so low accuracy there is expected by chance alone. Subgroups chosen in
  advance (sex, hospital, age band) are assessed in the app's subgroup panel instead.
- **Unused Features (WARN for the tree models).** Many predictors contribute little to the Random Forest and
  boosted trees. In this simulated cohort only a few predictors carry most of the signal, so this is expected.

## Disparity measures (Fairlearn, threshold 0.5)

Differences between hospitals and age bands in sensitivity and false-positive rate are large
(equalised odds difference 0.47–0.80), but each group has only about 28 test patients, so these figures are very
imprecise. The data are simulated without any intended unfairness; with real data, groups of this size could
not show whether a model is fair.

## How to repeat these checks

Both tools are Python libraries and need their own environments (Deepchecks 0.19 needs older versions of
scikit-learn and pandas than Fairlearn):

```bash
python3 -m venv .venv-fairlearn && .venv-fairlearn/bin/pip install fairlearn pandas
python3 -m venv .venv-deepchecks && .venv-deepchecks/bin/pip install "setuptools<81" deepchecks==0.19.1 scikit-learn==1.5.2 category_encoders==2.6.4 pandas==2.2.3 numpy==1.26.4
node tests/audit_app_runner.js /tmp/ummp-audit
.venv-fairlearn/bin/python tests/audit_fairlearn.py /tmp/ummp-audit
.venv-deepchecks/bin/python tests/audit_deepchecks.py /tmp/ummp-audit
```

The Fairlearn check exits with an error if any figure disagrees. The Deepchecks run takes about 2 minutes
and also writes its full HTML reports into the audit folder.
