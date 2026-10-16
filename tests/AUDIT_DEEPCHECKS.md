# Deepchecks audit

Deepchecks 0.19.1 (open-source tabular suites). Cohort: medical_shunt_cohort_large.csv (simulated), the app's Option A split (seed 42): 336 training and 84 test patients. Models are the app's own (preprocessing and model, run through Node).

Results are PASS, WARN (worth a look) or FAIL against Deepchecks' default thresholds, which are generic: a WARN or FAIL is a prompt to look, not necessarily a defect.

## 1. Data integrity (whole cohort)

Full report: `deepchecks_data_integrity.html` in the audit folder.

| Check | Result | Details |
|---|---|---|
| Single Value in Column | PASS | Passed for 15 relevant columns |
| Special Characters | PASS | Passed for 15 relevant columns |
| Mixed Nulls | PASS | Passed for 15 relevant columns |
| Mixed Data Types | PASS | 15 columns passed: found 0 columns with negligible types mix, and 15 columns without any types mix |
| String Mismatch | PASS | Passed for 1 relevant column |
| Data Duplicates | PASS | Found 0% duplicate data |
| String Length Out Of Bounds | PASS | Passed for 1 relevant column |
| Conflicting Labels | PASS | Ratio of samples with conflicting labels: 0% |
| Feature Label Correlation | PASS | Passed for 14 relevant columns |
| Feature-Feature Correlation | PASS | All correlations are less than 0.9 except pairs [] |
| Outlier Sample Detection | info | no condition (displayed in the HTML report) |
| Identifier Label Correlation - Train Dataset | not run | Dataset does not contain an index or a datetime |

## 2. Train-test validation (the app's split)

Full report: `deepchecks_train_test.html` in the audit folder.

| Check | Result | Details |
|---|---|---|
| Datasets Size Comparison | PASS | Test-Train size ratio is 0.25 |
| New Label Train Test | PASS | Found 0 new labels in test data: [] |
| New Category Train Test | PASS | Passed for 4 relevant features |
| String Mismatch Comparison | PASS | Passed for 5 relevant columns |
| Train Test Samples Mix | PASS | No samples mix found |
| Feature Label Correlation Change | PASS | Passed for 14 relevant columns |
| Feature Label Correlation Change | PASS | Passed for 14 relevant columns |
| Feature Drift | PASS | Passed for 14 columns out of 14 columns.
Found column "Site" has the highest categorical drift score: 0
Found column "Symptom_Duration_Months" has the highest numerical drift score |
| Label Drift | PASS | Label's drift score Cramer's V is 0 |
| Multivariate Drift | PASS | Found drift value of: 0.06, corresponding to a domain classifier AUC of: 0.53 |
| Date Train Test Leakage Duplicates | not run | Dataset does not contain a datetime |
| Date Train Test Leakage Overlap | not run | Dataset does not contain a datetime |
| Index Train Test Leakage | not run | Dataset does not contain an index |

## 3. Model evaluation: glm

Full report: `deepchecks_model_glm.html` in the audit folder.

| Check | Result | Details |
|---|---|---|
| Train Test Performance | FAIL | 2 scores failed. Found max degradation of 18.27% for metric Precision and class Non-Responder. |
| Simple Model Comparison | FAIL | Found classes with failed metric's gain: {'Responder': {'F1': '-3.58%'}} |
| Weak Segments Performance - Train Dataset | WARN | Found a segment with accuracy score of 0.429 in comparison to an average score of 0.774 in sampled data. |
| Weak Segments Performance - Test Dataset | WARN | Found a segment with accuracy score of 0 in comparison to an average score of 0.738 in sampled data. |
| ROC Report - Train Dataset | PASS | All classes passed, minimum AUC found is 0.8 for class Responder |
| ROC Report - Test Dataset | PASS | All classes passed, minimum AUC found is 0.8 for class Responder |
| Prediction Drift | PASS | Found model prediction Kolmogorov-Smirnov drift score of 0.14 |
| Unused Features - Train Dataset | PASS | Found 3 high variance unused features |
| Unused Features - Test Dataset | PASS | Found 3 high variance unused features |
| Model Inference Time - Train Dataset | PASS | Found average inference time (seconds): 1.72e-06 |
| Model Inference Time - Test Dataset | PASS | Found average inference time (seconds): 6.49e-06 |
| Confusion Matrix Report - Train Dataset | info | no condition (displayed in the HTML report) |
| Confusion Matrix Report - Test Dataset | info | no condition (displayed in the HTML report) |
| Calibration Metric - Train Dataset | info | no condition (displayed in the HTML report) |
| Calibration Metric - Test Dataset | info | no condition (displayed in the HTML report) |
| Regression Error Distribution - Train Dataset | not run | Check is irrelevant for classification tasks |
| Regression Error Distribution - Test Dataset | not run | Check is irrelevant for classification tasks |
| Boosting Overfit | not run | Check is relevant for Boosting models of type ('AdaBoostClassifier', 'GradientBoostingClassifier', 'LGBMClassifier', 'XGBClassifier', 'CatBo |

(8 batches of predictions computed by the app's engine.)

## 3. Model evaluation: rf

Full report: `deepchecks_model_rf.html` in the audit folder.

| Check | Result | Details |
|---|---|---|
| Train Test Performance | FAIL | 6 scores failed. Found max degradation of 42.54% for metric Precision and class Non-Responder. |
| Prediction Drift | FAIL | Found model prediction Kolmogorov-Smirnov drift score of 0.21 |
| Simple Model Comparison | FAIL | Found classes with failed metric's gain: {'Responder': {'F1': '-7.44%'}} |
| Weak Segments Performance - Test Dataset | WARN | Found a segment with accuracy score of 0 in comparison to an average score of 0.714 in sampled data. |
| Unused Features - Train Dataset | WARN | Found 11 high variance unused features |
| Unused Features - Test Dataset | WARN | Found 11 high variance unused features |
| ROC Report - Train Dataset | PASS | All classes passed, minimum AUC found is 0.98 for class Responder |
| ROC Report - Test Dataset | PASS | All classes passed, minimum AUC found is 0.78 for class Responder |
| Weak Segments Performance - Train Dataset | PASS | Found a segment with accuracy score of 0.812 in comparison to an average score of 0.881 in sampled data. |
| Model Inference Time - Train Dataset | PASS | Found average inference time (seconds): 1.6e-06 |
| Model Inference Time - Test Dataset | PASS | Found average inference time (seconds): 5.8e-06 |
| Confusion Matrix Report - Train Dataset | info | no condition (displayed in the HTML report) |
| Confusion Matrix Report - Test Dataset | info | no condition (displayed in the HTML report) |
| Calibration Metric - Train Dataset | info | no condition (displayed in the HTML report) |
| Calibration Metric - Test Dataset | info | no condition (displayed in the HTML report) |
| Regression Error Distribution - Train Dataset | not run | Check is irrelevant for classification tasks |
| Regression Error Distribution - Test Dataset | not run | Check is irrelevant for classification tasks |
| Boosting Overfit | not run | Check is relevant for Boosting models of type ('AdaBoostClassifier', 'GradientBoostingClassifier', 'LGBMClassifier', 'XGBClassifier', 'CatBo |

(8 batches of predictions computed by the app's engine.)

## 3. Model evaluation: gbm

Full report: `deepchecks_model_gbm.html` in the audit folder.

| Check | Result | Details |
|---|---|---|
| Train Test Performance | FAIL | 5 scores failed. Found max degradation of 41.26% for metric Precision and class Non-Responder. |
| Prediction Drift | FAIL | Found model prediction Kolmogorov-Smirnov drift score of 0.15 |
| Simple Model Comparison | FAIL | Found classes with failed metric's gain: {'Responder': {'F1': '-9.23%'}} |
| Weak Segments Performance - Test Dataset | WARN | Found a segment with accuracy score of 0.125 in comparison to an average score of 0.714 in sampled data. |
| Unused Features - Train Dataset | WARN | Found 9 high variance unused features |
| Unused Features - Test Dataset | WARN | Found 10 high variance unused features |
| ROC Report - Train Dataset | PASS | All classes passed, minimum AUC found is 0.93 for class Responder |
| ROC Report - Test Dataset | PASS | All classes passed, minimum AUC found is 0.75 for class Responder |
| Weak Segments Performance - Train Dataset | PASS | Found a segment with accuracy score of 0.808 in comparison to an average score of 0.869 in sampled data. |
| Model Inference Time - Train Dataset | PASS | Found average inference time (seconds): 2.107e-05 |
| Model Inference Time - Test Dataset | PASS | Found average inference time (seconds): 1.118e-05 |
| Confusion Matrix Report - Train Dataset | info | no condition (displayed in the HTML report) |
| Confusion Matrix Report - Test Dataset | info | no condition (displayed in the HTML report) |
| Calibration Metric - Train Dataset | info | no condition (displayed in the HTML report) |
| Calibration Metric - Test Dataset | info | no condition (displayed in the HTML report) |
| Regression Error Distribution - Train Dataset | not run | Check is irrelevant for classification tasks |
| Regression Error Distribution - Test Dataset | not run | Check is irrelevant for classification tasks |
| Boosting Overfit | not run | Check is relevant for Boosting models of type ('AdaBoostClassifier', 'GradientBoostingClassifier', 'LGBMClassifier', 'XGBClassifier', 'CatBo |

(8 batches of predictions computed by the app's engine.)

## 4. Repeated patients (every patient twice)

Deepchecks' Train Test Samples Mix looks for test records that also appear in the training set.

| Split | Result | Details |
|---|---|---|
| By row (a random split ignoring the patient) | FAIL | Percent of test data samples that appear in train data: 75% |
| By patient (the app's "keep each patient's records together") | PASS | No samples mix found |

