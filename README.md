# Universal Medical ML/NeuralNet Predictor

An interactive, in-browser client-side Machine Learning and Neural Network (multilayer perceptron) predictive modeling suite for **Clinical Medicine and Healthcare Research**, running 100% in the web browser with zero server dependencies and complete patient data privacy.

Conceived, designed and directed by **Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)** ([g_narenthiran@hotmail.com](mailto:g_narenthiran@hotmail.com)), Neurosurgery Research Listserv, United Kingdom (Implemented with AI pair programming: Google Antigravity & Anthropic Claude Code).

**&copy; 2026 Dr G Narenthiran. All rights reserved.**

> ⚠️ **Clinical & Research Disclaimer:** The Universal Medical ML/NeuralNet Predictor is designed as an interactive scientific computing and educational tool for clinical medical research across healthcare disciplines. It must not be utilized as the sole determinant for treatment or clinical interventions without independent medical evaluation and validation.

---

## 🔬 Universal Clinical Scope & Purpose

Translational medicine, clinical research, and electronic health record (EHR) analytics often involve heterogeneous multimodal datasets spanning:
- **Demographics & Physiological Vitals:** Age, BMI, Systolic/Diastolic BP, Heart Rate, Respiratory Rate.
- **Biomarkers & Laboratory Panels:** Serum Creatinine, HbA1c, Troponin, eGFR, Inflammatory Markers.
- **Clinical Severity Scores & Ordinal Scales:** NYHA Heart Failure Class, modified Rankin Scale (mRS), Glasgow Coma Scale (GCS), Charlson Comorbidity Index, Pre-op Gait Scores.
- **Diagnostic Classifications & Imaging Findings:** Binary treatment responsiveness (Responder vs Non-Responder), 1-year cardiovascular risk, mortality, length of stay, or continuous functional recovery scores.

The **Universal Medical ML/NeuralNet Predictor** provides clinical researchers, medical statisticians, and clinicians with a self-contained, client-side predictive suite capable of handling any tabular medical dataset.

---

## ⚙️ Core System Architecture & Workflow

### 1. Complete Client-Side Data Privacy (supports GDPR/HIPAA-aligned use)
- Runs locally in modern browsers via HTML5, Vanilla JavaScript, and TensorFlow.js.
- Patient health records and uploaded CSVs are processed solely in client memory—**zero data is transmitted to external servers**.
- A Content-Security-Policy in the page lets the browser contact only the app's own files (`connect-src 'self'`), so data cannot be sent to any other server.
- Checked on the deployed site (October 2026): during a full analysis every network request went to the app's own address, all to fetch its own files and sample data; nothing was sent out, and no third-party scripts, fonts or analytics are loaded.
- Files you choose to export can contain patient values, and say so before saving: models using k-NN (model or missing-value filling) or multiple imputation, and the "Save imputed data" file. Settings files and reports contain no patient values. **🧹 Clear session and patient data** discards everything held by the page.
- Keeping data in the browser supports GDPR/HIPAA-aligned workflows, but software alone cannot make a study compliant: lawful basis, de-identification, device security and governance remain the user's and institution's responsibility.

### 2. Flexible Variable Role & Scale Configuration
- **Variable Roles:**
  - `Meta / ID`: Identifiers excluded from modeling (e.g. `Patient_MRN`, `Study_ID`).
  - `Feature`: Clinical predictors and biomarkers.
  - `Target`: The primary clinical outcome (Binary, Multi-class, or Continuous).
- **Measurement Scales:**
  - `Continuous`: Interval / ratio parameters (Blood pressure, lab titers, dimensions, scores).
  - `Ordinal`: Ordered discrete clinical rankings (e.g. NYHA class 0–4, severity grades).
  - `Nominal`: Categorical attributes (Sex, comorbidities present/absent, treatment arms).

### 3. Tailored Missing Data Imputation Engine
Context-aware imputation strategies dynamically selectable based on variable scale:
- **Continuous:** *Median* (robust against clinical outliers), *Mean*, *K-Nearest Neighbors (KNN Imputation)*, *Mode*, *Constant*, or *Drop Rows*.
- **Ordinal:** *Mode / Most Frequent Rank*, *Median Rank*, *Constant Rank*, or *Drop Rows*.
- **Nominal:** *Mode / Most Frequent Category*, *New Category ("Missing" / "Unknown")*, or *Drop Rows*.

### 4. Neural Network & Machine Learning Algorithms with Advanced Fine-Tuning
Every model includes fine-tuning controls specifically designed to maximize predictive accuracy, adapt to patient cohort characteristics, and prevent overfitting:

- **Voting / Weighted Ensemble:**
  - Combines the neural network, Random Forest, Logistic/Ridge Regression, k-NN (and optionally Naive Bayes). This is a voting ensemble, not a cross-validated stacked Super Learner; performance weights use balanced accuracy (classification) or inverse MSE (regression) on an internal validation split of the training data.
  - **Fine-Tuning Controls:**
    - Combination Strategy: *Soft Voting* (probability blending), *Performance-Weighted* (dynamic validation weighting), or *Hard Majority Voting*.
    - Validation Weighting Power ($\gamma$): Linear, Quadratic ($\gamma=2$ sharply boosts best models), or Cubic.
    - Capacity Presets: *Balanced*, *Fast*, or *High-Capacity* (more trees and a larger neural network).
    - Dynamic constituent model inclusion checkboxes.
- **Neural Network (Multilayer Perceptron, TensorFlow.js):**
  - Early stopping on an internal validation split carved from the training data (the test set is never used in training). If TensorFlow.js fails, a linear fallback model is used and flagged in every results view, the batch CSV and the JSON report.
  - **Fine-Tuning Controls:**
    - Optimizer selection: *Adam* (Adaptive Moments), *RMSprop*, or *SGD* (Stochastic Gradient Descent).
    - Batch Size: *8* (Fine mini-batch), *16*, *32*, *64*, or *Full Batch* (entire cohort).
    - Architecture: Hidden Layer 1, Hidden Layer 2, and optional Hidden Layer 3 units.
    - Regularization: Dropout rate ($0$ to $0.5$) and L2 Weight Decay ($\lambda = 0, 0.0001, 0.001, 0.01$) to prevent overfitting on clinical cohorts.
    - Activation: *ReLU*, *Tanh*, or *Sigmoid*.
- **Random Forest:**
  - **Fine-Tuning Controls:**
    - Number of Estimator Trees ($5$ to $100$) and Maximum Tree Depth ($2$ to $15$).
    - Minimum Samples to Split ($2, 4, 8, 16$) for leaf node regularization against clinical outliers.
    - Feature Subsampling Ratio ($\sqrt{p}$, $50\%$, $75\%$, $\log_2(p)$, or $100\%$ bagging mode).
- **Logistic / Ridge Regression:**
  - **Fine-Tuning Controls:**
    - Learning rate and iterations ($20$ to $500$).
    - L2 Ridge Penalty ($\lambda = 0, 0.001, 0.01, 0.05, 0.1, 0.5$) for coefficient shrinkage.
    - Optimization Solver: *Momentum GD* (accelerated gradient descent) or *Standard GD*.
- **k-Nearest Neighbors (k-NN):**
  - **Fine-Tuning Controls:**
    - Number of Neighbors ($k = 1$ to $25$).
    - Distance Metric: *Euclidean ($L_2$)*, *Manhattan ($L_1$ Cityblock - robust to outliers)*, or *Chebyshev ($L_\infty$)*.
    - Neighbor Weighting: *Uniform* or *Distance-Weighted ($1/\text{dist}$)* to give closer patient cases higher influence.
- **Naive Bayes (Gaussian + Bernoulli):** Gaussian likelihoods for continuous features; Laplace-smoothed Bernoulli likelihoods for 0/1 indicator features (one-hot categories, binary flags).
  - **Fine-Tuning Controls:**
    - Variance Smoothing ($\epsilon = 10^{-9}, 10^{-4}, 10^{-2}, 0.1$) for numerical stability with sparse or noisy biomarkers.
    - Class Prior Calculation: *Empirical* (cohort frequency), *Balanced / Uniform* (equal $1/K$ priors to correct for class imbalance), or *Laplace-Smoothed*.

### 5. Training & Testing Validation Protocols: Option A vs Option B

The platform provides two rigorous clinical evaluation paradigms selectable in Step 4:

- **Option A: Internal Train / Test Split (with Retraining Choice)**
  - **Configurable Split Ratio (Default: 80% Train / 20% Test):**
    - Clinicians and researchers can dynamically configure the training partition ratio (50% to 95%) via an interactive slider, numeric inputs, or quick presets (`80/20`, `70/30`, `85/15`, `90/10`).
  - **Post-Testing Retraining Decision:**
    - **Yes &mdash; Two-Phase Retraining Protocol (Recommended):** The algorithm first benchmarks out-of-sample performance on the held-out test partition (e.g., 20%), reporting Test Accuracy, Macro F1, Precision, Recall, ROC-AUC, and Confusion Matrix. Once evaluated, it automatically retrains the final deployed model on **100% of all cohort data** ($N_{total}$) so real-time and batch predictions leverage all available clinical evidence.
    - **No &mdash; Single-Phase Protocol:** Benchmarks accuracy on the held-out test split, but keeps the deployed model strictly trained on the initial training partition only (e.g., 80%), without retraining on the remaining data.

  - **How records are divided:** randomly with the outcome balance kept (default); **by patient** (all records of one patient stay on the same side, for repeated visits or admissions); or **by date** (earlier records train, later records test, mimicking use on future patients). Cross-validation folds keep each patient's records together as well.

- **Option B: Separate Training &amp; Testing Datasets (Independent External Validation)**
  - **Dataset (i) &bull; Training Cohort:** Model parameters are fitted strictly on 100% of the primary training cohort.
  - **Dataset (ii) &bull; Testing Cohort:** Model accuracy is evaluated on an independent external validation dataset provided via dedicated CSV upload or built-in validation cohorts.
  - **Training-Only Encoding:** Category encodings and continuous scaling statistics calculated on Dataset (i) are applied to Dataset (ii), so no Dataset (ii) information enters the model. Rows with a missing outcome or an outcome class unseen in training are excluded and counted; missing or unrecognised predictor values are filled from Dataset (i) statistics and reported.

### 6. Flexible Model Testing Scope & Tournament Benchmarking

Clinicians and researchers can choose the exact scope of models to train and benchmark:
1. **Test One Model:** Select an individual algorithm (e.g. Neural Network or Random Forest) for focused training and single-model evaluation.
2. **Test More Than One Model (Custom Subset):** Pick any custom combination of algorithms via interactive card checkboxes (e.g. compare Ensemble vs. Random Forest vs. Logistic Regression).
3. **Test All Models (Full Tournament):** Benchmark all 6 available algorithms simultaneously in a head-to-head competition with a single click.

#### Multi-Model Diagnostic Results & Confusion Matrices:
- **Comparative Benchmark Leaderboard:** Ranks all tested models by a user-selected primary metric (default: Macro F1 for classification, $R^2$ for regression; alternatives: balanced accuracy, ROC-AUC, Brier score, accuracy, RMSE, MAE), and reports all metrics including the Wilson 95% CI for accuracy. The same policy applies in Step 4 and the 7-Step Pipeline. **Model selection:** by default the best model is chosen by **k-fold cross-validation inside the Training set** (missing-value filling, encoding and scaling refitted in every fold; in the 7-Step Pipeline SMOTENC as well), so the Testing set is used only once and the winner's test score is not selection-biased. Choosing by Testing-set results remains available (faster, but optimistic). Optional **automatic tuning** selects each model's key settings by the same leak-free cross-validation. A single split still does not establish general superiority.
- **Calibration & decision threshold:** for two-outcome predictions, a calibration plot with calibration slope and intercept (checked against statsmodels), a sensitivity/specificity/PPV/NPV table across thresholds, and a user-chosen decision threshold (default 0.5, to be set on clinical grounds, never tuned on the Testing set) that is applied consistently to evaluation and to new-patient predictions.
- **Uncertainty:** every test metric has a 95% confidence interval (percentile bootstrap of the test patients; Wilson interval for accuracy), and ROC-AUCs are compared with the top model by the DeLong test.
- **Curves:** ROC and precision-recall curves for all models, decision curve analysis (net benefit against "treat all" and "treat none"), calibration plots, and residual-versus-fitted plots for numerical outcomes.
- **Explainability:** permutation importance on the Testing set (model-agnostic), a per-patient "why this prediction?" explanation in Single Prediction, and odds ratios with 95% CIs and p-values from a logistic regression fitted to convergence (IRLS; checked against statsmodels).
- **Feature selection inside cross-validation:** in the 7-Step Pipeline the Step 6 top-70% selection for k-NN and Naive Bayes is repeated inside every fold, with a selection-stability summary; exclusions made in Steps 2–4 are flagged as post-selection.
- **Export / import of trained models:** the active model, with its fitted preprocessing, decision threshold, evaluation summary, seeds and software version, is saved as one JSON file and can be re-loaded (even without the training data) for predictions or external validation. Files contain no patient data, except where the method needs it (k-NN models, k-NN imputation), which is flagged before export. The file format (version 2) is a published schema: each model type writes and reads only its listed, type-checked fields (`ModelEngine.MODEL_FILE_SCHEMA`), and a model must predict a test patient before an import is accepted.
- **Reports:** both workflows export a JSON report recording the software version, validation design (split method, model-selection method, tuning, threshold), settings, seeds, exclusions and every model's results, including calibration.
- **Individual Confusion Matrices & Accuracy Cards:** Renders an individual diagnostic card for **every chosen model**, featuring:
  - **Prominent Test Accuracy** (and ROC-AUC / Macro F1 scores).
  - **Complete Confusion Matrix Table** displaying actual vs. predicted classifications with hit/miss cell coloration.
  - **One-Click Model Deployment:** The top-ranked model is auto-designated as active, but researchers can click *"Set as Active Predictor"* on any tested model card or leaderboard row to immediately deploy it for Single and Batch Inference.

### 7. Exact Scientific Reproducibility & Random Seed Management

Reproducibility is a fundamental prerequisite in scientific and clinical research. The platform provides full deterministic random seed control and provenance tracking:

- **Per-Model Seed Input & Randomization:**
  - Every machine learning and neural network model includes a **Random Seed (Reproducibility)** input control equipped with a 🎲 dice randomizer button.
  - Researchers can enter a custom integer seed (e.g. `42`, `1007`) or click the 🎲 dice button to generate a random seed.
  - If left blank, the system automatically generates a high-entropy seed and immediately populates the input field so the user always has a visible reference.
- **Option A Data Split Seed:**
  - An optional **Split Seed** control governs deterministic train/test cohort partitioning, guaranteeing identical patient allocations across validation experiments.
- **Deterministic Pseudo-Random Number Generation (PRNG):**
  - Powered by a client-side 32-bit **Mulberry32 PRNG** algorithm.
  - Ensures identical, platform-independent deterministic execution across browsers for:
    - Train/test data shuffling.
    - Random Forest bootstrap resamples and feature subsampling.
    - Neural network layer kernel and dropout initialization, plus a seeded per-epoch shuffle.
    - Ensemble derived sub-model seed propagation.
    - k-NN neighbor distance tie-breaking.
- **Prominent Output of Reproducibility Seeds for Future Reference:**
  - **Results Status Banner:** Displays the deployed model's exact seed.
  - **Benchmark Leaderboard:** Features a dedicated **Seed (Ref)** column with one-click clipboard copy (`🌱 [seed] 📋`).
  - **Multi-Model Confusion Matrix Cards:** Displays a seed badge with one-click copy on every tested model card.
  - **Single Model Metrics Summary:** Prominently presents a Reproducibility Seed metric card.
  - **Single Patient Case Predictor:** Highlights the active model seed within the outcome banner.
  - **Batch Prediction Export:** Automatically embeds the `Model_Seed` column in exported CSV files for permanent provenance in clinical study records.

### 8. Dual Inference Modes
- **Single Patient Predictor:**
  - Dynamically generated clinical form tailored to the features (sliders, rank selectors, categorical dropdowns).
  - Outputs predicted class/value, confidence score, probability distribution bar meters, and the deployed model's reproducibility seed.
- **Batch Prediction Facility:**
  - Upload an unseen cohort CSV with multiple patient records.
  - Computes predictions simultaneously for all rows.
  - Enriches the dataset with `Predicted_[Target]`, `Prediction_Confidence`, and `Model_Seed` and exports as `.csv`.

---

## 🆕 Version 2 additions

- **Architecture:** the application controller is split into ES modules (`src/`, entry `src/main.js`) with no build step needed to run; `tools/build.js` generates a single-file fallback (`dist/app.bundle.js`, used when `index.html` is opened directly from disk) and a content-hash import map so browsers never run stale code after an update.
- **Background computation:** training, cross-validation and the complete Option C bootstrap loop run in a Web Worker (`compute-worker.js`; TensorFlow.js on WebGL via OffscreenCanvas where available), keeping the page responsive; Force Stop terminates the worker immediately. Falls back to the page when workers are unavailable.
- **Gradient-boosted trees:** histogram gradient boosting (binary, multiclass, regression) with early stopping, in both the standard workflow and the 7-step pipeline; benchmarked as equivalent to scikit-learn's HistGradientBoosting.
- **Survival analysis tab:** Cox proportional-hazards model (Efron ties) with hazard ratios, Kaplan-Meier observed risks with Greenwood 95% confidence intervals on the log(−log) scale (checked against statsmodels), Harrell's C-index (bootstrap CI), time-dependent AUC (Uno, with bootstrap CI) and Brier score (Graf) at the horizon with inverse-probability-of-censoring weights, scaled Brier score, calibration at a chosen horizon against Kaplan-Meier, Kaplan-Meier curves by risk group and single-patient survival predictions; validated against statsmodels.
- **Option C – bootstrap internal validation:** uses all patients (recommended by TRIPOD for small cohorts). Every data-driven step is repeated inside each resample: imputation, encoding and scaling, the automatic tuning (3-fold, when switched on; not for the neural network and ensemble, which is stated) and, when models are compared by cross-validation, the choice of the best model (correcting the winner's curse). Regression-type models (logistic/ridge, naive Bayes) are reported with Harrell's optimism correction; flexible models (Random Forest, boosted trees, k-NN, neural network, ensemble) with Efron's .632+ estimate, because Harrell's correction is far too small for models that can memorise their training patients (on pure-noise data it left a Random Forest at AUC 0.89; .632+ gives 0.50). Both estimates are shown.
- **Shapley-value explanations:** "Why this prediction?" shows each predictor's contribution against the average prediction for 100 training patients: exact interventional TreeSHAP for the Random Forest and boosted trees, sampled Shapley values for the other models (both checked against brute-force enumeration). The one-at-a-time view remains as a quick view, and for models loaded from a file.
- **Subgroup (fairness) analysis:** AUC, observed vs predicted risk (O:E), Brier score, sensitivity, specificity and net benefit by sex, age band, site or any other column, on the Testing set, in both workflows.
- **Recalibration:** Platt scaling or isotonic regression fitted on out-of-fold cross-validation predictions from the training set, carried through to predictions and model export. With bootstrap validation (Option C) the recalibration is refitted on each resample's own cross-validated predictions, so the optimism correction covers it.
- **Multiple imputation:** chosen in Step 3 as the alternative to single imputation (one choice for the whole analysis). With MICE, the final models, every cross-validation fold (tuning, model selection, recalibration, Youden threshold: MICE with pooled models) and every bootstrap resample of Option C (MICE with one completed dataset per resample, "resample, then impute") use MICE; no single-imputation method is used. A labelled quick approximation (plain fill inside folds and resamples) is available for very large datasets. For the prediction models (Options A, B and C), MICE with predictive mean matching is fitted on the training patients only and without the outcome; one model is trained per imputed dataset (5 or 10) and the predictions are averaged. Testing and new patients with blanks are imputed with the same fitted imputation models (reproducibly), and exported models carry them. The imputation runs once per set of patients and settings: training further models reuses it, larger jobs run several imputations at a time on separate processor cores (with results identical to one run), and "Save imputed data" keeps it for a later session (that file contains patient-level data, and says so in its name). For the odds-ratio analysis, MICE with Rubin's rules. Optional "was missing" indicator predictors. The 7-step pipeline offers the same choice in its Step 1: MICE on the training split, screening (Steps 2–6) on the first completed dataset, every Step 7 model trained once per completed dataset (each SMOTENC-balanced when Step 5 applies it) with pooled predictions, and MICE repeated inside its cross-validation folds.
- **Minimum sample size:** Riley et al. criteria (as in pmsampsize) for binary and continuous outcomes.
- **How many predictors?** Where predictors are chosen (Step 2; pipeline Step 4), a live guide shows, for the training patients and the size of the smaller outcome group, the largest number of predictor parameters supported by the Riley criteria, by 10 and 20 events per parameter, and for flexible machine-learning models (about 200 per parameter; van der Ploeg et al. 2014), compared with the current selection (categorical predictors count categories − 1).
- **In plain language:** after training, a panel explains the active model's results for clinicians: what is predicted and for whom, what the probability means given its calibration, AUC in words, comparison with giving everyone the average risk, false negatives and positives per 100 patients at the chosen threshold, and the limits of the validation design.
- **Unusual patients:** single and batch predictions flag a patient whose combination of predictor values is unlike the training patients (Mahalanobis distance beyond the 99th / 99.9th percentile of the training patients), even when every value is within range. Only aggregate statistics are stored.
- **Calibration panel:** calibration-in-the-large (observed vs mean predicted, O:E), 95% intervals on each plotted group, and the Brier score with its reference (everyone given the observed rate) and scaled Brier.
- **Interface levels:** the start page offers Beginner (recommended defaults, fewer choices), Intermediate (adds validation and metric choices: model-selection method, ranking metric, decision threshold, recalibration) and Advanced (every setting, the default). Settings hidden at a level are reset to their recommended defaults, so nothing hidden affects the analysis; loading a settings file switches to Advanced. The level is remembered in the browser.
- **Analysis summary:** a bar under the step tabs always shows the dataset, outcome and positive class, records (and patients when a patient column is set), predictors, validation design, missing-data handling, recalibration and the active research model.
- **Enough data for each model?** Before training (Step 4; pipeline Step 7), each selected model is listed with the data it needs for the current predictors (logistic regression and naive Bayes 10–20 events per parameter plus the Riley criteria; random forest, boosted trees, neural network, k-NN and ensemble 200+), what the Training set has, and a verdict (enough data / borderline / exploratory only).
- **Exact Random Forest splits** (default below 5,000 patients).
- **Validation of a model on a separate cohort:** in Batch Prediction, a file that also contains the outcome column is used to validate the active model, trained here or imported from a file, exactly as saved: AUC, observed vs predicted risk (O:E), calibration slope and intercept, Brier score, sensitivity and specificity at the model's threshold (all with bootstrap 95% CIs), calibration plot, decision curve, subgroups, and a printable validation report. A model can thus be sent to another centre and validated there without either side sharing patient data.
- **Reporting:** a printable HTML report (methods, every model's results with confidence intervals, ROC and precision-recall curves, calibration, decision curve, confusion matrix, Option C tables, settings and seeds; "Print → Save as PDF" gives a PDF), a pre-filled checklist aligned with TRIPOD+AI with a draft Methods paragraph, and JSON reports with the full numerical detail. Chart points show their values on hover.
- **Accessibility:** labelled controls, skip link, visible keyboard focus, WCAG AA contrast in both themes.
- **Browser tests:** Playwright smoke tests of the main user journeys (including Option C, recalibration, subgroups and explanations) run in CI.
- **Safeguards:** values outside the training range are flagged in single and batch predictions (possible typing error or extrapolation); cross-validation uses no more folds than there are patients in the smaller outcome group; near-constant predictors are not rescaled; a one-line data summary on loading (missing values, columns excluded automatically); Youden's J threshold from cross-validated training predictions shown for information; a "research use only, not a medical device" notice on every exported file.
- **Security:** no inline event handlers; the Content-Security-Policy allows inline scripts only by hash and lets the page contact nothing but its own files.

## 📁 Pre-Loaded Sample Medical Cohorts

The suite includes built-in clinical datasets for instant experimentation:
0. **🧠 iNPH Shunt Response Cohort, simulated (`medical_shunt_cohort_large.csv`, 420 patients from three hospitals):** the recommended starting point. Same predictors as the mini demo below plus the hospital, with clinically plausible effect sizes (a positive tap test, DESH, a narrow callosal angle, high outflow resistance and shorter symptoms favour response; about two thirds respond) and realistic missing values (infusion-test results missing together when no infusion study was done). Models reach realistic discrimination (AUC about 0.75), not near-perfect scores. A separate simulated cohort of 150 patients from a fourth hospital (`medical_shunt_validation_cohort.csv`) is offered for external validation (Option B). Generated reproducibly by `tools/make_shunt_cohort.py`; the patients are not real.
1. **🫀 Cardiometabolic Risk Cohort (`universal_medical_cohort.csv`):** Multimodal cardiovascular and metabolic risk prediction (Age, BMI, BP, Creatinine, HbA1c, NYHA class, ECG ischemia) with missing cells.
2. **🧠 Shunt Response mini demo (`medical_shunt_cohort.csv`, 40 patients):** Hakim-Adams triad, Evans' Index, Callosal Angle, DESH, CSF dynamics ($R_{out}$, PVI), predicting surgical response.
3. **📈 Functional Recovery Score Cohort (`medical_recovery_score.csv`):** Continuous regression dataset predicting post-operative clinical improvement points.
4. **📂 Unseen Batch Cases (`medical_batch_unseen_cases.csv`):** 10 patient cases ready for immediate batch prediction.
5. **🎗️ Breast Cancer Wisconsin Diagnostic (`breast_cancer_wisconsin.csv`):** the public UCI dataset (569 patients, 30 continuous predictors, benign vs malignant). Large enough to show stable performance estimates and meaningful calibration; the sample cohorts above are small demonstrations (20–40 patients) on which models can look unrealistically good.
6. **⏱️ Shunt Revision Time-to-Event Cohort (`medical_shunt_revision_survival.csv`):** simulated cohort of 320 patients with time to shunt revision or censoring, for the Survival tab.

---

## ⚠️ Known Limitations & Statistical Notes

- **Single train/test split.** Final performance is estimated on one held-out split. With cross-validated model selection (default) the winner's test score is not selection-biased, but it is still imprecise for small test sets; if selection by Testing-set results is chosen, the winner's score is optimistic. Confirm the chosen model on an independent cohort (Option B) before reporting.
- **Splitting design.** Random, patient-grouped and date-based splits are built in; multicentre (site-based) validation needs a separate cohort (Option B). With a date-based split, cross-validation inside the training period uses folds that are not time-ordered.
- **Multiple imputation checks:** MICE is checked on simulated data with values missing at random. In 60 simulated datasets it recovered the true mean (bias −0.02, against −0.26 for complete cases) and correlation (0.69 against a true 0.71), and its 95% intervals covered the truth in 90% of datasets. Its pooled estimates agree with scikit-learn's IterativeImputer within 0.05 (`tests/audit_regressions.js`, `tests/reference_check.py`). Predictive mean matching can only fill gaps with values actually observed. When missing values cluster where few observed donors exist (for example, mostly in patients with high values), it slightly attenuates: about 1% in a mean and 3% in a regression slope in these simulations, with 5 donors as in R's mice. A convergence panel (imputed means by round and imputation, Gelman–Rubin R̂) is shown after training with MICE.
- **Remaining gaps:** exported models with multiple imputation contain observed predictor values of training patients (the imputation donors) and are flagged as such; random survival forests are not included; subgroup results have no confidence intervals and are not copied into the TRIPOD checklist automatically; in Option C, choices made by hand (which predictors to include, how to encode them) and the tuning of the neural network and ensemble are not repeated inside the resamples.
- **Bootstrap confidence intervals** on the leaderboard resample the Testing patients with each model held fixed: they describe the precision of that test estimate, not the variation from refitting the model. For a headline result in a small cohort, use Option C or an external cohort (Option B).
- **Shapley values** compare each predictor with values from real training patients, which can create unlikely combinations when predictors are strongly correlated; boosted-tree contributions are exact on the log-odds scale and shared out in percentage points.
- **Probabilities:** Random Forest and hard voting give vote shares, k-NN gives neighbour shares. Calibration is assessed (plot, slope, intercept, Brier score) and optional Platt/isotonic recalibration is available; calibration estimates are imprecise with fewer than about 100 patients with and without the outcome.
- **Survival model:** assumes proportional hazards (check that Kaplan-Meier curves do not cross); calibration is assessed at one horizon by grouped Kaplan-Meier estimates.
- **Exploratory 2×2 for continuous outcomes.** Cut at the training-data median, not a clinical threshold; these are not diagnostic accuracy measures.
- **Feature importance (MDI)** ranks predictors for this model and data; it is not evidence of causal or clinical effect, and it can favour features with many distinct values.
- **Ensemble** is a voting / performance-weighted ensemble, not a cross-validated stacked Super Learner.
- **Variance conventions.** Feature scaling and tree impurity use the population variance (÷ n, as in scikit-learn); classical statistics (skewness G₁, ANOVA, Pearson/χ² tests) use the standard sample formulas, and displayed SDs are sample SDs (÷ (n − 1)).
- **Collinearity.** Multi-level categorical features use the Generalised VIF (Fox & Monette), shown as GVIF^(1/df) so the VIF > 5 rule applies.
- **Univariate screening** (p < 0.05) is informational; selecting predictors by p-value can bias models and is left to the user's judgement.
- **Scale.** Model training, multiple imputation, and the whole Option C bootstrap loop (including its data preparation), run in a background worker (multiple imputation of 5,000 patients with 16 predictors takes about 2 s for 5 imputations on one core, less when they run in parallel); browsers slow background work down when the window is hidden or minimised; Option C with automatic tuning re-tunes the model in every resample and can take several minutes per model; elsewhere, data preparation (imputation, SMOTENC, fold preparation) still runs on the page and can briefly pause it for very large cohorts (tens of thousands of rows, or nominal variables with many levels).
- **Repeated patients.** If a patient can have several rows, say so: choose the patient column in the split settings (Option A, the pipeline) or in the Option C panel. Splits, cross-validation folds, the Option C bootstrap and the test-set confidence intervals then work on whole patients (on pure noise with three rows per patient, resampling rows reported an AUC of 0.99, resampling patients 0.50). The DeLong test still treats rows as independent.
- **Assessing a model.** Report it following TRIPOD+AI and assess risk of bias and applicability with PROBAST+AI; the app's checklist export pre-fills the TRIPOD+AI items it can.
- **Not a medical device.** For research and education only; not for individual clinical decisions without independent validation.

## 🏛️ For ethics committees

[`docs/ethics-committee-information-pack.html`](docs/ethics-committee-information-pack.html) describes, for research ethics and data-protection review, how patient data are handled (processed only in the browser, never transmitted; offline use), the intended use and regulatory status, what each export contains, and the accuracy and verification evidence. Open it in a browser and print it to PDF to submit it, or use the editable Word version [`docs/ethics-committee-information-pack.docx`](docs/ethics-committee-information-pack.docx).

## ✅ Verification Tests

From the project folder (Node.js; the reference check also needs Python 3.12 with SciPy, scikit-learn, statsmodels, scikit-survival and pmsampsize):

```bash
node tests/reference_check.js && python3 tests/reference_check.py   # 67 checks against SciPy / scikit-learn / statsmodels / scikit-survival / pmsampsize (incl. calibration, recalibration, DeLong, decision curve, IRLS odds ratios, Rubin's rules, t distribution, Cox model, C-index, Kaplan-Meier with Greenwood CIs, time-dependent AUC and Brier score, Riley sample size)
node tests/module_check.js                                         # every import between the src/ modules resolves
node tests/edge_cases.js                                           # robustness: single class, all-missing, constant features, imbalance, zero-variance target
node tests/audit_regressions.js                                    # one check per defect reproduced by the external audit (Oct 2026; responses: tests/AUDIT_GENSPARK_RESPONSE.md, tests/AUDIT_CHATGPT_RESPONSE.md, tests/INDEPENDENT_REVIEW_2026-10.md): model-file safety, k-NN export warning, isotonic ties, CSV headers/quotes/infinities, infeasible CV, non-estimable survival Brier
node tests/end_to_end.js                                           # 26 end-to-end checks: binary/multiclass/regression, missing values, unseen categories, rare outcome, tiny cohort, patient/date splits, calibration, threshold, NN fallback, batch = evaluation preprocessing, export/import of all model types; Option C on pure noise (tuning repeated, .632+ for flexible models, model-selection correction); .632+ no-information rates; TreeSHAP and sampled Shapley vs brute force; time-dependent AUC/Brier under censoring
python3 tests/benchmark_vs_python.py                               # app vs scikit-learn / PyTorch / imbalanced-learn (~2 min)
python3 tests/nn_forward_check.py                                  # exact forward pass of the trained TF.js network, recomputed in NumPy
npm ci && npx playwright install --with-deps && npm run test:ui   # 29 browser tests in Chromium, Firefox and WebKit (Safari's engine): every workflow, multiple imputation everywhere (folds, bootstrap, pipeline), interface levels, a network audit; yes/no, numerical and three-category outcomes; settings round-trip; export/re-import of a multiply imputed, recalibrated model; batch prediction; and an axe-core accessibility scan (WCAG 2.1 AA) of every main screen in both themes
node tools/build.js                                                # rebuild the file:// bundle and import-map versions after editing src/
```

These checks also run automatically on every push (GitHub Actions, `.github/workflows/tests.yml`). The browser suite needs Node's `@playwright/test` (installed by `npm ci`); the Python `playwright` package cannot run it. A local `test-results/` folder is only the output of the last local run.

**What these checks show, and what they do not.** Agreement with SciPy, scikit-learn, statsmodels, scikit-survival and pmsampsize is evidence that the app *computes* its statistics correctly. It is not evidence that a model built with it is clinically valid: that depends on the data, the study design and external validation. The app's TRIPOD+AI checklist helps *report* a study; risk of bias and applicability are *appraised* separately with PROBAST+AI (Moons et al., BMJ 2025). Neither is an automatic certification. The audits below and the AI-assisted reviews during development are tool-assisted checks, not an independent external audit; a fresh statistical and security review by two separate AI reviewers, working from the code alone, is reported in [`tests/INDEPENDENT_REVIEW_2026-10.md`](tests/INDEPENDENT_REVIEW_2026-10.md) (all confirmed defects fixed and regression-tested); an independent *human* statistical and security review is still advisable before high-stakes use.

**Open-source audit tools** ([`tests/AUDIT_REPORT.md`](tests/AUDIT_REPORT.md)): the app's own engine was also checked with **Fairlearn** (every figure of the subgroup panel recomputed: all agree) and **Deepchecks** (data integrity, train/test leakage and drift, repeated-patient mixing, and model evaluation of the app's trained models: no data or leakage problems; the model warnings reflect expected overfitting of flexible models and small subgroups, and are explained in the report). These need separate Python environments, so they are run by hand rather than on every push; the instructions are in the report.

`tests/gpu_cpu_check.html` checks, in the browser itself, that the GPU (WebGL) backend users get gives the same neural-network predictions as the CPU backend used by the automated tests. Serve the project folder (`python3 -m http.server`) and open `http://localhost:8000/tests/gpu_cpu_check.html`; it also reports whether the GPU computes in full float32 precision on that machine.

The benchmark runs the app's own engine (with the real TensorFlow.js network) on three public datasets over 10 paired train/test splits and writes [`tests/BENCHMARK_REPORT.md`](tests/BENCHMARK_REPORT.md). Latest result: preprocessing, k-NN and naive Bayes match scikit-learn to rounding; Random Forest, gradient-boosted trees (vs HistGradientBoosting), the neural network (vs PyTorch), logistic/ridge regression and the voting ensemble are equivalent within pre-specified margins (AUC ±0.02, accuracy/F1 ±0.03, R² ±0.05); SMOTENC matches imbalanced-learn in synthetic-data distribution and downstream AUC. Additional requirements: Python packages `torch` and `imbalanced-learn`.

**Browsers.** The automated browser tests run in Chromium (Chrome, Edge, Brave), Firefox and WebKit, the engine of Safari, on every push. Chrome or Edge on a computer is recommended; Safari itself (on a Mac or iPad) is covered only through WebKit on Linux, which is close but not identical.

## 🚀 How to Run

**Online:** open **https://rasan17.github.io/ai/** in any modern browser (Safari, Chrome, Edge, Firefox). Everything runs in your browser; uploaded data never leaves your computer.

**Locally:** the application is built entirely with client-side web technologies, so no installation, node packages or build steps are required.
1. Download the repository (`git clone https://github.com/Rasan17/ai.git`, or *Code → Download ZIP* on GitHub).
2. Serve the folder with any local web server and open it in the browser:
   ```bash
   cd ai
   python3 -m http.server 8080
   ```
   Then navigate to `http://localhost:8080`.
3. Double-clicking `index.html` also opens the app (it then uses the single-file bundle and runs training on the page instead of a background worker) and works with your own uploaded CSV files, but browsers block the built-in sample cohorts from loading over `file://`, so use a local server (or the online version) for those.

---

## 👨‍⚕️ Author & Copyright Attribution

- **Conception, Design & Direction:** **Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)**
- **Email:** [g_narenthiran@hotmail.com](mailto:g_narenthiran@hotmail.com)
- **Affiliation:** Neurosurgery Research Listserv, United Kingdom
- **Implementation:** Implemented with AI pair programming (Google Antigravity; statistical audit and corrections with Anthropic Claude Code)
- **Copyright:** &copy; 2026 Dr G Narenthiran. All rights reserved.
