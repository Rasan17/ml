# Fresh statistical and security review, October 2026

**What this was.** Two separate AI reviewers (Claude agents) each started from the code alone. Neither had seen the development conversation, the earlier audits or the author's reasoning.
- **Statistical reviewer:** briefed as a medical statistician. It recomputed metrics independently in Python and statsmodels.
- **Security reviewer:** briefed as an application-security tester. It ran the app in a headless browser and fuzzed every input: CSV files, settings files, model files and imputed-data files.

Both were told to be adversarial and to reproduce every finding. Each finding was then checked again before it was fixed. Every fixed defect has a regression check in [`audit_regressions.js`](audit_regressions.js), which runs on every push.

**What it was not.** This is not an independent human review. The reviewers were AI models of the same family as the development assistant, so they may share its blind spots. Before high-stakes use (clinical decisions, or publication of a model intended for practice), a qualified medical statistician and a security professional should still review the app and the specific analysis.

## Statistical review

| # | Finding | Verdict | Fix |
|---|---|---|---|
| 1 | Calibration slope and intercept blank ("separation") for over-confident models (slope below about 0.45) | **Confirmed** | Newton's method now starts from a neutral point (logit of the event rate, slope 0) with step-halving. It matches statsmodels in all 6 tested cases (slopes 0.23–1.12). Regression test against the separate IRLS implementation. |
| 2 | SMOTENC leaves predicted risks inflated (in the example O:E 0.23, net benefit negative); the UI implied the probabilities were fine | **Confirmed** | Two-outcome models trained on SMOTENC-balanced data get a prior-shift correction back to the real event rate (rankings and AUC unchanged). The out-of-fold predictions behind the Youden threshold are shifted too. Recalibration in Step 7, when chosen, does this instead. Multi-category outcomes carry a caution. The Step 5 text explains the distortion (van den Goorbergh et al., JAMIA 2022). |
| 3 | Option C confusion matrix titled "optimism-corrected" but measured on the training patients | **Confirmed** | Labelled "apparent", with a warning. Only the summary measures are corrected. |
| 4 | Pipeline cross-validation does not repeat the outcome-based screening of Steps 2–4 | Confirmed (design); **fixed later** | Steps 2–3 (correlation, VIF) never use the outcome. Step 4's univariate screening is now repeated inside every fold (predictors screened by significance are kept only if significant on the fold's training patients; manual overrides kept), with per-fold selection frequencies reported. The held-out test result was never affected. |
| 5 | Macro-F1 (and macro precision/recall) counted classes absent from both truth and predictions | **Confirmed** | Averaged over classes present, as in scikit-learn; bootstrap macro-F1 likewise. Test against scikit-learn's value. |
| 6 | Subgroup tertiles put blank values into the lowest group (`Number('') = 0`) | **Confirmed** | Blank values are reported as "(missing)". |
| — | Harrell-corrected calibration slope reported as the mean bootstrap slope on the original data | Valid | Now apparent slope minus its optimism. |
| — | Option B and external-validation CIs resampled rows even when patients repeat | Valid | They now resample patients when a patient column is set. |
| — | Survival horizon beyond the last training event | Valid | Warning shown. |
| — | Outcome labels trimmed in testing but not training | Valid | Trimmed in both. |
| — | MICE excludes the outcome | Defensible; documented | Rationale stated in the interface (Sisk et al., 2023). |
| — | Option A recalibrator fitted on split-model OOF predictions, applied to the retrained model | Noted | The retrained model is already flagged as untested; confirm on new patients. |
| — | .632+ uses a resample-level, not patient-level, out-of-bag average | Noted | An accepted approximation; unchanged. |

Checked and found correct by the reviewer:
- **Discrimination:** AUC with ties; Wilson CI; DeLong; balanced accuracy.
- **Probability scores:** binary and multiclass Brier; R², RMSE and MAE.
- **Decision rules:** net benefit; Youden.
- **Recalibration:** isotonic and Platt fitted out of fold.
- **Survival:** Harrell's C; Kaplan–Meier with Greenwood intervals; Cox with Efron ties and Breslow baseline; time-dependent AUC and IPCW Brier.
- **Sample size:** the Riley criteria (reproduces Riley et al., BMJ 2020: n = 662).
- **Leakage controls:** in Options A, B and C and in the pipeline folds.
- **Validation and imputation:** .632+ algebra; type-1 predictive mean matching in MICE.

## Security review

| # | Finding | Verdict | Fix |
|---|---|---|---|
| 1 | A crafted model file could inject HTML into the prediction banner (via a text "seed" in its evaluation summary, or markup in the ensemble's base-model list), including a redirect away from the app. No script could run (blocked by CSP). | **Confirmed, high** | Seed shown only if numeric; ensemble names escaped. The imported evaluation summary keeps only numbers, true/false and two fixed-value fields. Ensemble ids must be known model ids and weights numeric; otherwise the file is refused. |
| 2 | Downloadable reports had no security policy; the TRIPOD checklist inserted some numbers unescaped | **Confirmed (latent)** | Every TRIPOD value escaped (apostrophes included). All generated reports carry a strict CSP: no scripts, no external content. |
| 3 | A loaded imputed-data file was trusted without checking it matches the current patients | **Confirmed** | Used only if every observed value and outcome matches the current rows, row by row, and its structure is complete; otherwise the data are re-imputed. |
| 4 | Category values such as "constructor" or "__proto__" crashed analyses | **Confirmed** | Such values are refused when the file is read, with the row, column and value named. |
| 5 | Settings files could set controls outside the saved section; unvalidated ranking metric | **Confirmed** | Restores only controls inside the section they were saved from; ranking metrics validated. |
| 6 | Patient-derived details in shared files: exact boundary dates of a date split, each predictor's minimum and maximum, identifier columns mistakenly used as predictors | **Confirmed** | Dates stored to the month only in model files. Minimum and maximum disclosed in the ethics pack. New warning when a predictor looks like an identifier. |
| 7 | Browser-stored settings survive "Clear session"; the dialog said they hold no patient values | **Confirmed** | Clear session now offers to forget stored settings too, and describes them accurately (file names, column names, category labels). |
| 8 | CSP notes: `unsafe-eval` needed; framing possible on GitHub Pages | Noted | `unsafe-eval` confirmed necessary for TensorFlow.js. Framing protection needs an HTTP header GitHub Pages cannot set; low risk. |
| 9 | Crafted model files could exhaust memory (deep trees, oversized network weights) | Valid | Trees limited in depth and size; network weights must match their declared shapes and a size limit. |

Checked and found sound by the reviewer:
- **Untrusted text:** CSV-derived text is escaped everywhere it was fuzzed, in the page and in reports.
- **Content Security Policy:** inline handlers blocked; no requests to other hosts.
- **Imported files:** model deserialisation is schema-only, and TF.js model loading runs no code.
- **Page actions:** the delegated-action allowlist holds.
- **Downloads:** CSV formula protection works.
- **Logging and clearing:** no patient data are logged; Clear session clears memory.
- **Build and libraries:** the published bundle matches the source; no relevant CVEs in TensorFlow.js 4.22.0 or Chart.js 4.4.1.

## Second round: changes since the first review (19 October 2026)

Scope: everything changed after commit `6cb2eb3`. That covers:
- time-to-event outcomes in the standard workflow (Cox model, Options A/B/C, Steps 5–6, settings, model card);
- unified MICE;
- recalibration in Option C;
- MICE in the 7-step pipeline;
- MICE diagnostics;
- interface levels.

As before, two separate AI reviewers worked from the code alone, without editing it, one statistical and one security. Each confirmed defect was fixed and is covered by a test.

### Statistical review

| # | Finding | Verdict | Fix |
|---|---|---|---|
| 1 | The follow-up time could become a predictor: "Include all", or ticking its box, turned it back into a feature, and the Cox model did not remove it | **Confirmed, high** | The time and event columns are removed from the predictors inside the analysis. Their boxes are locked, and their roles can only be changed in the outcome panel. Browser test added. |
| 2 | Choosing a time-to-event outcome turned the previous outcome column into a predictor | **Confirmed** | It is excluded instead, with a visible note, and restored if the user switches back. Browser test added. |
| 3 | Option A ignored the split method chosen in Step 4 (by patient, by date) | **Confirmed** | Uses the workflow's split and reports it. |
| 4 | Option B and Step 6 weighted for censoring using the development cohort, biasing the Brier score when follow-up differs (simulated: 0.111 vs true 0.231) | **Confirmed** | The evaluated cohort's own censoring distribution is used for Options B and Step 6. |
| 5 | The JSON report gave hazard ratios per SD while labelling them "per unit", described every design as an 80/20 split, and omitted the Option C corrected values | **Confirmed** | Hazard ratios are on the same scale as the screen, the design is described by validation mode, and an optimism-corrected section was added. Browser test added. |
| 6 | The quick MICE approximation could act on resamples with single imputation, under a "single imputation" label | **Confirmed** | Applies only with MICE. |
| 7 | Minor reporting issues | **Confirmed, low** | Four changes: the apparent calibration slope is shown as a reference, no longer as "1 by construction"; no bootstrap CI is shown around apparent (uncorrected) values; the hazard-ratio CIs use the model's own interval (Rubin's t interval when pooled); test patients removed by a "drop rows" choice are reported. |

The reviewer checked and found sound:
- No leakage of the time or event into imputation.
- Options A/B/C structure.
- The Harrell bootstrap: full process repeated, sign of the optimism, censoring weights per resample, calibration slope.
- Rubin pooling on a common encoding.
- Row alignment.
- IPCW mathematics.
- Unified MICE in folds and resamples.
- In-fold Step 4 screening.

**New external check:** the per-resample optimism for the Cox bootstrap (C-index, time-dependent AUC at the horizon, calibration slope) is compared with statsmodels `PHReg` and scikit-survival. The app scores resamples with `ModelEngine.coxResampleOptimism`. The check covers 3 new reference values (70 in total) and agrees to about 1e-15.

### Security review

No exploitable issue was found in the changed code. Untrusted text is escaped in the outcome panel, Cox results, model card and batch output. Settings fields are validated, and CSV formula protection, CSP and network behaviour are unchanged.

| # | Finding | Verdict | Fix |
|---|---|---|---|
| 1 | Model files carried the MICE convergence trace; with one missing cell its value is one training patient's observed value | **Confirmed, low** | Trace left out of model files (diagnostic only). |
| 2 | A malformed trace in an imported model could stop the evaluation page from rendering | **Confirmed, low** | Only arrays of finite numbers are drawn. |

### Process change

Two new automated checks:
- **ESLint `no-undef` over `src/`, run in CI.** It catches a function used after being deleted, as happened once during this work. The module check only covers cross-module imports.
- **The Cox bootstrap reference check described above.**

### Runtime safeguards (added after the second round)

Several defects in this review came from interactions between features, which tests written for the expected routes did not cover. Three checks now run on every analysis:
1. **The outcome is never a predictor.** Encoding, single imputation and MICE refuse a predictor marked as the follow-up time or event (`DataEngine.assertNoOutcomeAsPredictor`).
2. **No patient on both sides of a split.** Every Option A split and every cross-validation fold is checked for shared records or shared patients (`DataEngine.assertDisjointSplit`).
3. **Possible leakage is flagged before training** in the standard workflow, the 7-step pipeline and for time-to-event outcomes (`src/safeguards.js`). The trigger is a single predictor with AUC ≥ 0.99, |rank correlation| ≥ 0.99 or C-index ≥ 0.97 on its own. The user must confirm to continue.

When first run on the built-in datasets, the leakage check flagged two older simulated demos: the 40-patient shunt cohort and the cardiometabolic cohort. In these, several predictors separate the outcome perfectly; in the first, for example, "DESH sign present" corresponds exactly to "responder". The check is correct; the demos are unrealistically clean. The 420-patient iNPH demo, its validation cohort, the survival demo and the public breast-cancer data are not flagged.
