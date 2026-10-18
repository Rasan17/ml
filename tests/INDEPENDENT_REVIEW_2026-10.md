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
