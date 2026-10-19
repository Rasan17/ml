# Response to the external (Genspark) audit, October 2026

Version 2.2 (build 2026.10.18) is a hardening release that addresses the audit's findings. Each reproduced
defect has a regression check in [`audit_regressions.js`](audit_regressions.js) or the browser tests, run
on every push.

| # | Finding | Verdict | What changed | Test |
|---|---|---|---|---|
| 1 | Imported model files can install code (typed-array constructor looked up by name; arbitrary fields restored onto models) | Confirmed | Typed arrays only from a fixed list of numeric types. Fields that would replace a model method or touch the prototype (`__proto__`, `constructor`, `prototype`) are refused. The preprocessing description is checked (names, encodings, positions, outcome classes). A model must predict a test patient with finite values before anything is installed (transactional import). Settings files: every restored variable choice must be a value the app itself can make. | `audit_regressions.js` (4 crafted files refused); browser test with a tampered model file |
| 1b | Delegated `data-action` handler resolves any global function | Confirmed | Only actions on a fixed list run. `module_check.js` fails if a page element names an action not on the list. | `module_check.js` |
| 1c | `'unsafe-eval'` in the page's security policy | Kept, documented | TensorFlow.js needs it at start-up; without it the neural network silently falls back to a linear model. With model files now restricted to plain data, no imported file can reach an evaluation path. | — |
| 2 | Export warning misses k-NN inside recalibrated or pooled models | Confirmed | k-NN is detected through every wrapper: recalibrated, multiply imputed, ensemble, and the neural-network fallback. | `audit_regressions.js` |
| 2b | k-NN imputation exports whole reference rows (identifiers, outcome) | Confirmed | Only the predictors that k-NN filling fills or measures distance on are exported. The confirmation lists exactly what the file holds, and the file records it (`patientDataContents`). | — |
| 2c | Session cleanup incomplete; "100% private" overstated | Confirmed | Loading a new dataset also clears the batch file, separate validation cohort, external validation, odds-ratio and permutation inputs, imputation cache and background jobs. New **🧹 Clear session and patient data** button. Badge now reads "Runs in your browser · no data sent", with a tooltip naming the exports that can contain patient values. | Browser test (Clear session) |
| 3 | Isotonic recalibration depends on the order of tied scores | Confirmed | Tied predictions are pooled before fitting (as in scikit-learn). | `audit_regressions.js` (tied block and 20 random permutations) |
| 4 | 7-step pipeline: imputation not refitted inside the inner CV folds | Confirmed (inner CV only; the outer test set was unaffected) | Folds are built from the raw training records, and the Step 1 imputation is refitted on each fold's learning rows. | End-to-end and browser pipeline tests |
| 5a | `effectiveFolds` forces 2 folds with one minority patient | Confirmed | Fewer than 2 patients in an outcome group is reported as "cross-validation not possible", with what to change. A fold lacking an outcome group also stops with an explanation. | `audit_regressions.js` |
| 5b | IPCW Brier returns a number (even 0) without censoring support | Confirmed | Returns "not estimable" (null) whenever a contributing patient has zero censoring support, or nobody contributes. The survival tab says why. | `audit_regressions.js` |
| 5c | Accuracy interval uses rows when patients have repeated records | Confirmed | Shows the bootstrap interval when one was computed (it resamples patients when a patient column is set, and says "bootstrap over patients"). Wilson's interval only otherwise. | — |
| 6 | CSV: renamed duplicate header collides (`a,a,a_2`) | Confirmed | Renamed headers never collide with any header in the file. A `__proto__` header is renamed. | `audit_regressions.js` |
| 6b | Malformed quotes, infinite numbers, formula protection on headers | Confirmed | An unclosed quote stops the upload with an explanation. Text after a closing quote is counted and reported. "Infinity", "-inf" and "1e999" are treated as missing. Exported CSV headers get the same formula protection as cells. The same checks and messages apply to training, separate-validation and batch uploads. | `audit_regressions.js` |
| 7 | Sample-size guides: records vs patients; candidate vs retained parameters; "Enough data" wording | Accepted | Patients are counted when a patient column is set; otherwise a note says counts assume one record per patient. In the pipeline the verdict uses all candidate parameters considered. Verdicts now read "meets these planning criteria" / "meets this guide (stability still not assured)", stated as planning guides under assumptions. | Browser test |
| 8 | Reference comparisons are implementation evidence, not clinical validity; TRIPOD+AI vs PROBAST+AI; tool-assisted vs independent audit | Accepted | README section "What these checks show, and what they do not". | — |
| 9 | Interface: task choice, clipped tab, terminology | Accepted | Start page offers Develop / Validate / Explore a demo. Step tabs wrap, so every tab is visible. "Deploy" replaced by "active research model" / "Use this model". Probabilities were already labelled by type (predicted probability, forest vote share, …), not "confidence". | Browser tests |
| 10 | Stale asynchronous results after a new dataset | Accepted | Each loaded dataset gets a run identifier; a training run started on an earlier dataset discards its results instead of installing them. | — |

## Architectural follow-ups (completed after the hardening release)

- ~~A single shared fold-preparation engine for both workflows.~~ **Done:** `buildFolds` in `src/validation-core.js` is the one cross-validation engine for the standard workflow, the 7-step pipeline and the folds inside bootstrap resamples. It owns the resampling (capping very large sets, feasibility, patient-grouped fold assignment, outcome coverage per fold); each workflow supplies only its own preprocessing for a fold, fitted on raw learning rows. The refactor was checked to give exactly the same folds as before (identical fingerprints of the fold data for both workflows), and a test guards against separate resampling code returning.
- ~~Versioned model-file schemas independent of internal object layouts.~~ **Done (model-file format 2):** each model type has a published schema (`ModelEngine.MODEL_FILE_SCHEMA`) listing the only fields written and read, each type-checked on loading; other internal properties never reach a file, and unknown fields in a file are ignored. Version 1 files still load. Recalibration settings are validated too. Tested on 13 model/outcome combinations, isotonic recalibration and a version-1 file (`audit_regressions.js`).
- ~~A persistent analysis-summary bar.~~ **Done:** a bar under the step tabs shows the dataset, outcome and positive class, records and patients, predictors, validation design, missing-data handling, recalibration and the active research model (browser test).

An independent statistical and security review of the corrected workflows remains advisable before high-stakes use, as the audit recommends.

## Second Genspark review (October 2026)

Several points repeat the first review and had already been fixed. The reviewer most likely examined the old version-1 repository, `Rasan17/ai`; the current software is in `Rasan17/ml`.

| Point | Status |
|---|---|
| Conflicting URLs (/ai vs /ml) | Already consistent in this repository: README, CITATION.cff and the app all point to `rasan17.github.io/ml` and `github.com/Rasan17/ml`. The /ai links belong to the old version-1 repository. |
| Licence | Kept as "All rights reserved" by the author's decision. |
| Zip clutter | `.gitignore` already excludes Office lock files, videos, PowerPoint files and the workflow guide. The README now gives the `git archive` command for a release zip containing tracked files only. |
| Emoji | **Done.** About 400 emoji were removed from the interface and generated reports: <ul><li>icon-only places (brand mark, upload boxes, random-seed buttons, model types, pipeline cards) use a small, consistent set of line icons drawn for the app, which follow the theme colours;</li><li>decorative emoji in text were removed;</li><li>where an emoji stood for a word (the seed symbol), the word is used;</li><li>warnings keep a monochrome ⚠ text symbol and ticks keep ✓.</li></ul> README headings were cleaned the same way. |
| Credentials and AI credits in header and footer | **Done.** The header now shows the title, tagline, the author's name and the privacy badge. Qualifications, contact and AI-tool credits are in the footer only. |
| Too many entry points on the first screen | **Done.** At the Beginner level (the default for first visits), the settings Save/Load buttons and the separate 7-step-pipeline launcher are hidden. Model import/export and Clear session remain, because the "Validate an existing model" path needs them. |
| Beginner default | Already the default (since the first review). |
| Survival as a separate tab | Already changed: time-to-event is an outcome type chosen in Step 2. |
| Model card; confirmation before single-patient prediction | Already in place. |
| Contrast | The axe-core scan (WCAG 2.1 AA, including colour contrast) passes on every main screen in both themes. |
| Competing risks | Not implemented. A limitation is added to the README: death before revision is treated as censoring, so revision risk is overestimated when deaths are common. |
| Large files, inline styles, `unsafe-eval`, innerHTML | Deferred, as in the DeepSeek response. `unsafe-eval` is required by TensorFlow.js and documented (confirmed by removing it). Text from data reaching `innerHTML` is escaped, and this has been fuzz-tested in two security reviews. |
| Subgroup CIs, clustered DeLong, random survival forests | Remain listed as known gaps (feature freeze). |
| Human statistician review; software paper | Agreed. They need the author, not code. |
