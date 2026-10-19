# Response to the DeepSeek review (October 2026)

Each point was checked against the code before acting. Features the review suggested adding were deferred, under the current feature freeze: verify and explain the existing methods before adding new ones.

## Fixed or added

| Point | Action |
|---|---|
| Frozen statistical golden tests on the demo cohorts | `tests/golden_check.js` runs fixed analyses and compares them with `tests/golden_values.json` (4 decimal places); it runs in CI. The analyses: <ul><li>420-patient iNPH demo, Option A: logistic regression, random forest, boosted trees, k-NN and naive Bayes (AUC, Brier, macro-F1, accuracy, calibration slope and intercept);</li><li>MICE on its training rows;</li><li>Riley sample size;</li><li>Cox model on the survival demo (coefficients, C-index, time-dependent AUC);</li><li>linear model on the recovery-score demo.</li></ul> |
| More ESLint rules | Added: unused variables and imports, `eqeqeq`, unreachable code, duplicate keys, fall-through, constant reassignment, implicit globals, redeclaration and shadowing. Findings fixed: an unused import, a dead variable in the sample-size panel and three shadowed names. None was a functional bug. |
| CSP test that external requests are blocked | New browser test: the page attempts to POST to another server. The request is refused, and a `connect-src` violation is recorded. |
| Provenance in the model file | Saved model files record the app version, TensorFlow.js version, compute backend and browser. The model card shows this as "Computing environment". |
| "Verify" re-run of stored predictions | Model files store three synthetic patients built from the training summaries (no real patient values) and the model's outputs for them. On import the outputs are recomputed. A mismatch beyond GPU rounding asks the user before the model is used. The model card reports the result. Browser test covers both an intact and an altered file. |
| Ensemble fallback provenance | The model card now reports an ensemble whose base neural network fell back to a linear model. The note was already kept in the file, but not shown. |
| MICE: predictors never observed | Reported in the MICE diagnostics panel: no observed values, filled with a constant, carries no information, exclude it. |
| Rounding must stay identical at training and prediction | Replaced by a single shared function (`DataEngine.scaleContinuous`) used by both. |
| `_ridgeSolve` near-singular fallback | Documented why it cannot be reached: the ridge penalty makes the matrix positive definite. |
| Hard warning for very few events | Before training, fewer than 5 events per predictor parameter now requires confirmation. Below 10, the warning after training remains. |
| Ordinal predictors in univariate tests | Labelled "Ordinal (ranks treated as equally spaced numbers)". |
| Settings stored in the browser | The settings bar now says remembered settings stay in this browser until cleared. |
| `prefers-reduced-motion` for the pulsing indicators | Added. |
| Step tabs and the keyboard | Left/Right, Home and End move between step tabs. |
| Oversized model files | Files over 300 MB are refused before parsing. |
| Archived library versions | SHA-256 checksums of `tf.min.js` and `chart.umd.min.js` in `THIRD_PARTY_NOTICES.md`. |
| README limitations | Added: no built-in external or prospective validation; no regulatory clearance; the no-data-out guarantee is enforced by CSP, not cryptography; small-sample caveats; code-structure notes. |

## Already in place

- **CI check that the build output (bundle, importmap hashes) matches `index.html`:** the first CI step.
- **file:// smoke test:** in the browser suite.
- **Malicious model files:** `audit_regressions.js` covers `__proto__` fields, the typed-array allowlist, a 300-level tree, markup in ensemble metadata and weight-size checks. JSON cannot express cyclic trees.
- **Class counts shown with the AUC.**
- **Vote and neighbour shares flagged as poorly calibrated probabilities.**
- **A user-facing message when a split leaves no test rows.**
- **Export warning that model files can contain patient data.**
- **NaN fallbacks for batch size and k.**
- **`CITATION.cff` and `tools/build.js` in the repository.**

## Not valid on checking

- **SMOTENC combined with class weights:** the app has no class-weight option, so the two are never combined.
- **TreeSHAP `includes` cost:** the lists searched hold only the features split on along one path, so they are at most the tree depth long, not the number of predictors. A Set would not help.
- **Replacing `document.write` with `import()` for the file:// fallback:** browsers block module and dynamic imports from file:// pages. That is why a classic script is used there, and the file:// test confirms it works.

## Deferred

Not started, by design:
- Splitting `models.js` and `data-engine.js` into modules.
- Moving inline styles to classes, so that `style-src 'unsafe-inline'` can be removed.
- A WASM TensorFlow.js backend to remove `unsafe-eval`.
- A nested-CV stacked learner.
- Permutation importance.
- Calibration by subgroup.
- Survival calibration plots at several horizons.
- A live training/testing/events strip.
- Full ARIA tablist roles.

These are refactors or new methods. The golden, reference and regression checks now in place should come first, so that any refactor can be shown not to change results. Zenodo archiving needs the author's account.
