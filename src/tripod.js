// tripod.js: pre-filled reporting checklist aligned with the TRIPOD+AI topics (Collins et al., BMJ 2024;385:e078378)
// and a draft Methods paragraph, generated from the analysis actually run. Items are described in our own
// words; authors must check them against the official checklist and complete the items marked for them.
import { AppState, APP_VERSION, MODEL_INFO, RESEARCH_USE_NOTICE, rankingMetricLabel } from './core.js';
import { describeSplit } from './cv-tuning.js';

const esc = s => String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = v => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : 'n/a');
const n3 = v => (typeof v === 'number' ? v.toFixed(3) : 'n/a');
const ci = (ev, k, f) => (ev && ev.ci && ev.ci[k] ? ` (95% CI ${f(ev.ci[k][0])}–${f(ev.ci[k][1])})` : '');

function collect(source) {
    const cfgs = AppState.columnConfigs || {};
    const target = Object.keys(cfgs).find(c => cfgs[c].role === 'target');
    if (source === 'pipeline') {
        const p = AppState.pipeline;
        const best = (p.benchmarkResults || [])[0];
        return {
            source, target, predictors: (p.featureCols || []).filter(c => !p.droppedFeatures.has(c)),
            excludedPredictors: Array.from(p.droppedFeatures || []), nTotal: (p.trainRows || []).length + (p.testRows || []).length,
            nTrain: (p.trainRows || []).length, nTest: (p.testRows || []).length, split: describeSplit(p.splitInfo), seed: p.seed,
            models: (p.benchmarkResults || []).map(b => ({ name: b.name, ev: b.eval })), best: best ? { name: best.name, ev: best.eval } : null,
            missing: p.completeCase ? `complete-case analysis (${p.excludedIncomplete} incomplete rows removed before the split)` : `single imputation (${p.imputeStrategy || 'median'} for continuous, most frequent for categorical predictors) fitted on the training set only`,
            imbalance: p.smotencResult && p.smotencResult.applied ? `SMOTENC applied to the training set only (+${p.smotencResult.syntheticCountGenerated} synthetic cases); the test set was not resampled` : 'no resampling (classes not imbalanced, or numerical outcome)',
            selection: best && best.eval.cv ? `${best.eval.cv.folds}-fold cross-validation within the training set (SMOTENC and scaling refitted in each fold)` : 'best test-set result (optimistic)',
            tuning: p.tuningReport ? 'key hyperparameters tuned by cross-validation within the training set' : 'prespecified protocol settings',
            validation: `single ${describeSplit(p.splitInfo)} into training and test sets`,
            excludedOutcome: p.excludedMissingTarget || 0
        };
    }
    const res = AppState.allTestedResults || [];
    const top = res[0] ? res[0].evalRes : AppState.evalResults;
    const oc = top && top.optimismCorrection;
    const protocol = top && top.protocol === 'option-c' ? `bootstrap internal validation (${(oc || {}).B || '?'} resamples; all patients used for training; imputation, encoding and scaling${oc && oc.tuning === 'repeated' ? ' and hyperparameter tuning' : ''} repeated in every resample; Harrell's optimism correction for regression-type models and the .632+ estimate for flexible models${oc && oc.selection && oc.selection.repeated ? '; the choice between models was also repeated in every resample' : ''})`
        : top && top.protocol === 'option-b' ? 'external validation on a separate testing dataset (Dataset ii)'
        : `single ${describeSplit(top && top.splitInfo)} (${top && top.trainPct ? top.trainPct : 80}/${top && top.testPct ? top.testPct : 20})`;
    return {
        source, target, predictors: Object.keys(cfgs).filter(c => cfgs[c].role === 'feature'), excludedPredictors: Object.keys(cfgs).filter(c => cfgs[c].role === 'meta'),
        nTotal: AppState.parsedData ? AppState.parsedData.rows.length : null, nTrain: top ? top.trainCount : null, nTest: top ? top.testCount : null,
        split: top && top.protocol === 'option-a' ? describeSplit(top.splitInfo) : null, seed: top ? top.splitSeed : null,
        models: res.map(r => ({ name: (MODEL_INFO[r.modelId] || {}).name || r.modelId, ev: r.evalRes, hp: r.hyperparams })),
        best: res[0] ? { name: (MODEL_INFO[res[0].modelId] || {}).name || res[0].modelId, ev: res[0].evalRes } : null,
        missing: AppState.completeCase ? 'complete-case analysis (rows with an empty cell removed)' : top && top.multipleImputation ? `multiple imputation by chained equations with predictive mean matching (${top.multipleImputation.m} imputations), fitted on the training patients without the outcome; one model was fitted per imputed dataset and predictions were pooled by averaging; test and new patients were imputed with the same imputation models` : `single imputation with the methods chosen per predictor, fitted on the training data only${AppState.missingIndicators ? ', with "was missing" indicator predictors' : ''}`,
        imbalance: 'no resampling in the standard workflow',
        selection: top && top.cv ? `${top.cv.folds}-fold cross-validation within the training set (imputation, encoding and scaling refitted in each fold)` : (res.length > 1 ? 'best test-set result (optimistic)' : 'single prespecified model'),
        tuning: AppState.tuningReport ? 'key hyperparameters tuned by cross-validation within the training set' : 'user-specified hyperparameters',
        validation: protocol,
        recalibration: top && top.recalibration ? `${top.recalibration.method === 'platt' ? 'Platt scaling' : 'isotonic regression'} fitted on out-of-fold training predictions` : 'none',
        threshold: top ? top.decisionThreshold : null
    };
}

function performanceText(ev) {
    if (!ev) return 'not available';
    if (ev.isClassification) {
        return [`accuracy ${pct(ev.accuracy)}${Array.isArray(ev.accuracyCI) ? ` (95% CI ${pct(ev.accuracyCI[0])}–${pct(ev.accuracyCI[1])})` : ''}`,
            typeof ev.rocAuc === 'number' ? `ROC-AUC ${n3(ev.rocAuc)}${ci(ev, 'rocAuc', n3)}` : null,
            typeof ev.brierScore === 'number' ? `Brier score ${n3(ev.brierScore)}${ci(ev, 'brierScore', n3)}` : null,
            ev.calibration && typeof ev.calibration.slope === 'number' ? `calibration slope ${ev.calibration.slope.toFixed(2)} and intercept ${typeof ev.calibration.intercept === 'number' ? ev.calibration.intercept.toFixed(2) : 'n/a'}` : null,
            ev.optimismCorrection ? '(optimism-corrected)' : null].filter(Boolean).join('; ');
    }
    return [`R² ${n3(ev.r2)}${ci(ev, 'r2', n3)}`, `RMSE ${n3(ev.rmse)}${ci(ev, 'rmse', n3)}`, `MAE ${n3(ev.mae)}${ci(ev, 'mae', n3)}`].join('; ');
}

function buildChecklist(source) {
    const d = collect(source);
    const AUTHOR = '<em class="tp-author">To be completed by the authors.</em>';
    const items = [
        ['Title', 'Identify the study as developing and/or evaluating a prediction model, the target population and the outcome.', `Outcome: "${esc(d.target)}". ${AUTHOR}`, 'author'],
        ['Abstract', 'Structured summary of objectives, data, methods, results and conclusions.', AUTHOR, 'author'],
        ['Background and objectives', 'Healthcare context, rationale, intended use and users of the model; study objectives.', AUTHOR, 'author'],
        ['Data sources', 'Study design and data sources, dates of data collection.', `Dataset: ${esc(AppState.fileName || 'n/a')} (${esc(d.nTotal ?? 'n/a')} records). ${AUTHOR} (design, setting, dates)`, 'partly'],
        ['Participants', 'Eligibility criteria, setting, treatments received.', `${d.excludedOutcome ? `${esc(d.excludedOutcome)} records without an outcome were excluded. ` : ''}${AUTHOR}`, 'partly'],
        ['Data preparation', 'Pre-processing, data quality checks, and whether these were consistent across training and evaluation.', `Encoding (one-hot for categorical, standardisation for continuous, rank scaling for ordinal) fitted on the training data only and applied unchanged to evaluation data. Excluded or identifier columns: ${esc(d.excludedPredictors.join(', ') || 'none')}.`, 'auto'],
        ['Outcome', 'Definition of the outcome, how and when it was assessed, blinding.', `Outcome variable "${esc(d.target)}". ${AUTHOR} (clinical definition, timing, assessors)`, 'partly'],
        ['Predictors', 'Definition of all predictors, how and when measured; reasons for their choice.', `${d.predictors.length} candidate predictors: ${esc(d.predictors.join(', '))}. ${AUTHOR} (definitions, timing)`, 'partly'],
        ['Sample size', 'How the study size was arrived at; justification against minimum sample size criteria.', `${esc(d.nTotal ?? 'n/a')} records; the app's Riley et al. minimum sample-size check is shown in Step 4. ${AUTHOR} (justification)`, 'partly'],
        ['Missing data', 'Amount of missing data and how it was handled.', esc(d.missing), 'auto'],
        ['Analytical methods: models', 'Type of models, model-building steps, hyperparameter tuning.', `Models: ${esc(d.models.map(m => m.name).join(', '))}. Hyperparameters: ${esc(d.tuning)}.`, 'auto'],
        ['Analytical methods: validation', 'Internal validation approach and how performance was estimated.', `Validation: ${esc(d.validation)}. Model selection: ${esc(d.selection)}.${d.seed !== null && d.seed !== undefined ? ` Random seed ${esc(d.seed)}.` : ''}`, 'auto'],
        ['Analytical methods: performance measures', 'Measures of discrimination, calibration and clinical utility, with uncertainty.', `Discrimination (ROC-AUC / C-index, DeLong comparison), calibration (plot, slope, intercept, Brier score), decision curve analysis; percentile bootstrap 95% CIs over test patients${d.source === 'standard' && d.threshold ? `; decision threshold ${esc(d.threshold)}` : ''}.`, 'auto'],
        ['Class imbalance', 'Methods used to handle class imbalance and their effect on calibration.', esc(d.imbalance) + (d.recalibration && d.recalibration !== 'none' ? `; recalibration: ${esc(d.recalibration)}` : ''), 'auto'],
        ['Fairness', 'Approaches to assess and address model fairness across groups.', `The app's subgroup panel reports, for a chosen column on the Testing set, the number of patients and events, AUC, observed and mean predicted risk (O:E), Brier score, sensitivity, specificity and net benefit in each group; results are not copied here automatically. ${AUTHOR} (state the groups examined, e.g. sex, age group, ethnicity, site, and what was found)`, 'partly'],
        ['Model output', 'What the model outputs (probabilities, classes) and how thresholds were chosen.', `Probabilities for each outcome class (or a numerical prediction); class assignment at a decision threshold of ${esc(d.threshold ?? 0.5)} chosen a priori, not tuned on the test set.`, 'auto'],
        ['Ethics, funding, conflicts, protocol, registration', 'Ethical approval, funding, competing interests, protocol and registration.', AUTHOR, 'author'],
        ['Open science', 'Availability of data, code and the model.', `Analysis performed in-browser with Universal Medical ML/NeuralNet Predictor v${APP_VERSION}; the trained model and settings can be exported as JSON files for sharing. ${AUTHOR} (data and code availability statements)`, 'partly'],
        ['Patient and public involvement', 'Details of any patient and public involvement.', AUTHOR, 'author'],
        ['Results: participants', 'Flow of participants, characteristics of training and evaluation data.', `Training ${esc(d.nTrain ?? 'n/a')}, evaluation ${esc(d.nTest ?? 'n/a')} records. ${AUTHOR} (flow diagram, baseline characteristics table)`, 'partly'],
        ['Results: model specification', 'Full model details so it can be used or evaluated by others.', 'Exportable trained-model file (preprocessing, parameters, threshold); for logistic models, coefficients / odds ratios are shown in the app.', 'auto'],
        ['Results: performance', 'Performance estimates with confidence intervals, for each model and the final model.', `Final model (${esc(d.best ? d.best.name : 'n/a')}): ${esc(performanceText(d.best && d.best.ev))}.<br>${d.models.map(m => `${esc(m.name)}: ${esc(performanceText(m.ev))}`).join('<br>')}`, 'auto'],
        ['Discussion: interpretation, limitations, usability', 'Interpretation, limitations (including bias and generalisability) and implications for practice.', `Note: single-centre/temporal limitations, small test sets, and that probabilities ${d.recalibration && d.recalibration !== 'none' ? 'were recalibrated' : 'were not recalibrated'}. ${AUTHOR}`, 'partly']
    ];
    return { d, items };
}

function methodsParagraph(d) {
    const parts = [];
    parts.push(`Prediction models for ${d.target} were developed using the Universal Medical ML/NeuralNet Predictor (version ${APP_VERSION}), which runs entirely in the web browser.`);
    parts.push(`${d.predictors.length} candidate predictors were considered (${d.predictors.join(', ')}).`);
    parts.push(`Missing predictor values were handled by ${d.missing}.`);
    parts.push(`Categorical predictors were one-hot encoded and continuous predictors standardised, with all preprocessing fitted on the training data only.`);
    parts.push(`The following models were compared: ${d.models.map(m => m.name).join(', ')}; hyperparameters were ${d.tuning}.`);
    parts.push(`Performance was assessed by ${d.validation}, with the final model chosen by ${d.selection}.`);
    if (d.imbalance && d.source === 'pipeline') parts.push(`Class imbalance: ${d.imbalance}.`);
    if (d.recalibration && d.recalibration !== 'none') parts.push(`Predicted probabilities were recalibrated by ${d.recalibration}.`);
    parts.push('Discrimination was summarised by the ROC-AUC (or R², RMSE and MAE for numerical outcomes) and calibration by calibration plots, slope, intercept and the Brier score; clinical utility was explored by decision curve analysis. 95% confidence intervals were obtained by percentile bootstrap of the evaluation patients, and AUCs were compared with the DeLong test.');
    return parts.join(' ');
}

function downloadTripodChecklist(source) {
    const { d, items } = buildChecklist(source);
    const statusLabel = { auto: 'Pre-filled', partly: 'Partly pre-filled', author: 'Authors' };
    const html = `<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRIPOD+AI Reporting Checklist (pre-filled)</title>
<style>body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:#0f172a;background:#fff;margin:24px auto;max-width:1000px;padding:0 16px}
h1{font-size:20px}h2{font-size:16px;margin-top:28px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #cbd5e1;padding:6px 8px;vertical-align:top;text-align:left}
th{background:#f1f5f9}.auto{color:#047857;font-weight:600}.partly{color:#b45309;font-weight:600}.author{color:#b91c1c;font-weight:600}.tp-author{color:#b91c1c}
.note{color:#475569;font-size:12.5px}@media print{body{margin:0}}</style></head><body>
<h1>Reporting checklist aligned with TRIPOD+AI, pre-filled from this analysis</h1>
<p style="border:2px solid #b45309;padding:8px 12px;font-weight:600">${esc(RESEARCH_USE_NOTICE)}</p>
<p class="note">Generated ${esc(new Date().toISOString().slice(0, 10))} by Universal Medical ML/NeuralNet Predictor v${esc(APP_VERSION)} (${d.source === 'pipeline' ? '7-step pipeline' : 'standard workflow'}). Items follow the topics of the TRIPOD+AI statement (Collins GS et al. BMJ 2024;385:e078378); wording is summarised. <strong>Check every item against the official checklist</strong> (tripod-statement.org) and complete those marked for the authors.</p>
<table><thead><tr><th>Topic</th><th>What to report (summary)</th><th>From this analysis</th><th>Status</th></tr></thead><tbody>
${items.map(([t, w, v, st]) => `<tr><td><strong>${esc(t)}</strong></td><td>${esc(w)}</td><td>${v}</td><td class="${st}">${statusLabel[st]}</td></tr>`).join('\n')}
</tbody></table>
<h2>Draft Methods paragraph</h2>
<p>${esc(methodsParagraph(d))}</p>
<p class="note">Edit to match your study and journal style. Ranking metric used to compare models: ${esc(rankingMetricLabel(Boolean(d.best && d.best.ev && d.best.ev.isClassification)))}.</p>
</body></html>`;
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `TRIPOD-AI_checklist_${source}_${new Date().toISOString().slice(0, 10)}.html`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

export { buildChecklist, methodsParagraph, downloadTripodChecklist };
