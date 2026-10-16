// cv-tuning.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { fitAndScore, fitModel } from './compute.js';
import { TUNE_MAX_ROWS, TUNABLE_MODELS, tuningGrid, makeFolds, assignRowFolds, capRows, effectiveFolds, buildRowFolds, runTuningCore } from './validation-core.js';
import { AppState, MODEL_INFO, RANKING_METRICS, escapeHtml, parseSeed, rankingMetricKey } from './core.js';
import { disposeModels, evalAbortOpts, fitWithYields } from './training-actions.js';
// =========================================================================
// AUTOMATIC TUNING (standard workflow, Step 4): k-fold cross-validation inside the Training set
// =========================================================================
// For each selected model a small grid of its key settings is compared by stratified k-fold
// cross-validation on the Training data only; the user's own settings are always one of the
// candidates, so tuning only changes them when another combination does better. The Testing set
// is never used. Imputation, encoding and scaling are refitted inside every fold.
const TUNE_LABELS = {
    k: 'k (neighbours)', weighting: 'Neighbour weighting', maxDepth: 'Max tree depth', minSamplesSplit: 'Min samples to split',
    lambdaL2: 'Ridge penalty (λ)', varSmoothing: 'Variance smoothing', dropoutRate: 'Dropout', l2Reg: 'L2 weight decay',
    learningRate: 'Learning rate', minSamplesLeaf: 'Min patients per leaf'
};

// Leak-free folds for the standard workflow: imputation, encoding and scaling are fitted on each fold's
// learning rows only and then applied to that fold's validation rows (as for the final Testing set)
async function buildStandardRowFolds(rawTrainRows, k, isClassification, seed, onFold) {
    const targetCol = Object.keys(AppState.columnConfigs).find(c => AppState.columnConfigs[c].role === 'target');
    return buildRowFolds(rawTrainRows, k, {
        configs: AppState.columnConfigs, analysis: AppState.columnAnalysis, isClassification, targetCol, seed,
        groupOf: splitGroupFunction(), decisionThreshold: readDecisionThreshold('decision-threshold')
    }, onFold);
}

// Compare the grid by mean cross-validated ranking metric over pre-built folds (see validation-core.js)
async function runTuningOnFolds(mId, baseHp, fd, isClassification, numClasses, watchdog, progress, ownOnly = false) {
    return runTuningCore(mId, baseHp, fd, isClassification, numClasses, {
        metricKey: rankingMetricKey(isClassification),
        metricDefs: isClassification ? RANKING_METRICS.classification : RANKING_METRICS.regression,
        fitAndScore, isAborted: () => Boolean(watchdog && watchdog.aborted), progress, ownOnly
    });
}

// What the confidence intervals under the leaderboard do and do not cover
function ciScopeNote() {
    return `<p class="tune-muted" style="margin-top: 6px;">The 95% confidence intervals come from resampling the Testing patients (bootstrap: 1,000 resamples, fewer for very large test sets) with each model held fixed: they show how precise this test estimate is, not how much the result would change if the model were refitted on other patients. For a headline result in a small cohort, prefer bootstrap internal validation (Option C) or an independent validation cohort (Option B).</p>`;
}

// Raw training rows for Option B (all of Dataset i with a usable outcome)
function optionBTrainRawRows(prep) {
    return AppState.parsedData.rows.filter(r => !DataEngine.isMissing(r[prep.targetCol]) &&
        (prep.isClassification || !isNaN(Number(r[prep.targetCol]))));
}

// Seed for the cross-validation folds: the Step 4 split seed
function splitSeedForCV() {
    const custom = document.querySelector('input[name="split-seed-mode"]:checked')?.value === 'custom';
    const v = custom ? parseSeed(document.getElementById('split-seed-input')?.value) : null;
    return v !== null ? v : 42;
}

// ---------------------------------------------------------------------------------------------
// How records are divided into Training and Testing sets: random (stratified by outcome), by patient
// (all records of one patient on the same side) or by date (earlier records train, later records test)
// ---------------------------------------------------------------------------------------------
function splitSpec(scope) {
    const pre = scope === 'pipeline' ? 'pipeline-' : '';
    const method = document.getElementById(`${pre}split-method`)?.value || 'random';
    return {
        method,
        groupCol: method === 'group' ? (document.getElementById(`${pre}split-group-col`)?.value || '') : '',
        dateCol: method === 'time' ? (document.getElementById(`${pre}split-date-col`)?.value || '') : ''
    };
}

// Grouping for cross-validation folds (patient-level splitting): null when records are independent
function splitGroupFunction() {
    const sp = splitSpec('standard');
    return sp.method === 'group' && sp.groupCol ? (r => String(r[sp.groupCol] ?? '').trim()) : null;
}
function pipelineGroupFunction() {
    const sp = splitSpec('pipeline');
    return sp.method === 'group' && sp.groupCol ? (r => String(r[sp.groupCol] ?? '').trim()) : null;
}

function splitRowsForWorkflow(rows, labels, trainRatio, isClassification, seed, scope) {
    const sp = splitSpec(scope);
    const target = Object.keys(AppState.columnConfigs).find(c => AppState.columnConfigs[c].role === 'target');
    if (sp.method === 'group') {
        if (!sp.groupCol || sp.groupCol === target) throw new Error('Choose the column that identifies each patient for the "keep each patient\'s records together" split.');
        const res = DataEngine.trainTestSplitGrouped(rows, labels, rows.map(r => r[sp.groupCol]), trainRatio, seed);
        res.info.column = sp.groupCol;
        return res;
    }
    if (sp.method === 'time') {
        if (!sp.dateCol || sp.dateCol === target) throw new Error('Choose the date column for the "by date" split.');
        const res = DataEngine.trainTestSplitTemporal(rows, labels, rows.map(r => r[sp.dateCol]), trainRatio);
        res.info.column = sp.dateCol;
        if (!res.testX.length) throw new Error(`No usable dates found in "${sp.dateCol}" for a date-based split.`);
        return res;
    }
    const res = DataEngine.trainTestSplit(rows, labels, trainRatio, isClassification, seed);
    res.info = { method: 'random' };
    return res;
}

// Plain description of the split, for results and reports
function describeSplit(info) {
    if (!info || info.method === 'random') return 'random split, balanced by outcome';
    if (info.method === 'group') return `split by patient ("${info.column}"): ${info.trainGroups} patient(s) in Training, ${info.testGroups} in Testing, none in both`;
    return `split by date ("${info.column}"): Training up to ${info.trainUntil}, Testing ${info.testFrom} to ${info.testUntil}${info.undated ? ` (${info.undated} record(s) without a usable date kept in Training)` : ''}`;
}

// Fill the patient-ID and date column choices (identifier/date-like columns listed first)
function populateSplitColumnSelects(headers) {
    const analysis = AppState.columnAnalysis || {};
    const looksDate = h => /date|time|admission|visit|year/i.test(h) || (analysis[h] && /calendar date/i.test(analysis[h].autoMetaReason || ''));
    const looksId = h => /id|mrn|patient|subject|record|number|no\b/i.test(h) || (analysis[h] && analysis[h].role === 'meta');
    const fill = (id, prefer) => {
        const el = document.getElementById(id);
        if (!el) return;
        const ordered = [...headers.filter(prefer), ...headers.filter(h => !prefer(h))];
        el.innerHTML = '<option value="">— choose a column —</option>' + ordered.map(h => `<option value="${escapeHtml(h)}">${escapeHtml(h)}</option>`).join('');
        const first = headers.find(prefer);
        if (first) el.value = first;
    };
    fill('split-group-col', looksId); fill('pipeline-split-group-col', looksId);
    fill('split-date-col', looksDate); fill('pipeline-split-date-col', looksDate);
    updateSplitMethodVisibility();
}

function updateSplitMethodVisibility() {
    ['', 'pipeline-'].forEach(pre => {
        const method = document.getElementById(`${pre}split-method`)?.value || 'random';
        const g = document.getElementById(`${pre}split-group-wrap`);
        const d = document.getElementById(`${pre}split-date-wrap`);
        if (g) g.style.display = method === 'group' ? '' : 'none';
        if (d) d.style.display = method === 'time' ? '' : 'none';
    });
}

// Bootstrap 95% CIs for a test evaluation (fewer resamples for very large test sets, to stay responsive)
async function bootstrapFor(ev, seed) {
    if (!ev) return null;
    await ModelEngine.yieldToBrowser();
    const n = (ev.actuals || []).length;
    const B = n > 5000 ? 200 : n > 1000 ? 500 : 1000;
    try { return ModelEngine.bootstrapCIs(ev, B, Number.isFinite(Number(seed)) ? Number(seed) : 2024); } catch (e) { return null; }
}

// Small "(lo–hi)" text for a bootstrap CI
function ciSpan(ev, key, pct = false, digits = 3) {
    const ci = ev && ev.ci && ev.ci[key];
    if (!ci) return '';
    const f = v => (pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(digits));
    return `<div class="ci-text" title="95% confidence interval (bootstrap of test patients)">${f(ci[0])}–${f(ci[1])}</div>`;
}

// Decision threshold for binary outcomes (probability of the positive class at or above which a patient is
// classed as positive). Chosen by the user on clinical grounds; never optimised on the Testing set.
function readDecisionThreshold(id) {
    const v = parseFloat(document.getElementById(id)?.value);
    return isFinite(v) && v > 0 && v < 1 ? v : 0.5;
}

// Prediction for a new patient with the deployed model's decision threshold
function thresholdedOutput(o) {
    const t = AppState.evalResults && AppState.evalResults.decisionThreshold;
    const prep = AppState.prepMeta;
    if (!o || !prep || !prep.isClassification || (prep.targetClasses || []).length !== 2 || !t || Math.abs(t - 0.5) < 1e-9) return o;
    return ModelEngine.applyDecisionThreshold(o, t);
}

// "CV score" cell: cross-validated ranking metric inside the Training set (± SD across folds)
function formatCvScore(ev, isClassification) {
    if (!ev || !ev.cv) return '<span style="color: var(--text-muted);">–</span>';
    const key = rankingMetricKey(isClassification);
    const v = ev.cv[key];
    if (typeof v !== 'number' || !isFinite(v)) return '<span style="color: var(--text-muted);">N/A</span>';
    const pctKey = ['macroF1', 'balancedAccuracy', 'accuracy'].includes(key);
    const sdTxt = key === ev.cv.metricKey && typeof ev.cv.sd === 'number' ? ` <span style="font-size: 10.5px; color: var(--text-muted);">± ${pctKey ? (ev.cv.sd * 100).toFixed(1) : ev.cv.sd.toFixed(3)}</span>` : '';
    return `<strong style="color: var(--accent-cyan);">${pctKey ? (v * 100).toFixed(1) + '%' : v.toFixed(3)}</strong>${sdTxt}`;
}

// ---------------------------------------------------------------------------------------------
// Recalibration (binary outcomes): a calibrator is fitted on the out-of-fold cross-validation predictions
// of the chosen settings (Training set only) and wrapped around the final model
// ---------------------------------------------------------------------------------------------
function readRecalibration(id) {
    const v = document.getElementById(id)?.value;
    return v === 'platt' || v === 'isotonic' ? v : 'none';
}

function fitRecalibrator(method, oof) {
    if (method === 'none' || !oof || !oof.p || oof.p.length < 20) return null;
    const events = oof.y.reduce((a, b) => a + b, 0);
    if (events < 5 || oof.y.length - events < 5) return null;
    return method === 'platt' ? ModelEngine.fitPlatt(oof.y, oof.p) : ModelEngine.fitIsotonic(oof.y, oof.p);
}

// Wraps a trained model with the calibrator and records the before/after evaluation summary
async function recalibrateAndEvaluate(model, calibrator, X, y, isClassification, targetClasses, evalOptions, abortOpts) {
    const before = await ModelEngine.evaluateAsync(model, X, y, isClassification, targetClasses, evalOptions, abortOpts);
    const wrapped = ModelEngine.wrapCalibrated(model, calibrator);
    const after = await ModelEngine.evaluateAsync(wrapped, X, y, isClassification, targetClasses, evalOptions, abortOpts);
    after.recalibration = {
        method: calibrator.type, fittedOn: `${calibrator.n} out-of-fold Training-set predictions`,
        before: { brierScore: before.brierScore, rocAuc: before.rocAuc, slope: before.calibration ? before.calibration.slope : null, intercept: before.calibration ? before.calibration.intercept : null }
    };
    return { model: wrapped, evalRes: after };
}

// Cross-validated performance of a model's final settings, attached to its results for model selection
function cvSummaryFrom(tuneResult) {
    const b = tuneResult.best;
    return { ...b.allMeans, folds: tuneResult.folds, sd: b.sd, metricKey: tuneResult.metricKey, rowsUsed: tuneResult.rowsUsed, grouped: tuneResult.grouped };
}

// Row folds (7-step pipeline): each fold re-runs SMOTENC and refits encoding/scaling on its own
// learning rows only, so no synthetic case is ever scored against the real patient it came from
// Step 6 selection repeated inside a fold: Random Forest scout on the fold's balanced learning rows, keep the
// top 70% by importance plus the user's reinstated predictors (same rule as the full pipeline)
async function selectTop70InFold(balancedRows, activeCols, targetCol, seed) {
    const p = AppState.pipeline;
    const cfg = {};
    activeCols.forEach(c => { cfg[c] = p.columnConfigs[c]; });
    cfg[targetCol] = p.columnConfigs[targetCol];
    const prep = DataEngine.prepareDataset(balancedRows, cfg, activeCols, targetCol);
    // Scout trained in the background worker when available
    const rf = await fitModel('rf', prep.isClassification, prep.targetClasses.length || 2, { numTrees: 40, maxDepth: 6, minSamplesSplit: 2, featureRatio: 'sqrt', seed: seed }, prep.X, prep.y);
    const imp = activeCols.map(col => {
        const meta = prep.featureMetadata[col];
        let sum = 0;
        if (meta) for (let j = 0; j < meta.length; j++) sum += rf.featureImportances[meta.startIndex + j] || 0;
        return { col, imp: sum };
    }).sort((a, b) => b.imp - a.imp);
    const k = Math.max(1, Math.round(activeCols.length * 0.30));
    const topCount = Math.max(1, activeCols.length - k);
    const reinstated = p.reinstatedFeatures || new Set();
    return imp.filter((r, i) => i < topCount || reinstated.has(r.col)).map(r => r.col);
}

async function buildPipelineFolds(trainRows, cols, k, seed, onFold, inFoldSelection = false) {
    const p = AppState.pipeline;
    const targetCol = p.targetCol;
    const isClassification = p.columnConfigs[targetCol]?.type !== 'continuous';
    const groupOf = pipelineGroupFunction();
    const cap = capRows(trainRows, isClassification, targetCol, seed, groupOf);
    const rows = cap.rows;
    const sampled = cap.sampled;
    const requestedFolds = k;
    k = effectiveFolds(rows, k, isClassification, targetCol);
    const fold = assignRowFolds(rows, k, isClassification, targetCol, seed, groupOf);
    const cfg = {};
    cols.forEach(c => { cfg[c] = p.columnConfigs[c]; });
    cfg[targetCol] = p.columnConfigs[targetCol];
    const sets = [];
    const selections = [];
    for (let f = 0; f < k; f++) {
        const tr = rows.filter((_, i) => fold[i] !== f);
        const va = rows.filter((_, i) => fold[i] === f);
        const balanced = DataEngine.applySMOTENC(tr, cols, targetCol, p.columnConfigs, seed).balancedRows;
        let foldCols = cols, foldCfg = cfg;
        if (inFoldSelection) {
            // Feature selection inside the fold, so the fold's validation patients never influence it
            foldCols = await selectTop70InFold(balanced, cols, targetCol, seed);
            selections.push(foldCols);
            foldCfg = {};
            foldCols.forEach(c => { foldCfg[c] = p.columnConfigs[c]; });
            foldCfg[targetCol] = p.columnConfigs[targetCol];
        }
        const prep = DataEngine.prepareDataset(balanced, foldCfg, foldCols, targetCol);
        prep.fillValues = { ...(p.imputer ? p.imputer.fillValues : {}) };
        const te = DataEngine.prepareTestingDataset(va, prep);
        sets.push({ Xtr: prep.X, ytr: prep.y, Xva: te.X, yva: te.y, targetClasses: prep.targetClasses });
        if (onFold) onFold(f + 1, k);
        await ModelEngine.yieldToBrowser(); // keep the page responsive between folds
    }
    return { sets, rowsUsed: rows.length, rowsTotal: trainRows.length, sampled, grouped: Boolean(groupOf), selections: inFoldSelection ? selections : null, decisionThreshold: readDecisionThreshold('pipeline-decision-threshold'), requestedFolds };
}

// Show the chosen settings in the model's settings panel (so they are visible and saved with Save settings)
function applyHyperparamsToForm(mId, hp) {
    const setVal = (id, v) => {
        const el = document.getElementById(id);
        if (!el || v === undefined) return;
        if (el.tagName === 'SELECT') {
            const opt = Array.from(el.options).find(o => Math.abs(parseFloat(o.value) - Number(v)) < 1e-15 || o.value === String(v));
            if (opt) el.value = opt.value;
        } else {
            el.value = v;
        }
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    };
    if (mId === 'knn') { setVal('knn-k-input', hp.k); setVal('knn-weighting-select', hp.weighting); }
    else if (mId === 'rf') { setVal('rf-depth-input', hp.maxDepth); setVal('rf-minsplit-select', hp.minSamplesSplit); }
    else if (mId === 'glm') { setVal('glm-lambda-select', hp.lambdaL2); }
    else if (mId === 'nb') { setVal('nb-smoothing-select', hp.varSmoothing); }
    else if (mId === 'nn') { setVal('nn-dropout-input', hp.dropoutRate); setVal('nn-l2-select', hp.l2Reg); }
    else if (mId === 'gbm') { setVal('gbm-lr-select', hp.learningRate); setVal('gbm-depth-input', hp.maxDepth); }
}

function fmtTuneValue(key, v) {
    if (key === 'weighting') return v === 'distance' ? 'distance-weighted' : 'equal (uniform)';
    if ((key === 'varSmoothing' || key === 'l2Reg') && Number(v) > 0 && Number(v) < 0.001) return Number(v).toExponential(0);
    return String(v);
}

function renderTuningReport(report, containerId = 'tuning-report-container') {
    const box = document.getElementById(containerId);
    if (!box) return;
    if (!report || !report.models.length) { box.innerHTML = ''; return; }
    const fmtNum = (key, v) => (['macroF1', 'balancedAccuracy', 'accuracy'].includes(key) ? (v * 100).toFixed(1) + '%' : v.toFixed(3));
    const fmtScore = (r, t) => (r && r.mean !== null
        ? `${fmtNum(t.metricKey, r.mean)}${r.sd !== null ? ` <span class="tune-sd">± ${fmtNum(t.metricKey, r.sd)}</span>` : ''}`
        : 'n/a');
    const describe = (t, params) => t.keys.map(k => `${TUNE_LABELS[k] || k}: <strong>${escapeHtml(fmtTuneValue(k, params[k]))}</strong>`).join(', ');
    const rows = report.models.map(t => {
        const name = (MODEL_INFO[t.modelId] || { name: t.modelId }).name;
        const changed = !t.best.isOwn;
        const sorted = [...t.results].sort((a, b) => (a.mean === null) - (b.mean === null) || (t.higherIsBetter ? b.mean - a.mean : a.mean - b.mean));
        return `
            <div class="tune-model">
                <div class="tune-model-head"><strong>${escapeHtml(name)}</strong>
                    <span class="pill" style="font-size: 10px; ${changed ? 'background: rgba(16,185,129,0.15); color: var(--accent-emerald);' : 'background: rgba(100,116,139,0.15); color: var(--text-muted);'}">${changed ? 'Settings changed' : (report.pipeline ? 'Protocol default kept' : 'Your settings kept')}</span></div>
                <div>Chosen: ${describe(t, t.best.params)} &mdash; cross-validated ${escapeHtml(t.metricLabel)} ${fmtScore(t.best, t)}</div>
                ${changed ? `<div class="tune-muted">${report.pipeline ? 'Protocol default' : 'Your settings'} (${describe(t, t.own.params)}) scored ${fmtScore(t.own, t)}.</div>` : ''}
                <div class="tune-muted">${t.results.length} combination(s) × ${t.folds} folds${t.requestedFolds > t.folds ? ` (reduced from ${t.requestedFolds}: the smaller outcome group has too few patients to place some in every fold)` : ''} on ${t.rowsUsed.toLocaleString('en-GB')}${t.sampled ? ` (a stratified sample of ${t.rowsTotal.toLocaleString('en-GB')})` : ''} training cases.</div>
                <details><summary>Show all combinations tried</summary>
                    <table class="data-table tune-table"><thead><tr>${t.keys.map(k => `<th>${escapeHtml(TUNE_LABELS[k] || k)}</th>`).join('')}<th>CV ${escapeHtml(t.metricLabel)} (mean ± SD)</th><th></th></tr></thead>
                    <tbody>${sorted.map(r => `<tr${r === t.best ? ' class="tune-best"' : ''}>${t.keys.map(k => `<td>${escapeHtml(fmtTuneValue(k, r.params[k]))}</td>`).join('')}<td>${fmtScore(r, t)}</td><td>${r === t.best ? '✓ chosen' : ''}${r.isOwn ? (report.pipeline ? ' (protocol default)' : ' (your settings)') : ''}</td></tr>`).join('')}</tbody></table>
                </details>
            </div>`;
    }).join('');
    box.innerHTML = `
        <div class="tune-report">
            <div class="tune-title">🎯 Automatic tuning report (${report.folds}-fold cross-validation inside the Training set)</div>
            ${report.smallSampleWarning ? `<div class="tune-warn">⚠️ ${report.smallSampleWarning}</div>` : ''}
            <p class="tune-muted">Each combination was trained on part of the Training set and scored on the remaining part, in turn, so every training case was used once for scoring. The best combination was then used to train the final model, which was scored once on the untouched Testing set (results below). ${report.pipeline
                ? `In each fold, SMOTENC balancing and encoding/scaling were re-done using only that fold's learning patients, so no synthetic patient was ever scored against the real patient it was made from. The protocol's default settings were always one of the candidates and are kept unless another combination does better.`
                : `In each fold, missing-value filling, encoding and scaling were redone using only that fold's learning patients, so the mini-test patients never influenced the preparation. The chosen settings are now shown in each model's settings panel.`} Small differences between combinations (within the ± SD) are not meaningful.${report.ensembleSkipped ? ` The Ensemble was not tuned: it uses ${report.pipeline ? 'the protocol\'s fixed base-model settings' : 'its base models\' settings'}.` : ''}</p>
            ${rows}
        </div>`;
}

export { ciScopeNote, readRecalibration, fitRecalibrator, recalibrateAndEvaluate, TUNE_MAX_ROWS, TUNABLE_MODELS, TUNE_LABELS, tuningGrid, makeFolds, assignRowFolds, capRows, buildStandardRowFolds, runTuningOnFolds, optionBTrainRawRows, splitSeedForCV, splitSpec, splitGroupFunction, pipelineGroupFunction, splitRowsForWorkflow, describeSplit, populateSplitColumnSelects, updateSplitMethodVisibility, bootstrapFor, ciSpan, readDecisionThreshold, thresholdedOutput, formatCvScore, cvSummaryFrom, selectTop70InFold, buildPipelineFolds, applyHyperparamsToForm, fmtTuneValue, renderTuningReport };
