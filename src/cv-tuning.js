// cv-tuning.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { fitAndScore, fitModel } from './compute.js';
import { TUNE_MAX_ROWS, TUNABLE_MODELS, tuningGrid, makeFolds, assignRowFolds, capRows, buildFolds, buildRowFolds, fitRecalibratorCore, runTuningCore } from './validation-core.js';
import { AppState, MODEL_INFO, RANKING_METRICS, escapeHtml, parseSeed, rankingMetricKey } from './core.js';
import { fitPooledModel, prepareMultipleImputation, readMultipleImputation } from './multiple-imputation.js';
/* global DataEngine, ModelEngine */
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
// With multiple imputation chosen (Step 3), each fold repeats the final procedure: MICE fitted on the fold's
// learning rows (m completed datasets), one model per dataset with pooled predictions, and the fold's validation
// rows imputed with the fold's chains. The quick approximation instead uses a plain fill (median / most frequent).
async function buildStandardRowFolds(rawTrainRows, k, isClassification, seed, onFold, isAborted = null) {
    const targetCol = Object.keys(AppState.columnConfigs).find(c => AppState.columnConfigs[c].role === 'target');
    const o = {
        configs: AppState.columnConfigs, analysis: AppState.columnAnalysis, isClassification, targetCol, seed,
        groupOf: splitGroupFunction(), decisionThreshold: readDecisionThreshold('decision-threshold')
    };
    const m = AppState.completeCase ? 0 : readMultipleImputation();
    if (!m) return buildRowFolds(rawTrainRows, k, o, onFold);
    if (miceQuickApproximation()) return buildRowFolds(rawTrainRows, k, { ...o, configs: plainFillConfigs(AppState.columnConfigs) }, onFold);
    const res = await buildFolds(rawTrainRows, k, o, async (tr, va, f) => {
        const mi = await prepareMultipleImputation(tr, AppState.columnConfigs, m, (seed + 104729 * (f + 1)) >>> 0, { isAborted });
        const te = DataEngine.prepareTestingDataset(va, mi.prep);
        return { set: { Xtr: mi.sets[0].X, ytr: mi.sets[0].y, Xva: te.X, yva: te.y, targetClasses: mi.prep.targetClasses, mi: mi.sets }, unknownClassCount: te.unknownClassCount };
    }, onFold);
    return { sets: res.sets, rowsUsed: res.rowsUsed, rowsTotal: res.rowsTotal, sampled: res.sampled, grouped: res.grouped, decisionThreshold: o.decisionThreshold, requestedFolds: res.requestedFolds, multipleImputation: m };
}

// Quick approximation (Step 3 option, for very large datasets): plain fill inside folds and resamples
function miceQuickApproximation() { return Boolean(document.getElementById('mi-quick-cv')?.checked); }
function plainFillConfigs(configs) {
    return Object.fromEntries(Object.entries(configs).map(([c, v]) => [c, v.role === 'feature' ? { ...v, imputationStrategy: v.type === 'continuous' ? 'median' : 'mode' } : v]));
}

// Scores one fold: a multiply imputed fold trains one model per completed dataset and scores their pooled
// predictions (validation rows carry one encoded version per imputation chain); other folds as usual
async function fitAndScoreAny(mId, isClassification, numClasses, hp, fold, evalOptions, opts = {}) {
    if (!fold.mi) return fitAndScore(mId, isClassification, numClasses, hp, fold, evalOptions, opts);
    const model = await fitPooledModel(mId, isClassification, numClasses, hp, fold.mi, opts);
    try {
        return await ModelEngine.evaluateAsync(model, fold.Xva, fold.yva, isClassification, fold.targetClasses, evalOptions);
    } finally {
        if (model.dispose) model.dispose();
    }
}

// Compare the grid by mean cross-validated ranking metric over pre-built folds (see validation-core.js)
async function runTuningOnFolds(mId, baseHp, fd, isClassification, numClasses, watchdog, progress, ownOnly = false) {
    return runTuningCore(mId, baseHp, fd, isClassification, numClasses, {
        metricKey: rankingMetricKey(isClassification),
        metricDefs: isClassification ? RANKING_METRICS.classification : RANKING_METRICS.regression,
        fitAndScore: fitAndScoreAny, isAborted: () => Boolean(watchdog && watchdog.aborted), progress, ownOnly
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
    // Option C (bootstrap): the patient column chosen in its own panel
    if (document.querySelector('input[name="validation-mode"]:checked')?.value === 'bootstrap') {
        const col = document.getElementById('bootstrap-group-col')?.value || '';
        return col ? (r => String(r[col] ?? '').trim()) : null;
    }
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
        DataEngine.assertDisjointSplit(res.trainX, res.testX, r => r[sp.groupCol]);
        return res;
    }
    if (sp.method === 'time') {
        if (!sp.dateCol || sp.dateCol === target) throw new Error('Choose the date column for the "by date" split.');
        const res = DataEngine.trainTestSplitTemporal(rows, labels, rows.map(r => r[sp.dateCol]), trainRatio);
        res.info.column = sp.dateCol;
        if (!res.testX.length) throw new Error(`No usable dates found in "${sp.dateCol}" for a date-based split.`);
        DataEngine.assertDisjointSplit(res.trainX, res.testX);
        return res;
    }
    const res = DataEngine.trainTestSplit(rows, labels, trainRatio, isClassification, seed);
    res.info = { method: 'random' };
    DataEngine.assertDisjointSplit(res.trainX, res.testX);
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
    // Option C: rows are independent patients unless the user names the patient column
    const bg = document.getElementById('bootstrap-group-col');
    if (bg) bg.innerHTML = '<option value="">Each row is a different patient</option>' + headers.map(h => `<option value="${escapeHtml(h)}">${escapeHtml(h)}</option>`).join('');
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
// groups (optional): the patient of each test row (patient-level split), for a patient-level bootstrap
async function bootstrapFor(ev, seed, groups = null) {
    if (!ev) return null;
    await ModelEngine.yieldToBrowser();
    const n = (ev.actuals || []).length;
    const B = n > 5000 ? 200 : n > 1000 ? 500 : 1000;
    try { return ModelEngine.bootstrapCIs(ev, B, Number.isFinite(Number(seed)) ? Number(seed) : 2024, groups); } catch (e) { return null; }
}

// Patient of each evaluated test row when the split keeps each patient's records together (else null)
function testPatientGroups(validRows, scope) {
    const sp = splitSpec(scope);
    if (sp.method !== 'group' || !sp.groupCol || !Array.isArray(validRows)) return null;
    return validRows.map(r => String(r[sp.groupCol] ?? '').trim());
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
    return fitRecalibratorCore(method, oof); // the same rule as inside bootstrap resamples (validation-core.js)
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
    // The raw (not yet imputed) training records go to the shared engine, and the Step 1 imputation is refitted
    // on each fold's learning rows, so a fold's validation patients never influence the values filled in
    const raw = p.trainRawRows && p.trainRawRows.length === trainRows.length ? p.trainRawRows : null;
    const cfg = {};
    cols.forEach(c => { cfg[c] = p.columnConfigs[c]; });
    cfg[targetCol] = p.columnConfigs[targetCol];
    // Step 4 univariate screening (it uses the outcome) repeated on each fold's training patients
    const scr = p.univScreen && p.univScreen.active ? p.univScreen : null;
    const activeSet = p.featureCols.filter(c => !p.droppedFeatures.has(c));
    const baseIsActive = activeSet.every(c => cols.includes(c));
    const screenedSelections = [];
    const screenCols = realRows => {
        if (!scr) return cols;
        const ruleSet = new Set(scr.rule);
        const res = DataEngine.runUnivariateTesting(realRows, scr.rule, targetCol, p.columnConfigs);
        const sigInFold = new Set(res.filter(r => r.significant).map(r => r.feature));
        const kept = cols.filter(c => !ruleSet.has(c) || sigInFold.has(c));
        // rule predictors dropped on the whole Training set but significant in this fold come back (full model sets only)
        const added = baseIsActive ? scr.rule.filter(c => !cols.includes(c) && sigInFold.has(c)) : [];
        const out = [...kept, ...added];
        screenedSelections.push(scr.rule.filter(c => out.includes(c)));
        return out.length ? out : cols;
    };
    const cfgFor = cs => { const o = {}; cs.forEach(c => { o[c] = p.columnConfigs[c]; }); o[targetCol] = p.columnConfigs[targetCol]; return o; };
    // The pipeline's own preprocessing for one fold: imputation, Step 4 screening, SMOTENC, optional top-70%
    // selection, encoding
    const prepareFold = async (tr, va, f) => {
        // Multiple imputation (Step 1): MICE refitted on the fold's learning rows; each completed dataset balanced
        // and encoded like the first; the fold is scored with pooled models (fitAndScoreAny)
        if (raw && p.miM) {
            const cfgMI = {};
            p.featureCols.forEach(c => { cfgMI[c] = { ...p.columnConfigs[c], role: 'feature' }; });
            cfgMI[targetCol] = { ...p.columnConfigs[targetCol], role: 'target' };
            const mi = DataEngine.miceFit(tr, cfgMI, { m: p.miM, seed: (seed + 104729 * (f + 1)) >>> 0 });
            const fills = { fillValues: Object.fromEntries(p.featureCols.map(c => [c, (p.imputer && p.imputer.fillValues[c]) ?? 0])) };
            const completed = mi.datasets.map(d => DataEngine.transformWithStrictImputer(d, fills, p.columnConfigs).rows);
            const baseCols = screenCols(completed[0]);
            const balanced = completed.map(d => DataEngine.applySMOTENC(d, baseCols, targetCol, p.columnConfigs, seed).balancedRows);
            let foldCols = baseCols, foldCfg = cfgFor(baseCols);
            if (inFoldSelection) {
                foldCols = await selectTop70InFold(balanced[0], baseCols, targetCol, seed);
                foldCfg = cfgFor(foldCols);
            }
            const prep = DataEngine.prepareDataset(balanced[0], foldCfg, foldCols, targetCol);
            prep.fillValues = {};
            prep.imputation = { configs: p.columnConfigs, mice: mi.imputer };
            const sets = balanced.map(rows => { const t = DataEngine.prepareTestingDataset(rows, prep); return { X: t.X, y: t.y }; });
            const te = DataEngine.prepareTestingDataset(va, prep);
            return { set: { Xtr: sets[0].X, ytr: sets[0].y, Xva: te.X, yva: te.y, targetClasses: prep.targetClasses, mi: sets }, unknownClassCount: te.unknownClassCount, meta: foldCols };
        }
        let foldImputer = p.imputer;
        if (raw) {
            foldImputer = DataEngine.fitStrictImputer(tr, p.featureCols, targetCol, p.columnConfigs, p.imputeStrategy || 'median');
            tr = DataEngine.transformWithStrictImputer(tr, foldImputer, p.columnConfigs).rows;
            va = DataEngine.transformWithStrictImputer(va, foldImputer, p.columnConfigs).rows;
        }
        const baseCols = screenCols(tr);
        const balanced = DataEngine.applySMOTENC(tr, baseCols, targetCol, p.columnConfigs, seed).balancedRows;
        let foldCols = baseCols, foldCfg = cfgFor(baseCols);
        if (inFoldSelection) {
            // Feature selection inside the fold, so the fold's validation patients never influence it
            foldCols = await selectTop70InFold(balanced, baseCols, targetCol, seed);
            foldCfg = cfgFor(foldCols);
        }
        const prep = DataEngine.prepareDataset(balanced, foldCfg, foldCols, targetCol);
        prep.fillValues = { ...(foldImputer ? foldImputer.fillValues : {}) };
        const te = DataEngine.prepareTestingDataset(va, prep);
        return { set: { Xtr: prep.X, ytr: prep.y, Xva: te.X, yva: te.y, targetClasses: prep.targetClasses }, unknownClassCount: te.unknownClassCount, meta: foldCols };
    };
    const res = await buildFolds(raw || trainRows, k, { isClassification, targetCol, seed, groupOf: pipelineGroupFunction() }, prepareFold, onFold);
    return { sets: res.sets, rowsUsed: res.rowsUsed, rowsTotal: res.rowsTotal, sampled: res.sampled, grouped: res.grouped, imputationRefitted: Boolean(raw), selections: inFoldSelection ? res.metas : null, screening: scr ? { rule: scr.rule, perFold: screenedSelections } : null, decisionThreshold: readDecisionThreshold('pipeline-decision-threshold'), requestedFolds: res.requestedFolds };
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

export { miceQuickApproximation, plainFillConfigs, testPatientGroups, ciScopeNote, readRecalibration, fitRecalibrator, recalibrateAndEvaluate, TUNE_MAX_ROWS, TUNABLE_MODELS, TUNE_LABELS, tuningGrid, makeFolds, assignRowFolds, capRows, buildStandardRowFolds, runTuningOnFolds, optionBTrainRawRows, splitSeedForCV, splitSpec, splitGroupFunction, pipelineGroupFunction, splitRowsForWorkflow, describeSplit, populateSplitColumnSelects, updateSplitMethodVisibility, bootstrapFor, ciSpan, readDecisionThreshold, thresholdedOutput, formatCvScore, cvSummaryFrom, selectTop70InFold, buildPipelineFolds, applyHyperparamsToForm, fmtTuneValue, renderTuningReport };
