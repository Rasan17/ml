// multiple-imputation.js: multiple imputation (MICE) for the prediction models in the standard workflow.
// The imputation models are fitted on the training patients only (without the outcome); one model is
// trained per completed dataset and the predictions are pooled (ModelEngine.createMIPooled). Testing and new
// patients are imputed with the same fitted chains by DataEngine.transformSingleRow.
/* global DataEngine, ModelEngine */
import { fitModel, miceFitRun } from './compute.js';
import { APP_VERSION, AppState } from './core.js';

// Number of imputations chosen in Step 4 (0 = single imputation)
function readMultipleImputation() {
    const v = document.getElementById('mi-select')?.value || 'single';
    return v === 'mice10' ? 10 : v === 'mice5' ? 5 : 0;
}

// Completed datasets already computed in this session, so that training another model on the same patients
// with the same settings reuses them instead of repeating the imputation. MICE is seeded, so a repeat run would
// give exactly the same datasets: reuse changes nothing but the waiting time. The key covers every input of the
// imputation (patients' values, variable settings, number of imputations, seed); any change re-imputes. Up to
// four entries are kept (Option A uses two: the Training set and all patients), oldest dropped first.
const MI_CACHE_SIZE = 4;
const miCache = new Map();
function miCacheKey(rows, columnConfigs, m, seed) {
    const text = JSON.stringify(rows) + '|' + JSON.stringify(columnConfigs);
    let h1 = 0x811c9dc5, h2 = 0x01000193 ^ text.length;
    for (let i = 0; i < text.length; i++) {
        const c = text.charCodeAt(i);
        h1 = Math.imul(h1 ^ c, 0x01000193);
        h2 = Math.imul(h2 ^ c, 0x5bd1e995);
        h2 ^= h2 >>> 15;
    }
    return `${m}|${seed}|${rows.length}|${text.length}|${(h1 >>> 0).toString(16)}${(h2 >>> 0).toString(16)}`;
}
function matchesRows(mi, rows, m) {
    if (!Array.isArray(mi.datasets) || mi.datasets.length !== m || !mi.imputer || !Array.isArray(mi.imputer.chains) || mi.imputer.chains.length !== m) return false;
    return mi.datasets.every(d => Array.isArray(d) && d.length === rows.length && d.every((r, i) => r && typeof r === 'object' &&
        Object.keys(rows[i]).every(k => DataEngine.isMissing(rows[i][k]) ? !DataEngine.isMissing(r[k]) || r[k] === rows[i][k] : String(r[k]) === String(rows[i][k]))));
}

function clearImputationCache() { miCache.clear(); updateImputedDataButtons(); }

// "Save imputed data" file: the cached imputations (completed datasets and the fitted imputation models),
// so a later session can skip the imputation. It holds patient-level data, unlike the settings file. On
// loading, entries are only ever used for exactly the same patients and settings (the cache key).
const MI_FILE_KIND = 'ummp-imputed-data';
const MI_FILE_FORMAT = 1; // changes whenever the imputation method changes, so old files are not reused

function updateImputedDataButtons() {
    const btn = document.getElementById('mi-save-btn');
    if (btn) {
        btn.disabled = miCache.size === 0;
        btn.title = miCache.size ? 'Download the imputed datasets of this session (contains patient data)' : 'Train a model with multiple imputation first';
    }
}

function saveImputedData() {
    if (!miCache.size) { alert('There are no imputed datasets yet: train a model with multiple imputation first.'); return; }
    const file = {
        kind: MI_FILE_KIND, format: MI_FILE_FORMAT, appVersion: APP_VERSION, savedAt: new Date().toISOString(),
        dataset: AppState.fileName || null,
        notice: 'CONTAINS PATIENT-LEVEL DATA: every patient\'s values, with the missing ones filled in. Store and share it only as the original dataset is permitted to be.',
        entries: [...miCache.entries()].map(([key, mi]) => ({ key, m: mi.datasets.length, patients: mi.datasets[0] ? mi.datasets[0].length : 0, mi }))
    };
    const base = String(AppState.fileName || 'dataset').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_').slice(0, 60) || 'dataset';
    const url = URL.createObjectURL(new Blob([JSON.stringify(file)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `${base}_imputed-data_PATIENT-DATA_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setImputedDataStatus(`Saved ${file.entries.length} set${file.entries.length === 1 ? '' : 's'} of imputed data. The file contains patient data: keep it as securely as the original dataset.`);
}

function loadImputedDataFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        let data;
        try { data = JSON.parse(e.target.result); } catch (err) { alert('This file could not be read as an imputed-data file (it is not valid JSON).'); return; }
        if (!data || data.kind !== MI_FILE_KIND || !Array.isArray(data.entries)) { alert('This is not an imputed-data file from this app.'); return; }
        if (data.format !== MI_FILE_FORMAT) { alert(`This imputed-data file was made by a version of the app with a different imputation method (${data.appVersion || 'unknown version'}), so it cannot be reused; the imputation will be run again when you train.`); return; }
        let added = 0;
        data.entries.forEach(en => {
            if (en && typeof en.key === 'string' && en.mi && Array.isArray(en.mi.datasets) && en.mi.imputer && Array.isArray(en.mi.imputer.chains)) { miCache.set(en.key, { ...en.mi, fromFile: true }); added++; }
        });
        while (miCache.size > MI_CACHE_SIZE) miCache.delete(miCache.keys().next().value);
        updateImputedDataButtons();
        setImputedDataStatus(`Loaded ${added} set${added === 1 ? '' : 's'} of imputed data${data.dataset ? ` (saved for "${data.dataset}")` : ''}. They are used automatically when you train on exactly the same patients with the same variable settings, number of imputations and seed; otherwise the imputation is run as usual.`);
    };
    reader.readAsText(file);
}

function setImputedDataStatus(html) {
    const box = document.getElementById('mi-file-status');
    if (box) { box.style.display = 'block'; box.textContent = html; }
}

/**
 * Multiply imputes raw training rows and returns the encoding fitted on all completed copies (prep, whose
 * X / y are the first copy, for counts and displays) and the encoded copies (sets) to train on.
 */
async function prepareMultipleImputation(rawRows, columnConfigs, m, seed, { onProgress = null, isAborted = null, onReuse = null } = {}) {
    // Plain copies: the rows are sent to the background worker
    const plainRows = rawRows.map(r => ({ ...r }));
    const key = miCacheKey(plainRows, columnConfigs, m, seed);
    let mi = miCache.get(key);
    // Imputed data loaded from a file are used only if they really are these patients: every observed value
    // (and the outcome) must match, row by row, and the structure must be complete; otherwise re-impute
    if (mi && mi.fromFile && !matchesRows(mi, plainRows, m)) { miCache.delete(key); mi = null; }
    if (mi) {
        miCache.delete(key); miCache.set(key, mi); // most recently used last
        if (onReuse) onReuse();
    } else {
        mi = await miceFitRun(plainRows, JSON.parse(JSON.stringify(columnConfigs)), { m, seed }, { onProgress, isAborted });
        if (isAborted && isAborted()) throw new Error('Training aborted by user.');
        miCache.set(key, mi);
        while (miCache.size > MI_CACHE_SIZE) miCache.delete(miCache.keys().next().value);
        updateImputedDataButtons();
    }
    await ModelEngine.yieldToBrowser();
    const stacked = [].concat(...mi.datasets);
    const prep = DataEngine.prepareFeaturesAndTarget(stacked, columnConfigs);
    prep.imputation = { configs: JSON.parse(JSON.stringify(columnConfigs)), mice: mi.imputer };
    const sets = mi.datasets.map(d => {
        const t = DataEngine.prepareTestingDataset(d, prep);
        return { X: t.X, y: t.y };
    });
    prep.X = sets[0].X;
    prep.y = sets[0].y;
    // Predictors with no observed value at all: MICE cannot impute them and fills a constant, so they carry no
    // information (reported in the MICE diagnostics)
    const allMissing = Object.keys(columnConfigs).filter(c => columnConfigs[c].role === 'feature' && plainRows.length && plainRows.every(r => DataEngine.isMissing(r[c])));
    prep.multipleImputation = { m, imputedColumns: mi.imputedColumns, allMissing, rows: rawRows.length, trace: mi.trace || null, traceScale: mi.traceScale || null };
    return { prep, sets, imputedColumns: mi.imputedColumns };
}

// Rough duration of the imputation step, for a warning before large runs (measured: 5,000 patients with
// 16 predictors take about 2 s for 5 imputations on one core, growing with patients, predictors and
// imputations; larger jobs run several imputations at a time, as in compute.js miceFitRun)
function estimateImputationSeconds(nRows, nPredictors, m) {
    const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2;
    const k = nRows * nPredictors * m >= 100000 ? Math.min(m, Math.max(1, cores - 1), 8) : 1;
    const oneCore = (nRows / 5000) * Math.pow(nPredictors / 16, 1.6) * (m / 5) * 2;
    return Math.round(oneCore * Math.ceil(m / k) / m + (k > 1 ? 1 : 0)); // + starting the extra workers
}

/** Trains one model per completed copy and returns the pooled model */
async function fitPooledModel(modelId, isClassification, numClasses, hp, sets, opts = {}) {
    const models = [];
    for (let l = 0; l < sets.length; l++) {
        if (opts.onImputation) opts.onImputation(l + 1, sets.length);
        models.push(await fitModel(modelId, isClassification, numClasses, hp, sets[l].X, sets[l].y, opts));
    }
    return ModelEngine.createMIPooled(models);
}

// Shown under the Step 4 choice: how long multiple imputation will take for the loaded cohort
function updateImputationEstimate() {
    const box = document.getElementById('mi-estimate');
    if (!box) return;
    const m = readMultipleImputation();
    const rows = AppState.parsedData && AppState.parsedData.rows ? AppState.parsedData.rows.length : 0;
    const nPred = AppState.columnConfigs ? Object.keys(AppState.columnConfigs).filter(c => AppState.columnConfigs[c].role === 'feature').length : 0;
    if (!m || !rows || !nPred) { box.style.display = 'none'; const q = document.getElementById('mi-quick-cv-row'); if (q) q.style.display = m ? 'flex' : 'none'; return; }
    const est = estimateImputationSeconds(rows, nPred, m);
    const quick = document.getElementById('mi-quick-cv-row');
    if (quick) quick.style.display = m ? 'flex' : 'none';
    const t = s => (s < 10 ? 'a few seconds' : s < 60 ? `about ${Math.round(s / 5) * 5} seconds` : `about ${Math.round(s / 60)} minute${Math.round(s / 60) === 1 ? '' : 's'}`);
    box.style.display = 'block';
    const foldRows = Math.round(rows * 0.8 * 0.8), foldSec = estimateImputationSeconds(foldRows, nPred, m), bootSec = estimateImputationSeconds(rows, nPred, 1);
    box.innerHTML = `For ${rows.toLocaleString('en-GB')} patients and ${nPred} predictors, the imputation takes ${t(est)} for the final models (twice with Option A when they are retrained on all patients), and each selected model is trained ${m} times. When cross-validation is used (tuning, model selection, recalibration) MICE is repeated in each of the 5 folds (${t(foldSec)} each, once for all models); with bootstrap validation (Option C) one MICE run is added to every resample (${t(bootSec)} each, for example ${t(bootSec * 200)} for 200 resamples per model). It runs in the background: the page stays usable and Force Stop works. Keep this window in front while it runs.`;
}

export { clearImputationCache, loadImputedDataFile, saveImputedData, updateImputedDataButtons, updateImputationEstimate, readMultipleImputation, prepareMultipleImputation, fitPooledModel, estimateImputationSeconds };
