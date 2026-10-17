// multiple-imputation.js: multiple imputation (MICE) for the prediction models in the standard workflow.
// The imputation models are fitted on the training patients only (without the outcome); one model is
// trained per completed dataset and the predictions are pooled (ModelEngine.createMIPooled). Testing and new
// patients are imputed with the same fitted chains by DataEngine.transformSingleRow.
/* global DataEngine, ModelEngine */
import { fitModel, miceFitRun } from './compute.js';
import { AppState } from './core.js';

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
function clearImputationCache() { miCache.clear(); }

/**
 * Multiply imputes raw training rows and returns the encoding fitted on all completed copies (prep, whose
 * X / y are the first copy, for counts and displays) and the encoded copies (sets) to train on.
 */
async function prepareMultipleImputation(rawRows, columnConfigs, m, seed, { onProgress = null, isAborted = null } = {}) {
    // Plain copies: the rows are sent to the background worker
    const plainRows = rawRows.map(r => ({ ...r }));
    const key = miCacheKey(plainRows, columnConfigs, m, seed);
    let mi = miCache.get(key);
    if (mi) {
        miCache.delete(key); miCache.set(key, mi); // most recently used last
    } else {
        mi = await miceFitRun(plainRows, JSON.parse(JSON.stringify(columnConfigs)), { m, seed }, { onProgress, isAborted });
        if (isAborted && isAborted()) throw new Error('Training aborted by user.');
        miCache.set(key, mi);
        while (miCache.size > MI_CACHE_SIZE) miCache.delete(miCache.keys().next().value);
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
    prep.multipleImputation = { m, imputedColumns: mi.imputedColumns, rows: rawRows.length };
    return { prep, sets, imputedColumns: mi.imputedColumns };
}

// Rough duration of the imputation step, for a warning before large runs (measured: 5,000 patients with
// 16 predictors take about 4 s for 5 imputations, growing with patients, predictors and imputations)
function estimateImputationSeconds(nRows, nPredictors, m) {
    return Math.round((nRows / 5000) * Math.pow(nPredictors / 16, 1.6) * (m / 5) * 4);
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
    if (!m || !rows || !nPred) { box.style.display = 'none'; return; }
    const est = estimateImputationSeconds(rows, nPred, m);
    const t = s => (s < 10 ? 'a few seconds' : s < 60 ? `about ${Math.round(s / 5) * 5} seconds` : `about ${Math.round(s / 60)} minute${Math.round(s / 60) === 1 ? '' : 's'}`);
    box.style.display = 'block';
    box.innerHTML = `⏱️ For ${rows.toLocaleString('en-GB')} patients and ${nPred} predictors, the imputation step takes ${t(est)} (twice with Option A when the final model is retrained on all patients; done once: training other models later on the same patients with the same settings reuses the imputed datasets), and each selected model is trained ${m} times. It runs in the background: the page stays usable and Force Stop works. Keep this window in front while it runs: browsers slow down background work in hidden or minimised windows.`;
}

export { clearImputationCache, updateImputationEstimate, readMultipleImputation, prepareMultipleImputation, fitPooledModel, estimateImputationSeconds };
