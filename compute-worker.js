// Background compute worker: trains models (and scores cross-validation folds) off the page's main thread,
// so the interface stays responsive and Force Stop can end a computation immediately.
// Messages in:  { id, op: 'fit' | 'fitEval' | 'fitEvalMulti' | 'bootstrap' | 'miceFit' | 'ping', payload }
// Messages out: { id, progress } during training, then { id, result } or { id, error }.
/* global importScripts, tf, ModelEngine, DataEngine */
'use strict';

self.window = self; // the engines attach themselves to window.*
// Versioned engine URLs (passed by the page) so an update is never masked by the browser cache
const qs = new URLSearchParams(self.location.search);
const ver = k => (qs.get(k) ? `?h=${qs.get(k)}` : '');
importScripts('tf.min.js', `data-engine.js${ver('de')}`, `models.js${ver('m')}`, `dist/validation-core.worker.js${ver('vc')}`);
/* global bootstrapOptimism */

async function fitModel(p, onProgress) {
    const m = ModelEngine.createModel(p.modelId, p.isClassification, p.numClasses, p.hp);
    if (p.modelId === 'nn' || p.modelId === 'ensemble') {
        await m.fit(p.X, p.y, null, null, (a, b, c) => onProgress({ a, b, c: c ? JSON.parse(JSON.stringify(c)) : null }));
    } else if (typeof m.fitAsync === 'function') {
        // Heartbeat for the page's progress monitor (Random Forest trees, logistic iterations)
        let last = 0;
        await m.fitAsync(p.X, p.y, { onYield: () => { const now = Date.now(); if (now - last > 500) { last = now; onProgress({ a: 'tick' }); } } });
    } else {
        m.fit(p.X, p.y);
    }
    return m;
}

// Fit-and-score helpers with the page's signatures, run here in the worker (used by the Option C loop)
const KEEP = ['accuracy', 'balancedAccuracy', 'macroF1', 'rocAuc', 'brierScore', 'r2', 'rmse', 'mae'];
async function localFitAndScore(modelId, isClassification, numClasses, hp, fold, evalOptions) {
    const m = await fitModel({ modelId, isClassification, numClasses, hp, X: fold.Xtr, y: fold.ytr }, () => {});
    try {
        return await ModelEngine.evaluateAsync(m, fold.Xva, fold.yva, isClassification, fold.targetClasses, evalOptions || {});
    } finally { if (m.dispose) m.dispose(); }
}
// evalOptions.recalibrator (optional): the fitted model is wrapped with it before scoring (Option C repeats
// recalibration inside every resample)
async function localFitAndScoreMulti(modelId, isClassification, numClasses, hp, X, y, evalSets, evalOptions) {
    const { recalibrator = null, ...evalOpts } = evalOptions || {};
    const fitted = await fitModel({ modelId, isClassification, numClasses, hp, X, y }, () => {});
    const m = recalibrator ? ModelEngine.wrapCalibrated(fitted, recalibrator) : fitted;
    try {
        const out = [];
        for (const set of evalSets) {
            const e = await ModelEngine.evaluateAsync(m, set.X, set.y, isClassification, set.targetClasses, evalOpts);
            const r = {};
            if (e) { KEEP.forEach(k => { if (e[k] !== undefined) r[k] = e[k]; }); if (e.calibration) r.calibrationSlope = e.calibration.slope; }
            out.push(r);
        }
        return out;
    } finally { if (m.dispose) m.dispose(); }
}

function extras(m) {
    return {
        trainingInfo: m.trainingInfo || null,
        usedFallback: Boolean(m.usedFallback),
        fallbackReason: m.fallbackReason || null,
        weightingNote: m.weightingNote || null,
        fallbackNote: m.fallbackNote || null
    };
}

self.onmessage = async (ev) => {
    const { id, op, payload } = ev.data || {};
    const progress = p => self.postMessage({ id, progress: p });
    try {
        if (op === 'ping') {
            if (typeof tf !== 'undefined' && tf.ready) await tf.ready();
            self.postMessage({ id, result: { backend: typeof tf !== 'undefined' && tf.getBackend ? tf.getBackend() : null } });
            return;
        }
        if (op === 'fit') {
            const m = await fitModel(payload, progress);
            const serialized = await ModelEngine.serializeModel(m);
            const result = { serialized, ...extras(m) };
            if (m.dispose) m.dispose();
            self.postMessage({ id, result });
            return;
        }
        if (op === 'fitEval') {
            const m = await fitModel(payload, progress);
            const e = await ModelEngine.evaluateAsync(m, payload.Xva, payload.yva, payload.isClassification, payload.targetClasses, payload.evalOptions || {});
            const keep = ['accuracy', 'balancedAccuracy', 'macroF1', 'rocAuc', 'brierScore', 'r2', 'rmse', 'mae', 'probPositive', 'actuals'];
            const result = {};
            if (e) keep.forEach(k => { if (e[k] !== undefined) result[k] = e[k]; });
            if (m.dispose) m.dispose();
            self.postMessage({ id, result });
            return;
        }
        if (op === 'miceFit') {
            // Multiple imputation of the training patients (DataEngine.miceFit), off the page
            const p = payload;
            const res = DataEngine.miceFit(p.rows, p.columnConfigs, { ...p.options, onProgress: (l, m, it, iters) => progress({ kind: 'mice', l, m, it, iters }) });
            self.postMessage({ id, result: res });
            return;
        }
        if (op === 'bootstrap') {
            // The whole Option C loop (validation-core.js) runs here, so the page stays responsive
            const p = payload;
            const tune = p.tune ? { ...p.tune, fitAndScore: localFitAndScore, groupOf: p.tune.groupCol ? (r => String(r[p.tune.groupCol] ?? '').trim()) : null } : null;
            const run = await bootstrapOptimism({
                ...p, tune, fitAndScoreMulti: localFitAndScoreMulti,
                onResample: (b, n, t0) => progress({ kind: 'resample', b, n, t0 }),
                onInnerProgress: (b, n, c, nc, f, nf) => progress({ kind: 'inner', b, n, c, nc, f, nf })
            });
            self.postMessage({ id, result: run });
            return;
        }
        if (op === 'fitEvalMulti') {
            // Train once, then score on several datasets (e.g. bootstrap sample and original data)
            const { recalibrator = null, ...evalOpts } = payload.evalOptions || {};
            const fitted = await fitModel(payload, progress);
            const m = recalibrator ? ModelEngine.wrapCalibrated(fitted, recalibrator) : fitted;
            const keep = ['accuracy', 'balancedAccuracy', 'macroF1', 'rocAuc', 'brierScore', 'r2', 'rmse', 'mae'];
            const out = [];
            for (const set of payload.evalSets) {
                const e = await ModelEngine.evaluateAsync(m, set.X, set.y, payload.isClassification, set.targetClasses, evalOpts);
                const r = {};
                if (e) { keep.forEach(k => { if (e[k] !== undefined) r[k] = e[k]; }); if (e.calibration) r.calibrationSlope = e.calibration.slope; }
                out.push(r);
            }
            if (m.dispose) m.dispose();
            self.postMessage({ id, result: out });
            return;
        }
        throw new Error(`Unknown operation: ${op}`);
    } catch (err) {
        self.postMessage({ id, error: err && err.message ? err.message : String(err) });
    }
};
