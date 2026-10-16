// Background compute worker: trains models (and scores cross-validation folds) off the page's main thread,
// so the interface stays responsive and Force Stop can end a computation immediately.
// Messages in:  { id, op: 'fit' | 'fitEval' | 'ping', payload }
// Messages out: { id, progress } during training, then { id, result } or { id, error }.
/* global importScripts, tf, ModelEngine */
'use strict';

self.window = self; // the engines attach themselves to window.*
importScripts('tf.min.js', 'data-engine.js', 'models.js');

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
        throw new Error(`Unknown operation: ${op}`);
    } catch (err) {
        self.postMessage({ id, error: err && err.message ? err.message : String(err) });
    }
};
