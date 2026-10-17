// compute.js: runs model training and cross-validation fits in a background Web Worker (compute-worker.js),
// so the page stays responsive and Force Stop takes effect at once (the worker is terminated). When workers
// are unavailable (e.g. the app opened directly from disk) or fail, the same code runs on the page instead.
/* global ModelEngine, DataEngine */
import { bootstrapOptimism } from './validation-core.js';

let worker = null;
let workerBroken = false;
let nextId = 1;
const pending = new Map();
let workerInfo = null;

// Worker URL carrying the content-hash versions of its scripts (from the page's import map), so the worker
// never runs stale cached engine code after an update
function versionedWorkerUrl() {
    try {
        const map = JSON.parse(document.querySelector('script[type="importmap"]').textContent).imports || {};
        const v = k => { const m = String(map[k] || '').match(/\?h=([0-9a-f]+)/); return m ? m[1] : ''; };
        return `compute-worker.js?h=${v('./compute-worker.js')}&de=${v('./data-engine.js')}&m=${v('./models.js')}&vc=${v('./dist/validation-core.worker.js')}`;
    } catch (e) {
        return 'compute-worker.js';
    }
}

function createWorker() {
    if (workerBroken || typeof Worker === 'undefined' || location.protocol === 'file:') return null;
    try {
        const w = new Worker(versionedWorkerUrl());
        w.onmessage = (ev) => {
            const { id, progress, result, error } = ev.data || {};
            const job = pending.get(id);
            if (!job) return;
            if (progress !== undefined) { if (job.onProgress) { try { job.onProgress(progress); } catch (e) { /* ignore */ } } return; }
            pending.delete(id);
            clearInterval(job.timer);
            if (error !== undefined) job.reject(new Error(error)); else job.resolve(result);
        };
        // A result that cannot be read (e.g. it could not be copied between threads): fail the jobs, never hang
        w.onmessageerror = () => {
            pending.forEach(job => { clearInterval(job.timer); job.reject(new Error('The background worker sent a result that could not be read.')); });
            pending.clear();
        };
        w.onerror = (e) => {
            // A worker that cannot start (e.g. blocked script) is not used again; pending jobs fall back
            workerBroken = true;
            e.preventDefault && e.preventDefault();
            pending.forEach(job => { clearInterval(job.timer); job.reject(Object.assign(new Error('Background worker unavailable'), { workerUnavailable: true })); });
            pending.clear();
            try { w.terminate(); } catch (err) { /* ignore */ }
            worker = null;
        };
        return w;
    } catch (e) {
        workerBroken = true;
        return null;
    }
}

function call(op, payload, { onProgress = null, isAborted = null } = {}) {
    if (!worker) worker = createWorker();
    if (!worker) return Promise.reject(Object.assign(new Error('Background worker unavailable'), { workerUnavailable: true }));
    const id = nextId++;
    return new Promise((resolve, reject) => {
        const job = { resolve, reject, onProgress, timer: null };
        // Force Stop: terminate the worker immediately (a new one is created for the next job)
        if (isAborted) {
            job.timer = setInterval(() => {
                if (isAborted()) cancelAll('Training aborted by user.');
            }, 150);
        }
        pending.set(id, job);
        worker.postMessage({ id, op, payload });
    });
}

function cancelAll(message = 'Training aborted by user.') {
    if (worker) { try { worker.terminate(); } catch (e) { /* ignore */ } }
    worker = null;
    pending.forEach(job => { clearInterval(job.timer); job.reject(new Error(message)); });
    pending.clear();
}

function workerAvailable() {
    return !workerBroken && typeof Worker !== 'undefined' && location.protocol !== 'file:';
}

// Which backend the worker's TensorFlow.js uses (for display); null if not started
async function describeWorker() {
    if (workerInfo) return workerInfo;
    try { workerInfo = await call('ping', {}); } catch (e) { workerInfo = null; }
    return workerInfo;
}

function applyExtras(model, r) {
    ['trainingInfo', 'usedFallback', 'fallbackReason', 'weightingNote', 'fallbackNote'].forEach(k => {
        if (r[k] !== undefined && r[k] !== null) model[k] = r[k];
    });
    return model;
}

// Local (main-thread) training, used when the worker is unavailable
async function fitLocal(modelId, isClassification, numClasses, hp, X, y, onProgress, isAborted) {
    const m = ModelEngine.createModel(modelId, isClassification, numClasses, hp);
    if (modelId === 'nn' || modelId === 'ensemble') {
        await m.fit(X, y, null, null, (a, b, c) => {
            if (isAborted && isAborted()) throw new Error('Training aborted by user.');
            if (onProgress) onProgress(a, b, c);
        });
    } else if (typeof m.fitAsync === 'function') {
        await m.fitAsync(X, y, { onYield: () => { if (isAborted && isAborted()) throw new Error('Training aborted by user.'); } });
    } else {
        m.fit(X, y);
    }
    return m;
}

/**
 * Trains a model and returns it ready for prediction on the page.
 * onProgress(a, b, c): neural network (epoch, total, logs) or ensemble (current, total, info).
 */
async function fitModel(modelId, isClassification, numClasses, hp, X, y, { onProgress = null, isAborted = null } = {}) {
    if (workerAvailable()) {
        try {
            const r = await call('fit', { modelId, isClassification, numClasses, hp, X, y },
                { onProgress: p => onProgress && onProgress(p.a, p.b, p.c), isAborted });
            return applyExtras(await ModelEngine.deserializeModel(r.serialized), r);
        } catch (e) {
            if (!e.workerUnavailable) throw e;
        }
    }
    return fitLocal(modelId, isClassification, numClasses, hp, X, y, onProgress, isAborted);
}

/** Trains on a fold's learning rows and returns the fold's validation metrics (no model is kept) */
async function fitAndScore(modelId, isClassification, numClasses, hp, fold, evalOptions, { onProgress = null, isAborted = null } = {}) {
    if (workerAvailable()) {
        try {
            return await call('fitEval', {
                modelId, isClassification, numClasses, hp, X: fold.Xtr, y: fold.ytr,
                Xva: fold.Xva, yva: fold.yva, targetClasses: fold.targetClasses, evalOptions
            }, { onProgress: p => onProgress && onProgress(p.a, p.b, p.c), isAborted });
        } catch (e) {
            if (!e.workerUnavailable) throw e;
        }
    }
    const m = await fitLocal(modelId, isClassification, numClasses, hp, fold.Xtr, fold.ytr, onProgress, isAborted);
    try {
        return await ModelEngine.evaluateAsync(m, fold.Xva, fold.yva, isClassification, fold.targetClasses, evalOptions);
    } finally {
        if (m.dispose) m.dispose();
    }
}

/** Trains once and returns metrics on each of several evaluation sets [{X, y, targetClasses}] */
async function fitAndScoreMulti(modelId, isClassification, numClasses, hp, X, y, evalSets, evalOptions, { onProgress = null, isAborted = null } = {}) {
    if (workerAvailable()) {
        try {
            return await call('fitEvalMulti', { modelId, isClassification, numClasses, hp, X, y, evalSets, evalOptions },
                { onProgress: p => onProgress && onProgress(p.a, p.b, p.c), isAborted });
        } catch (e) {
            if (!e.workerUnavailable) throw e;
        }
    }
    const m = await fitLocal(modelId, isClassification, numClasses, hp, X, y, onProgress, isAborted);
    try {
        const out = [];
        for (const set of evalSets) {
            const e = await ModelEngine.evaluateAsync(m, set.X, set.y, isClassification, set.targetClasses, evalOptions);
            const r = {};
            if (e) { ['accuracy', 'balancedAccuracy', 'macroF1', 'rocAuc', 'brierScore', 'r2', 'rmse', 'mae'].forEach(k => { if (e[k] !== undefined) r[k] = e[k]; }); if (e.calibration) r.calibrationSlope = e.calibration.slope; }
            out.push(r);
        }
        return out;
    } finally {
        if (m.dispose) m.dispose();
    }
}

/**
 * Option C: the whole bootstrap loop of one model (validation-core.js bootstrapOptimism), run inside the
 * worker so the page stays responsive; on the page when workers are unavailable. `payload` is the plain-data
 * part of the options (tune.groupCol names the patient-ID column, if any); onResample(b, n, t0) and
 * onInnerProgress(b, n, c, nc, f, nf) report progress.
 */
async function bootstrapRun(payload, { onResample = null, onInnerProgress = null, isAborted = null } = {}) {
    if (workerAvailable()) {
        try {
            return await call('bootstrap', payload, {
                isAborted,
                onProgress: p => {
                    if (p && p.kind === 'resample' && onResample) onResample(p.b, p.n, p.t0);
                    else if (p && p.kind === 'inner' && onInnerProgress) onInnerProgress(p.b, p.n, p.c, p.nc, p.f, p.nf);
                }
            });
        } catch (e) {
            if (!e.workerUnavailable) throw e;
        }
    }
    const t = payload.tune;
    return bootstrapOptimism({
        ...payload, fitAndScoreMulti, isAborted, onResample, onInnerProgress,
        tune: t ? { ...t, fitAndScore, groupOf: t.groupCol ? (r => String(r[t.groupCol] ?? '').trim()) : null } : null
    });
}

/**
 * Multiple imputation of training rows (DataEngine.miceFit) in the background, so the page stays responsive
 * and Force Stop works; on the page when workers are unavailable. Larger jobs are split across several
 * workers, each computing some of the imputations: every imputation is seeded on its own, so the merged
 * result is identical to a single run, only sooner. onProgress(l, m, it, iters, parallel): the imputation
 * and round reached (with `parallel` workers, l and it express the overall share of rounds completed).
 */
async function miceFitRun(rows, columnConfigs, options = {}, { onProgress = null, isAborted = null } = {}) {
    if (workerAvailable()) {
        const m = options.m || 5;
        const nFeat = Object.values(columnConfigs).filter(c => c.role === 'feature').length;
        const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2;
        const k = rows.length * nFeat * m >= 100000 ? Math.min(m, Math.max(1, cores - 1), 8) : 1;
        try {
            if (k > 1) return await miceFitParallel(rows, columnConfigs, options, k, { onProgress, isAborted });
            return await call('miceFit', { rows, columnConfigs, options }, {
                isAborted,
                onProgress: p => { if (p && p.kind === 'mice' && onProgress) onProgress(p.l, p.m, p.it, p.iters, 1); }
            });
        } catch (e) {
            if (!e.workerUnavailable) throw e;
        }
    }
    return DataEngine.miceFit(rows, columnConfigs, { ...options, onProgress });
}

// Runs the imputations in k separate workers (consecutive blocks of imputations, merged in order)
function miceFitParallel(rows, columnConfigs, options, k, { onProgress = null, isAborted = null } = {}) {
    const m = options.m || 5;
    const blocks = Array.from({ length: k }, () => []);
    for (let l = 0; l < m; l++) blocks[Math.floor(l * k / m)].push(l);
    const done = new Array(k).fill(0); // rounds completed by each worker
    const helpers = [];
    let timer = null;
    const stop = () => { clearInterval(timer); helpers.forEach(w => { try { w.terminate(); } catch (e) { /* ignore */ } }); };
    return new Promise((resolve, reject) => {
        const parts = new Array(k);
        let left = k, failed = false;
        const fail = err => { if (failed) return; failed = true; stop(); reject(err); };
        if (isAborted) timer = setInterval(() => { if (isAborted()) fail(new Error('Training aborted by user.')); }, 150);
        blocks.forEach((chains, b) => {
            let w;
            try { w = new Worker(versionedWorkerUrl()); } catch (e) { fail(Object.assign(new Error('Background worker unavailable'), { workerUnavailable: true })); return; }
            helpers.push(w);
            w.onmessage = ev => {
                const { progress, result, error } = ev.data || {};
                if (progress !== undefined) {
                    if (progress && progress.kind === 'mice') {
                        done[b] = (chains.indexOf(progress.l - 1)) * progress.iters + progress.it;
                        const total = done.reduce((a, v) => a + v, 0), iters = progress.iters;
                        const l = Math.max(1, Math.ceil(total / iters));
                        if (onProgress) { try { onProgress(l, m, total - (l - 1) * iters, iters, k); } catch (e) { /* ignore */ } }
                    }
                    return;
                }
                if (error !== undefined) { fail(new Error(error)); return; }
                parts[b] = result;
                try { w.terminate(); } catch (e) { /* ignore */ }
                if (--left === 0 && !failed) { clearInterval(timer); resolve(DataEngine.mergeMiceParts(parts)); }
            };
            w.onmessageerror = () => fail(new Error('The background worker sent a result that could not be read.'));
            w.onerror = e => { e.preventDefault && e.preventDefault(); fail(Object.assign(new Error('Background worker unavailable'), { workerUnavailable: true })); };
            w.postMessage({ id: 1, op: 'miceFit', payload: { rows, columnConfigs, options: { ...options, chains } } });
        });
    });
}

export { fitModel, fitAndScore, fitAndScoreMulti, bootstrapRun, miceFitRun, cancelAll, workerAvailable, describeWorker };
