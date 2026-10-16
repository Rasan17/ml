// compute.js: runs model training and cross-validation fits in a background Web Worker (compute-worker.js),
// so the page stays responsive and Force Stop takes effect at once (the worker is terminated). When workers
// are unavailable (e.g. the app opened directly from disk) or fail, the same code runs on the page instead.
/* global ModelEngine */

let worker = null;
let workerBroken = false;
let nextId = 1;
const pending = new Map();
let workerInfo = null;

function createWorker() {
    if (workerBroken || typeof Worker === 'undefined' || location.protocol === 'file:') return null;
    try {
        const w = new Worker('compute-worker.js');
        w.onmessage = (ev) => {
            const { id, progress, result, error } = ev.data || {};
            const job = pending.get(id);
            if (!job) return;
            if (progress !== undefined) { if (job.onProgress) { try { job.onProgress(progress); } catch (e) { /* ignore */ } } return; }
            pending.delete(id);
            clearInterval(job.timer);
            if (error !== undefined) job.reject(new Error(error)); else job.resolve(result);
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

export { fitModel, fitAndScore, cancelAll, workerAvailable, describeWorker };
