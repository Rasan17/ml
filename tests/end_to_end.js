// End-to-end test matrix for the data engine and models, run the way the app runs them:
// split raw rows -> fit imputation on the training rows only -> encode/scale -> train every model ->
// evaluate on the untouched test rows -> predict new patients (batch path) with the fitted preprocessing.
// Covers binary, multiclass and regression outcomes; mixed predictor types; missing values; unseen
// categories; rare outcomes; tiny cohorts; patient-grouped and date-based splits; calibration; the
// decision threshold; and the neural network's fallback when TensorFlow.js cannot train.
// Usage (from the project folder):  node tests/end_to_end.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const dir = path.join(__dirname, '..');
global.window = globalThis;
global.self = globalThis;
global.require = require;
const realLog = console.log;
console.log = () => {};   // TF.js prints a "use tfjs-node" banner on load
console.warn = () => {};
vm.runInThisContext(fs.readFileSync(path.join(dir, 'tf.min.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(dir, 'data-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(dir, 'models.js'), 'utf8'));
const DE = globalThis.DataEngine;
const ME = globalThis.ModelEngine;

let passed = 0;
let failed = 0;
const lines = [];
async function check(name, fn) {
    try {
        const msg = await fn();
        passed++;
        lines.push(`PASS  ${name}${msg ? `  (${msg})` : ''}`);
    } catch (e) {
        failed++;
        lines.push(`FAIL  ${name}: ${e && e.message ? e.message : e}`);
    }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }
const finite = v => typeof v === 'number' && isFinite(v);

// Seeded synthetic cohort with continuous, ordinal and nominal predictors and missing values
function cohort(n, seed, outcome, opts = {}) {
    let s = seed;
    const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
    const g = () => Math.sqrt(-2 * Math.log(rnd() || 1e-9)) * Math.cos(2 * Math.PI * rnd());
    const rows = [];
    for (let i = 0; i < n; i++) {
        const age = 70 + 8 * g();
        const icp = 12 + 4 * g();
        const gait = Math.max(0, Math.min(4, Math.round(2 + g())));
        const site = ['North', 'South', 'East'][Math.floor(rnd() * 3)];
        const z = 0.06 * (age - 70) + 0.3 * (icp - 12) - 0.4 * (gait - 2) + (site === 'North' ? 0.5 : 0);
        const p = 1 / (1 + Math.exp(-(z + (opts.shift || 0))));
        let y;
        if (outcome === 'binary') y = rnd() < p ? 'Yes' : 'No';
        else if (outcome === 'multi') y = z + 0.7 * g() > 0.8 ? 'Good' : z + 0.7 * g() < -0.8 ? 'Poor' : 'Fair';
        else y = (50 + 8 * z + 3 * g()).toFixed(2);
        const r = {
            PatientID: `PT${Math.floor(i / (opts.visitsPerPatient || 1))}`,
            VisitDate: `2024-${String(1 + (i % 12)).padStart(2, '0')}-${String(1 + (i % 28)).padStart(2, '0')}`,
            Age: age.toFixed(1), ICP: icp.toFixed(1), Gait: String(gait), Site: site, Outcome: y
        };
        if (opts.missing && rnd() < 0.1) r.ICP = '';
        if (opts.missing && rnd() < 0.05) r.Site = '';
        rows.push(r);
    }
    return rows;
}
function configs(outcome) {
    return {
        PatientID: { role: 'meta', type: 'nominal' },
        VisitDate: { role: 'meta', type: 'nominal' },
        Age: { role: 'feature', type: 'continuous', imputationStrategy: 'median' },
        ICP: { role: 'feature', type: 'continuous', imputationStrategy: 'median' },
        Gait: { role: 'feature', type: 'ordinal', imputationStrategy: 'median' },
        Site: { role: 'feature', type: 'nominal', imputationStrategy: 'mode' },
        Outcome: { role: 'target', type: outcome === 'regression' ? 'continuous' : 'nominal' }
    };
}

// The app's leak-free workflow (standard workflow, Option A)
function prepareWorkflow(rows, cfg, isClassification, split) {
    const analysis = DE.analyzeColumns(Object.keys(rows[0]), rows);
    const trainImp = DE.imputeDataset(split.trainX, cfg, analysis);
    const testImp = DE.imputeDataset(split.testX, cfg, analysis, trainImp.plan);
    const prep = DE.prepareFeaturesAndTarget(trainImp.imputedRows, cfg);
    prep.imputation = { configs: JSON.parse(JSON.stringify(cfg)), plan: trainImp.plan }; // as the app does
    const test = DE.prepareTestingDataset(testImp.imputedRows, prep);
    return { analysis, trainImp, testImp, prep, test };
}

const cfgHasBlank = r => ['Age', 'ICP', 'Gait', 'Site'].some(c => r[c] === '' || r[c] === undefined);
const MODELS = ['glm', 'rf', 'knn', 'nb', 'nn', 'ensemble'];
const HP = { glm: {}, rf: { numTrees: 20, maxDepth: 5 }, knn: { k: 7 }, nb: {}, nn: { epochs: 15, hiddenLayers: [16, 8], batchSize: 16 }, ensemble: { baseModels: ['rf', 'glm', 'knn'] } };

async function trainEval(id, w, isClassification) {
    const m = ME.createModel(id, isClassification, w.prep.targetClasses.length || 2, { ...HP[id], seed: 3 });
    if (id === 'nn' || id === 'ensemble') await m.fit(w.prep.X, w.prep.y);
    else if (m.fitAsync) await m.fitAsync(w.prep.X, w.prep.y);
    else m.fit(w.prep.X, w.prep.y);
    const ev = await ME.evaluateAsync(m, w.test.X, w.test.y, isClassification, w.prep.targetClasses,
        isClassification ? {} : { regressionThreshold: 50, thresholdSource: 'test' });
    return { m, ev };
}


// The resampling engine (src/validation-core.js) is loaded as-is, with the app's local (no worker)
// training functions, so the test runs the same code as Option C in the browser
function loadValidationCore() {
    const code = fs.readFileSync(path.join(dir, 'src', 'validation-core.js'), 'utf8').replace(/^export\s*\{[^}]*\};?\s*$/gm, '');
    return new Function(`${code}\nreturn { bootstrapResamples, bootstrapResamplesGrouped, bootstrapOptimism, selectionOptimism, tuningGrid, noInformationRates, plus632, youdenThreshold, effectiveFolds, buildRowFolds };`)();
}
async function fitLocal(id, isClf, nc, hp, X, y) {
    const m = ME.createModel(id, isClf, nc, hp);
    if (id === 'nn' || id === 'ensemble') await m.fit(X, y);
    else if (m.fitAsync) await m.fitAsync(X, y);
    else m.fit(X, y);
    return m;
}
const KEEP = ['accuracy', 'balancedAccuracy', 'macroF1', 'rocAuc', 'brierScore', 'r2', 'rmse', 'mae'];
async function localFitAndScore(id, isClf, nc, hp, fold, evalOptions) {
    const m = await fitLocal(id, isClf, nc, hp, fold.Xtr, fold.ytr);
    const ev = await ME.evaluateAsync(m, fold.Xva, fold.yva, isClf, fold.targetClasses, evalOptions);
    if (m.dispose) m.dispose();
    return ev;
}
async function localFitAndScoreMulti(id, isClf, nc, hp, X, y, sets, evalOptions) {
    const m = await fitLocal(id, isClf, nc, hp, X, y);
    const out = [];
    for (const set of sets) {
        const e = await ME.evaluateAsync(m, set.X, set.y, isClf, set.targetClasses, evalOptions);
        const r = {};
        KEEP.forEach(k => { if (e && e[k] !== undefined) r[k] = e[k]; });
        if (e && e.calibration) r.calibrationSlope = e.calibration.slope;
        out.push(r);
    }
    if (m.dispose) m.dispose();
    return out;
}
// Pure-noise cohort: no predictor is related to the outcome, so any honest estimate of AUC is about 0.5
function noiseCohort(n, p, seed) {
    let s = seed;
    const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
    const g = () => Math.sqrt(-2 * Math.log(rnd() || 1e-9)) * Math.cos(2 * Math.PI * rnd());
    const rows = [];
    for (let i = 0; i < n; i++) {
        const r = { Outcome: rnd() < 0.5 ? 'Yes' : 'No' };
        for (let j = 1; j <= p; j++) r[`N${j}`] = g().toFixed(3);
        rows.push(r);
    }
    const cfg = { Outcome: { role: 'target', type: 'nominal' } };
    for (let j = 1; j <= p; j++) cfg[`N${j}`] = { role: 'feature', type: 'continuous', imputationStrategy: 'median' };
    return { rows, cfg };
}
async function optionC(VC, rows, cfg, modelId, hp, B, tune) {
    const analysis = DE.analyzeColumns(Object.keys(rows[0]), rows);
    const imp = DE.imputeDataset(rows, cfg, analysis);
    const prep = DE.prepareFeaturesAndTarget(imp.imputedRows, cfg);
    const metricDefs = { macroF1: { label: 'Macro F1', higherIsBetter: true }, rocAuc: { label: 'ROC-AUC', higherIsBetter: true }, brierScore: { label: 'Brier', higherIsBetter: false }, accuracy: { label: 'Accuracy', higherIsBetter: true }, balancedAccuracy: { label: 'Balanced accuracy', higherIsBetter: true } };
    // Apparent performance: the whole process (tuning included) run once on all patients
    let hpAll = hp;
    if (tune) {
        const one = await VC.bootstrapOptimism({ rows, boots: [rows.map((_, i) => i)], configs: cfg, analysis, isClassification: true, modelId, hp, evalOptions: {}, keys: ['rocAuc'],
            fitAndScoreMulti: localFitAndScoreMulti, tune: { k: 3, seed: 5, metricKey: 'rocAuc', metricDefs, doTune: true, fitAndScore: localFitAndScore } });
        hpAll = one.perResample[0].hp;
    }
    const m = await fitLocal(modelId, true, 2, hpAll, prep.X, prep.y);
    const app = await ME.evaluateAsync(m, prep.X, prep.y, true, prep.targetClasses, {});
    const run = await VC.bootstrapOptimism({ rows, boots: VC.bootstrapResamples(rows.length, B, 42), configs: cfg, analysis, isClassification: true, modelId,
        hp: tune === 'inside' ? hp : hpAll, evalOptions: {}, keys: ['rocAuc', 'brierScore'], fitAndScoreMulti: localFitAndScoreMulti,
        tune: tune === 'inside' ? { k: 3, seed: 42, metricKey: 'rocAuc', metricDefs, doTune: true, fitAndScore: localFitAndScore } : null });
    const e = VC.plus632('rocAuc', app.rocAuc, run.oobMean.rocAuc, VC.noInformationRates(app, true).rocAuc);
    if (m.dispose) m.dispose();
    return { apparent: app.rocAuc, corrected: app.rocAuc - run.optimism.rocAuc, p632: e ? e.value : null, run, hpAll };
}

(async () => {
    await tf.setBackend('cpu');
    await tf.ready();

    for (const outcome of ['binary', 'multi', 'regression']) {
        const isClf = outcome !== 'regression';
        await check(`${outcome}: every model trains and gives valid test metrics (mixed predictors, missing values)`, async () => {
            const rows = cohort(400, 11, outcome, { missing: true });
            const cfg = configs(outcome);
            const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.8, isClf, 42);
            const w = prepareWorkflow(rows, cfg, isClf, split);
            assert(w.test.X.length === split.testX.length, 'test rows were lost');
            const out = [];
            for (const id of MODELS) {
                if (id === 'nb' && !isClf) continue;
                const { m, ev } = await trainEval(id, w, isClf);
                if (isClf) {
                    assert(finite(ev.accuracy) && ev.accuracy >= 0 && ev.accuracy <= 1, `${id}: accuracy`);
                    assert(finite(ev.macroF1) && finite(ev.brierScore), `${id}: F1/Brier`);
                    if (outcome === 'binary') assert(finite(ev.rocAuc) && ev.calibration && ev.calibration.bins.length > 0, `${id}: AUC/calibration`);
                    out.push(`${id} ${(ev.accuracy * 100).toFixed(0)}%`);
                } else {
                    assert(finite(ev.r2) && finite(ev.rmse) && finite(ev.mae), `${id}: regression metrics`);
                    out.push(`${id} R² ${ev.r2.toFixed(2)}`);
                }
                if (id === 'nn') assert(!m.usedFallback, 'neural network fell back unexpectedly');
                if (m.dispose) m.dispose();
            }
            return out.join(', ');
        });
    }

    await check('Batch prediction of new patients uses the fitted preprocessing (same predictions as the evaluation path)', async () => {
        const rows = cohort(300, 5, 'binary', { missing: true });
        const cfg = configs('binary');
        const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.8, true, 7);
        const w = prepareWorkflow(rows, cfg, true, split);
        const { m } = await trainEval('glm', w, true);
        const batch = split.testX.map(r => DE.transformSingleRow(r, w.prep, []));
        const outsBatch = m.predictOutputs ? m.predictOutputs(batch) : batch.map(x => m.predictSingle(x));
        const outsEval = m.predictOutputs ? m.predictOutputs(w.test.X) : w.test.X.map(x => m.predictSingle(x));
        let same = 0;
        outsBatch.forEach((o, i) => { if (o.pred === outsEval[i].pred && Math.abs(o.probabilities[1] - outsEval[i].probabilities[1]) < 1e-9) same++; });
        assert(same === outsBatch.length, `${outsBatch.length - same} prediction(s) differ`);
        const blanks = split.testX.filter(r => cfgHasBlank(r)).length;
        return `${same}/${outsBatch.length} identical, ${blanks} with blank predictors`;
    });

    await check('Unseen category and blank values in a new patient are handled (no crash, issues reported)', async () => {
        const rows = cohort(200, 9, 'binary');
        const cfg = configs('binary');
        const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.8, true, 1);
        const w = prepareWorkflow(rows, cfg, true, split);
        const { m } = await trainEval('rf', w, true);
        const issues = [];
        const x = DE.transformSingleRow({ Age: '', ICP: '15', Gait: '9', Site: 'Mars' }, w.prep, issues);
        const o = m.predictSingle(x);
        assert(x.every(finite), 'non-finite encoded value');
        assert(issues.length >= 2, 'expected the blank and unseen inputs to be reported');
        assert(o.pred === 0 || o.pred === 1, 'invalid prediction');
        return `${issues.length} input issue(s) reported`;
    });

    await check('Rare outcome (5% events): training and evaluation do not crash', async () => {
        const rows = cohort(400, 21, 'binary', { shift: -4 });
        const events = rows.filter(r => r.Outcome === 'Yes').length;
        const cfg = configs('binary');
        const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.8, true, 3);
        const w = prepareWorkflow(rows, cfg, true, split);
        for (const id of ['glm', 'rf', 'nb']) {
            const { ev } = await trainEval(id, w, true);
            assert(finite(ev.balancedAccuracy), `${id}: balanced accuracy`);
        }
        return `${events} events in ${rows.length}`;
    });

    await check('Very small cohort (12 patients) runs end to end', async () => {
        const rows = cohort(12, 4, 'binary');
        const cfg = configs('binary');
        const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.75, true, 2);
        const w = prepareWorkflow(rows, cfg, true, split);
        for (const id of ['glm', 'knn', 'rf']) {
            const { ev } = await trainEval(id, w, true);
            assert(ev && finite(ev.accuracy), `${id}: accuracy`);
        }
        return `${split.trainX.length} train / ${split.testX.length} test`;
    });

    await check('Patient-grouped split: no patient in both Training and Testing sets', async () => {
        const rows = cohort(300, 8, 'binary', { visitsPerPatient: 3 });
        const res = DE.trainTestSplitGrouped(rows, rows.map(r => r.Outcome), rows.map(r => r.PatientID), 0.8, 42);
        const trainIds = new Set(res.trainX.map(r => r.PatientID));
        const overlap = res.testX.filter(r => trainIds.has(r.PatientID)).length;
        assert(overlap === 0, `${overlap} test record(s) belong to training patients`);
        assert(res.trainX.length + res.testX.length === rows.length, 'records lost');
        const frac = res.testX.length / rows.length;
        assert(frac > 0.15 && frac < 0.25, `test fraction ${frac.toFixed(2)}`);
        return `${res.info.trainGroups} / ${res.info.testGroups} patients`;
    });

    await check('Date-based split: every Training date is on or before every Testing date', async () => {
        const rows = cohort(240, 6, 'binary');
        rows[0].VisitDate = '';                 // undated record stays in Training
        rows[1].VisitDate = '15/12/2024';       // UK day-first format
        const res = DE.trainTestSplitTemporal(rows, rows.map(r => r.Outcome), rows.map(r => r.VisitDate), 0.8);
        const t = r => DE.parseDateValue(r.VisitDate);
        const maxTrain = Math.max(...res.trainX.map(t).filter(v => v !== null));
        const minTest = Math.min(...res.testX.map(t));
        assert(maxTrain < minTest, 'a Training date is later than a Testing date');
        assert(res.info.undated === 1 && res.trainX.includes(rows[0]), 'undated record handling');
        assert(DE.parseDateValue('15/12/2024') === Date.UTC(2024, 11, 15), 'day-first date parsing');
        return `Training to ${res.info.trainUntil}, Testing from ${res.info.testFrom}`;
    });

    await check('Calibration: well-calibrated probabilities give slope ≈ 1 and intercept ≈ 0', async () => {
        let s = 99;
        const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
        const p = [], y = [];
        for (let i = 0; i < 20000; i++) { const q = 0.05 + 0.9 * rnd(); p.push(q); y.push(rnd() < q ? 1 : 0); }
        const c = ME.calibrationSummary(y, p);
        assert(Math.abs(c.slope - 1) < 0.08 && Math.abs(c.intercept) < 0.06, `slope ${c.slope}, intercept ${c.intercept}`);
        // Over-confident probabilities (pushed towards 0 and 1) must give a slope below 1
        const over = p.map(q => 1 / (1 + Math.exp(-2 * Math.log(q / (1 - q)))));
        const c2 = ME.calibrationSummary(y, over);
        assert(c2.slope < 0.6, `over-confident slope ${c2.slope}`);
        return `slope ${c.slope}, intercept ${c.intercept}; over-confident slope ${c2.slope}`;
    });

    await check('Decision threshold re-labels predictions consistently (evaluation and new patients)', async () => {
        const rows = cohort(300, 12, 'binary');
        const cfg = configs('binary');
        const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.8, true, 5);
        const w = prepareWorkflow(rows, cfg, true, split);
        const m = ME.createModel('glm', true, 2, { seed: 1 });
        m.fit(w.prep.X, w.prep.y);
        const ev5 = ME.evaluate(m, w.test.X, w.test.y, true, w.prep.targetClasses, {});
        const ev3 = ME.evaluate(m, w.test.X, w.test.y, true, w.prep.targetClasses, { decisionThreshold: 0.3 });
        const manual = ev3.probPositive.filter(v => v >= 0.3).length;
        const predPos = ev3.predictions.filter(v => v === 1).length;
        assert(manual === predPos, 'threshold not applied to every prediction');
        assert(ev3.decisionThreshold === 0.3 && ev5.decisionThreshold === 0.5, 'threshold not recorded');
        const one = ME.applyDecisionThreshold(m.predictSingle(w.test.X[0]), 0.3);
        assert(one.pred === (ev3.probPositive[0] >= 0.3 ? 1 : 0), 'single-patient path disagrees');
        const sens = e => e.confMatrix[1][1] / (e.confMatrix[1][0] + e.confMatrix[1][1]);
        assert(sens(ev3) >= sens(ev5), 'a lower threshold should not reduce sensitivity');
        return `positive at 0.5: ${ev5.predictions.filter(v => v === 1).length}, at 0.3: ${predPos}`;
    });

    await check('Neural network falls back to a linear model when TensorFlow.js cannot train', async () => {
        const rows = cohort(150, 3, 'binary');
        const cfg = configs('binary');
        const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.8, true, 9);
        const w = prepareWorkflow(rows, cfg, true, split);
        const original = tf.sequential;
        tf.sequential = () => { throw new Error('simulated WebGL failure'); };
        try {
            const { m, ev } = await trainEval('nn', w, true);
            assert(m.usedFallback === true, 'fallback flag not set');
            assert(finite(ev.accuracy), 'fallback model did not evaluate');
            return `fallback reason: ${m.fallbackReason}`;
        } finally {
            tf.sequential = original;
        }
    });

    await check('Exported models reload with identical predictions (all six model types, via JSON)', async () => {
        const rows = cohort(200, 15, 'binary', { missing: true });
        const cfg = configs('binary');
        const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.8, true, 4);
        const w = prepareWorkflow(rows, cfg, true, split);
        const done = [];
        for (const id of MODELS) {
            const { m } = await trainEval(id, w, true);
            const json = JSON.stringify(await ME.serializeModel(m));
            const back = await ME.deserializeModel(JSON.parse(json));
            const a = w.test.X.map(x => m.predictSingle(x)), b = w.test.X.map(x => back.predictSingle(x));
            const p1 = o => (o.probabilities ? Number(o.probabilities[1] || 0) : o.pred); // k-NN lists only classes among the neighbours
            const maxDiff = Math.max(...a.map((o, i) => Math.abs(p1(o) - p1(b[i]))));
            assert(a.every((o, i) => o.pred === b[i].pred) && maxDiff < 1e-6, `${id}: predictions changed (max diff ${maxDiff})`);
            done.push(id);
            if (m.dispose) m.dispose();
            if (back.dispose) back.dispose();
        }
        // A recalibrated (Platt) model keeps its calibrator through export and reload
        const base = ME.createModel('rf', true, 2, { numTrees: 10, seed: 2 });
        base.fit(w.prep.X, w.prep.y);
        const probs = w.prep.X.map(x => Number(base.predictSingle(x).probabilities[1] || 0));
        const cal = ME.fitPlatt(w.prep.y.map(v => (v === 1 ? 1 : 0)), probs);
        const wrapped = ME.wrapCalibrated(base, cal);
        const back = await ME.deserializeModel(JSON.parse(JSON.stringify(await ME.serializeModel(wrapped))));
        const same = w.test.X.every(x => Math.abs(wrapped.predictSingle(x).probabilities[1] - back.predictSingle(x).probabilities[1]) < 1e-12);
        assert(same && back.predictSingle(w.test.X[0]).probabilityType === 'recalibrated', 'recalibrated model changed after reload');
        done.push('recalibrated RF');
        return done.join(', ');
    });

    await check('Survival: time-dependent AUC and Brier score (IPCW) equal the ordinary measures without censoring and recover them under censoring', async () => {
        const rng = DE.createPRNG(5);
        const n = 3000, tau = 1;
        const x = Array.from({ length: n }, () => rng() * 2 - 1);
        const T = x.map(v => -Math.log(rng() || 1e-12) / Math.exp(1.2 * v)); // exponential, hazard exp(1.2x)
        const S = x.map(v => Math.exp(-Math.exp(1.2 * v) * tau));           // true survival at tau
        const yb = T.map(t => (t <= tau ? 1 : 0));
        // No censoring: the IPCW versions reduce to the binary AUC and Brier score of "event by tau"
        const none = ME.censoringKM(T, T.map(() => 1));
        const aucNone = ME.timeDependentAUC(T, T.map(() => 1), x, tau, none);
        const aucBin = ME._calculateBinaryAUC(yb, x);
        const bsNone = ME.ipcwBrier(T, T.map(() => 1), S, tau, none);
        const bsBin = yb.reduce((a, v, i) => a + (v - (1 - S[i])) ** 2, 0) / n;
        assert(Math.abs(aucNone - aucBin) < 1e-4 && /* the binary AUC is rounded to 4 decimals */ Math.abs(bsNone - bsBin) < 1e-12, `no-censoring identity: AUC ${aucNone} vs ${aucBin}, Brier ${bsNone} vs ${bsBin}`);
        // Independent censoring (~35%): the weighted estimates stay close to the uncensored truth
        const C = x.map(() => -Math.log(rng() || 1e-12) / 0.45);
        const obsT = T.map((t, i) => Math.min(t, C[i])), obsE = T.map((t, i) => (t <= C[i] ? 1 : 0));
        const cens = ME.censoringKM(obsT, obsE);
        const aucC = ME.timeDependentAUC(obsT, obsE, x, tau, cens), bsC = ME.ipcwBrier(obsT, obsE, S, tau, cens);
        const censored = obsE.filter(e => e === 0).length / n;
        assert(Math.abs(aucC - aucBin) < 0.02 && Math.abs(bsC - bsBin) < 0.01, `with censoring: AUC ${aucC.toFixed(3)} vs ${aucBin.toFixed(3)}, Brier ${bsC.toFixed(4)} vs ${bsBin.toFixed(4)}`);
        return `no censoring exact; ${(censored * 100).toFixed(0)}% censored: AUC ${aucC.toFixed(3)} vs true ${aucBin.toFixed(3)}, Brier ${bsC.toFixed(4)} vs ${bsBin.toFixed(4)}`;
    });

    await check('Option C in the background worker: compute-worker.js runs the bootstrap loop and gives the same result as the page', async () => {
        // Run the real worker script in a simulated worker scope (importScripts loads the project files)
        const sent = [];
        const scope = {
            location: { search: '' },
            postMessage: m => sent.push(m),
            importScripts: (...files) => files.forEach(f => {
                if (/tf\.min\.js/.test(f)) return; // TensorFlow.js is already loaded in this process
                vm.runInThisContext(fs.readFileSync(path.join(dir, f.split('?')[0]), 'utf8'));
            })
        };
        const workerCode = fs.readFileSync(path.join(dir, 'compute-worker.js'), 'utf8');
        new Function('self', 'importScripts', `${workerCode}\nreturn self;`)(Object.assign(globalThis.__workerScope = {}, scope), scope.importScripts);
        const ws = globalThis.__workerScope;
        ws.location = scope.location; ws.postMessage = scope.postMessage;
        const VC = loadValidationCore();
        const { rows, cfg } = noiseCohort(60, 5, 41);
        const analysis = DE.analyzeColumns(Object.keys(rows[0]), rows);
        const metricDefs = { rocAuc: { label: 'ROC-AUC', higherIsBetter: true } };
        const payload = { rows, boots: VC.bootstrapResamples(rows.length, 12, 3), configs: cfg, analysis, isClassification: true, modelId: 'glm', hp: { seed: 1 },
            evalOptions: {}, keys: ['rocAuc', 'brierScore'], tune: { k: 3, seed: 3, groupCol: null, metricKey: 'rocAuc', metricDefs, doTune: true } };
        await ws.onmessage({ data: { id: 1, op: 'bootstrap', payload: JSON.parse(JSON.stringify(payload)) } });
        const done = sent.find(m => m.id === 1 && (m.result || m.error));
        assert(done && done.result, `worker failed: ${done && done.error}`);
        const progress = sent.filter(m => m.progress && m.progress.kind === 'resample').length;
        const page = await VC.bootstrapOptimism({ ...payload, fitAndScoreMulti: localFitAndScoreMulti, tune: { ...payload.tune, fitAndScore: localFitAndScore } });
        const d = Math.abs(done.result.optimism.rocAuc - page.optimism.rocAuc) + Math.abs(done.result.oobMean.rocAuc - page.oobMean.rocAuc);
        assert(d < 1e-12 && done.result.B === page.B && progress === page.B, `worker and page differ (${d}; B ${done.result.B} vs ${page.B}; ${progress} progress messages)`);
        return `${done.result.B} resamples with tuning repeated, ${progress} progress messages, identical optimism (${done.result.optimism.rocAuc.toFixed(4)})`;
    });

    await check('Multiple imputation for prediction: leak-free, reproducible, survives export, and helps when data are missing at random', async () => {
        // Missing at random: X1 (strongly predictive) is missing more often when X2 is high; X2 carries information about X1
        const rng = DE.createPRNG(77);
        const g = () => Math.sqrt(-2 * Math.log(rng() || 1e-12)) * Math.cos(2 * Math.PI * rng());
        const make = n => Array.from({ length: n }, (_, i) => {
            const x2 = g(), x1 = 0.8 * x2 + 0.6 * g(), x3 = g();
            const y = rng() < 1 / (1 + Math.exp(-(1.6 * x1 + 0.3 * x3))) ? 'Yes' : 'No';
            return { ID: `P${i}`, X1: rng() < 1 / (1 + Math.exp(-(1.5 * x2 - 0.2))) ? '' : x1.toFixed(3), X2: x2.toFixed(3), X3: x3.toFixed(3), Outcome: y };
        });
        const train = make(600), test = make(1500);
        const cfg = { ID: { role: 'meta', type: 'nominal' }, Outcome: { role: 'target', type: 'nominal' } };
        ['X1', 'X2', 'X3'].forEach(c => { cfg[c] = { role: 'feature', type: 'continuous', imputationStrategy: 'median' }; });
        const analysis = DE.analyzeColumns(Object.keys(train[0]), train);
        // Single (median) imputation, as before
        const sImp = DE.imputeDataset(train, cfg, analysis);
        const sPrep = DE.prepareFeaturesAndTarget(sImp.imputedRows, cfg);
        sPrep.imputation = { configs: cfg, plan: sImp.plan };
        const sTest = DE.prepareTestingDataset(test, sPrep);
        const sModel = ME.createModel('glm', true, 2, { seed: 1 }); sModel.fit(sPrep.X, sPrep.y);
        const sEv = await ME.evaluateAsync(sModel, sTest.X, sTest.y, true, sPrep.targetClasses, {});
        // Multiple imputation: m models on m completed training sets, pooled predictions
        const mi = DE.miceFit(train, cfg, { m: 5, seed: 11 });
        const mPrep = DE.prepareFeaturesAndTarget([].concat(...mi.datasets), cfg);
        mPrep.imputation = { configs: cfg, plan: sImp.plan, mice: mi.imputer };
        const models = mi.datasets.map(d => { const t = DE.prepareTestingDataset(d, mPrep); const md = ME.createModel('glm', true, 2, { seed: 1 }); md.fit(t.X, t.y); return md; });
        const pooled = ME.createMIPooled(models);
        const mTest = DE.prepareTestingDataset(test, mPrep);
        const mEv = await ME.evaluateAsync(pooled, mTest.X, mTest.y, true, mPrep.targetClasses, {});
        assert(mTest.X.some(x => x.__mi && x.__mi.length === 5), 'test patients with missing values were not imputed 5 times');
        // Leak-free: a test patient's outcome does not affect their prediction
        const flipped = test.map(r => ({ ...r, Outcome: r.Outcome === 'Yes' ? 'No' : 'Yes' }));
        const fTest = DE.prepareTestingDataset(flipped, mPrep);
        const p1 = X => X.map(x => pooled.predictSingle(x).probabilities[1]);
        const a = p1(mTest.X), b = p1(fTest.X);
        assert(a.every((v, i) => Math.abs(v - b[i]) < 1e-15), 'predictions changed when the test outcomes changed');
        // Reproducible: the same patient gets the same prediction every time
        const again = p1(DE.prepareTestingDataset(test, mPrep).X);
        assert(a.every((v, i) => v === again[i]), 'predictions for patients with missing values are not reproducible');
        // Export and reload (model and preprocessing as JSON) give identical predictions
        const back = await ME.deserializeModel(JSON.parse(JSON.stringify(await ME.serializeModel(pooled))));
        const prepBack = JSON.parse(JSON.stringify({ ...mPrep, X: undefined, y: undefined }));
        const c = p1(DE.prepareTestingDataset(test, prepBack).X).map((v, i) => v); // eslint-disable-line no-unused-vars
        const c2 = DE.prepareTestingDataset(test, prepBack).X.map(x => back.predictSingle(x).probabilities[1]);
        assert(a.every((v, i) => Math.abs(v - c2[i]) < 1e-12), 'pooled model or imputer changed after export and reload');
        // Better or equal performance than median imputation when data are missing at random
        assert(mEv.rocAuc >= sEv.rocAuc - 0.005 && mEv.brierScore <= sEv.brierScore + 0.005, `MI AUC ${mEv.rocAuc} Brier ${mEv.brierScore} vs median ${sEv.rocAuc} ${sEv.brierScore}`);
        const missShare = train.filter(r => r.X1 === '').length / train.length;
        return `${Math.round(missShare * 100)}% of X1 missing (MAR); test AUC ${mEv.rocAuc.toFixed(3)} with MI vs ${sEv.rocAuc.toFixed(3)} with median, Brier ${mEv.brierScore.toFixed(3)} vs ${sEv.brierScore.toFixed(3)}; outcome-blind, reproducible, export-safe`;
    });

    await check('Multiple imputation in the background worker gives the same result as the page, and stays fast for large cohorts', async () => {
        const sent = [];
        const scope = { location: { search: '' }, postMessage: m => sent.push(m), importScripts: () => {} };
        const ws = {};
        new Function('self', 'importScripts', `${fs.readFileSync(path.join(dir, 'compute-worker.js'), 'utf8')}\nreturn self;`)(Object.assign(ws, scope), scope.importScripts);
        ws.postMessage = scope.postMessage;
        const rng = DE.createPRNG(21);
        const g = () => Math.sqrt(-2 * Math.log(rng() || 1e-12)) * Math.cos(2 * Math.PI * rng());
        const make = n => Array.from({ length: n }, () => {
            const r = { Y: rng() < 0.4 ? 'Yes' : 'No', Cat: ['A', 'B', 'C'][Math.floor(rng() * 3)] };
            for (let j = 0; j < 15; j++) r[`X${j}`] = rng() < 0.1 ? '' : g().toFixed(3);
            return r;
        });
        const cfg = { Y: { role: 'target', type: 'nominal' }, Cat: { role: 'feature', type: 'nominal' } };
        for (let j = 0; j < 15; j++) cfg[`X${j}`] = { role: 'feature', type: 'continuous' };
        const small = make(300);
        await ws.onmessage({ data: { id: 7, op: 'miceFit', payload: { rows: JSON.parse(JSON.stringify(small)), columnConfigs: cfg, options: { m: 3, seed: 5 } } } });
        const done = sent.find(m => m.id === 7 && (m.result || m.error));
        assert(done && done.result, `worker failed: ${done && done.error}`);
        const page = DE.miceFit(small, cfg, { m: 3, seed: 5 });
        assert(JSON.stringify(done.result.datasets) === JSON.stringify(page.datasets) && JSON.stringify(done.result.imputer) === JSON.stringify(page.imputer), 'worker and page imputations differ');
        const progress = sent.filter(m => m.progress && m.progress.kind === 'mice').length;
        // Speed guard: 5,000 patients, 16 predictors, 5 imputations (about 4 s on a laptop; was 24 s)
        const t0 = Date.now();
        DE.miceFit(make(5000), cfg, { m: 5 });
        const secs = (Date.now() - t0) / 1000;
        assert(secs < 30, `multiple imputation of 5,000 patients took ${secs.toFixed(1)} s`);
        return `identical in worker and page (${progress} progress messages); 5,000 patients imputed 5 times in ${secs.toFixed(1)} s`;
    });

    await check('Safeguards: Youden threshold, fewer folds for a rare outcome, out-of-range inputs, near-constant predictors', async () => {
        const VC = loadValidationCore();
        const rng = DE.createPRNG(4);
        // Youden's J equals a brute-force search over every threshold
        const p = Array.from({ length: 200 }, () => Math.round(rng() * 50) / 50), y = p.map(v => (rng() < v ? 1 : 0));
        const yd = VC.youdenThreshold({ p, y });
        let bestJ = -1;
        [...new Set(p)].forEach(t => { const P = y.filter(v => v).length, N = y.length - P; const tp = p.filter((v, i) => v >= t && y[i]).length, fp = p.filter((v, i) => v >= t && !y[i]).length; bestJ = Math.max(bestJ, tp / P - fp / N); });
        assert(Math.abs(yd.J - bestJ) < 1e-12, `Youden J ${yd.J} vs brute force ${bestJ}`);
        // Rare outcome: 3 events cannot fill 5 folds; folds are reduced so each holds an event
        const rows = Array.from({ length: 60 }, (_, i) => ({ A: String(rng()), B: String(rng()), Y: i < 3 ? 'Yes' : 'No' }));
        const cfg = { A: { role: 'feature', type: 'continuous', imputationStrategy: 'median' }, B: { role: 'feature', type: 'continuous', imputationStrategy: 'median' }, Y: { role: 'target', type: 'nominal' } };
        const fd = await VC.buildRowFolds(rows, 5, { configs: cfg, analysis: DE.analyzeColumns(['A', 'B', 'Y'], rows), isClassification: true, targetCol: 'Y', seed: 1 });
        assert(fd.sets.length === 3 && fd.requestedFolds === 5 && fd.sets.every(f => f.yva.some(v => v === 1)), `folds ${fd.sets.length}, events per fold ${fd.sets.map(f => f.yva.filter(v => v === 1).length)}`);
        // Out-of-range inputs are flagged (Age 650 with training 60-90); blanks are not
        const train = Array.from({ length: 50 }, () => ({ Age: String(60 + Math.round(rng() * 30)), Const: '5.000000001', Y: rng() < 0.5 ? 'a' : 'b' }));
        train[0].Const = '5.000000002';
        const c2 = { Age: { role: 'feature', type: 'continuous', imputationStrategy: 'median' }, Const: { role: 'feature', type: 'continuous', imputationStrategy: 'median' }, Y: { role: 'target', type: 'nominal' } };
        const prep = DE.prepareFeaturesAndTarget(train, c2);
        const ranges = [];
        const x = DE.transformSingleRow({ Age: '650', Const: '5' }, prep, [], ranges);
        const r2 = [];
        DE.transformSingleRow({ Age: '', Const: '5.000000001' }, prep, [], r2);
        assert(ranges.length === 2 && ranges[0].column === 'Age' && r2.length === 0, `range flags ${JSON.stringify(ranges)} / ${JSON.stringify(r2)}`);
        // A near-constant predictor is not blown up by dividing by a tiny spread
        assert(x.every(v => Number.isFinite(v) && Math.abs(v) < 1e6), `encoded values ${x}`);
        return `Youden J exact; 5 folds reduced to 3 for 3 events; Age 650 flagged; near-constant predictor encoded as ${x[1]}`;
    });

    await check('Repeated patients: Option C resampling whole patients removes the optimism of resampling rows', async () => {
        const VC = loadValidationCore();
        // Pure noise, but every patient appears 3 times (e.g. three admissions with the same findings)
        const base = noiseCohort(60, 6, 31);
        const rows = [], cfg = { ...base.cfg, PID: { role: 'meta', type: 'nominal' } };
        base.rows.forEach((r, i) => { for (let k = 0; k < 3; k++) rows.push({ ...r, PID: `P${i}` }); });
        const analysis = DE.analyzeColumns(Object.keys(rows[0]), rows);
        const imp = DE.imputeDataset(rows, cfg, analysis), prep = DE.prepareFeaturesAndTarget(imp.imputedRows, cfg);
        const m = await fitLocal('rf', true, 2, { numTrees: 40, seed: 1 }, prep.X, prep.y);
        const app = await ME.evaluateAsync(m, prep.X, prep.y, true, prep.targetClasses, {});
        const est = async boots => {
            const run = await VC.bootstrapOptimism({ rows, boots, configs: cfg, analysis, isClassification: true, modelId: 'rf', hp: { numTrees: 40, seed: 1 }, evalOptions: {}, keys: ['rocAuc'], fitAndScoreMulti: localFitAndScoreMulti });
            return VC.plus632('rocAuc', app.rocAuc, run.oobMean.rocAuc, 0.5).value;
        };
        const byRow = await est(VC.bootstrapResamples(rows.length, 25, 7));
        const byPatient = await est(VC.bootstrapResamplesGrouped(rows.map(r => r.PID), 25, 7));
        assert(Math.abs(byPatient - 0.5) < 0.1 && byRow - byPatient > 0.1, `.632+ AUC by row ${byRow.toFixed(3)}, by patient ${byPatient.toFixed(3)}`);
        // Test-set confidence intervals resample patients when told the patient of each row
        const ev = { actuals: prep.y, predictions: prep.y, isClassification: true, numClasses: 2, probPositive: prep.X.map(() => 0.5) };
        const ciP = ME.bootstrapCIs(ev, 200, 1, rows.map(r => r.PID));
        assert(ciP && ciP.clustered, 'patient-level bootstrap of the test set was not used');
        return `pure noise with 3 rows per patient: .632+ AUC ${byRow.toFixed(2)} resampling rows vs ${byPatient.toFixed(2)} resampling patients`;
    });

    await check('Reproducibility: the same seed gives identical predictions for every model type', async () => {
        const rows = cohort(200, 13, 'binary', { missing: true });
        const cfg = configs('binary');
        const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.8, true, 42);
        const w = prepareWorkflow(rows, cfg, true, split);
        const out = [];
        for (const id of ['glm', 'rf', 'gbm', 'knn', 'nb', 'nn', 'ensemble']) {
            const hp = { ...(HP[id] || {}), seed: 9 };
            const run = async () => { const m = await fitLocal(id, true, 2, hp, w.prep.X, w.prep.y); const p = w.test.X.map(x => Number(m.predictSingle(x).probabilities ? m.predictSingle(x).probabilities[1] || 0 : m.predictSingle(x).pred)); if (m.dispose) m.dispose(); return p; };
            const a = await run(), b = await run();
            const d = Math.max(...a.map((v, i) => Math.abs(v - b[i])));
            assert(d === 0, `${id}: predictions differ between two runs with the same seed (max ${d})`);
            out.push(id);
        }
        return `identical: ${out.join(', ')}`;
    });

    await check('Shapley explanations: exact TreeSHAP (RF, boosted trees) and sampled Shapley (logistic) match brute-force enumeration', async () => {
        const rows = cohort(200, 31, 'binary', { missing: true });
        const cfg = configs('binary');
        const split = DE.trainTestSplit(rows, rows.map(r => r.Outcome), 0.8, true, 42);
        const w = prepareWorkflow(rows, cfg, true, split);
        const groups = [], names = w.prep.featureCols;
        names.forEach((col, g) => { const m = w.prep.featureMetadata[col]; for (let j = 0; j < m.length; j++) groups[m.startIndex + j] = g; });
        const P = names.length, bg = w.prep.X.slice(0, 25), x = w.test.X[3];
        // Brute force: Shapley value from all 2^P subsets of predictors, v(S) = mean over the background
        const brute = f => {
            const hybrid = (S, z) => x.map((v, j) => ((S >> groups[j]) & 1 ? v : z[j]));
            const v = new Array(1 << P).fill(0).map((_, S) => bg.reduce((a, z) => a + f(hybrid(S, z)), 0) / bg.length);
            const phi = new Array(P).fill(0);
            for (let S = 0; S < (1 << P); S++) for (let i = 0; i < P; i++) {
                if ((S >> i) & 1) continue;
                let k = 0; for (let t = S; t; t >>= 1) k += t & 1;
                const fact = n => (n <= 1 ? 1 : n * fact(n - 1));
                phi[i] += fact(k) * fact(P - k - 1) / fact(P) * (v[S | (1 << i)] - v[S]);
            }
            return phi;
        };
        const out = [];
        const rf = ME.createModel('rf', true, 2, { numTrees: 15, maxDepth: 5, seed: 4 });
        rf.fit(w.prep.X, w.prep.y);
        const tsRf = ME.treeShap(rf, x, bg, groups, P, 1);
        const bRf = brute(z => Number(rf.predictSingle(z).probabilities[1] || 0));
        const dRf = Math.max(...bRf.map((v, i) => Math.abs(v - tsRf.phi[i])));
        const sumRf = Math.abs(tsRf.phi.reduce((a, b) => a + b, 0) - (Number(rf.predictSingle(x).probabilities[1] || 0) - tsRf.base));
        assert(dRf < 1e-12 && sumRf < 1e-12, `RF TreeSHAP differs from brute force by ${dRf} (sum error ${sumRf})`);
        out.push(`RF max diff ${dRf.toExponential(1)}`);
        const gb = ME.createModel('gbm', true, 2, { nEstimators: 40, maxDepth: 3, seed: 4 });
        if (gb.fitAsync) await gb.fitAsync(w.prep.X, w.prep.y); else gb.fit(w.prep.X, w.prep.y);
        const tsGb = ME.treeShap(gb, x, bg, groups, P, 1);
        const bGb = brute(z => gb._raw(z)[0]);
        const dGb = Math.max(...bGb.map((v, i) => Math.abs(v - tsGb.phi[i])));
        assert(dGb < 1e-10 && Math.abs(tsGb.value - gb._raw(x)[0]) < 1e-10, `boosted-tree TreeSHAP differs from brute force by ${dGb}`);
        out.push(`boosted trees max diff ${dGb.toExponential(1)}`);
        const glm = ME.createModel('glm', true, 2, { seed: 4 });
        if (glm.fitAsync) await glm.fitAsync(w.prep.X, w.prep.y); else glm.fit(w.prep.X, w.prep.y);
        const f = z => Number(glm.predictSingle(z).probabilities[1] || 0);
        const ss = ME.sampledShapley(rows2 => rows2.map(f), x, bg, groups, P, 4000, 7);
        const bG = brute(f);
        const dG = Math.max(...bG.map((v, i) => Math.abs(v - ss.phi[i])));
        const sumG = Math.abs(ss.phi.reduce((a, b) => a + b, 0) - (f(x) - ss.base));
        assert(dG < 0.01 && sumG < 1e-9, `sampled Shapley differs from brute force by ${dG}`);
        out.push(`logistic (sampled) max diff ${dG.toFixed(4)}`);
        // A multiply imputed patient with blanks: the contributions add up to the pooled prediction shown
        const mi = DE.miceFit(split.trainX, cfg, { m: 3, seed: 2 });
        const miPrep = DE.prepareFeaturesAndTarget([].concat(...mi.datasets), cfg);
        miPrep.imputation = { configs: cfg, mice: mi.imputer };
        const pooled = ME.createMIPooled(mi.datasets.map(d => { const t = DE.prepareTestingDataset(d, miPrep); const md = ME.createModel('rf', true, 2, { numTrees: 15, seed: 3 }); md.fit(t.X, t.y); return md; }));
        const blank = { ...split.testX[0], ICP: '', Site: '' };
        const xm = DE.transformSingleRow(blank, miPrep);
        const shown = Number(pooled.predictSingle(xm).probabilities[1] || 0);
        const sm = ME.sampledShapley(r2 => pooled.predictOutputs(r2).map(o => Number(o.probabilities[1] || 0)), xm, miPrep.X.slice(0, 25), groups.length ? DE.prepareFeaturesAndTarget(mi.datasets[0], cfg).featureCols.flatMap((c, g) => { const m2 = miPrep.featureMetadata[c]; return Array(m2.length).fill(g); }) : groups, P, 100, 3);
        const sumMI = Math.abs(sm.base + sm.phi.reduce((a2, b2) => a2 + b2, 0) - shown);
        assert(xm.__mi && Math.abs(sm.value - shown) < 1e-12 && sumMI < 1e-9, `imputed patient: explanation value ${sm.value} vs prediction ${shown} (sum error ${sumMI})`);
        out.push('multiply imputed patient: contributions add up to the prediction shown');
        return out.join('; ');
    });

    await check('Option C on pure noise with tuning repeated inside each resample: corrected AUC is about 0.5', async () => {
        const VC = loadValidationCore();
        const { rows, cfg } = noiseCohort(80, 8, 17);
        const inside = await optionC(VC, rows, cfg, 'glm', { seed: 1 }, 40, 'inside');
        const once = await optionC(VC, rows, cfg, 'glm', { seed: 1 }, 40, 'once');
        const varied = new Set(inside.run.perResample.filter(Boolean).map(p => p.hp.lambdaL2)).size;
        assert(Math.abs(inside.corrected - 0.5) < 0.08, `corrected AUC ${inside.corrected.toFixed(3)} is not close to 0.5`);
        assert(varied > 1, 'the tuning was not repeated inside the resamples (the same setting every time)');
        assert(inside.apparent > 0.6, `apparent AUC ${inside.apparent.toFixed(3)} should be optimistic on noise`);
        return `apparent ${inside.apparent.toFixed(3)}; corrected ${inside.corrected.toFixed(3)} with tuning repeated vs ${once.corrected.toFixed(3)} with tuning done once; ${varied} different ridge penalties chosen`;
    });

    await check('.632+ bootstrap on pure noise: flexible models (RF, boosted trees, k-NN) are estimated at about AUC 0.5', async () => {
        const VC = loadValidationCore();
        const { rows, cfg } = noiseCohort(80, 8, 17);
        const out = [];
        for (const [id, hp] of [['rf', { numTrees: 50, seed: 1 }], ['gbm', { seed: 1 }], ['knn', { k: 5 }]]) {
            const r = await optionC(VC, rows, cfg, id, hp, 30, 'inside');
            assert(typeof r.p632 === 'number' && Math.abs(r.p632 - 0.5) < 0.08, `${id}: .632+ AUC ${r.p632} is not close to 0.5`);
            out.push(`${id}: apparent ${r.apparent.toFixed(2)}, Harrell ${r.corrected.toFixed(2)}, .632+ ${r.p632.toFixed(2)}`);
        }
        return out.join('; ');
    });

    await check('.632+ no-information rates match brute-force pairing of every prediction with every outcome', async () => {
        const VC = loadValidationCore();
        let s = 99;
        const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
        const n = 37;
        const yb = Array.from({ length: n }, () => (rnd() < 0.4 ? 1 : 0)), pb = yb.map(() => rnd());
        const evB = { actuals: yb, predictions: pb.map(p => (p >= 0.5 ? 1 : 0)), probPositive: pb, numClasses: 2 };
        const ni = VC.noInformationRates(evB, true);
        let acc = 0, brier = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { acc += yb[i] === evB.predictions[j] ? 1 : 0; brier += (pb[j] - yb[i]) ** 2; }
        assert(Math.abs(ni.accuracy - acc / n / n) < 1e-12 && Math.abs(ni.brierScore - brier / n / n) < 1e-12 && ni.rocAuc === 0.5, 'binary no-information rates');
        const yr = Array.from({ length: n }, () => 10 * rnd()), hr = yr.map(v => v + 3 * (rnd() - 0.5));
        const nr = VC.noInformationRates({ actuals: yr, predictions: hr }, false);
        let mse = 0, mae = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { mse += (yr[i] - hr[j]) ** 2; mae += Math.abs(yr[i] - hr[j]); }
        assert(Math.abs(nr.rmse - Math.sqrt(mse / n / n)) < 1e-9 && Math.abs(nr.mae - mae / n / n) < 1e-9, 'regression no-information rates');
        // .632+ limits: no overfitting gives the .632 weighting; complete overfitting gives the out-of-bag value
        const a = VC.plus632('rocAuc', 0.8, 0.8, 0.5), b = VC.plus632('rocAuc', 1.0, 0.5, 0.5);
        assert(Math.abs(a.value - 0.8) < 1e-12 && Math.abs(b.value - 0.5) < 1e-12 && Math.abs(b.weight - 1) < 1e-12, '.632+ limits');
        return 'accuracy, Brier, RMSE and MAE exact; weights 0.632 to 1';
    });

    await check('Option C model-selection correction: choosing the best of several models on noise is not rewarded', async () => {
        const VC = loadValidationCore();
        const { rows, cfg } = noiseCohort(80, 8, 23);
        const analysis = DE.analyzeColumns(Object.keys(rows[0]), rows);
        const metricDefs = { rocAuc: { label: 'ROC-AUC', higherIsBetter: true } };
        const boots = VC.bootstrapResamples(rows.length, 30, 42);
        const runs = [];
        for (const id of ['glm', 'nb', 'knn']) {
            const r = await VC.bootstrapOptimism({ rows, boots, configs: cfg, analysis, isClassification: true, modelId: id, hp: { seed: 1, k: 15 }, evalOptions: {}, keys: ['rocAuc'],
                fitAndScoreMulti: localFitAndScoreMulti, tune: { k: 3, seed: 42, metricKey: 'rocAuc', metricDefs, doTune: false, fitAndScore: localFitAndScore } });
            runs.push({ modelId: id, perResample: r.perResample });
        }
        const sel = VC.selectionOptimism(runs, ['rocAuc'], true);
        assert(sel.B >= 25 && typeof sel.optimism.rocAuc === 'number' && sel.optimism.rocAuc > 0, `selection optimism ${sel.optimism.rocAuc}`);
        return `${sel.B} resamples; optimism of choose-then-fit ${sel.optimism.rocAuc.toFixed(3)}; chosen ${Object.entries(sel.chosenShare).map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(', ')}`;
    });

    console.log = realLog;
    console.log(lines.join('\n'));
    console.log(`\n${passed}/${passed + failed} end-to-end checks passed`);
    process.exit(failed ? 1 : 0);
})().catch(e => {
    console.log = realLog;
    console.log('end_to_end failed: ' + (e && e.stack ? e.stack : e));
    process.exit(1);
});
