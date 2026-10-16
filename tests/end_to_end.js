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

    console.log = realLog;
    console.log(lines.join('\n'));
    console.log(`\n${passed}/${passed + failed} end-to-end checks passed`);
    process.exit(failed ? 1 : 0);
})().catch(e => {
    console.log = realLog;
    console.log('end_to_end failed: ' + (e && e.stack ? e.stack : e));
    process.exit(1);
});
