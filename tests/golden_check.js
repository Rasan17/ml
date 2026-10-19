// Golden-value regression check: fixed analyses of the built-in demo cohorts, compared with frozen results
// (tests/golden_values.json, 4 decimal places). Any change to parsing, imputation, encoding, scaling, the models
// or the metrics that alters a result fails this check; if the change is intended, regenerate with --update and
// say why in the commit message.
// Usage (from the project folder):  node tests/golden_check.js            (check)
//                                    node tests/golden_check.js --update   (rewrite the frozen values)
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const dir = path.join(__dirname, '..');
global.window = globalThis;
global.self = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(dir, 'data-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(dir, 'models.js'), 'utf8'));
const DE = globalThis.DataEngine, ME = globalThis.ModelEngine;
const GOLDEN = path.join(__dirname, 'golden_values.json');
const r4 = v => (typeof v === 'number' && isFinite(v) ? Number(v.toFixed(4)) : v);

function load(file) { return DE.parseCSV(fs.readFileSync(path.join(dir, file), 'utf8')); }
function configsFor(p, target, meta) {
    const cfg = {};
    p.headers.forEach(h => {
        const num = p.rows.every(r => DE.isMissing(r[h]) || isFinite(Number(r[h])));
        cfg[h] = { role: h === target ? 'target' : meta.includes(h) ? 'meta' : 'feature', type: h === target ? 'nominal' : num ? 'continuous' : 'nominal', imputationStrategy: num ? 'median' : 'mode' };
    });
    return cfg;
}

(async () => {
    const out = {};
    // 1. iNPH demo (420 patients): Option A as the app runs it (split first, imputation fitted on training rows)
    {
        const p = load('medical_shunt_cohort_large.csv');
        const cfg = configsFor(p, 'Shunt_Response', ['Patient_ID']);
        const split = DE.trainTestSplit(p.rows, p.rows.map(r => r.Shunt_Response), 0.8, true, 42);
        const analysis = DE.analyzeColumns(p.headers, p.rows);
        const trI = DE.imputeDataset(split.trainX, cfg, analysis), teI = DE.imputeDataset(split.testX, cfg, analysis, trI.plan);
        const prep = DE.prepareFeaturesAndTarget(trI.imputedRows, cfg);
        const test = DE.prepareTestingDataset(teI.imputedRows, prep);
        out.inph_split = { train: prep.X.length, test: test.X.length, dims: prep.totalDimensions };
        const HP = { glm: {}, rf: { numTrees: 30, maxDepth: 6 }, gbm: {}, knn: { k: 15 }, nb: {} };
        for (const id of Object.keys(HP)) {
            const m = ME.createModel(id, true, 2, { ...HP[id], seed: 7 });
            if (m.fitAsync) await m.fitAsync(prep.X, prep.y); else m.fit(prep.X, prep.y);
            const ev = await ME.evaluateAsync(m, test.X, test.y, true, prep.targetClasses, {});
            out[`inph_${id}`] = { rocAuc: r4(ev.rocAuc), brier: r4(ev.brierScore), macroF1: r4(ev.macroF1), accuracy: r4(ev.accuracy),
                calSlope: r4(ev.calibration && ev.calibration.slope), calIntercept: r4(ev.calibration && ev.calibration.intercept) };
        }
        // Multiple imputation (MICE) on the training rows: pooled summary of the imputed values
        const mi = DE.miceFit(split.trainX, cfg, { m: 3, iterations: 5, seed: 11 });
        const rout = mi.datasets.flatMap(d => d.map(r => Number(r.Rout_mmHg_min_ml)));
        out.inph_mice = { imputedColumns: mi.imputedColumns.slice().sort(), routMean: r4(rout.reduce((a, b) => a + b, 0) / rout.length) };
        // Riley minimum sample size for the 420-patient demo's outcome and 16 parameters
        const prev = p.rows.filter(r => r.Shunt_Response === 'Responder').length / p.rows.length;
        const rs = ME.rileySampleSize({ type: 'binary', p: 16, prevalence: prev, r2: 0.15 });
        out.inph_riley = { n: rs.n, criteria: (rs.criteria || []).map(c => c.n) };
    }
    // 2. Time-to-event demo: Cox model on all patients (single imputation), C-index and hazard ratios
    {
        const p = load('medical_shunt_revision_survival.csv');
        const cfg = configsFor(p, 'Shunt_Revision', ['Patient_ID', 'Months_To_Revision_Or_Censoring']);
        const imp = DE.imputeDataset(p.rows, cfg, DE.analyzeColumns(p.headers, p.rows));
        const prep = DE.prepareFeaturesAndTarget(imp.imputedRows, cfg);
        const t = imp.imputedRows.map(r => Number(r.Months_To_Revision_Or_Censoring)), e = imp.imputedRows.map(r => (r.Shunt_Revision === 'Yes' ? 1 : 0));
        const cox = ME.createCox().fit(prep.X, t, e);
        const lp = prep.X.map(x => cox.linearPredictor(x));
        out.survival_cox = { converged: cox.converged, beta: cox.beta.map(r4), cIndex: r4(ME.concordanceIndex(t, e, lp)),
            tdAuc24: r4(ME.timeDependentAUC(t, e, lp, 24, ME.censoringKM(t, e))) };
    }
    // 3. Regression demo: ridge/linear model R²
    {
        const p = load('medical_recovery_score.csv');
        const target = p.headers[p.headers.length - 1];
        const cfg = configsFor(p, target, [p.headers[0]]); cfg[target].type = 'continuous';
        const split = DE.trainTestSplit(p.rows, p.rows.map(r => r[target]), 0.8, false, 42);
        const analysis = DE.analyzeColumns(p.headers, p.rows);
        const trI = DE.imputeDataset(split.trainX, cfg, analysis), teI = DE.imputeDataset(split.testX, cfg, analysis, trI.plan);
        const prep = DE.prepareFeaturesAndTarget(trI.imputedRows, cfg), test = DE.prepareTestingDataset(teI.imputedRows, prep);
        const m = ME.createModel('glm', false, 1, { seed: 7 }); m.fit(prep.X, prep.y);
        const ev = await ME.evaluateAsync(m, test.X, test.y, false, [], { regressionThreshold: 0, thresholdSource: 'test' });
        out.recovery_glm = { r2: r4(ev.r2), rmse: r4(ev.rmse), mae: r4(ev.mae) };
    }

    if (process.argv.includes('--update')) {
        fs.writeFileSync(GOLDEN, JSON.stringify(out, null, 2) + '\n');
        console.log(`Golden values written to ${path.relative(process.cwd(), GOLDEN)}`);
        return;
    }
    const want = JSON.parse(fs.readFileSync(GOLDEN, 'utf8'));
    const diffs = [];
    const cmp = (a, b, where) => {
        if (typeof a === 'number' && typeof b === 'number') { if (Math.abs(a - b) > 1.5e-4) diffs.push(`${where}: expected ${b}, got ${a}`); return; }
        if (Array.isArray(a) || Array.isArray(b) || (a && typeof a === 'object') || (b && typeof b === 'object')) {
            const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
            keys.forEach(k => cmp((a || {})[k], (b || {})[k], `${where}.${k}`));
            return;
        }
        if (a !== b) diffs.push(`${where}: expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);
    };
    cmp(out, want, 'golden');
    if (diffs.length) { console.log(diffs.join('\n')); console.log(`\n${diffs.length} golden value(s) changed`); process.exit(1); }
    console.log(`${Object.keys(want).length}/${Object.keys(want).length} golden analyses unchanged (4 decimal places)`);
})().catch(e => { console.error(e); process.exit(1); });
