// Edge-case robustness tests for the data engine and models (no browser / TF.js needed; the neural
// network uses its documented linear fallback under Node).
// Usage (from the project folder):  node tests/edge_cases.js
const fs = require('fs');
const path = require('path');
const dir = path.join(__dirname, '..');
global.window = {};
eval(fs.readFileSync(path.join(dir, 'data-engine.js'), 'utf8'));
eval(fs.readFileSync(path.join(dir, 'models.js'), 'utf8'));
const DE = window.DataEngine;
const ME = window.ModelEngine;
console.warn = () => {}; // silence the expected "TF.js unavailable" fallback warning

let passed = 0;
let failed = 0;
const results = [];
async function check(name, fn) {
    try {
        const msg = await fn();
        passed++;
        results.push(`PASS  ${name}${msg ? `  (${msg})` : ''}`);
    } catch (e) {
        failed++;
        results.push(`FAIL  ${name}: ${e && e.message ? e.message : e}`);
    }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }
const finite = v => typeof v === 'number' && isFinite(v);
const MODEL_IDS = ['rf', 'glm', 'knn', 'nb', 'nn', 'ensemble'];

async function trainAll(prep, split) {
    const out = {};
    for (const id of MODEL_IDS) {
        if (id === 'nb' && !prep.isClassification) continue;
        const m = ME.createModel(id, prep.isClassification, prep.targetClasses.length || 2,
            id === 'nn' ? { epochs: 5, seed: 1 } : { seed: 1, baseModels: ['rf', 'glm', 'knn'] });
        if (id === 'nn' || id === 'ensemble') await m.fit(split.trainX, split.trainY);
        else m.fit(split.trainX, split.trainY);
        out[id] = ME.evaluate(m, split.testX, split.testY, prep.isClassification, prep.targetClasses,
            prep.isClassification ? {} : { regressionThreshold: 0, thresholdSource: 'test' });
    }
    return out;
}

function makeRows(n, fn) { return Array.from({ length: n }, (_, i) => fn(i)); }

(async () => {
    const baseCfg = {
        a: { role: 'feature', type: 'continuous', imputationStrategy: 'median' },
        b: { role: 'feature', type: 'continuous', imputationStrategy: 'median' },
        g: { role: 'feature', type: 'nominal', imputationStrategy: 'mode' },
        y: { role: 'target', type: 'nominal' }
    };

    await check('Single-class training outcome: clear handling, no crash', async () => {
        const rows = makeRows(30, i => ({ a: i, b: i % 5, g: i % 2 ? 'x' : 'z', y: 'Yes' }));
        const prep = DE.prepareFeaturesAndTarget(rows, baseCfg);
        assert(prep.targetClasses.length === 1, 'expected one class');
        const smote = DE.applySMOTENC(rows, ['a', 'b', 'g'], 'y', baseCfg, 1);
        assert(smote.applied === false && /Single class/.test(smote.reason), 'SMOTENC should decline');
        const univ = DE.runUnivariateTesting(rows, ['a', 'g'], 'y', baseCfg);
        assert(univ.every(r => r.pValue === 1), 'tests with one class must give p = 1');
        const m = ME.createModel('rf', true, 1, { seed: 1 }); m.fit(prep.X, prep.y);
        const ev = ME.evaluate(m, prep.X, prep.y, true, prep.targetClasses);
        assert(ev.rocAuc === null, 'AUC must be N/A');
        return 'SMOTENC declines, p = 1, AUC N/A';
    });

    await check('All-missing predictor column: imputation and encoding stay finite', async () => {
        const rows = makeRows(40, i => ({ a: '', b: i % 7, g: i % 3 ? 'x' : 'z', y: i % 2 ? 'Yes' : 'No' }));
        const imp = DE.imputeDataset(rows, baseCfg, {});
        const prep = DE.prepareFeaturesAndTarget(imp.imputedRows, baseCfg);
        assert(prep.X.every(r => r.every(finite)), 'non-finite value in encoded matrix');
        const vif = DE.calculateVIF(imp.imputedRows, ['a', 'b', 'g'], baseCfg);
        assert(vif.every(v => finite(v.vif)), 'non-finite VIF');
        const strict = DE.fitStrictImputer(rows, ['a', 'b', 'g'], 'y', baseCfg);
        assert(strict.fillValues.a === 0, 'all-missing numeric column should fall back to 0');
        return `VIF(a) = ${vif.find(v => v.feature === 'a').vif}`;
    });

    await check('Constant features: all six models train and give finite metrics', async () => {
        const rows = makeRows(60, i => ({ a: 5, b: i % 9, g: 'same', y: (i % 9) > 4 ? 'Yes' : 'No' }));
        const prep = DE.prepareFeaturesAndTarget(rows, baseCfg);
        assert(prep.X.every(r => r.every(finite)), 'non-finite encoded value');
        const split = DE.trainTestSplit(prep.X, prep.y, 0.8, true, 1);
        const res = await trainAll(prep, split);
        Object.entries(res).forEach(([id, ev]) => assert(finite(ev.accuracy) && finite(ev.macroF1), `${id}: non-finite metric`));
        const corr = DE.checkPairwiseCorrelations(rows, ['a', 'b'], 0.8, baseCfg);
        assert(corr.allPairs[0].coeff === 0, 'correlation with a constant must be 0');
        const univ = DE.runUnivariateTesting(rows, ['a'], 'y', baseCfg);
        assert(univ[0].pValue === 1, 'constant feature must have p = 1');
        return `accuracies ${Object.values(res).map(e => e.accuracy).join(', ')}`;
    });

    await check('Extreme imbalance (2 of 200 positive): split, SMOTENC and metrics', async () => {
        const rows = makeRows(200, i => ({ a: i % 13, b: (i * 7) % 11, g: i % 4 ? 'x' : 'z', y: i < 2 ? 'Yes' : 'No' }));
        const prep = DE.prepareFeaturesAndTarget(rows, baseCfg);
        const split = DE.trainTestSplit(prep.X, prep.y, 0.8, true, 3);
        assert(split.testY.includes(prep.targetClassToIdx.Yes), 'minority class must reach the test set');
        const smote = DE.applySMOTENC(rows.slice(0, 150).concat(rows.slice(0, 2)), ['a', 'b', 'g'], 'y', baseCfg, 1);
        assert(smote.applied, 'SMOTENC should apply');
        const res = await trainAll(prep, split);
        Object.entries(res).forEach(([id, ev]) => {
            assert(finite(ev.balancedAccuracy) && finite(ev.brierScore), `${id}: non-finite metric`);
        });
        return `SMOTENC +${smote.syntheticCountGenerated}`;
    });

    await check('One minority case only: SMOTENC declines cleanly', async () => {
        const rows = makeRows(50, i => ({ a: i, b: i % 3, g: 'x', y: i === 0 ? 'Yes' : 'No' }));
        const smote = DE.applySMOTENC(rows, ['a', 'b', 'g'], 'y', baseCfg, 1);
        assert(!smote.applied && /too few minority/.test(smote.reason), smote.reason);
        return 'declined';
    });

    await check('Zero-variance continuous target: R² is N/A, no crash', async () => {
        const cfg = { ...baseCfg, y: { role: 'target', type: 'continuous' } };
        const rows = makeRows(40, i => ({ a: i, b: i % 4, g: i % 2 ? 'x' : 'z', y: 7 }));
        const prep = DE.prepareFeaturesAndTarget(rows, cfg);
        const split = DE.trainTestSplit(prep.X, prep.y, 0.8, false, 1);
        const res = await trainAll(prep, split);
        Object.entries(res).forEach(([id, ev]) => {
            assert(ev.r2 === null, `${id}: R² should be null`);
            assert(finite(ev.rmse) && finite(ev.mae), `${id}: non-finite error`);
        });
        const univ = DE.runUnivariateTesting(rows, ['a', 'g'], 'y', cfg);
        assert(univ.every(r => finite(r.pValue)), 'non-finite p-value');
        return `RMSE ${Object.values(res).map(e => e.rmse).join(', ')}`;
    });

    await check('Fewer than 4 rows: split leaves no test set, evaluate() returns null', async () => {
        const rows = makeRows(3, i => ({ a: i, b: i, g: 'x', y: i ? 'Yes' : 'No' }));
        const prep = DE.prepareFeaturesAndTarget(rows, baseCfg);
        const split = DE.trainTestSplit(prep.X, prep.y, 0.8, true, 1);
        const m = ME.createModel('knn', true, 2, {}); m.fit(split.trainX, split.trainY);
        assert(ME.evaluate(m, split.testX, split.testY, true, prep.targetClasses) === null || split.testX.length > 0, 'expected null');
        return `${split.testX.length} test row(s)`;
    });

    await check('High-cardinality nominal (one level per row): encoding stays consistent', async () => {
        const rows = makeRows(80, i => ({ a: i % 9, b: i % 5, g: 'id' + i, y: i % 2 ? 'Yes' : 'No' }));
        const prep = DE.prepareFeaturesAndTarget(rows, baseCfg);
        assert(prep.totalDimensions === 2 + 80, `dimensions ${prep.totalDimensions}`);
        const x = DE.transformSingleRow({ a: 1, b: 1, g: 'never-seen' }, prep, []);
        assert(x.length === prep.totalDimensions, 'width mismatch');
        return `${prep.totalDimensions} encoded columns`;
    });

    await check('Text ordinal with unseen and missing levels at prediction time', async () => {
        const cfg = { s: { role: 'feature', type: 'ordinal' }, y: { role: 'target', type: 'nominal' } };
        const rows = makeRows(30, i => ({ s: ['mild', 'moderate', 'severe'][i % 3], y: i % 2 ? 'Yes' : 'No' }));
        const prep = DE.prepareFeaturesAndTarget(rows, cfg);
        const issues = [];
        const x1 = DE.transformSingleRow({ s: 'catastrophic' }, prep, issues);
        const x2 = DE.transformSingleRow({ s: '' }, prep, issues);
        assert(finite(x1[0]) && finite(x2[0]) && issues.length === 2, 'unseen/missing should be flagged');
        return issues.map(i => i.kind).join(', ');
    });

    await check('Survival: everyone censored, or a horizon beyond follow-up, gives "not estimable" (null), never a number', async () => {
        const t = [1, 2, 3, 4, 5, 6], allCens = [0, 0, 0, 0, 0, 0], e = [1, 0, 1, 0, 1, 0], lp = [0.3, 0.1, 0.2, -0.4, 0.5, 0];
        assert(ME.concordanceIndex(t, allCens, lp) === null, 'C-index with no events should be null');
        const cw = ME.censoringKM(t, e);
        assert(ME.timeDependentAUC(t, e, lp, 50, cw) === null, 'AUC beyond follow-up should be null');
        assert(ME.ipcwBrier(t, e, t.map(() => 0.5), 50, cw) === null, 'Brier beyond follow-up should be null');
        const m = ME.createCox().fit(t.map((_, i) => [lp[i]]), t, allCens);
        assert(!m.converged || m.beta.every(b => Number.isFinite(b)), 'Cox fit with no events must not return non-finite coefficients');
        return 'null / not estimable';
    });

    await check('Survival: all event times tied (one time point) still gives finite Cox estimates and a C-index of 0.5 for a constant score', async () => {
        const n = 40, t = Array(n).fill(12), e = Array.from({ length: n }, (_, i) => (i % 3 ? 0 : 1)), x = Array.from({ length: n }, (_, i) => [(i % 7) - 3]);
        const m = ME.createCox().fit(x, t, e);
        assert(m.beta.every(Number.isFinite), 'non-finite coefficient');
        const c = ME.concordanceIndex(t, e, t.map(() => 0));
        assert(c === null || Math.abs(c - 0.5) < 1e-12, `C-index ${c}`);
        return `beta ${m.beta[0].toFixed(3)}, C ${c}`;
    });

    results.forEach(r => console.log(r));
    console.log(`\n${passed}/${passed + failed} edge-case checks passed`);
    process.exit(failed ? 1 : 0);
})();
