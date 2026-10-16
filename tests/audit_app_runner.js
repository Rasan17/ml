// Runs the app's own engine for the independent audits (tests/audit_fairlearn.py, tests/audit_deepchecks.py):
// the iNPH demo cohort is split as in Option A (seed 42), imputation, encoding and scaling are fitted on the
// training patients, three models are trained, and the test predictions, the app's own subgroup figures
// (src/subgroups.js) and the fitted models are written to a folder for the Python tools.
// Usage: node tests/audit_app_runner.js <output folder>
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const dir = path.join(__dirname, '..');
const out = process.argv[2];
if (!out) { console.error('usage: node tests/audit_app_runner.js <output folder>'); process.exit(1); }
fs.mkdirSync(out, { recursive: true });
global.window = globalThis; global.self = globalThis; global.require = require;
const log = console.log; console.log = () => {}; console.warn = () => {};
['tf.min.js', 'data-engine.js', 'models.js'].forEach(f => vm.runInThisContext(fs.readFileSync(path.join(dir, f), 'utf8')));
const DE = globalThis.DataEngine, ME = globalThis.ModelEngine;

// The app's subgroup code, loaded as-is (its page-only helper is replaced by a stub)
const subCode = fs.readFileSync(path.join(dir, 'src', 'subgroups.js'), 'utf8').replace(/^import[^\n]*$/gm, '').replace(/^export\s*\{[^}]*\};?\s*$/gm, '');
const SG = new Function('escapeHtml', `${subCode}\nreturn { groupLabels, groupMetrics };`)(s => String(s));

function configsFor(headers, rows) {
    const a = DE.analyzeColumns(headers, rows);
    const cfg = {};
    headers.forEach(h => { cfg[h] = { role: a[h].role, type: a[h].type, imputationStrategy: a[h].suggestedImputation }; });
    return { cfg, analysis: a };
}

(async () => {
    const tf = globalThis.tf; await tf.setBackend('cpu'); await tf.ready();
    const parsed = DE.parseCSV(fs.readFileSync(path.join(dir, 'medical_shunt_cohort_large.csv'), 'utf8'));
    const rows = parsed.rows;
    const { cfg, analysis } = configsFor(parsed.headers, rows);
    const target = Object.keys(cfg).find(c => cfg[c].role === 'target');
    const split = DE.trainTestSplit(rows, rows.map(r => r[target]), 0.8, true, 42);
    const trImp = DE.imputeDataset(split.trainX, cfg, analysis);
    const teImp = DE.imputeDataset(split.testX, cfg, analysis, trImp.plan);
    const prep = DE.prepareFeaturesAndTarget(trImp.imputedRows, cfg);
    prep.imputation = { configs: cfg, plan: trImp.plan };
    const test = DE.prepareTestingDataset(teImp.imputedRows, prep);
    const train = DE.prepareTestingDataset(trImp.imputedRows, prep);
    const positive = prep.targetIdxToClass[1];
    const result = { cohort: 'medical_shunt_cohort_large.csv', target, positiveClass: positive, nTrain: train.X.length, nTest: test.X.length, models: {} };
    const rawTestRows = test.validRows.map(r => split.testX[teImp.imputedRows.indexOf(r)] || r);
    for (const [id, hp] of [['glm', { seed: 1 }], ['rf', { numTrees: 100, maxDepth: 6, seed: 1 }], ['gbm', { seed: 1 }]]) {
        const m = ME.createModel(id, true, 2, hp);
        if (m.fitAsync) await m.fitAsync(prep.X, prep.y); else m.fit(prep.X, prep.y);
        const probs = test.X.map(x => Number(m.predictSingle(x).probabilities[1] || 0));
        const outputs = test.X.map(x => m.predictSingle(x));
        const groups = {}, app = {};
        for (const col of ['Sex', 'Site', 'Age']) {
            const labels = SG.groupLabels(rawTestRows, col);
            groups[col] = labels;
            app[col] = {};
            [...new Set(labels)].forEach(g => {
                const idx = labels.map((v, i) => (v === g ? i : -1)).filter(i => i >= 0);
                app[col][g] = SG.groupMetrics(idx.map(i => test.y[i]), idx.map(i => outputs[i]), true, true, 0.5);
            });
        }
        result.models[id] = { yTrue: test.y, prob: probs, threshold: 0.5, groups, appSubgroupMetrics: app };
        fs.writeFileSync(path.join(out, `model_${id}.json`), JSON.stringify({ model: await ME.serializeModel(m), prep: { ...prep, X: undefined, y: undefined } }));
        if (m.dispose) m.dispose();
    }
    fs.writeFileSync(path.join(out, 'app_results.json'), JSON.stringify(result));
    // Raw rows of the same split, for the data checks (Deepchecks)
    const toCsv = list => [parsed.headers.join(','), ...list.map(r => parsed.headers.map(h => {
        const v = r[h] === undefined || r[h] === null ? '' : String(r[h]);
        return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
    }).join(','))].join('\n');
    fs.writeFileSync(path.join(out, 'train.csv'), toCsv(split.trainX));
    fs.writeFileSync(path.join(out, 'test.csv'), toCsv(split.testX));
    // Repeated-patient demonstration: every patient twice (two visits), split by row and by patient
    const visits = [];
    rows.forEach(r => { visits.push({ ...r, Visit: '1' }); visits.push({ ...r, Visit: '2' }); });
    const byRow = DE.trainTestSplit(visits, visits.map(r => r[target]), 0.8, true, 42);
    const byPatient = DE.trainTestSplitGrouped(visits, visits.map(r => r[target]), visits.map(r => r.Patient_ID), 0.8, 42);
    fs.writeFileSync(path.join(out, 'visits_row_train.csv'), toCsv(byRow.trainX));
    fs.writeFileSync(path.join(out, 'visits_row_test.csv'), toCsv(byRow.testX));
    fs.writeFileSync(path.join(out, 'visits_patient_train.csv'), toCsv(byPatient.trainX));
    fs.writeFileSync(path.join(out, 'visits_patient_test.csv'), toCsv(byPatient.testX));
    log(`audit data written to ${out}: ${result.nTrain} training and ${result.nTest} test patients, models ${Object.keys(result.models).join(', ')}`);
})().catch(e => { console.log = log; console.log('audit runner failed: ' + (e && e.stack ? e.stack : e)); process.exit(1); });
