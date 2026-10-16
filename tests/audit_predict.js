// Predictions from a model saved by tests/audit_app_runner.js, for raw patient rows (JSON list), using the
// app's own preprocessing (imputation, encoding, scaling) and model. Prints the positive-class probabilities.
// Usage: node tests/audit_predict.js <model json> <rows json>
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const dir = path.join(__dirname, '..');
global.window = globalThis; global.self = globalThis; global.require = require;
const log = console.log; console.log = () => {}; console.warn = () => {};
['tf.min.js', 'data-engine.js', 'models.js'].forEach(f => vm.runInThisContext(fs.readFileSync(path.join(dir, f), 'utf8')));

(async () => {
    const tf = globalThis.tf; await tf.setBackend('cpu'); await tf.ready();
    const saved = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
    const rows = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
    const model = await globalThis.ModelEngine.deserializeModel(saved.model);
    const probs = rows.map(r => {
        const x = globalThis.DataEngine.transformSingleRow(r, saved.prep);
        return Number(model.predictSingle(x).probabilities[1] || 0);
    });
    log(JSON.stringify(probs));
})();
