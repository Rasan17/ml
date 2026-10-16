// Runs the app's own engine (data-engine.js + models.js, with the real TensorFlow.js neural network on
// the CPU backend) on the datasets and splits written by benchmark_vs_python.py, mirroring the Option A
// workflow: impute (fitted on train) -> encode/scale (fitted on train) -> train -> predict on test.
// Usage: node tests/benchmark_app_runner.js <input.json> <output.json>   (called by benchmark_vs_python.py)
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const dir = path.join(__dirname, '..');
global.window = globalThis;
global.self = globalThis;
global.require = require;
const originalLog = console.log;
console.log = () => {};   // TF.js prints a "use tfjs-node" banner on load
console.warn = () => {};
vm.runInThisContext(fs.readFileSync(path.join(dir, 'tf.min.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(dir, 'data-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(dir, 'models.js'), 'utf8'));
const DE = window.DataEngine;
const ME = window.ModelEngine;

const [inputPath, outputPath] = process.argv.slice(2);
const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));

// Hyper-parameters shared with the Python side (also written into the report)
const H = input.hyperparams;

function toRows(columns, matrix) {
    return matrix.map(vals => {
        const r = {};
        columns.forEach((c, j) => { r[c] = vals[j] === null ? '' : vals[j]; });
        return r;
    });
}

// Probabilities as a full-length vector per class index (RF / k-NN only report classes that received votes)
function outputsOf(model, X, k) {
    return ME.predictOutputs(model, X).map(o => ({
        pred: o.pred,
        probs: o.probabilities ? Array.from({ length: k }, (_, c) => Number(o.probabilities[c] || 0)) : null
    }));
}

(async () => {
    await tf.setBackend('cpu');
    await tf.ready();
    const out = { tfjs: tf.version['tfjs-core'], backend: tf.getBackend(), datasets: {}, smotenc: [] };

    for (const ds of input.datasets) {
        const isClf = ds.task !== 'regression';
        const columns = [...ds.features, 'target'];
        const cfg = {};
        ds.features.forEach(f => { cfg[f] = { role: 'feature', type: 'continuous', imputationStrategy: 'median' }; });
        cfg.target = { role: 'target', type: isClf ? 'nominal' : 'continuous' };
        out.datasets[ds.name] = [];

        for (const sp of ds.splits) {
            const all = toRows(columns, ds.rows.map((r, i) => [...r, isClf ? String(ds.y[i]) : ds.y[i]]));
            const trainRows = sp.train.map(i => all[i]);
            const testRows = sp.test.map(i => all[i]);

            // Option A preprocessing: fitted on the training rows only
            const trainImp = DE.imputeDataset(trainRows, cfg, {});
            const testImp = DE.imputeDataset(testRows, cfg, {}, trainImp.plan);
            const prep = DE.prepareFeaturesAndTarget(trainImp.imputedRows, cfg);
            const test = DE.prepareTestingDataset(testImp.imputedRows, prep);
            const k = prep.targetClasses.length || 2;
            const seed = sp.seed;
            const res = {
                seed,
                classes: prep.targetClasses,
                trainX: prep.X, testX: test.X, trainY: prep.y, testY: test.y,
                models: {}
            };

            // k-NN: every metric / weighting (deterministic -> exact comparison)
            for (const metric of ['euclidean', 'manhattan', 'chebyshev']) {
                for (const weighting of ['uniform', 'distance']) {
                    const m = new KNNModel(isClf, H.knn_k, metric, weighting, seed);
                    m.fit(prep.X, prep.y);
                    res.models[`knn_${metric}_${weighting}`] = outputsOf(m, test.X, k);
                }
            }
            const rf = new RandomForest(isClf, H.rf_trees, H.rf_depth, 2, 'sqrt', seed);
            await rf.fitAsync(prep.X, prep.y);
            res.models.rf = outputsOf(rf, test.X, k);

            const glm = new GeneralizedLinearModel(isClf, k, H.glm_lr, H.glm_epochs, H.glm_lambda, 'momentum', seed);
            glm.fit(prep.X, prep.y);
            res.models.glm = outputsOf(glm, test.X, k);

            const nn = new DeepNeuralNetwork(isClf, k, {
                hiddenLayers: H.nn_hidden, epochs: H.nn_epochs, learningRate: H.nn_lr, batchSize: H.nn_batch,
                dropoutRate: H.nn_dropout, l2Reg: H.nn_l2, activation: 'relu', optimizer: 'adam', seed
            });
            await nn.fit(prep.X, prep.y);
            res.models.nn = outputsOf(nn, test.X, k);
            res.nnInfo = { usedFallback: Boolean(nn.usedFallback), training: nn.trainingInfo };
            nn.dispose();

            if (isClf) {
                const nb = new GaussianNaiveBayes(1e-9, 'empirical', seed);
                nb.fit(prep.X, prep.y);
                res.models.nb = outputsOf(nb, test.X, k);
            }

            const ens = new EnsembleModel(isClf, k, {
                voting: 'soft', baseModels: ['nn', 'rf', 'glm'], capacity: 'balanced', seed
            });
            await ens.fit(prep.X, prep.y);
            res.models.ensemble = outputsOf(ens, test.X, k);
            ens.dispose();

            const gbm = new GradientBoostingModel(isClf, k, {
                nEstimators: H.gbm_rounds, learningRate: H.gbm_lr, maxDepth: H.gbm_depth, l2: H.gbm_l2, maxBins: H.gbm_bins,
                minSamplesLeaf: H.gbm_min_leaf, subsample: 1, earlyStopping: true, validationFraction: 0.15, patience: 15, seed
            });
            await gbm.fitAsync(prep.X, prep.y);
            res.models.gbm = outputsOf(gbm, test.X, k);

            out.datasets[ds.name].push(res);
            process.stderr.write(`  app: ${ds.name} split seed ${seed} done\n`);
        }
    }

    // SMOTENC on mixed continuous + nominal data (training rows only)
    for (const job of input.smotenc) {
        const cfg = {};
        job.continuous.forEach(f => { cfg[f] = { role: 'feature', type: 'continuous' }; });
        job.nominal.forEach(f => { cfg[f] = { role: 'feature', type: 'nominal' }; });
        cfg.target = { role: 'target', type: 'nominal' };
        const cols = [...job.continuous, ...job.nominal, 'target'];
        const rows = job.trainRows.map(v => {
            const r = {};
            cols.forEach((c, j) => { r[c] = v[j]; });
            return r;
        });
        const res = DE.applySMOTENC(rows, [...job.continuous, ...job.nominal], 'target', cfg, job.seed);
        out.smotenc.push({
            seed: job.seed,
            applied: res.applied,
            synthetic: res.balancedRows.slice(rows.length).map(r => cols.map(c => r[c])),
            originalCount: rows.length,
            balancedCounts: res.balancedCounts || res.classCounts
        });
    }

    fs.writeFileSync(outputPath, JSON.stringify(out));
    console.log = originalLog;
})().catch(e => {
    process.stderr.write(`benchmark_app_runner failed: ${e && e.stack ? e.stack : e}\n`);
    process.exit(1);
});
