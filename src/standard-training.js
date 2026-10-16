// standard-training.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { bootstrapRun, fitModel } from './compute.js';
import { estimateImputationSeconds, fitPooledModel, prepareMultipleImputation, readMultipleImputation } from './multiple-imputation.js';
import { AppState, MODEL_INFO, RANKING_METRICS, compareEvalResults, describeNNTraining, escapeHtml, fmtR2, parseSeed, rankingMetricKey, rankingMetricLabel, trainingMedian } from './core.js';
import { TUNABLE_MODELS, applyHyperparamsToForm, bootstrapFor, buildStandardRowFolds, cvSummaryFrom, fitRecalibrator, optionBTrainRawRows, readDecisionThreshold, readRecalibration, recalibrateAndEvaluate, renderTuningReport, runTuningOnFolds, splitRowsForWorkflow, splitSeedForCV, testPatientGroups } from './cv-tuning.js';
import { renderEvaluationDashboard } from './evaluation-dashboard.js';
import { saveSettingsToBrowser } from './settings.js';
import { DOM } from './shell.js';
import { disposeModels, evalAbortOpts, markBaselineDeployed, setupTrainingLossChart, updateLossChart } from './training-actions.js';
import { getHyperparamsForModel } from './training-setup.js';
import { FLEXIBLE_MODELS, INNER_TUNE_MAX_FOLDS, bootstrapResamples, bootstrapResamplesGrouped, noInformationRates, plus632, selectionOptimism, youdenThreshold } from './validation-core.js';

async function startModelTraining() {
    if (!AppState.imputedData) {
        alert("Please execute Data Imputation first.");
        return;
    }

    const watchdog = AppState.watchdogs ? AppState.watchdogs.step4 : null;

    try {
        DOM.trainModelBtn.disabled = true;
        DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan); display: inline-flex; align-items: center; gap: 6px;">⚙️ Initializing multi-model tournament and data partitions...</span>`;

        // Prepare features and target from training cohort
        let prep = DataEngine.prepareFeaturesAndTarget(AppState.imputedData.imputedRows, AppState.columnConfigs);
        // New patients' blank predictors are filled with the same fitted imputation as the training data
        prep.imputation = { configs: JSON.parse(JSON.stringify(AppState.columnConfigs)), plan: AppState.imputedData.plan };
        AppState.prepMeta = prep;

        // Determine active validation protocol (A: split, B: separate test file, C: bootstrap internal validation)
        const valMode = document.querySelector('input[name="validation-mode"]:checked')?.value || AppState.validationMode || 'split';
        const isOptionC = valMode === 'bootstrap';
        const isOptionA = !isOptionC && valMode !== 'separate';

        // Determine models to test
        let modelsToTest = (AppState.selectedModelIds && AppState.selectedModelIds.length > 0)
            ? AppState.selectedModelIds.slice()
            : [AppState.selectedModelId || 'nn'];

        // Continuous regression guard: Naive Bayes is classification-only
        if (!prep.isClassification) {
            modelsToTest = modelsToTest.filter(mId => mId !== 'nb');
        }
        if (modelsToTest.length === 0) modelsToTest = ['nn'];

        if (watchdog) {
            watchdog.start(`Initializing tournament for ${modelsToTest.length} model(s)...`, modelsToTest.length, (reason) => {
                console.warn("Step 4 Model Training stopped:", reason);
                DOM.trainModelBtn.disabled = false;
                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-rose);">⚠️ Model training stopped: ${escapeHtml(reason || 'cancelled')}</span>`;
            });
        }

        // Clear any loss curve from a previous run so the chart never shows another model's history
        if (AppState.charts.loss) {
            AppState.charts.loss.destroy();
            AppState.charts.loss = null;
        }

        const testedResults = [];
        const autoTune = Boolean(document.getElementById('autotune-check')?.checked);
        const tuneFolds = parseInt(document.getElementById('autotune-folds')?.value, 10) || 5;
        const tuningReport = { folds: tuneFolds, models: [], ensembleSkipped: autoTune && modelsToTest.includes('ensemble') };
        // Model selection by cross-validation inside the Training set (recommended), or by Testing-set results
        const selectByCV = modelsToTest.length > 1 && (document.querySelector('input[name="selection-mode"]:checked')?.value || 'cv') === 'cv';
        tuningReport.selectByCV = selectByCV;
        // Optional recalibration of two-outcome probabilities (fitted on out-of-fold Training-set predictions)
        const recalMethod = prep.isClassification && (prep.targetClasses || []).length === 2 ? readRecalibration('recalibration-select') : 'none';
        // Progress and Force Stop for the multiple-imputation step (it runs in the background worker)
        const miProgress = (what, nRows, m) => {
            const est = estimateImputationSeconds(nRows, prep.featureCols.length, m);
            const t0 = Date.now();
            const say = msg => { DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">${msg}</span>`; if (watchdog) watchdog.tick(1, msg); };
            say(`🧩 Multiple imputation of ${what}: ${nRows.toLocaleString('en-GB')} patients, ${m} imputations${est >= 10 ? ` (about ${est < 60 ? `${Math.max(10, Math.round(est / 5) * 5)} s` : `${Math.round(est / 60)} min`})` : ''}...`);
            return {
                isAborted: () => Boolean(watchdog && watchdog.aborted),
                onProgress: (l, mm, it, iters) => {
                    const done = ((l - 1) * iters + it) / (mm * iters);
                    const left = done > 0 ? ((Date.now() - t0) / done) * (1 - done) / 1000 : null;
                    say(`🧩 Multiple imputation of ${what}: imputation ${l}/${mm}, round ${it}/${iters}${left !== null && left >= 5 ? ` (about ${left < 60 ? `${Math.round(left / 5) * 5} s` : `${Math.round(left / 60)} min`} left)` : ''}...`);
                }
            };
        };
        const foldCache = {};
        const foldsFor = async (rawTrainRows, k, label, i) => {
            if (!foldCache[k]) {
                foldCache[k] = await buildStandardRowFolds(rawTrainRows, k, prep.isClassification, splitSeedForCV(), (f, nf) => {
                    const msg = `🎯 Preparing ${nf} cross-validation folds (imputation and scaling refitted inside each fold): fold ${f}/${nf}...`;
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">${msg}</span>`;
                    if (watchdog) watchdog.tick(i + 1, msg);
                });
            }
            return foldCache[k];
        };
        // Returns the settings to train with and, when selecting by CV, the cross-validated scores
        const tuneIfRequested = async (mId, hyperparams, rawTrainRows, numClasses, i) => {
            const doTune = autoTune && TUNABLE_MODELS.has(mId);
            if (!doTune && !selectByCV && recalMethod === 'none') return { hp: hyperparams, cv: null, oof: null };
            const mInfo = MODEL_INFO[mId] || { name: mId.toUpperCase() };
            const k = (mId === 'nn' || mId === 'ensemble') ? Math.min(tuneFolds, 3) : tuneFolds; // slow models: at most 3 folds
            const fd = await foldsFor(rawTrainRows, k, mInfo.name, i);
            const t = await runTuningOnFolds(mId, hyperparams, fd, prep.isClassification, numClasses, watchdog,
                (c, nc, f, nf, eta) => {
                    const what = doTune ? `Tuning ${mInfo.name}: combination ${c}/${nc}, fold ${f}/${nf}` : `Cross-validating ${mInfo.name}: fold ${f}/${nf}`;
                    const msg = `🎯 [${i + 1}/${modelsToTest.length}] ${what}${(eta || '').replace(/&middot;/g, '·')} (Training set only)...`;
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">${msg}</span>`;
                    if (watchdog) watchdog.tick(i + 1, msg);
                }, !doTune);
            if (doTune) {
                tuningReport.models.push(t);
                applyHyperparamsToForm(mId, t.bestHp);
            }
            return { hp: t.bestHp, cv: selectByCV ? cvSummaryFrom(t) : null, oof: t.best.oof || null };
        };

        if (isOptionC) {
            // =============================================================
            // OPTION C: BOOTSTRAP OPTIMISM-CORRECTED INTERNAL VALIDATION (Harrell)
            // =============================================================
            // Every data-driven step is repeated inside each resample (validation-core.js): imputation,
            // encoding and scaling; the automatic tuning, when switched on; and the cross-validation used to
            // choose between models, so that picking the best of several models is corrected as well.
            const targetCol = prep.targetCol;
            const allRaw = AppState.parsedData.rows.filter(r => !DataEngine.isMissing(r[targetCol]) &&
                (prep.isClassification || !isNaN(Number(r[targetCol]))));
            const B = parseInt(document.getElementById('bootstrap-reps')?.value, 10) || 100;
            const seedC = splitSeedForCV();
            // The same resamples for every model (fair comparison), reproducible from the seed
            const groupColC = document.getElementById('bootstrap-group-col')?.value || '';
            const boots = groupColC
                ? bootstrapResamplesGrouped(allRaw.map(r => String(r[groupColC] ?? '').trim()), B, seedC)
                : bootstrapResamples(allRaw.length, B, seedC);
            const evalOptsC = {
                regressionThreshold: prep.isClassification ? null : trainingMedian(prep.y),
                decisionThreshold: readDecisionThreshold('decision-threshold'),
                thresholdSource: 'median of all outcomes'
            };
            const keys = prep.isClassification ? ['accuracy', 'balancedAccuracy', 'macroF1', 'rocAuc', 'brierScore'] : ['r2', 'rmse', 'mae'];
            const metricKey = rankingMetricKey(prep.isClassification);
            const metricDefs = prep.isClassification ? RANKING_METRICS.classification : RANKING_METRICS.regression;
            const slowModel = id => id === 'nn' || id === 'ensemble';
            // The choice between models is repeated inside the resamples when it is made by cross-validation
            // and every model is fast enough to be cross-validated there
            const repeatSelection = selectByCV && modelsToTest.length > 1 && !modelsToTest.some(slowModel);
            const innerK = Math.min(tuneFolds, INNER_TUNE_MAX_FOLDS);
            const runs = [];
            for (let i = 0; i < modelsToTest.length; i++) {
                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");
                const mId = modelsToTest[i];
                const mInfo = MODEL_INFO[mId] || { name: mId.toUpperCase(), icon: '⚙️' };
                const baseHp = getHyperparamsForModel(mId, prep); // the user's own settings, before any tuning
                const tuned = await tuneIfRequested(mId, baseHp, allRaw, prep.targetClasses.length || 2, i);
                const hp = tuned.hp;
                const isAborted = () => Boolean(watchdog && watchdog.aborted);
                const heartbeat = (a, b) => { if (watchdog) watchdog.tick(i + 1, `${mInfo.name}: training on all ${prep.X.length} patients...`); };
                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-emerald);">🚀 [${i + 1}/${modelsToTest.length}] Training ${mInfo.name} on all ${prep.X.length} patients...</span>`;
                const model = await fitModel(mId, prep.isClassification, prep.targetClasses.length || 2, hp, prep.X, prep.y, { isAborted, onProgress: heartbeat });
                const evalRes = await ModelEngine.evaluateAsync(model, prep.X, prep.y, prep.isClassification, prep.targetClasses, evalOptsC, evalAbortOpts(watchdog));
                const wasTuned = autoTune && TUNABLE_MODELS.has(mId);
                const tuneInside = wasTuned && !slowModel(mId);
                const inner = (tuneInside || repeatSelection)
                    ? { k: innerK, seed: seedC, groupCol: groupColC || null, metricKey, metricDefs, doTune: tuneInside }
                    : null;
                const Bm = slowModel(mId) ? Math.min(B, 50) : B;
                const what = tuneInside ? ' (tuning repeated inside each resample)' : (inner ? ' (cross-validation repeated inside each resample)' : '');
                const say = msg => { DOM.trainStatus.innerHTML = `<span style="color: var(--accent-emerald);">${msg}</span>`; if (watchdog) watchdog.tick(i + 1, msg); };
                say(`🔁 [${i + 1}/${modelsToTest.length}] ${mInfo.name}: bootstrap resample 1/${Bm}${what}...`);
                // The whole loop runs in the background worker (on the page if workers are unavailable)
                const run = await bootstrapRun({
                    rows: allRaw, boots, maxResamples: Bm, configs: AppState.columnConfigs, analysis: AppState.columnAnalysis,
                    isClassification: prep.isClassification, modelId: mId, hp: tuneInside ? baseHp : hp, evalOptions: evalOptsC, keys,
                    tune: inner
                }, {
                    isAborted,
                    onInnerProgress: (b, n, c, nc, f, nf) => { if (watchdog) watchdog.tick(i + 1, `${mInfo.name}: resample ${b}/${n}, ${tuneInside ? `combination ${c}/${nc}, ` : ''}fold ${f}/${nf}`); },
                    onResample: (b, n, t0) => {
                        const left = ((Date.now() - t0) / b) * (n - b) / 1000;
                        say(`🔁 [${i + 1}/${modelsToTest.length}] ${mInfo.name}: bootstrap resample ${Math.min(b + 1, n)}/${n}${what} (about ${left < 60 ? Math.max(5, Math.round(left / 5) * 5) + ' s' : Math.round(left / 60) + ' min'} left)...`);
                    }
                });
                const apparent = {}, corrected = {}, harrell = {}, est632 = {};
                const apparentSlope = evalRes.calibration && typeof evalRes.calibration.slope === 'number' ? evalRes.calibration.slope : null;
                // Flexible models can memorise their training patients, which defeats Harrell's correction;
                // for them the .632+ estimate (out-of-bag patients only) is the headline figure
                const method = FLEXIBLE_MODELS.has(mId) ? '632plus' : 'harrell';
                const noInfo = noInformationRates(evalRes, prep.isClassification);
                keys.forEach(k => {
                    apparent[k] = typeof evalRes[k] === 'number' ? evalRes[k] : null;
                    harrell[k] = apparent[k] !== null && run.optimism[k] !== null ? Number((apparent[k] - run.optimism[k]).toFixed(4)) : null;
                    const e = plus632(k, apparent[k], run.oobMean[k], noInfo[k]);
                    est632[k] = e ? Number(e.value.toFixed(4)) : null;
                    corrected[k] = method === '632plus' && est632[k] !== null ? est632[k] : harrell[k];
                    if (corrected[k] !== null) evalRes[k] = corrected[k]; // headline metrics are corrected
                });
                // How often the resamples chose the same settings as the full data (stability of the tuning)
                let sameSettingsShare = null;
                if (tuneInside) {
                    const done = run.perResample.filter(Boolean);
                    const sig = h => JSON.stringify(Object.keys(h).sort().map(k => [k, h[k]]));
                    sameSettingsShare = done.length ? done.filter(p => sig(p.hp) === sig(hp)).length / done.length : null;
                }
                evalRes.optimismCorrection = {
                    B: run.B, apparent, optimism: run.optimism, corrected, harrell, est632, oobMean: run.oobMean, noInfo, method,
                    calibrationSlope: (() => { const v = method === '632plus' ? run.oobCalibrationSlope : run.calibrationSlope; return typeof v === 'number' ? Number(v.toFixed(3)) : null; })(), apparentSlope,
                    tuning: !wasTuned ? 'none' : (tuneInside ? 'repeated' : 'once'), innerFolds: inner ? innerK : null, sameSettingsShare
                };
                evalRes.protocol = 'option-c';
                if (readMultipleImputation()) evalRes.multipleImputationSkipped = 'Multiple imputation is not combined with bootstrap internal validation (it would multiply the work in every resample); single imputation (the Step 3 choices) was refitted inside each resample. Use Option A or B for multiple imputation.';
                evalRes.accuracyCI = null; // a CI of the apparent accuracy would be misleading
                evalRes.ci = null;
                evalRes.trainCount = prep.X.length;
                evalRes.testCount = prep.X.length;
                evalRes.totalCount = prep.X.length;
                evalRes.splitSeed = seedC;
                if (groupColC) evalRes.bootstrapGroupCol = groupColC;
                evalRes.seed = hp.seed;
                if (tuned.cv) evalRes.cv = tuned.cv;
                if (prep.isClassification && (prep.targetClasses || []).length === 2 && evalRes.protocol !== 'option-c') evalRes.youden = youdenThreshold(tuned.oof);
                if (recalMethod !== 'none') evalRes.recalibrationSkipped = 'Recalibration is not combined with bootstrap internal validation (it would need to be repeated inside every resample); use Option A or B to recalibrate.';
                if (model.trainingInfo) evalRes.nnNote = describeNNTraining(model.trainingInfo);
                if (model.usedFallback) evalRes.fallbackNote = `TensorFlow.js training failed (${model.fallbackReason}); a linear (logistic/ridge) fallback model was evaluated instead of a neural network.`;
                testedResults.push({ modelId: mId, info: mInfo, evalRes, model, retrained: true, prep, hyperparams: hp, seed: hp.seed });
                runs.push({ modelId: mId, perResample: run.perResample });
            }
            // Choosing the best of several models (the winner's curse): optimism of the whole choose-then-fit process
            const selection = modelsToTest.length < 2 ? null
                : repeatSelection ? { repeated: true, nModels: modelsToTest.length, ...selectionOptimism(runs, keys, metricDefs[metricKey].higherIsBetter), noInfoByModel: Object.fromEntries(testedResults.map(r => [r.modelId, r.evalRes.optimismCorrection.noInfo])) }
                    : { repeated: false, nModels: modelsToTest.length, reason: selectByCV ? 'slow' : 'not-cv' };
            testedResults.forEach(r => { r.evalRes.optimismCorrection.selection = selection; });
            AppState.prepMeta = prep;
            AppState.lastTestRows = null; // no separate test set
            AppState.orRawRows = allRaw;  // raw rows matching the deployed encoding (odds ratios / MICE)
        } else if (isOptionA) {
            // =============================================================
            // OPTION A: INTERNAL TRAIN / TEST SPLIT WITH RETRAINING CHOICE
            // =============================================================
            const trainPctInput = document.getElementById('train-pct-input');
            const trainPct = trainPctInput ? parseFloat(trainPctInput.value) : 80;
            const trainRatio = Math.min(0.95, Math.max(0.50, (trainPct || 80) / 100));
            const testPct = Math.round((1 - trainRatio) * 100);

            // Option A Reproducibility Seed for train/test split (Default 42 vs Custom Seed)
            const splitModeRadio = document.querySelector('input[name="split-seed-mode"]:checked');
            const isSplitDefault = splitModeRadio ? splitModeRadio.value === 'default' : true;
            let splitSeed = 42;
            if (!isSplitDefault) {
                const splitSeedInput = document.getElementById('split-seed-input');
                const parsed = splitSeedInput ? parseSeed(splitSeedInput.value) : null;
                splitSeed = parsed !== null ? parsed : 42;
                if (parsed === null && splitSeedInput) splitSeedInput.value = 42; // show the seed actually used
            }

            // LEAK-FREE OPTION A: split the raw patient records FIRST, then fit imputation, encoding and
            // scaling on the training partition only and apply them unchanged to the test partition.
            const targetCol = prep.targetCol;
            const rawRows = AppState.parsedData.rows.filter(r => !DataEngine.isMissing(r[targetCol]) &&
                (prep.isClassification || !isNaN(Number(r[targetCol]))));
            const rowSplit = splitRowsForWorkflow(
                rawRows,
                rawRows.map(r => String(r[targetCol]).trim()),
                trainRatio,
                prep.isClassification,
                splitSeed,
                'standard'
            );
            const trainImp = DataEngine.imputeDataset(rowSplit.trainX, AppState.columnConfigs, AppState.columnAnalysis);
            const testImp = DataEngine.imputeDataset(rowSplit.testX, AppState.columnConfigs, AppState.columnAnalysis, trainImp.plan);
            let splitPrep = DataEngine.prepareFeaturesAndTarget(trainImp.imputedRows, AppState.columnConfigs);
            splitPrep.imputation = { configs: JSON.parse(JSON.stringify(AppState.columnConfigs)), plan: trainImp.plan };
            let testData = DataEngine.prepareTestingDataset(testImp.imputedRows, splitPrep);
            let split = { trainX: splitPrep.X, trainY: splitPrep.y, testX: testData.X, testY: testData.y };
            AppState.trainTestData = split;
            AppState.lastTestRows = testImp.imputedRows; // for permutation importance
            AppState.baselineTestRawRows = testImp.imputedRows.length === rowSplit.testX.length ? rowSplit.testX : null; // for the pipeline comparison

            if (split.testX.length === 0) {
                throw new Error(`The test partition is empty (${rawRows.length} usable rows at a ${Math.round(trainRatio * 100)}/${testPct} split). Use a larger cohort, a smaller training percentage, or Option B.`);
            }
            if (split.trainX.length < 5) {
                throw new Error(`Only ${split.trainX.length} training case(s) after splitting; at least 5 are required.`);
            }
            const splitThreshold = prep.isClassification ? null : trainingMedian(split.trainY);

            const shouldRetrain = DOM.retrainYes ? DOM.retrainYes.checked : true;

            // Multiple imputation (Step 4 choice): imputation models fitted on the Training rows only; the
            // Testing rows are imputed with them (several times each) and the predictions pooled
            const miM = readMultipleImputation();
            let miSplit = null, miFull = null;
            if (miM) {
                miSplit = await prepareMultipleImputation(rowSplit.trainX, AppState.columnConfigs, miM, splitSeed, miProgress('the Training set', rowSplit.trainX.length, miM));
                splitPrep = miSplit.prep;
                testData = DataEngine.prepareTestingDataset(rowSplit.testX, splitPrep);
                split = { trainX: miSplit.sets[0].X, trainY: miSplit.sets[0].y, testX: testData.X, testY: testData.y };
                AppState.trainTestData = split;
                AppState.lastTestRows = rowSplit.testX; // raw: imputed per chain when scored
                AppState.baselineTestRawRows = rowSplit.testX;
                if (shouldRetrain) {
                    miFull = await prepareMultipleImputation(rawRows, AppState.columnConfigs, miM, splitSeed, miProgress('all patients (for the final models)', rawRows.length, miM));
                    prep = miFull.prep;
                }
            }
            const miLabel = l => (miM ? ` (imputed dataset ${l}/${miM})` : '');

            for (let i = 0; i < modelsToTest.length; i++) {
                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");

                const mId = modelsToTest[i];
                const mInfo = MODEL_INFO[mId] || { name: mId.toUpperCase(), icon: '⚙️' };
                const tuned = await tuneIfRequested(mId, getHyperparamsForModel(mId, prep), rowSplit.trainX, splitPrep.targetClasses.length || 2, i);
                const hyperparams = tuned.hp;

                if (watchdog) {
                    watchdog.tick(i + 1, `Fitting ${mInfo.name} on ${Math.round(trainRatio * 100)}% split (${split.trainX.length} cases)...`);
                }

                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">🚀 [${i + 1}/${modelsToTest.length}] Training ${mInfo.name} on ${Math.round(trainRatio * 100)}% split (${split.trainX.length} cases &bull; Seed: ${hyperparams.seed})...</span>`;
                await new Promise(r => setTimeout(r, 40));

                if (modelsToTest.length === 1 && mId === 'nn') {
                    setupTrainingLossChart();
                }

                let valModel = await (miSplit ? fitPooledModel : fitModel)(mId, prep.isClassification, splitPrep.targetClasses.length || 2, hyperparams, miSplit ? miSplit.sets : split.trainX, ...(miSplit ? [] : [split.trainY]), {
                    onImputation: (l) => { DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">🚀 [${i + 1}/${modelsToTest.length}] Training ${mInfo.name}${miLabel(l)}...</span>`; if (watchdog) watchdog.tick(i + 1, `${mInfo.name}${miLabel(l)}`); },
                    isAborted: () => Boolean(watchdog && watchdog.aborted),
                    onProgress: (a, b, c) => {
                        if (a === 'tick') { if (watchdog) watchdog.tick(i + 1, `${mInfo.name}: training...`); return; }
                        if (mId === 'nn') {
                            const epoch = a, total = b, logs = c || {};
                            if (typeof logs.loss !== 'number') return;
                        DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Epoch ${epoch}/${total} &mdash; Train Loss: ${logs.loss.toFixed(4)} ${logs.val_loss !== undefined ? `| Internal Val Loss: ${logs.val_loss.toFixed(4)}` : ''}</span>`;
                        if (modelsToTest.length === 1) updateLossChart(epoch, logs.loss, logs.val_loss);
                        if (watchdog && (epoch % 4 === 0 || epoch === total)) {
                            watchdog.tick(i + 1, `${mInfo.name}: Epoch ${epoch}/${total} (Loss: ${logs.loss.toFixed(4)})`);
                        }
                        } else if (mId === 'ensemble') {
                            const curr = a, total = b, info = c || {};
                        DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Sub-Model ${curr}/${total} (${info.stage || ''})...</span>`;
                        if (watchdog) {
                            watchdog.tick(i + 1, `${mInfo.name}: Sub-Model ${curr}/${total} (${info.stage || ''})`);
                        }
                        }
                    }
                });

                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");

                // Benchmark out-of-sample accuracy on held-out test split
                const evalOptsA = {
                    regressionThreshold: splitThreshold,
                    decisionThreshold: readDecisionThreshold('decision-threshold'),
                    thresholdSource: 'median of the training-split outcomes'
                };
                const calibrator = fitRecalibrator(recalMethod, tuned.oof);
                let evalRes;
                if (calibrator) {
                    const rc = await recalibrateAndEvaluate(valModel, calibrator, split.testX, split.testY, prep.isClassification, splitPrep.targetClasses, evalOptsA, evalAbortOpts(watchdog));
                    valModel = rc.model;
                    evalRes = rc.evalRes;
                } else {
                    evalRes = await ModelEngine.evaluateAsync(valModel, split.testX, split.testY, prep.isClassification, splitPrep.targetClasses, evalOptsA, evalAbortOpts(watchdog));
                    if (recalMethod !== 'none') evalRes.recalibrationSkipped = 'Recalibration was not applied: too few out-of-fold predictions or events to fit it reliably.';
                }
                evalRes.protocol = 'option-a';
                evalRes.ci = await bootstrapFor(evalRes, hyperparams.seed, testPatientGroups(testData.validRows, 'standard'));
                evalRes.splitInfo = rowSplit.info;
                if (tuned.cv) evalRes.cv = tuned.cv;
                if (prep.isClassification && (prep.targetClasses || []).length === 2 && evalRes.protocol !== 'option-c') evalRes.youden = youdenThreshold(tuned.oof);
                evalRes.testImputedCells = testImp.audit.filter(a => a.type === 'impute_column').reduce((acc, a) => acc + a.imputedCount, 0);
                evalRes.testDroppedRows = miM ? 0 : rowSplit.testX.length - testImp.imputedRows.length;
                if (miM) evalRes.multipleImputation = { m: miM, imputedColumns: miSplit.imputedColumns };
                evalRes.unknownClassCount = testData.unknownClassCount;
                evalRes.retrainOn100 = shouldRetrain;
                evalRes.trainRatio = trainRatio;
                evalRes.trainPct = Math.round(trainRatio * 100);
                evalRes.testPct = testPct;
                evalRes.trainCount = split.trainX.length;
                evalRes.testCount = split.testX.length;
                // Rows actually available for the 100% retrain (after any 'drop rows' imputation choices)
                evalRes.totalCount = prep.X.length;
                evalRes.splitSeed = splitSeed;
                evalRes.seed = hyperparams.seed;
                if (valModel.trainingInfo) evalRes.nnNote = describeNNTraining(valModel.trainingInfo);
                if (valModel.usedFallback) {
                    evalRes.fallbackNote = `TensorFlow.js training failed (${valModel.fallbackReason}); a linear (logistic/ridge) fallback model was evaluated instead of a neural network.`;
                }
                if (valModel.weightingNote) evalRes.weightingNote = valModel.weightingNote;
                if (valModel.fallbackNote) evalRes.fallbackNote = valModel.fallbackNote;

                let activeModelInstance = valModel;

                if (shouldRetrain) {
                    if (watchdog) {
                        watchdog.tick(i + 1, `Retraining ${mInfo.name} on 100% of cohort (N=${prep.X.length})...`);
                    }
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-amber); font-weight: 600;">✓ [${i + 1}/${modelsToTest.length}] ${mInfo.name} benchmarked. Retraining on 100% of cohort (N=${prep.X.length})...</span>`;
                    await new Promise(r => setTimeout(r, 40));

                    const fullModel = await (miFull ? fitPooledModel : fitModel)(mId, prep.isClassification, prep.targetClasses.length || 2, hyperparams, miFull ? miFull.sets : prep.X, ...(miFull ? [] : [prep.y]), {
                        onImputation: (l) => { if (watchdog) watchdog.tick(i + 1, `${mInfo.name} 100% retrain${miLabel(l)}`); },
                        isAborted: () => Boolean(watchdog && watchdog.aborted),
                        onProgress: (a, b, c) => {
                            if (a === 'tick') { if (watchdog) watchdog.tick(i + 1, `${mInfo.name}: training...`); return; }
                            if (mId === 'nn') {
                                const epoch = a, total = b, logs = c || {};
                                if (typeof logs.loss !== 'number') return;
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name} 100% Retrain: Epoch ${epoch}/${total} &mdash; Loss: ${logs.loss.toFixed(4)}</span>`;
                            if (watchdog && (epoch % 4 === 0 || epoch === total)) {
                                watchdog.tick(i + 1, `${mInfo.name} 100% Retrain: Epoch ${epoch}/${total}`);
                            }
                            } else if (mId === 'ensemble') {
                                const curr = a, total = b, info = c || {};
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name} 100% Retrain: Base ${curr}/${total} (${info.stage || ''})...</span>`;
                            if (watchdog) {
                                watchdog.tick(i + 1, `${mInfo.name} 100% Retrain: Base ${curr}/${total} (${info.stage || ''})`);
                            }
                            }
                        }
                    });
                    activeModelInstance = calibrator ? ModelEngine.wrapCalibrated(fullModel, calibrator) : fullModel;
                    // The split-phase model has been evaluated and is replaced: free its GPU memory
                    disposeModels([valModel]);
                }

                testedResults.push({
                    modelId: mId,
                    info: mInfo,
                    evalRes: evalRes,
                    model: activeModelInstance,
                    retrained: shouldRetrain,
                    prep: shouldRetrain ? prep : splitPrep,
                    hyperparams,
                    seed: hyperparams.seed,
                    splitSeed: splitSeed
                });
            }
            // Predictions must use the encoding the deployed models were fitted with:
            // full-cohort encoding after 100% retraining, otherwise the training-split encoding.
            AppState.prepMeta = shouldRetrain ? prep : splitPrep;
            AppState.orRawRows = shouldRetrain ? rawRows : rowSplit.trainX;
        } else {
            // =============================================================
            // OPTION B: SEPARATE TRAINING (i) AND TESTING (ii) DATASETS
            // =============================================================
            if (!AppState.separateTestRawRows || AppState.separateTestRawRows.length === 0) {
                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-amber);">⚠️ Option B requires a separate testing cohort (Dataset ii). Please upload a test CSV or click 'Load Built-In Validation Cohort'.</span>`;
                DOM.trainModelBtn.disabled = false;
                if (watchdog) watchdog.forceAbort("Missing external test cohort.");
                alert("Option B requires a separate testing cohort (Dataset ii). Please upload a test CSV or load the built-in validation cohort in the Option B card.");
                return;
            }

            // Complete-case option: apply the same rule to the testing file (drop rows with an empty predictor cell)
            let optionBRows = AppState.separateTestRawRows;
            let completeCaseDropped = 0;
            if (AppState.completeCase) {
                const predictors = prep.featureCols;
                optionBRows = optionBRows.filter(r => predictors.every(c => !DataEngine.isMissing(r[c])));
                completeCaseDropped = AppState.separateTestRawRows.length - optionBRows.length;
            }
            // Multiple imputation (Step 4 choice): fitted on Dataset (i) only, then applied to Dataset (ii)
            const miMB = readMultipleImputation();
            let miB = null;
            if (miMB) {
                const rowsB = optionBTrainRawRows(prep);
                miB = await prepareMultipleImputation(rowsB, AppState.columnConfigs, miMB, 42, miProgress('Training Dataset (i)', rowsB.length, miMB));
                prep = miB.prep;
            }
            // Transform separate test data using training metadata (strict zero leakage)
            const testPrep = DataEngine.prepareTestingDataset(optionBRows, prep);
            AppState.lastTestRows = optionBRows; // for permutation importance
            AppState.orRawRows = optionBTrainRawRows(prep);
            testPrep.completeCaseDropped = completeCaseDropped;
            AppState.separateTestData = testPrep;

            if (!testPrep.totalEvaluated || testPrep.totalEvaluated === 0) {
                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-rose);">Testing cohort evaluation failed: 0 valid test cases found.</span>`;
                DOM.trainModelBtn.disabled = false;
                if (watchdog) watchdog.forceAbort("0 valid test cases in external cohort.");
                alert(`Could not extract any valid test cases with matching predictors and target. Missing target cases: ${testPrep.missingTargetCount}, unknown classes: ${testPrep.unknownClassCount}. Please check the testing CSV.`);
                return;
            }

            for (let i = 0; i < modelsToTest.length; i++) {
                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");

                const mId = modelsToTest[i];
                const mInfo = MODEL_INFO[mId] || { name: mId.toUpperCase(), icon: '⚙️' };
                const tuned = await tuneIfRequested(mId, getHyperparamsForModel(mId, prep), optionBTrainRawRows(prep), prep.targetClasses.length || 2, i);
                const hyperparams = tuned.hp;

                if (watchdog) {
                    watchdog.tick(i + 1, `Fitting ${mInfo.name} on 100% of Dataset (i) (${prep.X.length} cases)...`);
                }

                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-purple);">🚀 [${i + 1}/${modelsToTest.length}] Fitting ${mInfo.name} on 100% of Training Dataset (i) (${prep.X.length} cases &bull; Seed: ${hyperparams.seed})...</span>`;
                await new Promise(r => setTimeout(r, 40));

                if (modelsToTest.length === 1 && mId === 'nn') {
                    setupTrainingLossChart();
                }

                let model = await (miB ? fitPooledModel : fitModel)(mId, prep.isClassification, prep.targetClasses.length || 2, hyperparams, miB ? miB.sets : prep.X, ...(miB ? [] : [prep.y]), {
                    onImputation: (l) => { DOM.trainStatus.innerHTML = `<span style="color: var(--accent-purple);">🚀 [${i + 1}/${modelsToTest.length}] Fitting ${mInfo.name} (imputed dataset ${l}/${miMB})...</span>`; if (watchdog) watchdog.tick(i + 1, `${mInfo.name} (imputed dataset ${l}/${miMB})`); },
                    isAborted: () => Boolean(watchdog && watchdog.aborted),
                    onProgress: (a, b, c) => {
                        if (a === 'tick') { if (watchdog) watchdog.tick(i + 1, `${mInfo.name}: training...`); return; }
                        if (mId === 'nn') {
                            const epoch = a, total = b, logs = c || {};
                            if (typeof logs.loss !== 'number') return;
                        DOM.trainStatus.innerHTML = `<span style="color: var(--accent-purple);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Epoch ${epoch}/${total} &mdash; Train Loss: ${logs.loss.toFixed(4)} ${logs.val_loss !== undefined ? `| Internal Val Loss: ${logs.val_loss.toFixed(4)}` : ''}</span>`;
                        if (modelsToTest.length === 1) updateLossChart(epoch, logs.loss, logs.val_loss);
                        if (watchdog && (epoch % 4 === 0 || epoch === total)) {
                            watchdog.tick(i + 1, `${mInfo.name}: Epoch ${epoch}/${total}`);
                        }
                        } else if (mId === 'ensemble') {
                            const curr = a, total = b, info = c || {};
                        DOM.trainStatus.innerHTML = `<span style="color: var(--accent-purple);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Base ${curr}/${total} (${info.stage || ''})...</span>`;
                        if (watchdog) {
                            watchdog.tick(i + 1, `${mInfo.name}: Base ${curr}/${total} (${info.stage || ''})`);
                        }
                        }
                    }
                });

                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");

                // Evaluate on independent Dataset (ii)
                const evalOptsB = {
                    regressionThreshold: prep.isClassification ? null : trainingMedian(prep.y),
                    decisionThreshold: readDecisionThreshold('decision-threshold'),
                    thresholdSource: 'median of the Dataset (i) training outcomes'
                };
                const calibratorB = fitRecalibrator(recalMethod, tuned.oof);
                let evalRes;
                if (calibratorB) {
                    const rc = await recalibrateAndEvaluate(model, calibratorB, testPrep.X, testPrep.y, prep.isClassification, prep.targetClasses, evalOptsB, evalAbortOpts(watchdog));
                    model = rc.model;
                    evalRes = rc.evalRes;
                } else {
                    evalRes = await ModelEngine.evaluateAsync(model, testPrep.X, testPrep.y, prep.isClassification, prep.targetClasses, evalOptsB, evalAbortOpts(watchdog));
                    if (recalMethod !== 'none') evalRes.recalibrationSkipped = 'Recalibration was not applied: too few out-of-fold predictions or events to fit it reliably.';
                }
                evalRes.protocol = 'option-b';
                if (miB) evalRes.multipleImputation = { m: miMB, imputedColumns: miB.imputedColumns };
                evalRes.ci = await bootstrapFor(evalRes, hyperparams.seed);
                if (tuned.cv) evalRes.cv = tuned.cv;
                if (prep.isClassification && (prep.targetClasses || []).length === 2 && evalRes.protocol !== 'option-c') evalRes.youden = youdenThreshold(tuned.oof);
                evalRes.retrainOn100 = false;
                evalRes.trainCount = prep.X.length;
                evalRes.testCount = testPrep.totalEvaluated;
                evalRes.totalCount = prep.X.length + testPrep.totalEvaluated;
                evalRes.testProvided = testPrep.totalProvided;
                evalRes.missingTargetCount = testPrep.missingTargetCount;
                evalRes.unknownClassCount = testPrep.unknownClassCount;
                evalRes.imputedFeatureCells = testPrep.imputedFeatureCells;
                evalRes.completeCaseDropped = testPrep.completeCaseDropped;
                evalRes.seed = hyperparams.seed;
                if (model.trainingInfo) evalRes.nnNote = describeNNTraining(model.trainingInfo);
                if (model.usedFallback) {
                    evalRes.fallbackNote = `TensorFlow.js training failed (${model.fallbackReason}); a linear (logistic/ridge) fallback model was evaluated instead of a neural network.`;
                }
                if (model.weightingNote) evalRes.weightingNote = model.weightingNote;
                if (model.fallbackNote) evalRes.fallbackNote = model.fallbackNote;

                testedResults.push({
                    modelId: mId,
                    info: mInfo,
                    evalRes: evalRes,
                    model: model,
                    retrained: false,
                    prep,
                    hyperparams,
                    seed: hyperparams.seed
                });
            }
        }

        // Sort tested models by out-of-sample performance (macro-F1 then AUC for classification; R² for regression)
        testedResults.sort((a, b) => compareEvalResults(a.evalRes, b.evalRes, prep.isClassification));

        const previousResults = AppState.allTestedResults;
        AppState.allTestedResults = testedResults;
        // Designate top-performing model as champion
        const champion = testedResults[0];
        AppState.trainedModel = champion.model;
        AppState.deployedModelId = champion.modelId;
        AppState.evalResults = champion.evalRes;
        AppState.selectedModelId = champion.modelId;
        // Remember the encoding the Step 4 models use, then mark the standard workflow as the source
        AppState.baselinePrepMeta = AppState.prepMeta;
        markBaselineDeployed();
        // The previous run's models are no longer reachable (the new leader is now deployed)
        disposeModels((previousResults || []).map(r => r.model));
        AppState.tuningReport = autoTune ? tuningReport : null;
        renderTuningReport(AppState.tuningReport);
        testedResults.forEach(r => {
            const t = tuningReport.models.find(m => m.modelId === r.modelId);
            if (t) r.evalRes.tuning = { folds: t.folds, metric: t.metricLabel, chosen: t.best.params, cvMean: t.best.mean, cvSD: t.best.sd, ownSettingsKept: t.best.isOwn };
        });
        saveSettingsToBrowser(); // remember these choices for this dataset

        if (watchdog) {
            watchdog.finish(`Tournament Complete: Evaluated ${testedResults.length} Models. Top-ranked on ${rankingMetricLabel(prep.isClassification)}: ${champion.info.name}`);
        }

        // Update train status banner
        if (testedResults.length === 1) {
            const singleRes = champion.evalRes;
            const accText = singleRes.isClassification
                ? `${isOptionC ? 'Optimism-corrected accuracy' : 'Test Accuracy'}: ${(singleRes.accuracy * 100).toFixed(1)}% (ROC-AUC: ${singleRes.rocAuc !== null && singleRes.rocAuc !== undefined ? singleRes.rocAuc.toFixed(3) : 'N/A'})`
                : `${isOptionC ? 'Optimism-corrected R²' : 'Test R²'}: ${fmtR2(singleRes.r2)} (RMSE: ${singleRes.rmse.toFixed(3)})`;
            DOM.trainStatus.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 3px;">
                    <span style="color: var(--accent-emerald); font-weight: 700;">
                        ✓ Successfully Evaluated &amp; Deployed ${champion.info.name}!
                    </span>
                    <span style="font-size: 11.5px; color: var(--text-secondary);">
                        Reported Benchmark: <strong>${accText}</strong> &bull; Reproducibility Seed: <code style="color: var(--accent-purple); font-weight: 700; background: rgba(168,85,247,0.12); padding: 1px 6px; border-radius: 4px;">🌱 ${champion.evalRes.seed !== null ? champion.evalRes.seed : 'Auto'}</code> &bull; Active Predictor: <strong>${champion.info.name} (${champion.retrained ? '100% Cohort Retrained' : (isOptionA ? `${singleRes.trainPct}% Split Model` : 'Dataset i Fitted')})</strong>
                    </span>
                </div>
            `;
        } else {
            const topAccText = champion.evalRes.isClassification
                ? `Macro F1: ${(champion.evalRes.macroF1 * 100).toFixed(1)}%, Accuracy: ${(champion.evalRes.accuracy * 100).toFixed(1)}%`
                : `R²: ${fmtR2(champion.evalRes.r2)}`;
            DOM.trainStatus.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 3px;">
                    <span style="color: var(--accent-emerald); font-weight: 700;">
                        🏆 Tournament Complete: Tested &amp; Evaluated ${testedResults.length} Models!
                    </span>
                    <span style="font-size: 11.5px; color: var(--text-secondary);">
                        Top-ranked on ${escapeHtml(rankingMetricLabel(prep.isClassification))}: <strong>${champion.info.name}</strong> [${topAccText} &bull; Seed: <code style="color: var(--accent-purple); font-weight: 700;">🌱 ${champion.evalRes.seed !== null ? champion.evalRes.seed : 'Auto'}</code>] &bull; All model seeds are listed below.
                    </span>
                    <span style="font-size: 11px; color: var(--accent-amber);">
                        ${champion.evalRes.cv
                            ? '✓ The leader was chosen by cross-validation inside the Training set, so its Testing-set score is an honest estimate. Confirm it on an independent cohort (Option B) before clinical use.'
                            : '⚠️ The leader was chosen using the same test cases it is scored on, so its test score is optimistically biased. Choose by cross-validation (option above) or confirm the chosen model on an independent cohort (Option B).'}
                    </span>
                </div>
            `;
        }

        DOM.trainModelBtn.disabled = false;
        DOM.toStep5Btn.disabled = false;
        DOM.toStep6Btn.disabled = false;

        renderEvaluationDashboard();
        AppState.baselineTrainedAt = new Date();
        return true;
    } catch (e) {
        if (watchdog && !watchdog.aborted) {
            watchdog.forceAbort(`Training failed: ${e.message}`);
        }
        DOM.trainStatus.innerHTML = `<span style="color: var(--accent-rose);">Training halted: ${e.message}</span>`;
        DOM.trainModelBtn.disabled = false;
        if (e.message !== "Training aborted by user." && e.message !== "Operation halted by user.") {
            alert(`Training error: ${e.message}`);
        }
        return false;
    }
}

export { startModelTraining };
