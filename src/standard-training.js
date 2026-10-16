// standard-training.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { fitModel } from './compute.js';
import { AppState, MODEL_INFO, compareEvalResults, describeNNTraining, displayModelName, escapeHtml, fmtR2, parseSeed, rankingMetricKey, rankingMetricLabel, renderExploratoryCutoffTable, summariseHeldOutSet, trainingMedian } from './core.js';
import { TUNABLE_MODELS, applyHyperparamsToForm, bootstrapFor, buildStandardRowFolds, ciSpan, cvSummaryFrom, describeSplit, formatCvScore, optionBTrainRawRows, readDecisionThreshold, renderTuningReport, runTuningOnFolds, splitRowsForWorkflow, splitSeedForCV } from './cv-tuning.js';
import { renderCalibrationPanel, renderExplainPanel, renderOddsRatioPanel } from './panels.js';
import { renderPipelineLeaderboardOnly } from './pipeline.js';
import { updatePipelineActiveDeploymentStatus, updatePredictionProvenanceBar } from './pipeline-results.js';
import { saveSettingsToBrowser } from './settings.js';
import { DOM } from './shell.js';
import { renderSinglePredictionForm } from './single-prediction.js';
import { getHyperparamsForModel } from './training-setup.js';


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
        const prep = DataEngine.prepareFeaturesAndTarget(AppState.imputedData.imputedRows, AppState.columnConfigs);
        // New patients' blank predictors are filled with the same fitted imputation as the training data
        prep.imputation = { configs: JSON.parse(JSON.stringify(AppState.columnConfigs)), plan: AppState.imputedData.plan };
        AppState.prepMeta = prep;

        // Determine active validation protocol (Option A vs Option B)
        const isOptionA = DOM.valModeA ? DOM.valModeA.checked : (AppState.validationMode !== 'separate');

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
            if (!doTune && !selectByCV) return { hp: hyperparams, cv: null };
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
            return { hp: t.bestHp, cv: selectByCV ? cvSummaryFrom(t) : null };
        };

        if (isOptionA) {
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
            const splitPrep = DataEngine.prepareFeaturesAndTarget(trainImp.imputedRows, AppState.columnConfigs);
            splitPrep.imputation = { configs: JSON.parse(JSON.stringify(AppState.columnConfigs)), plan: trainImp.plan };
            const testData = DataEngine.prepareTestingDataset(testImp.imputedRows, splitPrep);
            const split = { trainX: splitPrep.X, trainY: splitPrep.y, testX: testData.X, testY: testData.y };
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

                const valModel = await fitModel(mId, prep.isClassification, splitPrep.targetClasses.length || 2, hyperparams, split.trainX, split.trainY, {
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
                const evalRes = await ModelEngine.evaluateAsync(valModel, split.testX, split.testY, prep.isClassification, splitPrep.targetClasses, {
                    regressionThreshold: splitThreshold,
                    decisionThreshold: readDecisionThreshold('decision-threshold'),
                    thresholdSource: 'median of the training-split outcomes'
                }, evalAbortOpts(watchdog));
                evalRes.protocol = 'option-a';
                evalRes.ci = await bootstrapFor(evalRes, hyperparams.seed);
                evalRes.splitInfo = rowSplit.info;
                if (tuned.cv) evalRes.cv = tuned.cv;
                evalRes.testImputedCells = testImp.audit.filter(a => a.type === 'impute_column').reduce((acc, a) => acc + a.imputedCount, 0);
                evalRes.testDroppedRows = rowSplit.testX.length - testImp.imputedRows.length;
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

                    const fullModel = await fitModel(mId, prep.isClassification, prep.targetClasses.length || 2, hyperparams, prep.X, prep.y, {
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
                    activeModelInstance = fullModel;
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
            // Transform separate test data using training metadata (strict zero leakage)
            const testPrep = DataEngine.prepareTestingDataset(optionBRows, prep);
            AppState.lastTestRows = optionBRows; // for permutation importance
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

                const model = await fitModel(mId, prep.isClassification, prep.targetClasses.length || 2, hyperparams, prep.X, prep.y, {
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
                const evalRes = await ModelEngine.evaluateAsync(model, testPrep.X, testPrep.y, prep.isClassification, prep.targetClasses, {
                    regressionThreshold: prep.isClassification ? null : trainingMedian(prep.y),
                    decisionThreshold: readDecisionThreshold('decision-threshold'),
                    thresholdSource: 'median of the Dataset (i) training outcomes'
                }, evalAbortOpts(watchdog));
                evalRes.protocol = 'option-b';
                evalRes.ci = await bootstrapFor(evalRes, hyperparams.seed);
                if (tuned.cv) evalRes.cv = tuned.cv;
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
                ? `Test Accuracy: ${(singleRes.accuracy * 100).toFixed(1)}% (ROC-AUC: ${singleRes.rocAuc !== null && singleRes.rocAuc !== undefined ? singleRes.rocAuc.toFixed(3) : 'N/A'})`
                : `Test R²: ${fmtR2(singleRes.r2)} (RMSE: ${singleRes.rmse.toFixed(3)})`;
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

// Shows the ranking metric that applies to the current outcome type in both selectors
function syncRankingSelectors(isClassification) {
    const value = `${isClassification ? 'clf' : 'reg'}:${rankingMetricKey(isClassification)}`;
    const otherPrefix = isClassification ? 'reg:' : 'clf:';
    document.querySelectorAll('.ranking-metric-select').forEach(sel => {
        // Metrics for the other outcome type cannot be chosen (selecting one would have no effect
        // and the selector would snap back), so they are disabled while these results are shown
        Array.from(sel.options).forEach(opt => { opt.disabled = opt.value.startsWith(otherPrefix); });
        sel.value = value;
    });
}

// Frees GPU (WebGL) memory held by trained neural networks / ensembles that are no longer reachable.
// The currently deployed model is never disposed.
function disposeModels(models) {
    (models || []).forEach(m => {
        if (m && m !== AppState.trainedModel && typeof m.dispose === 'function') {
            try { m.dispose(); } catch (e) { /* ignore */ }
        }
    });
}

// Fits a non-neural model. Random Forest and logistic/ridge regression yield to the browser between
// trees/iterations so the UI repaints and Force Stop can interrupt them; other models fit synchronously.
async function fitWithYields(model, X, y, watchdog) {
    if (typeof model.fitAsync === 'function') {
        await model.fitAsync(X, y, {
            onYield: () => {
                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");
            }
        });
    } else {
        model.fit(X, y);
    }
}

// Lets Force Stop interrupt long (chunked) predictions during evaluation, e.g. k-NN on large cohorts
function evalAbortOpts(watchdog) {
    return {
        onYield: () => {
            if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");
        }
    };
}

// Re-sorts already-computed results after the ranking metric changes (no retraining, no redeployment)
function rerankExistingResults() {
    if (AppState.allTestedResults && AppState.allTestedResults.length > 0 && AppState.prepMeta) {
        const isClf = AppState.allTestedResults[0].evalRes.isClassification;
        AppState.allTestedResults.sort((a, b) => compareEvalResults(a.evalRes, b.evalRes, isClf));
        if (AppState.evalResults) renderEvaluationDashboard();
    }
    if (AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0) {
        const isClf = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous';
        AppState.pipeline.benchmarkResults.sort((a, b) => compareEvalResults(a.eval, b.eval, isClf));
        renderPipelineLeaderboardOnly();
    }
}

function setupTrainingLossChart() {
    const ctx = DOM.trainLossChartCanvas.getContext('2d');
    if (AppState.charts.loss) {
        AppState.charts.loss.destroy();
    }

    const isDark = AppState.theme === 'dark';
    AppState.charts.loss = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2
                },
                {
                    label: 'Internal Validation Loss (held out from training data)',
                    data: [],
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' },
                    ticks: { color: isDark ? '#94a3b8' : '#64748b' },
                    title: { display: true, text: 'Epoch', color: isDark ? '#94a3b8' : '#64748b' }
                },
                y: {
                    grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' },
                    ticks: { color: isDark ? '#94a3b8' : '#64748b' },
                    title: { display: true, text: 'Loss', color: isDark ? '#94a3b8' : '#64748b' }
                }
            },
            plugins: {
                legend: { labels: { color: isDark ? '#f8fafc' : '#0f172a' } },
                title: {
                    display: true,
                    // The optional 100% retrain is not plotted
                    text: 'Evaluation-phase fit (single neural network run); 100% retrain not plotted',
                    color: isDark ? '#94a3b8' : '#64748b',
                    font: { size: 11, weight: 'normal' }
                }
            }
        }
    });
}

function updateLossChart(epoch, trainLoss, valLoss) {
    if (!AppState.charts.loss) return;
    const chart = AppState.charts.loss;
    chart.data.labels.push(epoch);
    chart.data.datasets[0].data.push(trainLoss);
    if (valLoss !== undefined) {
        chart.data.datasets[1].data.push(valLoss);
    }
    chart.update('none');
}

function deployTestedModel(modelId) {
    if (!AppState.allTestedResults) return;
    const targetItem = AppState.allTestedResults.find(r => r.modelId === modelId);
    if (!targetItem) return;

    AppState.deployedModelId = modelId;
    AppState.trainedModel = targetItem.model;
    AppState.evalResults = targetItem.evalRes;
    AppState.selectedModelId = modelId;
    markBaselineDeployed();

    // Re-render dashboard to refresh badges and active buttons
    renderEvaluationDashboard();

    // Update single prediction form headers if open
    if (AppState.prepMeta) {
        renderSinglePredictionForm();
    }

    showSeedToast(`✓ Deployed "${displayModelName(modelId, targetItem.model)}" as the active predictor for Steps 5 & 6`);
}

// A Step 4 (standard workflow) model is now active: record that, and restore the encoding it was
// trained with (a previously deployed pipeline model may have replaced AppState.prepMeta)
function markBaselineDeployed() {
    AppState.deployedSource = 'standard';
    AppState.deployedPipelineModel = null;
    if (AppState.baselinePrepMeta) AppState.prepMeta = AppState.baselinePrepMeta;
    if (typeof updatePipelineActiveDeploymentStatus === 'function') updatePipelineActiveDeploymentStatus();
    if (typeof updatePredictionProvenanceBar === 'function') updatePredictionProvenanceBar();
    if (AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0 && DOM.pipelineLeaderboardContainer) {
        renderPipelineLeaderboardOnly();
    }
}
window.deployTestedModel = deployTestedModel;

function showSeedToast(msg) {
    const existing = document.querySelector('.seed-toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = 'seed-toast';
    toast.setAttribute('role', 'status');
    toast.innerHTML = '<span>📋</span> <span></span>';
    toast.lastElementChild.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2600);
}

function copySeedToClipboard(seed) {
    if (seed === undefined || seed === null) return;
    const seedStr = String(seed);
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(seedStr).then(() => {
            showSeedToast(`Seed "${seedStr}" copied to clipboard! (Keep as reference)`);
        }).catch(() => {
            prompt("Copy reproducibility seed:", seedStr);
        });
    } else {
        prompt("Copy reproducibility seed:", seedStr);
    }
}
window.copySeedToClipboard = copySeedToClipboard;

function renderEvaluationDashboard() {
    const evalRes = AppState.evalResults;
    const prep = AppState.prepMeta;
    const allResults = AppState.allTestedResults || [];
    DOM.resultsDashboard.style.display = 'flex';

    const isOptionB = evalRes.protocol === 'option-b';
    syncRankingSelectors(evalRes.isClassification);
    const trainPct = evalRes.trainPct || 80;
    const testPct = evalRes.testPct || 20;
    const trainCount = evalRes.trainCount || Math.round(evalRes.totalCount * 0.8);
    const testCount = evalRes.testCount || (evalRes.totalCount - trainCount);
    const totalCount = evalRes.totalCount || evalRes.totalSamples;

    // Protocol Status Banner
    if (DOM.retrainStatusBanner) {
        if (isOptionB) {
            DOM.retrainStatusBanner.innerHTML = `
                <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-purple); display: flex; align-items: center; gap: 6px;">
                            <span>🛡️</span> Option B: Independent External Validation Cohort
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                            Tested on separate external <strong>Dataset (ii) (${testCount} test cases)</strong> with zero data leakage. All models trained on 100% of <strong>Dataset (i) (${trainCount} patients)</strong>.
                        </div>
                    </div>
                    <span class="pill" style="background: rgba(168, 85, 247, 0.18); color: var(--accent-purple); border: 1px solid rgba(168, 85, 247, 0.4); font-size: 12px; font-weight: 700;">
                        External Cohort Validated
                    </span>
                </div>
            `;
        } else if (evalRes.retrainOn100) {
            DOM.retrainStatusBanner.innerHTML = `
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.28); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-emerald); display: flex; align-items: center; gap: 6px;">
                            <span>✓</span> Option A: Two-Phase Retraining Protocol &bull; 100% Retrained Models Active
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                            Accuracy evaluated on held-out <strong>${testPct}% test split (${testCount} cases)</strong> after fitting on <strong>${trainPct}% (${trainCount} cases)</strong>. Deployed models were then fully retrained on <strong>100% of cohort data (${totalCount} cases)</strong>.
                        </div>
                    </div>
                    <span class="pill pill-target" style="background: rgba(16, 185, 129, 0.18); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.4); font-size: 12px; font-weight: 700;">
                        100% Retrained Models Active
                    </span>
                </div>
            `;
        } else {
            DOM.retrainStatusBanner.innerHTML = `
                <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.28); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-cyan); display: flex; align-items: center; gap: 6px;">
                            <span>ℹ️</span> Option A: Single-Phase Protocol &bull; Split-Trained Models Active
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                            Evaluated on held-out <strong>${testPct}% test split (${testCount} cases)</strong>. Models maintained strictly on the <strong>${trainPct}% training split (${trainCount} cases)</strong> without 100% retraining.
                        </div>
                    </div>
                    <span class="pill" style="background: rgba(6, 182, 212, 0.18); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.4); font-size: 12px; font-weight: 700;">
                        ${trainPct}% Split Models Active
                    </span>
                </div>
            `;
        }
    }

    renderOddsRatioPanel('odds-ratio-container', AppState.prepMeta);

    // Explainability panel (permutation importance on the Testing set)
    renderExplainPanel('explain-panel-container',
        (allResults.length ? allResults : []).map(r => ({ name: (MODEL_INFO[r.modelId] || { name: r.modelId }).name, model: r.model, prep: r.prep, inSample: Boolean(r.retrained) })),
        AppState.lastTestRows, Boolean(evalRes && evalRes.isClassification));

    // Calibration and decision-threshold panel (binary outcomes)
    renderCalibrationPanel('calibration-panel-container',
        (allResults.length ? allResults : [{ modelId: AppState.selectedModelId, evalRes }]).map(r => ({
            name: (MODEL_INFO[r.modelId] || { name: r.modelId || 'Model' }).name, ev: r.evalRes
        })), Boolean(evalRes && evalRes.isClassification));

    // Methodological notes that affect how the metrics should be read
    if (DOM.retrainStatusBanner) {
        const notes = [];
        const resultsForNotes = allResults.length ? allResults.map(r => r.evalRes) : [evalRes];
        if (evalRes && evalRes.splitInfo && evalRes.splitInfo.method !== 'random') {
            notes.push(`Training/Testing ${escapeHtml(describeSplit(evalRes.splitInfo))}.${evalRes.splitInfo.method === 'time' ? ' Cross-validation inside the Training period uses folds that are not time-ordered.' : ''}`);
        }
        if (evalRes && evalRes.isClassification && prep && Array.isArray(prep.y) && prep.y.length && prep.X && prep.X[0]) {
            const counts = {};
            prep.y.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
            const minority = Math.min(...Object.values(counts));
            const nPred = prep.X[0].length;
            const epv = minority / Math.max(1, nPred);
            if (epv < 10) {
                notes.push(`⚠️ <strong>Small sample for the number of predictors:</strong> the smaller outcome group has ${minority} patient(s) for ${nPred} encoded predictor column(s) (${epv.toFixed(1)} per predictor; at least 10–20 is a common rule of thumb). Models may overfit; consider fewer predictors or more patients, and treat the results as exploratory.`);
            }
        }
        if (evalRes && evalRes.cv) {
            notes.push(`Best model chosen by ${evalRes.cv.folds}-fold cross-validation inside the Training set${evalRes.cv.grouped ? ' (each patient\'s records kept in one fold)' : ''}; missing-value filling, encoding and scaling were refitted inside every fold.`);
        }
        if (prep && prep.isClassification && prep.positiveClass !== null && prep.positiveClass !== undefined) {
            notes.push(`Positive (event) class for ROC-AUC, Brier score, sensitivity and TP/FN: <strong>${escapeHtml(prep.positiveClass)}</strong>. Change it in Step 2 (target card) if this is not the outcome of interest.`);
        }
        if (AppState.completeCase) {
            notes.push(isOptionB
                ? `Complete-case analysis: training rows with an empty cell were removed${evalRes.completeCaseDropped ? `; ${evalRes.completeCaseDropped} Dataset (ii) row(s) with an empty predictor cell were also removed` : ''}.`
                : 'Complete-case analysis: rows with an empty cell were removed from both the training and test splits (no imputation).');
        }
        if (isOptionB && evalRes.imputedFeatureCells > 0) {
            notes.push(`${evalRes.imputedFeatureCells} missing/unrecognised predictor cell(s) in Dataset (ii) were filled from training-cohort statistics.`);
        }
        if (isOptionB && (evalRes.missingTargetCount > 0 || evalRes.unknownClassCount > 0)) {
            notes.push(`Dataset (ii) rows excluded: ${evalRes.missingTargetCount || 0} with a missing outcome, ${evalRes.unknownClassCount || 0} with an outcome class not seen in training.`);
        }
        if (!isOptionB) {
            notes.push(`Option A: records were split first; imputation, category encoding and scaling were fitted on the training split only and applied unchanged to the test split.${(!evalRes.splitInfo || evalRes.splitInfo.method === 'random')
                ? ' The split is random at row level: if patients contribute several rows, choose "keep each patient\'s records together"; for temporal data choose "by date"; for multicentre data use an external cohort (Option B).'
                : ''}`);
            if (evalRes.testImputedCells > 0) {
                notes.push(`${evalRes.testImputedCells} missing predictor value(s) in the test split were filled with training-derived values.`);
            }
            if (evalRes.unknownClassCount > 0) {
                notes.push(`${evalRes.unknownClassCount} test row(s) had an outcome class absent from the training split and were excluded.`);
            }
        }
        if (prep && prep.isClassification) {
            notes.push(`Class "probabilities" are uncalibrated: Random Forest and hard voting give vote shares, k-NN gives neighbour shares. The Brier score (lower is better) summarises probability accuracy; a calibration plot is recommended before clinical use.`);
        }
        summariseHeldOutSet(evalRes).forEach(n => notes.push(n));
        resultsForNotes.forEach(r => {
            if (r && r.fallbackNote) notes.push(`⚠️ ${escapeHtml(r.fallbackNote)}`);
            if (r && r.weightingNote) notes.push(`Ensemble: ${escapeHtml(r.weightingNote)}`);
            if (r && r.nnNote) notes.push(`Neural network: ${escapeHtml(r.nnNote)}`);
        });
        if (notes.length > 0) {
            DOM.retrainStatusBanner.innerHTML += `
                <div style="font-size: 11.5px; color: var(--text-secondary); margin: -6px 0 14px 0; padding: 8px 12px; border-left: 3px solid var(--accent-amber); background: var(--bg-surface-subtle); border-radius: var(--radius-sm);">
                    ${Array.from(new Set(notes)).map(n => `<div>• ${n}</div>`).join('')}
                </div>
            `;
        }
    }

    const cohortDesc = isOptionB ? `External Testing Cohort (Dataset ii: ${testCount} Patients)` : `Held-out ${testPct}% Test Cohort (${testCount} Patients)`;

    // RENDER BASED ON NUMBER OF TESTED MODELS
    if (allResults.length > 1) {
        // MULTI-MODEL TOURNAMENT / COMPARISON MODE
        if (DOM.dashboardSingleHeading) DOM.dashboardSingleHeading.style.display = 'none';
        if (DOM.metricsSummaryRow) DOM.metricsSummaryRow.style.display = 'none';
        if (DOM.evaluationDetailContainer) DOM.evaluationDetailContainer.style.display = 'none';

        // 1. Benchmark Leaderboard Table
        if (DOM.multiModelLeaderboardContainer) {
            DOM.multiModelLeaderboardContainer.style.display = 'block';
            let lbRows = '';
            const anyCV = allResults.some(r => r.evalRes && r.evalRes.cv);
            allResults.forEach((resItem, idx) => {
                const mInfo = resItem.info;
                const rEval = resItem.evalRes;
                const isChampion = idx === 0;
                const isDeployed = AppState.deployedModelId === resItem.modelId;

                const rankBadge = isChampion ? '🥇 #1' : (idx === 1 ? '🥈 #2' : (idx === 2 ? '🥉 #3' : `#${idx + 1}`));
                const scoreDisplay = rEval.isClassification 
                    ? `<strong style="color: var(--accent-cyan); font-size: 14.5px;">${(rEval.accuracy * 100).toFixed(1)}%</strong>${Array.isArray(rEval.accuracyCI) && rEval.accuracyCI.length === 2 ? `<div style="font-size: 10px; color: var(--text-muted);">95% CI ${(rEval.accuracyCI[0] * 100).toFixed(0)}–${(rEval.accuracyCI[1] * 100).toFixed(0)}%</div>` : ''}`
                    : `<strong style="color: var(--accent-cyan); font-size: 14.5px;">R²: ${fmtR2(rEval.r2)}</strong>${ciSpan(rEval, 'r2')}`;

                const rocAucDisplay = rEval.isClassification 
                    ? (rEval.rocAuc !== null ? `${rEval.rocAuc.toFixed(3)}${ciSpan(rEval, 'rocAuc')}` : '<span style="color:var(--text-muted)">N/A</span>')
                    : `RMSE: ${rEval.rmse.toFixed(3)}${ciSpan(rEval, 'rmse')}`;

                const f1Display = rEval.isClassification 
                    ? `${(rEval.macroF1 * 100).toFixed(1)}%${ciSpan(rEval, 'macroF1', true)}`
                    : `MAE: ${rEval.mae.toFixed(3)}${ciSpan(rEval, 'mae')}`;

                const precDisplay = rEval.isClassification 
                    ? `${(rEval.macroPrecision * 100).toFixed(1)}%`
                    : `MSE: ${rEval.mse.toFixed(3)}`;

                const recDisplay = rEval.isClassification 
                    ? `${(rEval.macroRecall * 100).toFixed(1)}%`
                    : '-';
                const balDisplay = rEval.isClassification && typeof rEval.balancedAccuracy === 'number'
                    ? `${(rEval.balancedAccuracy * 100).toFixed(1)}%${ciSpan(rEval, 'balancedAccuracy', true)}` : '-';
                const brierDisplay = rEval.isClassification && typeof rEval.brierScore === 'number'
                    ? `${rEval.brierScore.toFixed(3)}${ciSpan(rEval, 'brierScore')}` : '-';

                const seedDisplay = (rEval.seed !== null && rEval.seed !== undefined)
                    ? `<span class="seed-pill" title="Click to copy reproducibility seed: ${rEval.seed}" onclick="window.copySeedToClipboard(${rEval.seed})">🌱 ${rEval.seed} <span class="copy-icon">📋</span></span>`
                    : `<span style="font-size: 11px; color: var(--text-muted); font-family: monospace;">N/A</span>`;

                const deployBtn = isDeployed
                    ? `<span class="pill pill-target" style="background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); font-weight: 700; border-color: rgba(16, 185, 129, 0.4);">✓ Active Predictor</span>`
                    : `<button class="btn-preset-action deploy-model-btn" onclick="window.deployTestedModel('${resItem.modelId}')" style="padding: 4px 10px; font-size: 11px; cursor: pointer;">Deploy Model</button>`;

                lbRows += `
                    <tr style="${isDeployed ? 'background: rgba(6, 182, 212, 0.05);' : ''}">
                        <td><span class="leaderboard-rank ${isChampion ? 'rank-1' : ''}">${rankBadge}</span></td>
                        ${anyCV ? `<td>${formatCvScore(rEval, prep.isClassification)}</td>` : ''}
                        <td>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 16px;">${mInfo.icon}</span>
                                <div>
                                    <div style="font-weight: 700; color: var(--text-primary);">${escapeHtml(displayModelName(resItem.modelId, resItem.model))}</div>
                                    <div style="font-size: 11px; color: var(--text-muted);">${resItem.retrained ? '100% Cohort Retrained' : (isOptionB ? 'External Validated' : 'Split-Trained')}</div>
                                </div>
                            </div>
                        </td>
                        <td><span class="pill ${mInfo.pillClass}" style="font-size: 11px;">${mInfo.tag}</span></td>
                        <td>${scoreDisplay}</td>
                        <td>${rocAucDisplay}</td>
                        <td>${f1Display}</td>
                        <td>${precDisplay}</td>
                        <td>${recDisplay}</td>
                        <td>${balDisplay}</td>
                        <td>${brierDisplay}</td>
                        <td>${seedDisplay}</td>
                        <td>${deployBtn}</td>
                    </tr>
                `;
            });

            DOM.multiModelLeaderboardContainer.innerHTML = `
                <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-color); border-radius: var(--radius-lg); padding: 18px 20px; margin-bottom: 24px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px;">
                        <div>
                            <h3 style="font-size: 16px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
                                <span>🏆</span> Benchmark Leaderboard &mdash; ${allResults.length} Tested Models Ranked by ${escapeHtml(rankingMetricLabel(prep.isClassification))}
                            </h3>
                            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
                                ${anyCV
                                    ? `Models are ranked by <strong>cross-validated ${escapeHtml(rankingMetricLabel(prep.isClassification))} inside the Training set</strong> (CV column); the Testing set (${cohortDesc}) was not used to choose. The top-ranked model (${allResults[0].info.name}) is set as the active predictor, and its Testing-set score is an honest estimate. The other models' test scores are shown for information; picking a different winner from them would reintroduce optimism. Report all metrics, not only the winner. Click any seed pill to copy it.`
                                    : `Comparative performance on ${cohortDesc}, ranked by <strong>${escapeHtml(rankingMetricLabel(prep.isClassification))}</strong> (change the primary metric above). The top-ranked model (${allResults[0].info.name}) is set as the active predictor. A single train/test split does not show that one model is generally superior, and because the leader was selected on these same test cases its score is optimistically biased. Choosing by cross-validation (option above) avoids this. Report all metrics, not only the winner. Click any seed pill to copy it.`}
                            </div>
                        </div>
                        <span class="pill pill-target" style="font-size: 12px; font-weight: 700;">
                            ${allResults.length} Models Tested
                        </span>
                    </div>
                    <div class="table-container">
                        <table class="leaderboard-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    ${anyCV ? `<th title="Score inside the Training set by cross-validation (mean ± SD across folds); used to choose the best model">CV ${escapeHtml(rankingMetricLabel(prep.isClassification))}</th>` : ''}
                                    <th>Model Algorithm</th>
                                    <th>Paradigm</th>
                                    <th>${prep.isClassification ? 'Test Accuracy' : 'Test R²'}</th>
                                    <th>${prep.isClassification ? 'ROC-AUC' : 'RMSE'}</th>
                                    <th>${prep.isClassification ? 'Macro F1' : 'MAE'}</th>
                                    <th>${prep.isClassification ? 'Precision' : 'MSE'}</th>
                                    <th>${prep.isClassification ? 'Recall' : '-'}</th>
                                    <th>${prep.isClassification ? 'Balanced Acc' : '-'}</th>
                                    <th>${prep.isClassification ? 'Brier ↓' : '-'}</th>
                                    <th>Seed (Ref)</th>
                                    <th>Status / Deploy</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${lbRows}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        }

        // 2. Individual Confusion Matrices & Accuracy Cards for ALL Chosen Models
        if (DOM.multiModelCmContainer) {
            DOM.multiModelCmContainer.style.display = 'block';
            let cmCardsHtml = '';
            allResults.forEach((resItem, idx) => {
                const mInfo = resItem.info;
                const rEval = resItem.evalRes;
                const isChampion = idx === 0;
                const isDeployed = AppState.deployedModelId === resItem.modelId;

                let cmContent = '';
                if (rEval.isClassification) {
                    const cm = rEval.confMatrix;
                    const classes = prep.targetClasses;
                    cmContent = `
                        <div class="table-container" style="margin-top: 10px;">
                            <table class="cm-table" style="font-size: 11.5px;">
                                <thead>
                                    <tr>
                                        <th style="padding: 6px 8px;">Actual \\ Pred</th>
                                        ${classes.map(c => `<th style="padding: 6px 8px;">${escapeHtml(c)}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${classes.map((actualClass, r) => `
                                        <tr>
                                            <th style="text-align: left; padding: 6px 8px;">${escapeHtml(actualClass)}</th>
                                            ${classes.map((_, c) => {
                                                const val = cm[r][c];
                                                const isHit = r === c;
                                                return `<td class="${isHit ? 'cm-cell-hit' : 'cm-cell-miss'}" style="padding: 6px 8px; font-weight: 700;">${val}</td>`;
                                            }).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    `;
                } else {
                    cmContent = `
                        ${renderExploratoryCutoffTable(rEval, true)}
                        <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: var(--text-secondary); padding: 0 4px;">
                            <span>RMSE: <strong>${rEval.rmse.toFixed(3)}</strong></span>
                            <span>MAE: <strong>${rEval.mae.toFixed(3)}</strong></span>
                        </div>
                    `;
                }

                const deployBtn = isDeployed
                    ? `<button class="btn-preset-action" disabled style="background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.4); color: var(--accent-emerald); font-weight: 700; width: 100%;">✓ Active Predictor (Steps 5 &amp; 6)</button>`
                    : `<button class="btn-preset-action deploy-model-btn" onclick="window.deployTestedModel('${resItem.modelId}')" style="width: 100%; font-weight: 600; cursor: pointer;">⭐ Set as Active Predictor</button>`;

                cmCardsHtml += `
                    <div class="multi-cm-card ${isDeployed ? 'active-deployed' : ''}">
                        <div class="multi-cm-card-header">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 18px;">${mInfo.icon}</span>
                                <div>
                                    <div class="cm-card-title">${escapeHtml(displayModelName(resItem.modelId, resItem.model))}</div>
                                    <span class="pill ${mInfo.pillClass}" style="font-size: 10px; padding: 1px 6px;">${mInfo.tag}</span>
                                </div>
                            </div>
                            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
                                ${isChampion ? `<span class="champion-badge">🏆 Top-ranked (${escapeHtml(rankingMetricLabel(prep.isClassification))})</span>` : `<span class="pill" style="font-size: 10.5px;">Rank #${idx + 1}</span>`}
                                ${isDeployed ? `<span class="pill pill-target" style="font-size: 10px; background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald);">⭐ Active</span>` : ''}
                            </div>
                        </div>

                        <div class="cm-card-metrics">
                            <div class="cm-metric-item">
                                <span class="cm-m-label">${rEval.isClassification ? 'Test Accuracy' : 'R² Score'}</span>
                                <span class="cm-m-val">${rEval.isClassification ? `${(rEval.accuracy * 100).toFixed(1)}%` : fmtR2(rEval.r2)}</span>
                            </div>
                            <div class="cm-metric-item">
                                <span class="cm-m-label">${rEval.isClassification ? 'ROC-AUC' : 'RMSE'}</span>
                                <span class="cm-m-val" style="color: var(--accent-purple);">${rEval.isClassification ? (rEval.rocAuc !== null ? rEval.rocAuc.toFixed(3) : 'N/A') : rEval.rmse.toFixed(3)}</span>
                            </div>
                            <div class="cm-metric-item">
                                <span class="cm-m-label">${rEval.isClassification ? 'Macro F1' : 'MAE'}</span>
                                <span class="cm-m-val" style="color: var(--accent-amber);">${rEval.isClassification ? `${(rEval.macroF1 * 100).toFixed(1)}%` : rEval.mae.toFixed(3)}</span>
                            </div>
                        </div>

                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; padding-top: 6px; border-top: 1px dashed var(--border-color);">
                            <div class="cm-seed-card-badge" title="Click to copy reproducibility seed: ${rEval.seed}" onclick="window.copySeedToClipboard(${rEval.seed})">
                                <span>🌱 Seed:</span> <strong>${rEval.seed !== null && rEval.seed !== undefined ? rEval.seed : 'N/A'}</strong> <span style="font-size: 10px; opacity: 0.8;">📋</span>
                            </div>
                            ${rEval.splitSeed ? `<span style="font-size: 10.5px; color: var(--text-muted); font-family: monospace;">Split: ${rEval.splitSeed}</span>` : ''}
                        </div>

                        <div style="font-size: 12px; font-weight: 700; color: var(--text-primary); margin-top: 10px;">
                            ${rEval.isClassification ? 'Confusion Matrix:' : 'Error metrics &amp; exploratory 2×2 split:'}
                        </div>
                        ${cmContent}

                        <div style="margin-top: 12px;">
                            ${deployBtn}
                        </div>
                    </div>
                `;
            });

            DOM.multiModelCmContainer.innerHTML = `
                <div style="margin-bottom: 20px;">
                    <h3 style="font-size: 16px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
                        <span>📊</span> Confusion Matrix &amp; Diagnostic Accuracy Breakdown &mdash; All Chosen Models
                    </h3>
                    <p style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
                        Individual confusion matrices and test metrics for every model chosen in this benchmark. Each card includes its exact reproducibility seed (click to copy).
                    </p>
                </div>
                <div class="multi-cm-grid">
                    ${cmCardsHtml}
                </div>
            `;
        }
    } else {
        // SINGLE MODEL TESTING MODE
        if (DOM.multiModelLeaderboardContainer) DOM.multiModelLeaderboardContainer.style.display = 'none';
        if (DOM.multiModelCmContainer) DOM.multiModelCmContainer.style.display = 'none';
        if (DOM.dashboardSingleHeading) DOM.dashboardSingleHeading.style.display = 'block';
        if (DOM.metricsSummaryRow) DOM.metricsSummaryRow.style.display = 'grid';
        if (DOM.evaluationDetailContainer) DOM.evaluationDetailContainer.style.display = 'block';

        const activeInfo = MODEL_INFO[AppState.deployedModelId || AppState.selectedModelId] || { name: 'Model', icon: '🧠' };
        if (DOM.dashboardSingleHeading) {
            DOM.dashboardSingleHeading.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span>${activeInfo.icon}</span> Single Model Diagnostic Performance: <strong>${activeInfo.name}</strong>
                </div>
            `;
        }

        const testAccLabel = isOptionB ? 'External Test Accuracy (Dataset ii)' : `Test Accuracy (${testPct}% Split)`;
        const testR2Label = isOptionB ? 'External Test R² (Dataset ii)' : `Test R² Score (${testPct}% Split)`;

        if (evalRes.isClassification) {
            DOM.metricsSummaryRow.innerHTML = `
                <div class="metric-pill-card">
                    <span class="m-title">${testAccLabel}</span>
                    <span class="m-val">${(evalRes.accuracy * 100).toFixed(1)}%</span>
                    ${Array.isArray(evalRes.accuracyCI) && evalRes.accuracyCI.length === 2 ? `<span style="font-size: 11px; color: var(--text-muted);">95% CI ${(evalRes.accuracyCI[0] * 100).toFixed(1)}–${(evalRes.accuracyCI[1] * 100).toFixed(1)}% (Wilson)</span>` : ''}
                </div>
                ${typeof evalRes.balancedAccuracy === 'number' ? `
                <div class="metric-pill-card">
                    <span class="m-title">Balanced Accuracy</span>
                    <span class="m-val">${(evalRes.balancedAccuracy * 100).toFixed(1)}%</span>
                </div>` : ''}
                ${typeof evalRes.brierScore === 'number' ? `
                <div class="metric-pill-card">
                    <span class="m-title">Brier Score (lower is better)</span>
                    <span class="m-val">${evalRes.brierScore.toFixed(3)}</span>
                </div>` : ''}
                <div class="metric-pill-card">
                    <span class="m-title">Macro F1-Score</span>
                    <span class="m-val">${(evalRes.macroF1 * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">Macro Precision</span>
                    <span class="m-val">${(evalRes.macroPrecision * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">Macro Recall</span>
                    <span class="m-val">${(evalRes.macroRecall * 100).toFixed(1)}%</span>
                </div>
                ${evalRes.rocAuc !== null ? `
                    <div class="metric-pill-card">
                        <span class="m-title">ROC AUC</span>
                        <span class="m-val">${(evalRes.rocAuc).toFixed(3)}</span>
                    </div>
                ` : ''}
                <div class="metric-pill-card" style="cursor: pointer;" onclick="window.copySeedToClipboard(${evalRes.seed})" title="Click to copy reproducibility seed: ${evalRes.seed}">
                    <span class="m-title">Reproducibility Seed</span>
                    <span class="m-val" style="color: var(--accent-purple); font-family: monospace;">🌱 ${evalRes.seed !== null && evalRes.seed !== undefined ? evalRes.seed : 'Auto'} 📋</span>
                </div>
            `;

            // Confusion Matrix Table
            const cm = evalRes.confMatrix;
            const classes = prep.targetClasses;
            let cmHtml = `
                <div style="display: flex; flex-direction: column; gap: 16px;">
                    <h3 style="font-size: 15px; color: var(--text-primary);">Confusion Matrix (${cohortDesc})</h3>
                    <div class="table-container" style="max-width: 500px;">
                        <table class="cm-table">
                            <thead>
                                <tr>
                                    <th>Actual \\ Predicted</th>
                                    ${classes.map(c => `<th>${escapeHtml(c)}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${classes.map((actualClass, r) => `
                                    <tr>
                                        <th style="text-align: left;">${escapeHtml(actualClass)}</th>
                                        ${classes.map((_, c) => {
                                            const val = cm[r][c];
                                            const isHit = r === c;
                                            return `<td class="${isHit ? 'cm-cell-hit' : 'cm-cell-miss'}">${val}</td>`;
                                        }).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
            DOM.evaluationDetailContainer.innerHTML = cmHtml;
        } else {
            // Regression Metrics
            DOM.metricsSummaryRow.innerHTML = `
                <div class="metric-pill-card">
                    <span class="m-title">${testR2Label}</span>
                    <span class="m-val">${fmtR2(evalRes.r2)}</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">RMSE</span>
                    <span class="m-val">${evalRes.rmse.toFixed(3)}</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">MAE</span>
                    <span class="m-val">${evalRes.mae.toFixed(3)}</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">MSE</span>
                    <span class="m-val">${evalRes.mse.toFixed(3)}</span>
                </div>
                <div class="metric-pill-card" style="cursor: pointer;" onclick="window.copySeedToClipboard(${evalRes.seed})" title="Click to copy reproducibility seed: ${evalRes.seed}">
                    <span class="m-title">Reproducibility Seed</span>
                    <span class="m-val" style="color: var(--accent-purple); font-family: monospace;">🌱 ${evalRes.seed !== null && evalRes.seed !== undefined ? evalRes.seed : 'Auto'} 📋</span>
                </div>
            `;

            const regCmHtml = renderExploratoryCutoffTable(evalRes, false);

            DOM.evaluationDetailContainer.innerHTML = `
                ${regCmHtml}
                <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
                    Model validated on ${cohortDesc} with R² = ${fmtR2(evalRes.r2)} and Mean Absolute Error = ${evalRes.mae}. ${isOptionB ? `Fitted strictly on 100% of Training Dataset (i) (${trainCount} patients).` : (evalRes.retrainOn100 ? `Active model was then retrained on 100% of data (${totalCount} patients).` : `Active model maintained strictly on ${trainPct}% training split (${trainCount} patients; 100% retraining declined).`)}
                </p>
                <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 6px;">
                    💡 <em>R², RMSE and MAE are the primary performance measures for a continuous outcome. If the outcome has a prespecified clinical decision threshold, analyse it as a binary outcome (Nominal) instead of relying on the exploratory split above.</em>
                </div>
            `;
        }
    }
}

export { startModelTraining, syncRankingSelectors, disposeModels, fitWithYields, evalAbortOpts, rerankExistingResults, setupTrainingLossChart, updateLossChart, deployTestedModel, markBaselineDeployed, showSeedToast, copySeedToClipboard, renderEvaluationDashboard };
