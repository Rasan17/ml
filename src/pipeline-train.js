// pipeline-train.js: part of the Universal Medical ML/NeuralNet Predictor application controller (split from pipeline.js)
import { fitModel } from './compute.js';
import { youdenThreshold } from './validation-core.js';
import { AppState, actionAttrs, compareEvalResults, describeNNTraining, displayModelName, escapeHtml, fmtR2, parseSeed, rankingMetricLabel, summariseHeldOutSet, trainingMedian } from './core.js';
import { TUNABLE_MODELS, TUNE_MAX_ROWS, bootstrapFor, buildPipelineFolds, ciScopeNote, ciSpan, cvSummaryFrom, fitRecalibrator, formatCvScore, readDecisionThreshold, readRecalibration, recalibrateAndEvaluate, renderTuningReport, runTuningOnFolds, testPatientGroups } from './cv-tuning.js';
import { renderCalibrationPanel, renderExplainPanel } from './panels.js';
import { renderPipelineStep5 } from './pipeline-balance.js';
import { renderPipelineBenchmarkResults, renderTooGoodToBeTrueCheck } from './pipeline-results.js';
import { saveSettingsToBrowser } from './settings.js';
import { DOM } from './shell.js';
import { disposeModels, evalAbortOpts, syncRankingSelectors } from './training-actions.js';
import { renderSubgroupPanel } from './subgroups.js';

// =========================================================================
// STEP 7: Algorithm-Specific Training & Constraints
// =========================================================================
async function executePipelineStep7() {
    const runDataset = AppState.datasetRun; // results are installed only if the same dataset is still loaded
    if (!AppState.pipeline.rfImportance || AppState.pipeline.rfImportance.length === 0) {
        alert("Please complete Step 6: Random Forest Scout & Feature Ranking first.");
        return;
    }

    const userSeed = DOM.pipelineEvalSeedInput ? parseSeed(DOM.pipelineEvalSeedInput.value) : 42;
    AppState.pipeline.evalSeed = userSeed === null ? 42 : userSeed;
    if (DOM.pipelineEvalSeedInput) DOM.pipelineEvalSeedInput.value = AppState.pipeline.evalSeed;

    if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.disabled = true;
    if (DOM.pipelineResultsContainer) DOM.pipelineResultsContainer.style.display = 'none';

    const watchdog = AppState.watchdogs ? AppState.watchdogs.pipeline : null;
    if (watchdog) {
        watchdog.start("Initializing constrained clinical pipeline...", 5, (reason) => {
            console.warn("Pipeline Step 7 training stopped:", reason);
            if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.disabled = false;
        });
    } else if (DOM.pipelineTrainStatus) {
        DOM.pipelineTrainStatus.style.display = 'block';
    }

    try {
        await new Promise(r => setTimeout(r, 60));

        // Recompute Step 5 so the balanced rows match the current split and feature set
        renderPipelineStep5();
        if (!AppState.pipeline.balancedTrainRows) {
            throw new Error("Training rows are unavailable. Please re-run Step 1.");
        }

        const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
        if (activeCols.length === 0) {
            throw new Error("No active predictor features remain. Restore at least one feature in Step 4.");
        }
        
        // Top 70% features with safe fallback
        let top70Cols = AppState.pipeline.top70Features;
        if (!top70Cols || top70Cols.length === 0) {
            const k = Math.max(1, Math.round(activeCols.length * 0.30));
            const topCount = Math.max(1, activeCols.length - k);
            top70Cols = activeCols.slice(0, topCount);
        }
        top70Cols = top70Cols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
        if (top70Cols.length === 0) top70Cols = [...activeCols];

        const targetCol = AppState.pipeline.targetCol;
        const isClassification = AppState.pipeline.columnConfigs[targetCol]?.type !== 'continuous';
        // Model plan: Naive Bayes is classification-only, so regression runs one model fewer
        const plannedModels = isClassification ? ['knn', 'nb', 'rf', 'gbm', 'nn', 'glm', 'ensemble'] : ['knn', 'rf', 'gbm', 'nn', 'glm', 'ensemble'];
        const totalModels = plannedModels.length;
        const stepOf = id => plannedModels.indexOf(id) + 1;
        if (watchdog) watchdog.totalSteps = totalModels;

        // 1. Prepare standard full dataset (for DNN, GLM, Ensemble)
        const fullConfigs = {};
        activeCols.forEach(c => fullConfigs[c] = AppState.pipeline.columnConfigs[c]);
        fullConfigs[targetCol] = AppState.pipeline.columnConfigs[targetCol];

        const fullPrep = DataEngine.prepareDataset(
            AppState.pipeline.balancedTrainRows,
            fullConfigs,
            activeCols,
            targetCol
        );
        // New patients' blank predictors are filled with the pipeline's training-only fill values
        fullPrep.fillValues = { ...(AppState.pipeline.imputer ? AppState.pipeline.imputer.fillValues : {}) };
        // Single source for the class count: the encoder's class list
        const numClasses = isClassification ? fullPrep.targetClasses.length : 1;

        // Transform unaltered Test set using full training metadata
        const fullTestTrans = DataEngine.prepareTestingDataset(AppState.pipeline.testRows, fullPrep);
        // Exploratory 2×2 cutoff for continuous outcomes: median of the (real, non-synthetic) training outcomes
        const pipeEvalOptions = {
            decisionThreshold: readDecisionThreshold('pipeline-decision-threshold'),
            regressionThreshold: isClassification ? null : trainingMedian(AppState.pipeline.trainRows.map(r => Number(r[targetCol]))),
            thresholdSource: 'median of the training-split outcomes'
        };

        // 2. Prepare reduced Top 70% dataset (strictly for k-NN & Naive Bayes)
        const top70Configs = {};
        top70Cols.forEach(c => top70Configs[c] = AppState.pipeline.columnConfigs[c]);
        top70Configs[targetCol] = AppState.pipeline.columnConfigs[targetCol];

        const top70Prep = DataEngine.prepareDataset(
            AppState.pipeline.balancedTrainRows,
            top70Configs,
            top70Cols,
            targetCol
        );
        top70Prep.fillValues = fullPrep.fillValues;
        const top70TestTrans = DataEngine.prepareTestingDataset(AppState.pipeline.testRows, top70Prep);

        const benchmarkList = [];

        // Protocol default settings; optional automatic tuning (cross-validation inside the real
        // Training rows, imputation and SMOTENC re-applied within each fold) may replace them
        const evalSeed = AppState.pipeline.evalSeed;
        const pHp = {
            knn: { k: 5, metric: 'euclidean', weighting: 'distance', seed: evalSeed },
            nb: { varSmoothing: 1e-4, priorMode: 'empirical', seed: evalSeed },
            rf: { numTrees: 40, maxDepth: 6, minSamplesSplit: 2, featureRatio: 'sqrt', seed: evalSeed },
            gbm: { nEstimators: 200, learningRate: 0.1, maxDepth: 3, minSamplesLeaf: 5, l2: 1, subsample: 0.8, seed: evalSeed },
            nn: { epochs: 25, learningRate: 0.005, hiddenLayers: [48, 24], dropoutRate: 0.15, l2Reg: 0.001, activation: 'relu', optimizer: 'adam', batchSize: 16, seed: evalSeed },
            glm: { lr: 0.05, epochs: 150, lambdaL2: 0.02, solver: 'momentum', seed: evalSeed }
        };
        const pipeTune = Boolean(document.getElementById('pipeline-autotune-check')?.checked);
        const pipeFolds = parseInt(document.getElementById('pipeline-autotune-folds')?.value, 10) || 5;
        const pipeSelectCV = (document.querySelector('input[name="pipeline-selection-mode"]:checked')?.value || 'cv') === 'cv';
        const pipeTuningReport = { folds: pipeFolds, models: [], pipeline: true, ensembleSkipped: pipeTune, selectByCV: pipeSelectCV };
        pHp.ensemble = { voting: 'soft', baseModels: ['nn', 'rf', 'glm'], capacity: 'balanced', seed: evalSeed };
        const cvByModel = {};
        const oofByModel = {};
        const recalPipe = isClassification && numClasses === 2 ? readRecalibration('pipeline-recalibration-select') : 'none';
        if (pipeTune || pipeSelectCV || recalPipe !== 'none') {
            const nTrainReal = AppState.pipeline.trainRows.length;
            if (pipeTune && nTrainReal < 100) {
                pipeTuningReport.smallSampleWarning = `Only ${nTrainReal} training patients: each fold scores very few patients, so differences between settings are mostly chance. Tuning is unlikely to help much with a cohort this small; the protocol defaults are a sound choice.`;
            }
            const foldCache = {};
            const foldsFor = async (cols, k, label, stepId, inFold = false) => {
                const key = `${k}|${inFold ? 'sel' : ''}|${cols.join('␟')}`;
                if (!foldCache[key]) {
                    foldCache[key] = await buildPipelineFolds(AppState.pipeline.trainRows, cols, k, evalSeed, (f, nf) => {
                        if (watchdog) watchdog.tick(stepOf(stepId), `🎯 Preparing cross-validation folds for ${label}: fold ${f}/${nf} (imputation, SMOTENC, scaling${inFold ? ' and top-70% selection' : ''} inside each fold)...`);
                    }, inFold);
                }
                return foldCache[key];
            };
            const plan = (pipeSelectCV || recalPipe !== 'none') ? plannedModels.slice() : plannedModels.filter(id => id !== 'ensemble');
            if (watchdog) watchdog.tick(1, `🎯 ${pipeTune ? 'Automatic tuning' : 'Cross-validation'} of ${plan.length} models on ${Math.min(nTrainReal, TUNE_MAX_ROWS).toLocaleString('en-GB')} training patients: this takes several times longer than training alone (the neural network is slowest). A time estimate appears once each model's first trial has run.`);
            await ModelEngine.yieldToBrowser();
            for (let t = 0; t < plan.length; t++) {
                const id = plan[t];
                const reduced = id === 'knn' || id === 'nb';
                const k = (id === 'nn' || id === 'ensemble') ? Math.min(pipeFolds, 3) : pipeFolds;
                const label = { knn: 'k-NN', nb: 'Naive Bayes', rf: 'Random Forest', gbm: 'Gradient-Boosted Trees', nn: 'Neural Network', glm: 'Logistic / Ridge Regression', ensemble: 'Ensemble' }[id];
                const doTune = pipeTune && TUNABLE_MODELS.has(id);
                // k-NN / Naive Bayes: the Step 6 top-70% selection is repeated inside every fold
                const fdModel = await foldsFor(activeCols, k, label, id, reduced);
                if (reduced && fdModel.selections) pipeTuningReport.selectionStability = fdModel.selections;
                const res = await runTuningOnFolds(id, pHp[id], fdModel, isClassification, numClasses, watchdog, (c, nc, f, nf, eta) => {
                    const what = doTune ? `Tuning ${label} (${t + 1}/${plan.length}): combination ${c}/${nc}, fold ${f}/${nf}` : `Cross-validating ${label} (${t + 1}/${plan.length}): fold ${f}/${nf}`;
                    if (watchdog) watchdog.tick(stepOf(id), `🎯 ${what}${(eta || '').replace(/&middot;/g, '·')} (Training set only)...`);
                }, !doTune);
                pHp[id] = res.bestHp;
                if (doTune) pipeTuningReport.models.push(res);
                if (pipeSelectCV) cvByModel[id] = cvSummaryFrom(res);
                if (res.best && res.best.oof) oofByModel[id] = res.best.oof;
            }
        }
        const tunedTag = id => (pipeTuningReport.models.some(m => m.modelId === id) ? ' Settings tuned by cross-validation within the Training set.' : '');

        // A. k-NN Model (Strict constraint: Bottom 30% features dropped)
        if (watchdog) watchdog.tick(stepOf('knn'), `Training Model ${stepOf('knn')}/${totalModels}: k-NN (Top 70% features only)...`);
        await new Promise(r => setTimeout(r, 60));
        const isAborted = () => Boolean(watchdog && watchdog.aborted);
        // Training runs in the background worker when available (same models and settings as before)
        const knn = await fitModel('knn', isClassification, numClasses, { ...pHp.knn, seed: evalSeed }, top70Prep.X, top70Prep.y, { isAborted });
        const knnEval = await ModelEngine.evaluateAsync(knn, top70TestTrans.X, top70TestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions, evalAbortOpts(watchdog));
        const nReinstated = Array.from(AppState.pipeline.reinstatedFeatures || []).filter(f => top70Cols.includes(f)).length;
        const reducedLabel = nReinstated ? `Top 70% + ${nReinstated} reinstated (${top70Cols.length} features)` : `Top 70% (${top70Cols.length} features)`;
        knnEval.featuresUsed = reducedLabel;
        knnEval.constraintNote = `k = ${pHp.knn.k} (${pHp.knn.weighting === 'distance' ? 'distance-weighted' : 'equal-weighted'}). Bottom 30% RF-importance features removed to reduce distance noise in high dimensions.` + (nReinstated ? ` ${nReinstated} of them reinstated by the user.` : '') + tunedTag('knn');
        benchmarkList.push({ id: 'knn', name: 'k-Nearest Neighbors (k-NN)', model: knn, prep: top70Prep, eval: knnEval });

        // B. Gaussian Naive Bayes (Strict constraint: Bottom 30% features dropped)
        if (isClassification) {
            if (watchdog) watchdog.tick(stepOf('nb'), `Training Model ${stepOf('nb')}/${totalModels}: Naive Bayes (Top 70% features only)...`);
            await new Promise(r => setTimeout(r, 60));
            const nb = await fitModel('nb', isClassification, numClasses, { ...pHp.nb, seed: evalSeed }, top70Prep.X, top70Prep.y, { isAborted });
            const nbEval = await ModelEngine.evaluateAsync(nb, top70TestTrans.X, top70TestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions, evalAbortOpts(watchdog));
            nbEval.featuresUsed = reducedLabel;
            nbEval.constraintNote = "Bottom 30% RF-importance features removed (fewer noisy inputs; does not make features conditionally independent)." + (nReinstated ? ` ${nReinstated} of them reinstated by the user.` : '') + (tunedTag('nb') ? ` Variance smoothing ${pHp.nb.varSmoothing}.` + tunedTag('nb') : '');
            benchmarkList.push({ id: 'nb', name: 'Naive Bayes (Gaussian + Bernoulli)', model: nb, prep: top70Prep, eval: nbEval });
        }

        // B2. Random Forest (all surviving features). Tree splits are scale-free and the forest
        // selects features per split, so no RF-importance pruning is applied.
        if (watchdog) watchdog.tick(stepOf('rf'), `Training Model ${stepOf('rf')}/${totalModels}: Random Forest (all surviving features)...`);
        await new Promise(r => setTimeout(r, 60));
        const rfFinal = await fitModel('rf', isClassification, numClasses, { ...pHp.rf, seed: evalSeed }, fullPrep.X, fullPrep.y, {
            isAborted, onProgress: () => { if (watchdog) watchdog.tick(stepOf('rf'), `Training Model ${stepOf('rf')}/${totalModels}: Random Forest (all surviving features)...`); }
        });
        const rfEval = await ModelEngine.evaluateAsync(rfFinal, fullTestTrans.X, fullTestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions, evalAbortOpts(watchdog));
        rfEval.featuresUsed = `All Surviving (${activeCols.length} features)`;
        rfEval.constraintNote = `${pHp.rf.numTrees} trees, max depth ${pHp.rf.maxDepth}, min ${pHp.rf.minSamplesSplit} samples to split, √p features per split; probabilities are tree vote shares.` + tunedTag('rf');
        benchmarkList.push({ id: 'rf', name: 'Random Forest', model: rfFinal, prep: fullPrep, eval: rfEval });

        // B3. Gradient-boosted trees (all surviving features): shallow trees, shrinkage, row subsampling and
        // early stopping on an internal validation split of the training rows (the test set is never used)
        if (watchdog) watchdog.tick(stepOf('gbm'), `Training Model ${stepOf('gbm')}/${totalModels}: Gradient-Boosted Trees (all surviving features)...`);
        await new Promise(r => setTimeout(r, 60));
        const gbm = await fitModel('gbm', isClassification, numClasses, { ...pHp.gbm, seed: evalSeed }, fullPrep.X, fullPrep.y, {
            isAborted, onProgress: () => { if (watchdog) watchdog.tick(stepOf('gbm'), `Training Model ${stepOf('gbm')}/${totalModels}: Gradient-Boosted Trees...`); }
        });
        const gbmEval = await ModelEngine.evaluateAsync(gbm, fullTestTrans.X, fullTestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions, evalAbortOpts(watchdog));
        gbmEval.featuresUsed = `All Surviving (${activeCols.length} features)`;
        gbmEval.constraintNote = `Up to ${pHp.gbm.nEstimators} rounds of depth-${pHp.gbm.maxDepth} trees, learning rate ${pHp.gbm.learningRate}, at least ${pHp.gbm.minSamplesLeaf} patients per leaf, ${Math.round(pHp.gbm.subsample * 100)}% row subsampling and early stopping on an internal validation split.` + (gbm.trainingInfo && gbm.trainingInfo.bestRounds ? ` ${gbm.trainingInfo.bestRounds} rounds kept.` : '') + tunedTag('gbm');
        benchmarkList.push({ id: 'gbm', name: 'Gradient-Boosted Trees', model: gbm, prep: fullPrep, eval: gbmEval });

        // C. Neural network (multilayer perceptron; strict constraint: L2 Weight Decay + Dropout)
        if (watchdog) watchdog.tick(stepOf('nn'), `Training Model ${stepOf('nn')}/${totalModels}: Neural Network (MLP; L2 + dropout + early stopping)...`);
        await new Promise(r => setTimeout(r, 60));
        const dnnParams = { ...pHp.nn };
        // The test split is never shown to the model: the NN carves its own validation split from the training rows
        const dnn = await fitModel('nn', isClassification, numClasses, dnnParams, fullPrep.X, fullPrep.y, {
            isAborted,
            onProgress: (ep, tot) => {
                if (ep === 'tick') return;
                if (watchdog && (ep % 4 === 0 || ep === tot)) watchdog.tick(stepOf('nn'), `Training Model ${stepOf('nn')}/${totalModels}: Neural Network (Epoch ${ep}/${tot})...`);
            }
        });
        const dnnEval = await ModelEngine.evaluateAsync(dnn, fullTestTrans.X, fullTestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions, evalAbortOpts(watchdog));
        dnnEval.featuresUsed = `All Surviving (${activeCols.length} features)`;
        dnnEval.constraintNote = dnn.usedFallback
            ? `⚠️ TensorFlow.js failed (${dnn.fallbackReason}); a linear fallback model was trained instead of a neural network.`
            : `L2 weight decay (${pHp.nn.l2Reg}) and dropout (${pHp.nn.dropoutRate}); ${dnn.trainingInfo ? describeNNTraining(dnn.trainingInfo) : ''}` + tunedTag('nn');
        benchmarkList.push({ id: 'nn', name: displayModelName('nn', dnn), model: dnn, prep: fullPrep, eval: dnnEval });

        // D. Logistic / Ridge Regression (Strict constraint: L2 penalization against collinearity)
        if (watchdog) watchdog.tick(stepOf('glm'), `Training Model ${stepOf('glm')}/${totalModels}: Logistic / Ridge Regression (L2 lambda = ${pHp.glm.lambdaL2})...`);
        await new Promise(r => setTimeout(r, 60));
        const glm = await fitModel('glm', isClassification, numClasses, { ...pHp.glm, seed: evalSeed }, fullPrep.X, fullPrep.y, { isAborted });
        const glmEval = await ModelEngine.evaluateAsync(glm, fullTestTrans.X, fullTestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions, evalAbortOpts(watchdog));
        glmEval.featuresUsed = `All Surviving (${activeCols.length} features)`;
        glmEval.constraintNote = `Applied Ridge L2 penalization (λ = ${pHp.glm.lambdaL2}) to stabilize against minor collinearity.` + tunedTag('glm');
        benchmarkList.push({ id: 'glm', name: 'Logistic / Ridge Regression', model: glm, prep: fullPrep, eval: glmEval });

        // E. Ensemble Super Learner (Strict constraint: StandardScaler fitted on Train and applied to Test)
        if (watchdog) watchdog.tick(stepOf('ensemble'), `Training Model ${stepOf('ensemble')}/${totalModels}: Voting Ensemble...`);
        await new Promise(r => setTimeout(r, 60));
        // Same base models for classification and regression (consistent with the Step 4 default)
        const ensembleConfig = { ...pHp.ensemble };
        const ensemble = await fitModel('ensemble', isClassification, numClasses, ensembleConfig, fullPrep.X, fullPrep.y, {
            isAborted,
            onProgress: (curr, tot, info) => {
                if (curr === 'tick' || !watchdog) return;
                watchdog.tick(stepOf('ensemble'), `Training Model ${stepOf('ensemble')}/${totalModels}: Ensemble (${(info && info.stage) || `Base ${curr}/${tot}`})...`);
            }
        });
        const ensEval = await ModelEngine.evaluateAsync(ensemble, fullTestTrans.X, fullTestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions, evalAbortOpts(watchdog));
        ensEval.featuresUsed = `All Surviving (${activeCols.length} features)`;
        if (ensemble.fallbackNote) ensEval.fallbackNote = ensemble.fallbackNote;
        ensEval.constraintNote = `Equal-weight soft vote of ${ensembleConfig.baseModels.map(b => b.toUpperCase()).join(', ')}; scaling fitted on Train only.`;
        benchmarkList.push({ id: 'ensemble', name: 'Voting / Weighted Ensemble', model: ensemble, prep: fullPrep, eval: ensEval });

        // SMOTENC trains the models on artificially balanced rows, which inflates every predicted risk of the rarer
        // outcome (van den Goorbergh et al., JAMIA 2022). Without recalibration, the predictions are shifted back to
        // the real event rate of the training patients (prior-shift correction: logit p − log(odds_balanced /
        // odds_real)); ranking (AUC) is unchanged. With recalibration, the calibrator fitted on real out-of-fold
        // predictions does this instead.
        const smoteApplied = Boolean(AppState.pipeline.smotencResult && AppState.pipeline.smotencResult.applied);
        let priorShift = null; // applied to out-of-fold predictions too (Youden threshold on the corrected scale)
        if (smoteApplied && isClassification && numClasses === 2 && recalPipe === 'none') {
            const pos = String(fullPrep.targetClasses[1]);
            const rate = rows => rows.filter(r => String(r[targetCol]).trim() === pos).length / Math.max(1, rows.length);
            const real = rate(AppState.pipeline.trainRows), bal = rate(AppState.pipeline.balancedTrainRows);
            if (real > 0 && real < 1 && bal > 0 && bal < 1) {
                const shift = Math.log(real / (1 - real)) - Math.log(bal / (1 - bal));
                priorShift = shift;
                const cal = { type: 'platt', a: shift, b: 1, n: AppState.pipeline.trainRows.length };
                for (const b of benchmarkList) {
                    const td = DataEngine.prepareTestingDataset(AppState.pipeline.testRows, b.prep);
                    const rc = await recalibrateAndEvaluate(b.model, cal, td.X, td.y, isClassification, fullPrep.targetClasses, pipeEvalOptions, evalAbortOpts(watchdog));
                    ['featuresUsed', 'constraintNote', 'fallbackNote'].forEach(k => { if (b.eval[k] !== undefined) rc.evalRes[k] = b.eval[k]; });
                    rc.evalRes.recalibration.method = 'prior-shift';
                    rc.evalRes.recalibration.fittedOn = `the real event rate of the training patients (${(real * 100).toFixed(1)}%, against ${(bal * 100).toFixed(1)}% after SMOTENC balancing)`;
                    rc.evalRes.constraintNote = (rc.evalRes.constraintNote || '') + ` Probabilities corrected from the SMOTENC-balanced rate (${(bal * 100).toFixed(1)}%) back to the real rate (${(real * 100).toFixed(1)}%).`;
                    b.model = rc.model;
                    b.eval = rc.evalRes;
                }
            }
        } else if (smoteApplied && isClassification && numClasses > 2 && recalPipe === 'none') {
            benchmarkList.forEach(b => { b.eval.constraintNote = (b.eval.constraintNote || '') + ' Caution: trained on SMOTENC-balanced rows, so predicted probabilities over-state the rarer outcome groups; use them for ranking, not as risks.'; });
        }

        // Optional recalibration: calibrator fitted on each model's out-of-fold Training-set predictions
        if (recalPipe !== 'none') {
            for (const b of benchmarkList) {
                const cal = fitRecalibrator(recalPipe, oofByModel[b.id]);
                if (!cal) { b.eval.recalibrationSkipped = 'Recalibration was not applied: too few out-of-fold predictions or events to fit it reliably.'; continue; }
                const td = DataEngine.prepareTestingDataset(AppState.pipeline.testRows, b.prep);
                const rc = await recalibrateAndEvaluate(b.model, cal, td.X, td.y, isClassification, fullPrep.targetClasses, pipeEvalOptions, evalAbortOpts(watchdog));
                ['featuresUsed', 'constraintNote', 'fallbackNote'].forEach(k => { if (b.eval[k] !== undefined) rc.evalRes[k] = b.eval[k]; });
                rc.evalRes.constraintNote = (rc.evalRes.constraintNote || '') + ` Probabilities recalibrated (${cal.type === 'platt' ? 'Platt scaling' : 'isotonic regression'}) on out-of-fold Training-set predictions.`;
                b.model = rc.model;
                b.eval = rc.evalRes;
            }
        }
        // Cross-validated scores (model selection inside the Training set) when that mode is chosen
        benchmarkList.forEach(b => { if (cvByModel[b.id]) b.eval.cv = cvByModel[b.id]; });
        const shifted = o => (o && priorShift !== null ? { ...o, p: o.p.map(v => { const q = Math.min(1 - 1e-12, Math.max(1e-12, v)); return 1 / (1 + Math.exp(-(Math.log(q / (1 - q)) + priorShift))); }) } : o);
        if (isClassification && numClasses === 2) benchmarkList.forEach(b => { const y = youdenThreshold(shifted(oofByModel[b.id])); if (y) b.eval.youden = y; });
        // Bootstrap 95% confidence intervals for every test metric
        for (const b of benchmarkList) {
            if (watchdog) watchdog.tick(totalModels, `Computing 95% confidence intervals (bootstrap): ${b.name}...`);
            b.eval.ci = await bootstrapFor(b.eval, evalSeed, testPatientGroups(fullTestTrans.validRows, 'pipeline'));
        }
        // Store ranked best-first, so benchmarkResults[0] is genuinely the top model everywhere it is used
        benchmarkList.sort((a, b) => compareEvalResults(a.eval, b.eval, isClassification));
        if (AppState.datasetRun !== runDataset) throw new Error('A different dataset was loaded while the models were training, so these results were discarded.');
        disposeModels((AppState.pipeline.benchmarkResults || []).map(b => b.model));
        AppState.pipeline.benchmarkResults = benchmarkList;
        AppState.pipeline.tuningReport = pipeTune ? pipeTuningReport : null;
        AppState.pipeline.cvSelectionStability = pipeTuningReport.selectionStability || null;
        renderTuningReport(AppState.pipeline.tuningReport, 'pipeline-tuning-report-container');

        // Render Leaderboard & Confusion Matrices
        renderPipelineBenchmarkResults(benchmarkList, fullPrep.targetClasses, isClassification);

        if (DOM.pipelineResultsContainer) DOM.pipelineResultsContainer.style.display = 'block';

        if (watchdog) watchdog.finish(`All ${benchmarkList.length} Clinical Models Evaluated Successfully!`);
        saveSettingsToBrowser(); // remember these choices for this dataset

    } catch (err) {
        console.error("Pipeline Step 7 training error:", err);
        if (watchdog && !watchdog.aborted) {
            watchdog.forceAbort(`Pipeline error: ${err.message}`);
        }
        alert(`Pipeline training error: ${err.message}`);
    } finally {
        if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.disabled = false;
    }
}

function renderPipelineLeaderboardOnly() {
    if (!AppState.pipeline.benchmarkResults || !DOM.pipelineLeaderboardContainer) return;
    const isClassification = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous';
    const sorted = [...AppState.pipeline.benchmarkResults].sort((a, b) => compareEvalResults(a.eval, b.eval, isClassification));

    let tableHtml = `
        <p style="font-size: 12px; color: var(--text-secondary); margin-bottom: 8px;">
            ${sorted[0] && sorted[0].eval.cv
                ? `Ranked by <strong>cross-validated ${escapeHtml(rankingMetricLabel(isClassification))} inside the Training set</strong> (CV column); the held-out ${Math.round((1 - (AppState.pipeline.trainRatio || 0.8)) * 100)}% Testing set was not used to choose, so the top model's test score is an honest estimate. The other models' test scores are for information only.`
                : `Ranked by <strong>${escapeHtml(rankingMetricLabel(isClassification))}</strong> on the held-out ${Math.round((1 - (AppState.pipeline.trainRatio || 0.8)) * 100)}% test split (change the ranking metric above). The top model is selected on these same test cases, so its score is optimistically biased; choose by cross-validation or confirm on an independent cohort.`}
        </p>`;
    tableHtml += `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Algorithm</th>
                    ${sorted[0] && sorted[0].eval.cv ? `<th title="Score inside the Training set by cross-validation (mean ± SD); used to choose the best model">CV ${escapeHtml(rankingMetricLabel(isClassification))}</th>` : ''}
                    <th>Features Subset</th>
                    ${isClassification ? '<th title="How close the predicted probabilities are to what happened (0 = perfect; lower is better)">Brier ↓</th><th>Test Accuracy</th>' : ''}
                    <th>${isClassification ? 'F1-Score (Macro)' : 'R²'}</th>
                    <th>${isClassification ? 'Precision' : 'MAE'}</th>
                    <th>${isClassification ? 'Recall' : 'RMSE'}</th>
                    <th>AUC-ROC</th>
                    <th>${isClassification ? 'Balanced Acc' : '-'}</th>
                    ${isClassification ? '' : '<th>-</th><th>MSE</th>'}
                    <th>Algorithm Constraint</th>
                    <th style="text-align: right; width: 140px;">Deployment</th>
                </tr>
            </thead>
            <tbody>
    `;

    sorted.forEach(item => {
        const ev = item.eval;
        const f1 = isClassification ? (ev.macroF1 !== undefined ? (ev.macroF1 * 100).toFixed(1) + '%' + ciSpan(ev, 'macroF1', true) : 'N/A') : fmtR2(ev.r2) + ciSpan(ev, 'r2');
        const prec = isClassification ? (ev.macroPrecision !== undefined ? (ev.macroPrecision * 100).toFixed(1) + '%' : 'N/A') : `${ev.mae}${ciSpan(ev, 'mae')}`;
        const rec = isClassification ? (ev.macroRecall !== undefined ? (ev.macroRecall * 100).toFixed(1) + '%' : 'N/A') : `${ev.rmse}${ciSpan(ev, 'rmse')}`;
        const auc = isClassification ? (ev.rocAuc !== null && ev.rocAuc !== undefined ? ev.rocAuc.toFixed(3) + ciSpan(ev, 'rocAuc') : 'N/A') : 'N/A';
        const acc = isClassification ? (ev.accuracy !== undefined ? (ev.accuracy * 100).toFixed(1) + '%' + (Array.isArray(ev.accuracyCI) ? `<div class="ci-text" title="Wilson 95% confidence interval">${(ev.accuracyCI[0] * 100).toFixed(1)}%–${(ev.accuracyCI[1] * 100).toFixed(1)}%</div>` : '') : 'N/A') : `${ev.mse}`;
        const bal = isClassification && typeof ev.balancedAccuracy === 'number' ? (ev.balancedAccuracy * 100).toFixed(1) + '%' + ciSpan(ev, 'balancedAccuracy', true) : '-';
        const brier = isClassification && typeof ev.brierScore === 'number' ? ev.brierScore.toFixed(3) + ciSpan(ev, 'brierScore') : '-';

        const isDeployed = AppState.deployedSource === '7step' && AppState.deployedModelId === item.id;

        tableHtml += `
            <tr style="${isDeployed ? 'background: rgba(16, 185, 129, 0.05);' : ''}">
                <td><strong>${escapeHtml(item.name)}</strong></td>
                ${sorted[0] && sorted[0].eval.cv ? `<td>${formatCvScore(ev, isClassification)}</td>` : ''}
                <td><span class="pill pill-target" style="font-size: 11px;">${ev.featuresUsed}</span></td>
                ${isClassification ? `<td><strong>${brier}</strong></td><td><strong>${acc}</strong></td>` : ''}
                <td><strong style="color: var(--accent-emerald); font-size: 13.5px;">${f1}</strong></td>
                <td>${prec}</td>
                <td>${rec}</td>
                <td><strong style="color: var(--accent-cyan);">${auc}</strong></td>
                <td>${bal}</td>
                ${isClassification ? '' : `<td>${brier}</td><td><strong>${acc}</strong></td>`}
                <td style="font-size: 11.5px; color: var(--text-secondary);">${escapeHtml(ev.constraintNote)}</td>
                <td style="text-align: right;">
                    ${isDeployed ? `
                        <span class="pill" style="background: rgba(16, 185, 129, 0.15); color: var(--tag-emerald); font-weight: 700; border-color: rgba(16, 185, 129, 0.3);">
                            ✓ Active
                        </span>
                    ` : `
                        <button class="btn btn-secondary btn-sm" ${actionAttrs('deployPipelineModel', item.id)} title="Make this the active research model for Steps 5 and 6">
                            Use this model
                        </button>
                    `}
                </td>
            </tr>
        `;
    });

    tableHtml += `</tbody></table>${sorted.some(it => it.eval && it.eval.ci) ? ciScopeNote() : ''}`;
    const caveats = new Set();
    sorted.forEach(item => {
        summariseHeldOutSet(item.eval).forEach(n => caveats.add(n));
        if (item.eval.fallbackNote) caveats.add(`⚠️ ${escapeHtml(item.eval.fallbackNote)}`);
    });
    if (sorted[0] && sorted[0].eval.cv) {
        if (AppState.pipeline.droppedFeatures && AppState.pipeline.droppedFeatures.size) {
            caveats.add(`Predictors removed in Steps 2–4 (${AppState.pipeline.droppedFeatures.size}) were chosen once on the whole Training set, partly from their association with the outcome, and not re-chosen inside each cross-validation fold: the cross-validated scores used to pick and tune models are therefore somewhat optimistic. The held-out Testing-set results are not affected.`);
        }
        const stab = AppState.pipeline.cvSelectionStability;
        if (stab && stab.length) {
            const freq = {};
            stab.forEach(list => list.forEach(c => { freq[c] = (freq[c] || 0) + 1; }));
            const always = Object.keys(freq).filter(c => freq[c] === stab.length);
            const some = Object.keys(freq).filter(c => freq[c] < stab.length).map(c => `${escapeHtml(c)} (${freq[c]}/${stab.length})`);
            caveats.add(`Feature selection inside cross-validation: for k-NN and Naive Bayes, the Step 6 top-70% selection was repeated inside each of the ${stab.length} folds. Selected in every fold: ${always.length ? always.map(escapeHtml).join(', ') : 'none'}${some.length ? `; in some folds: ${some.join(', ')}` : ''}. Predictors that are selected only sometimes are unstable choices.`);
        }
        if (AppState.pipeline.droppedFeatures && AppState.pipeline.droppedFeatures.size > 0) {
            caveats.add(`Post-selection note: ${AppState.pipeline.droppedFeatures.size} predictor(s) were excluded in Steps 2–4 using the whole Training set before cross-validation, so the cross-validated scores are slightly optimistic. The Testing-set results are not affected.`);
        }
    }
    if (isClassification && AppState.pipeline.trainRows && AppState.pipeline.trainRows.length) {
        const counts = {};
        AppState.pipeline.trainRows.forEach(r => { const k = String(r[AppState.pipeline.targetCol]); counts[k] = (counts[k] || 0) + 1; });
        const minority = Math.min(...Object.values(counts));
        const nPred = (AppState.pipeline.featureCols || []).filter(c => !AppState.pipeline.droppedFeatures.has(c)).length;
        if (nPred > 0 && minority / nPred < 10) {
            caveats.add(`⚠️ Small sample for the number of predictors: the smaller outcome group has ${minority} real training patient(s) for ${nPred} predictor(s) (${(minority / nPred).toFixed(1)} per predictor; at least 10–20 is a common rule of thumb). SMOTENC does not add real information, so treat the results as exploratory.`);
        }
    }
    if (isClassification && sorted[0] && sorted[0].eval.positiveClass !== null && sorted[0].eval.positiveClass !== undefined) {
        caveats.add(`Positive (event) class for AUC, Brier, sensitivity and PPV: <strong>${escapeHtml(sorted[0].eval.positiveClass)}</strong> (set in Step 2).`);
    }
    if (caveats.size > 0) {
        tableHtml += `<div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 10px; padding: 8px 12px; border-left: 3px solid var(--accent-amber); background: var(--bg-surface-subtle);">
            ${[...caveats].map(c => `<div>• ${c}</div>`).join('')}
        </div>`;
    }
    tableHtml += renderTooGoodToBeTrueCheck(sorted, isClassification);
    DOM.pipelineLeaderboardContainer.innerHTML = tableHtml;
    renderCalibrationPanel('pipeline-calibration-container', sorted.map(it => ({ name: it.name, ev: it.eval })), isClassification);
    renderExplainPanel('pipeline-explain-container', sorted.map(it => ({ name: it.name, model: it.model, prep: it.prep, inSample: false })), AppState.pipeline.testRows, isClassification);
    renderSubgroupPanel('pipeline-subgroup-container', sorted.map(it => ({ name: it.name, model: it.model, prep: it.prep, threshold: it.eval && it.eval.decisionThreshold })), AppState.pipeline.testRows, isClassification);
    syncRankingSelectors(isClassification);
}

export { executePipelineStep7, renderPipelineLeaderboardOnly };
