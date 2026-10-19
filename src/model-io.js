// model-io.js: part of the Universal Medical ML/NeuralNet Predictor application controller
/* global tf, DataEngine, ModelEngine */
import { APP_VERSION, AppState, RESEARCH_USE_NOTICE, displayModelName, prepTypicality, rankingMetricLabel } from './core.js';
import { describeSplit } from './cv-tuning.js';
import { goToStep } from './navigation.js';
import { updatePredictionProvenanceBar } from './pipeline-results.js';
import { SETTINGS_APP } from './settings.js';
import { DOM } from './shell.js';
import { renderSinglePredictionForm } from './single-prediction.js';
import { showSeedToast } from './training-actions.js';
import { scheduleAnalysisSummary } from './analysis-summary.js';
// =========================================================================
// EXPORT / IMPORT OF THE DEPLOYED (TRAINED) MODEL
// One JSON file: fitted preprocessing (encoding, scaling, imputation), model parameters, decision
// threshold, evaluation summary, seeds and software version. Patient-level training data are not
// included, except where the method needs them (k-NN models and k-NN imputation), which is flagged.
// =========================================================================
const MODEL_FILE_FORMAT = 'ummp-trained-model';
// Version 2: model parameters follow ModelEngine.MODEL_FILE_SCHEMA (only listed, type-checked fields); version 1
// files (earlier releases) use the same field names and still load
const MODEL_FILE_VERSION = 2;

function sanitizedPrep(prep) {
    const keep = ['featureCols', 'featureMetadata', 'totalDimensions', 'targetCol', 'isClassification', 'targetClasses',
        'targetClassToIdx', 'targetIdxToClass', 'positiveClass', 'fillValues', 'encodedFeatureNames', 'derivedIndicators', 'typicality'];
    prepTypicality(prep); // aggregate statistics only (mean and inverse covariance), for the unusual-patient check
    const out = {};
    keep.forEach(k => { if (prep[k] !== undefined) out[k] = JSON.parse(JSON.stringify(prep[k])); });
    let knnImputation = false, knnColumns = [];
    if (prep.imputation && prep.imputation.plan) {
        const cols = prep.imputation.plan.columns || {};
        knnImputation = Object.values(cols).some(c => c && c.strategy === 'knn');
        // k-NN filling needs, for each training patient, only the predictors it fills and measures distance
        // on: every other column (identifiers, outcome, excluded variables) is left out of the file
        const needed = new Set();
        Object.keys(cols).forEach(c => { if (cols[c] && cols[c].strategy === 'knn') { needed.add(c); (cols[c].numericCols || []).forEach(n => needed.add(n)); } });
        const refRows = knnImputation ? (prep.imputation.plan.referenceRows || []).map(r => { const o = {}; needed.forEach(c => { if (r[c] !== undefined) o[c] = r[c]; }); return o; }) : [];
        out.imputation = {
            configs: JSON.parse(JSON.stringify(prep.imputation.configs)),
            plan: { columns: JSON.parse(JSON.stringify(cols)), referenceRows: JSON.parse(JSON.stringify(refRows)) }
        };
        knnColumns = [...needed];
    }
    // Multiple imputation: the fitted imputation models, including their donor values (observed values of
    // training patients, one predictor at a time), are needed to impute new patients
    const miceImputation = Boolean(prep.imputation && prep.imputation.mice);
    if (miceImputation) {
        out.imputation = { ...(out.imputation || { configs: JSON.parse(JSON.stringify(prep.imputation.configs)) }), mice: JSON.parse(JSON.stringify(prep.imputation.mice)) };
        // The convergence trace is a diagnostic only and is left out: with one missing cell its "mean imputed value"
        // would be a single training patient's observed value
        if (prep.multipleImputation) { const { trace, traceScale, ...miInfo } = prep.multipleImputation; out.multipleImputation = JSON.parse(JSON.stringify(miInfo)); }
    }
    return { prep: out, knnImputation: knnImputation || miceImputation, miceImputation, knnFilling: knnImputation, knnColumns };
}

function evaluationSummary(ev) {
    if (!ev) return null;
    const drop = new Set(['actuals', 'predictions', 'probPositive', 'probMatrix', 'validRows', 'confMatrix']);
    const out = {};
    Object.keys(ev).forEach(k => { if (!drop.has(k) && typeof ev[k] !== 'function') out[k] = ev[k]; });
    const res = JSON.parse(JSON.stringify(out));
    // A date split records the dates of individual records at its boundaries: only the month goes into a shared file
    if (res.splitInfo) ['trainUntil', 'testFrom', 'testUntil'].forEach(k => { if (typeof res.splitInfo[k] === 'string') res.splitInfo[k] = res.splitInfo[k].slice(0, 7); });
    return res;
}

// Whether a serialised model holds training patients' values: a k-NN model anywhere inside it, including inside
// recalibrated, multiply imputed (pooled), ensemble or neural-network-fallback wrappers
function modelContainsKnn(ser) {
    if (!ser || typeof ser !== 'object') return false;
    if (ser.type === 'knn') return true;
    if (ser.type === 'ensemble') return (ser.models || []).some(it => modelContainsKnn(it.model));
    if (ser.type === 'mi-pooled') return (ser.models || []).some(modelContainsKnn);
    if (ser.type === 'calibrated') return modelContainsKnn(ser.base);
    if (ser.type === 'nn') return modelContainsKnn(ser.fallbackGLM);
    return false;
}

async function exportTrainedModel() {
    if (AppState.outcomeType === 'survival') { alert('Saving a Cox (time-to-event) model as a model file is not available yet. The Survival Analysis Report (Step 4) records the hazard ratios, the design and the performance; the settings file (header) lets you refit the same model.'); return; }
    const model = AppState.trainedModel, prep = AppState.prepMeta;
    if (!model || !prep) { alert('Train or deploy a model first.'); return; }
    const btn = document.getElementById('model-export-btn');
    try {
        if (btn) btn.disabled = true;
        const ser = await ModelEngine.serializeModel(model);
        const { prep: cleanPrep, knnImputation, miceImputation, knnFilling, knnColumns } = sanitizedPrep(prep);
        const knnModel = modelContainsKnn(ser);
        const containsPatientData = knnImputation || knnModel;
        // What the file holds, stated exactly (no identifiers or outcomes of individual patients beyond this)
        const manifest = [];
        if (knnModel) manifest.push(`• k-Nearest Neighbours model: every training patient's encoded predictor values and outcome (${prep.featureCols.length} predictor(s), outcome "${prep.targetCol}")`);
        if (knnFilling) manifest.push(`• k-NN missing-value filling: every training patient's values of ${knnColumns.length} predictor(s) (${knnColumns.slice(0, 6).join(', ')}${knnColumns.length > 6 ? ', …' : ''}); no identifiers or outcomes`);
        if (miceImputation) manifest.push('• Multiple imputation: up to 500 observed values per predictor from training patients (donors), each predictor separately; no identifiers or outcomes');
        if (containsPatientData && !confirm('This model file will contain patient-level data from the training set:\n\n' + manifest.join('\n') +
            '\n\nIt also contains the model parameters, preprocessing and summary results (no other patient values). Share it only where your data-governance rules allow. Continue?')) return;
        const name = AppState.deployedSource === '7step' && AppState.deployedPipelineModel ? AppState.deployedPipelineModel.name
            : AppState.deployedSource === 'imported' && AppState.importedModelInfo ? AppState.importedModelInfo.name
            : displayModelName(AppState.deployedModelId || AppState.selectedModelId, model);
        const bundle = {
            format: MODEL_FILE_FORMAT, formatVersion: MODEL_FILE_VERSION, notice: RESEARCH_USE_NOTICE,
            software: { name: SETTINGS_APP, version: APP_VERSION },
            exportedAt: new Date().toISOString(),
            model: { id: AppState.deployedModelId || AppState.selectedModelId, name, workflow: AppState.deployedSource || 'standard', seed: model.seed ?? null },
            trainingDataset: { name: AppState.fileName || (AppState.importedModelInfo && AppState.importedModelInfo.dataset) || null, outcome: prep.targetCol },
            containsPatientData,
            patientDataContents: containsPatientData ? manifest.map(m => m.replace(/^• /, '')) : [],
            decisionThreshold: AppState.evalResults ? AppState.evalResults.decisionThreshold ?? null : null,
            evaluation: evaluationSummary(AppState.evalResults),
            environment: computingEnvironment(),
            preprocessing: cleanPrep,
            parameters: ser
        };
        { const inputs = selfCheckInputs(prep); bundle.selfCheck = { inputs, outputs: selfCheckOutputs(model, prep, inputs) }; }
        const blob = new Blob([JSON.stringify(bundle)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const base = String(bundle.trainingDataset.name || 'model').replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_-]+/g, '_');
        a.href = url;
        a.download = `${base}_trained-model_${bundle.model.id || 'model'}_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showSeedToast(`✓Exported "${name}"${containsPatientData ? ' (contains training patients\' predictor values)' : ''}`);
    } catch (e) {
        alert(`The model could not be exported: ${e.message || e}`);
    } finally {
        if (btn) btn.disabled = !AppState.trainedModel;
    }
}

// A model file is untrusted input: its preprocessing description is checked before anything is used
function validateImportedPrep(prep) {
    const bad = what => { throw new Error(`The file's preprocessing information is invalid (${what}).`); };
    if (!prep || typeof prep !== 'object') bad('missing');
    if (!Array.isArray(prep.featureCols) || !prep.featureCols.length || !prep.featureCols.every(c => typeof c === 'string')) bad('predictor names');
    if (typeof prep.targetCol !== 'string') bad('outcome name');
    if (typeof prep.isClassification !== 'boolean') bad('outcome type');
    if (!Number.isInteger(prep.totalDimensions) || prep.totalDimensions < 1 || prep.totalDimensions > 5000) bad('number of encoded columns');
    if (!prep.featureMetadata || typeof prep.featureMetadata !== 'object') bad('predictor encoding');
    prep.featureCols.forEach(c => {
        const md = prep.featureMetadata[c];
        if (!md || typeof md !== 'object' || !['nominal', 'ordinal', 'continuous'].includes(md.type)) bad(`encoding of "${c}"`);
        if (!Number.isInteger(md.startIndex) || md.startIndex < 0 || md.startIndex >= prep.totalDimensions) bad(`position of "${c}"`);
        if (md.type === 'nominal' && (!Array.isArray(md.categories) || !md.categories.every(v => typeof v === 'string'))) bad(`categories of "${c}"`);
    });
    if (prep.isClassification && (!Array.isArray(prep.targetClasses) || prep.targetClasses.length < 2)) bad('outcome categories');
    ['__proto__', 'constructor', 'prototype'].forEach(k => { if (Object.prototype.hasOwnProperty.call(prep, k)) bad(k); });
    // Unusual-patient statistics: used only if well formed (otherwise the check is skipped for this model)
    const t = prep.typicality, num = v => typeof v === 'number' && isFinite(v);
    const okT = t && Array.isArray(t.mean) && t.mean.length === prep.totalDimensions && t.mean.every(num) && Array.isArray(t.inv) && t.inv.length === t.mean.length
        && t.inv.every(r => Array.isArray(r) && r.length === t.mean.length && r.every(num)) && num(t.q99) && num(t.q999);
    prep.typicality = okT ? t : null;
}

// The evaluation summary in a model file is untrusted: only numbers, true/false and nested lists or objects of
// them are kept (no text, which could reach the page or reports), apart from two fields with fixed values
function sanitizeEvaluation(e, depth = 0) {
    const ENUMS = { protocol: ['option-a', 'option-b', 'option-c', 'pipeline'], probabilityType: ['model', 'vote', 'neighbours', 'recalibrated', 'blend'] };
    const clean = (v, d, key) => {
        if (typeof v === 'number') return isFinite(v) ? v : undefined;
        if (typeof v === 'boolean' || v === null) return v;
        if (typeof v === 'string') return key && ENUMS[key] && ENUMS[key].includes(v) ? v : undefined;
        if (d > 4) return undefined;
        if (Array.isArray(v)) return v.length <= 10000 ? v.map(x => clean(x, d + 1)).filter(x => x !== undefined) : undefined;
        if (typeof v === 'object') {
            const o = {};
            Object.keys(v).slice(0, 500).forEach(k => {
                if (k === '__proto__' || k === 'constructor' || k === 'prototype') return;
                const c = clean(v[k], d + 1, k);
                if (c !== undefined) o[k] = c;
            });
            return o;
        }
        return undefined;
    };
    const out = clean(e && typeof e === 'object' && !Array.isArray(e) ? e : {}, depth);
    return out || {};
}

// The imported model must predict a typical patient (all predictors at their fitted defaults) with valid output
// The computing environment a model was trained and saved in (no patient data): app version, TensorFlow.js
// version and backend (GPU arithmetic differs slightly between devices), browser
function computingEnvironment() {
    const env = { app: APP_VERSION };
    try { if (typeof tf !== 'undefined') { env.tfjs = tf.version && tf.version.tfjs; env.backend = tf.getBackend && tf.getBackend(); } } catch (e) { /* TF.js unavailable */ }
    try { env.browser = navigator.userAgent; } catch (e) { /* not available */ }
    return env;
}

// Reproducibility self-check: three synthetic patients built from the training summaries (a typical patient,
// and every numerical predictor one SD above or below its mean, categories at their most common value; no
// real patient's values) and the model's outputs for them. Re-run on import: the same file must give the
// same predictions.
function selfCheckInputs(prep) {
    return [0, 1, -1].map(k => {
        const row = {};
        prep.featureCols.forEach(c => {
            const m = prep.featureMetadata[c] || {};
            row[c] = m.type === 'continuous' && typeof m.mean === 'number' ? Number((m.mean + k * (m.std || 0)).toPrecision(8)) : '';
        });
        return row;
    });
}
function outputVector(out) {
    if (out && typeof out === 'object') return [out.pred, ...Object.keys(out.probabilities || {}).sort().map(k => out.probabilities[k])].map(Number);
    return [Number(out)];
}
function selfCheckOutputs(model, prep, inputs) {
    return inputs.map(r => outputVector(model.predictSingle(DataEngine.transformSingleRow(r, prep))));
}

function checkImportedModel(model, prep) {
    if (!model || typeof model.predictSingle !== 'function') throw new Error('The file does not contain a usable model.');
    const row = {};
    prep.featureCols.forEach(c => { row[c] = ''; });
    const x = DataEngine.transformSingleRow(row, prep);
    if (!Array.isArray(x) && !ArrayBuffer.isView(x)) throw new Error('The model\'s preprocessing could not be applied.');
    const out = model.predictSingle(x);
    const nums = out && typeof out === 'object' ? [out.pred, ...Object.values(out.probabilities || {})] : [out];
    if (!nums.every(v => typeof v === 'number' && isFinite(v))) throw new Error('The model gives invalid predictions.');
}

async function importTrainedModel(file) {
    try {
        if (file.size > 300 * 1024 * 1024) throw new Error('The file is larger than 300 MB, too large for a model file from this app.');
        const bundle = JSON.parse(await file.text());
        if (!bundle || bundle.format !== MODEL_FILE_FORMAT) throw new Error('This is not a trained-model file exported from this app.');
        if (bundle.formatVersion !== undefined && bundle.formatVersion > MODEL_FILE_VERSION) throw new Error('This model file was made by a newer version of the app; please use the latest version.');
        const prep = bundle.preprocessing;
        validateImportedPrep(prep);
        const model = await ModelEngine.deserializeModel(bundle.parameters);
        // Nothing is installed until the model has predicted a typical patient without error
        checkImportedModel(model, prep);
        // Reproducibility self-check: the recorded predictions for three synthetic patients must be reproduced
        let selfCheckNote = '';
        const sc = bundle.selfCheck;
        if (sc && Array.isArray(sc.inputs) && Array.isArray(sc.outputs) && sc.inputs.length === sc.outputs.length && sc.inputs.length <= 10) {
            const now = selfCheckOutputs(model, prep, sc.inputs.map(r => (r && typeof r === 'object' ? Object.fromEntries(prep.featureCols.map(c => [c, r[c] ?? ''])) : {})));
            const diff = Math.max(0, ...now.flatMap((v, i) => v.map((x, j) => Math.abs(x - Number((sc.outputs[i] || [])[j])))));
            const env = bundle.environment || {};
            if (!(diff <= 1e-6)) {
                const gpu = (env.backend && env.backend !== 'cpu') || (typeof tf !== 'undefined' && tf.getBackend && tf.getBackend() !== 'cpu');
                if (!(diff <= 1e-3 && gpu)) {
                    if (!confirm(`Reproducibility check: this model's predictions differ from those recorded when it was saved (largest difference ${isFinite(diff) ? diff.toPrecision(2) : 'not comparable'}).\n\nThe file may have been altered, or saved by a different version of the app (saved with ${env.app || 'an unknown version'}, this is ${APP_VERSION}). Use it anyway?`)) return;
                    selfCheckNote = 'predictions differ from those recorded at saving';
                } else selfCheckNote = `reproduced to within ${diff.toPrecision(2)} (GPU arithmetic)`;
            } else selfCheckNote = 'reproduced exactly';
        }
        AppState.trainedModel = model;
        AppState.prepMeta = prep;
        const evalIn = sanitizeEvaluation(bundle.evaluation);
        const thr = [bundle.decisionThreshold, evalIn.decisionThreshold].find(v => typeof v === 'number' && v > 0 && v < 1);
        AppState.evalResults = { ...evalIn, decisionThreshold: thr ?? null, isClassification: prep.isClassification };
        AppState.deployedSource = 'imported';
        AppState.deployedModelId = bundle.model && bundle.model.id;
        AppState.importedSelfCheck = selfCheckNote;
        AppState.importedModelInfo = { environment: bundle.environment && typeof bundle.environment === 'object' ? { app: String(bundle.environment.app || ''), tfjs: String(bundle.environment.tfjs || ''), backend: String(bundle.environment.backend || '') } : null, name: (bundle.model && bundle.model.name) || 'Imported model', exportedAt: bundle.exportedAt, dataset: bundle.trainingDataset && bundle.trainingDataset.name, version: bundle.software && bundle.software.version };
        if (DOM.toStep5Btn) DOM.toStep5Btn.disabled = false;
        if (DOM.toStep6Btn) DOM.toStep6Btn.disabled = false;
        updateModelExportButton();
        renderSinglePredictionForm();
        updatePredictionProvenanceBar();
        goToStep(5);
        showSeedToast(`✓Imported "${AppState.importedModelInfo.name}" — ready for single and batch predictions`);
    } catch (e) {
        alert(`The model could not be imported: ${e.message || e}`);
    }
}

function updateModelExportButton() {
    scheduleAnalysisSummary(); // the active model may have changed
    const btn = document.getElementById('model-export-btn');
    if (btn) { btn.disabled = !AppState.trainedModel; btn.title = AppState.trainedModel ? 'Save the active model (with its fitted preprocessing) as a file' : 'Train or deploy a model first'; }
    const card = document.getElementById('model-card-btn');
    if (card) { card.disabled = !AppState.trainedModel; card.title = AppState.trainedModel ? 'One-page summary of the active model: intended use, population, performance and limitations' : 'Train or deploy a model first'; }
}

function downloadPipelineProtocolReport() {
    if (!AppState.pipeline.benchmarkResults || AppState.pipeline.benchmarkResults.length === 0) {
        alert("No pipeline results available to download.");
        return;
    }

    const report = {
        title: "Strict 7-Step Clinical Machine Learning Protocol Report",
        software: { name: SETTINGS_APP, version: APP_VERSION, url: location.href.split('?')[0] },
        validation_design: {
            split: describeSplit(AppState.pipeline.splitInfo),
            train_fraction: AppState.pipeline.trainRatio,
            split_seed: AppState.pipeline.seed,
            model_selection: AppState.pipeline.benchmarkResults[0] && AppState.pipeline.benchmarkResults[0].eval.cv
                ? `${AppState.pipeline.benchmarkResults[0].eval.cv.folds}-fold cross-validation inside the training set${AppState.pipeline.benchmarkResults[0].eval.cv.grouped ? ' (patient-grouped folds)' : ''}`
                : 'best test-set score (optimistic)',
            settings_tuning: AppState.pipeline.tuningReport ? 'automatic (cross-validation inside the training set)' : 'protocol defaults',
            decision_threshold: AppState.pipeline.benchmarkResults[0] ? AppState.pipeline.benchmarkResults[0].eval.decisionThreshold : null,
            test_set_use: 'held-out, never resampled; used once to report performance'
        },
        conceived_and_directed_by: "Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)",
        generated_at: new Date().toISOString(),
        notice: RESEARCH_USE_NOTICE,
        dataset_name: AppState.fileName,
        target_variable: AppState.pipeline.targetCol,
        reproducibility_seed: AppState.pipeline.evalSeed,
        automatic_tuning: AppState.pipeline.tuningReport ? {
            method: `${AppState.pipeline.tuningReport.folds}-fold stratified cross-validation within the training set; imputation, SMOTENC and encoding/scaling refitted inside each fold; neural network at most 3 folds; test set not used`,
            ranking_metric: AppState.pipeline.tuningReport.models[0] ? AppState.pipeline.tuningReport.models[0].metricLabel : null,
            small_sample_warning: AppState.pipeline.tuningReport.smallSampleWarning || null,
            models: AppState.pipeline.tuningReport.models.map(t => ({
                model: t.modelId, folds: t.folds, rows_used: t.rowsUsed, protocol_default: t.own ? t.own.params : null,
                chosen: t.best.params, protocol_default_kept: t.best.isOwn, cv_mean: t.best.mean, cv_sd: t.best.sd,
                candidates: t.results.map(r => ({ params: r.params, cv_mean: r.mean, cv_sd: r.sd }))
            }))
        } : 'not used (protocol default settings)',
        pipeline_steps: {
            step_1_split_and_strict_imputation: {
                split_method: describeSplit(AppState.pipeline.splitInfo),
                train_records: AppState.pipeline.trainRows.length,
                test_records: AppState.pipeline.testRows.length,
                rows_excluded_missing_outcome: AppState.pipeline.excludedMissingTarget || 0,
                complete_case_analysis: Boolean(AppState.pipeline.completeCase),
                rows_excluded_incomplete_predictors: AppState.pipeline.excludedIncomplete || 0,
                split_seed: AppState.pipeline.seed,
                imputation_strategy: AppState.pipeline.imputeStrategy,
                leak_free_audit: AppState.pipeline.imputer.fillValues
            },
            step_2_skewness_and_correlation: {
                continuous_features_analyzed: Object.keys(AppState.pipeline.normalityMap),
                skewness_screen_rule: '|g1| <= 1 treated as approximately symmetric (rule of thumb, not a formal normality test)',
                skewness_assessment: AppState.pipeline.normalityMap,
                high_correlation_pairs: AppState.pipeline.corrPairs.filter(p => p.isHigh)
            },
            step_3_vif_check: {
                vif_scores: AppState.pipeline.vifResults,
                excluded_features_all_steps: Array.from(AppState.pipeline.droppedFeatures)
            },
            step_4_univariate_testing: {
                baseline_rankings: AppState.pipeline.univariateResults
            },
            step_5_smotenc_imbalance_handling: {
                applied: AppState.pipeline.smotencResult?.applied,
                reason: AppState.pipeline.smotencResult?.reason,
                minority_percentage_before: AppState.pipeline.smotencResult?.minorityPctBefore,
                minority_percentage_after: AppState.pipeline.smotencResult?.minorityPctAfter,
                synthetic_instances_generated: AppState.pipeline.smotencResult?.syntheticCountGenerated || 0
            },
            step_6_rf_feature_scout: {
                feature_importance_scores: AppState.pipeline.rfImportance,
                top_70_features: AppState.pipeline.top70Features,
                bottom_30_features: AppState.pipeline.bottom30Features,
                reinstated_bottom_30_features: Array.from(AppState.pipeline.reinstatedFeatures || [])
            },
            step_7_ranking_metric: rankingMetricLabel(AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous'),
            step_7_model_selection_note: AppState.pipeline.benchmarkResults[0] && AppState.pipeline.benchmarkResults[0].eval.cv
                ? 'Models are ranked by cross-validated performance inside the training set (SMOTENC and scaling refitted in each fold); the test split was used only to report performance, so the top model\'s test score is not selection-biased. Other models\' test scores are for information. A single split does not establish general superiority.'
                : 'Models are ranked best-first on the held-out test split; the top model is selected on that same split, so its score is optimistically biased. A single split does not establish general superiority.',
            step_7_constrained_model_benchmark: AppState.pipeline.benchmarkResults.map(b => ({
                model_id: b.id,
                model_name: b.name,
                features_used: b.eval.featuresUsed,
                constraint_note: b.eval.constraintNote,
                neural_network_fell_back_to_linear_model: Boolean(b.model && b.model.usedFallback),
                nn_training: b.model && b.model.trainingInfo ? b.model.trainingInfo : undefined,
                metrics: b.eval.isClassification ? {
                    f1_score_macro: b.eval.macroF1,
                    precision_macro: b.eval.macroPrecision,
                    recall_macro: b.eval.macroRecall,
                    balanced_accuracy: b.eval.balancedAccuracy,
                    auc_roc: b.eval.rocAuc,
                    brier_score: b.eval.brierScore,
                    accuracy: b.eval.accuracy,
                    accuracy_95ci_wilson: b.eval.accuracyCI,
                    probability_type: b.eval.probabilityType,
                    positive_class: b.eval.positiveClass,
                    classes_missing_from_test: b.eval.classesMissingFromTest
                } : {
                    r2: b.eval.r2,
                    rmse: b.eval.rmse,
                    mae: b.eval.mae,
                    mse: b.eval.mse,
                    exploratory_cutoff: b.eval.threshold,
                    exploratory_cutoff_source: b.eval.thresholdSource
                },
                cross_validated_training_scores: b.eval.cv || null,
                decision_threshold: b.eval.decisionThreshold,
                calibration: b.eval.calibration || null,
                confusion_matrix: b.eval.confMatrix
            }))
        }
    };

    const jsonString = JSON.stringify(report, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Clinical_ML_7Step_Protocol_Report_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

export { MODEL_FILE_FORMAT, sanitizedPrep, evaluationSummary, modelContainsKnn, exportTrainedModel, importTrainedModel, updateModelExportButton, downloadPipelineProtocolReport };
