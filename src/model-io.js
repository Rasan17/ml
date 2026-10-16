// model-io.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { APP_VERSION, AppState, displayModelName, rankingMetricLabel } from './core.js';
import { describeSplit } from './cv-tuning.js';
import { goToStep } from './navigation.js';
import { updatePredictionProvenanceBar } from './pipeline-results.js';
import { SETTINGS_APP } from './settings.js';
import { DOM } from './shell.js';
import { renderSinglePredictionForm } from './single-prediction.js';
import { showSeedToast } from './standard-training.js';


// =========================================================================
// EXPORT / IMPORT OF THE DEPLOYED (TRAINED) MODEL
// One JSON file: fitted preprocessing (encoding, scaling, imputation), model parameters, decision
// threshold, evaluation summary, seeds and software version. Patient-level training data are not
// included, except where the method needs them (k-NN models and k-NN imputation), which is flagged.
// =========================================================================
const MODEL_FILE_FORMAT = 'ummp-trained-model';

function sanitizedPrep(prep) {
    const keep = ['featureCols', 'featureMetadata', 'totalDimensions', 'targetCol', 'isClassification', 'targetClasses',
        'targetClassToIdx', 'targetIdxToClass', 'positiveClass', 'fillValues', 'encodedFeatureNames'];
    const out = {};
    keep.forEach(k => { if (prep[k] !== undefined) out[k] = JSON.parse(JSON.stringify(prep[k])); });
    let knnImputation = false;
    if (prep.imputation && prep.imputation.plan) {
        const cols = prep.imputation.plan.columns || {};
        knnImputation = Object.values(cols).some(c => c && c.strategy === 'knn');
        out.imputation = {
            configs: JSON.parse(JSON.stringify(prep.imputation.configs)),
            plan: { columns: JSON.parse(JSON.stringify(cols)), referenceRows: knnImputation ? JSON.parse(JSON.stringify(prep.imputation.plan.referenceRows || [])) : [] }
        };
    }
    return { prep: out, knnImputation };
}

function evaluationSummary(ev) {
    if (!ev) return null;
    const drop = new Set(['actuals', 'predictions', 'probPositive', 'probMatrix', 'validRows', 'confMatrix']);
    const out = {};
    Object.keys(ev).forEach(k => { if (!drop.has(k) && typeof ev[k] !== 'function') out[k] = ev[k]; });
    return JSON.parse(JSON.stringify(out));
}

function modelContainsKnn(ser) {
    if (!ser) return false;
    if (ser.type === 'knn') return true;
    if (ser.type === 'ensemble') return (ser.models || []).some(it => modelContainsKnn(it.model));
    return false;
}

async function exportTrainedModel() {
    const model = AppState.trainedModel, prep = AppState.prepMeta;
    if (!model || !prep) { alert('Train or deploy a model first.'); return; }
    const btn = document.getElementById('model-export-btn');
    try {
        if (btn) btn.disabled = true;
        const ser = await ModelEngine.serializeModel(model);
        const { prep: cleanPrep, knnImputation } = sanitizedPrep(prep);
        const containsPatientData = knnImputation || modelContainsKnn(ser);
        if (containsPatientData && !confirm('This model file will contain the training patients\' predictor values (needed by ' +
            (modelContainsKnn(ser) ? 'k-Nearest Neighbours' : 'k-NN missing-value filling') + '). Share it only where your data-governance rules allow. Continue?')) return;
        const name = AppState.deployedSource === '7step' && AppState.deployedPipelineModel ? AppState.deployedPipelineModel.name
            : AppState.deployedSource === 'imported' && AppState.importedModelInfo ? AppState.importedModelInfo.name
            : displayModelName(AppState.deployedModelId || AppState.selectedModelId, model);
        const bundle = {
            format: MODEL_FILE_FORMAT, formatVersion: 1,
            software: { name: SETTINGS_APP, version: APP_VERSION },
            exportedAt: new Date().toISOString(),
            model: { id: AppState.deployedModelId || AppState.selectedModelId, name, workflow: AppState.deployedSource || 'standard', seed: model.seed ?? null },
            trainingDataset: { name: AppState.fileName || (AppState.importedModelInfo && AppState.importedModelInfo.dataset) || null, outcome: prep.targetCol },
            containsPatientData,
            decisionThreshold: AppState.evalResults ? AppState.evalResults.decisionThreshold ?? null : null,
            evaluation: evaluationSummary(AppState.evalResults),
            preprocessing: cleanPrep,
            parameters: ser
        };
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
        showSeedToast(`✓ Exported "${name}"${containsPatientData ? ' (contains training patients\' predictor values)' : ''}`);
    } catch (e) {
        alert(`The model could not be exported: ${e.message || e}`);
    } finally {
        if (btn) btn.disabled = !AppState.trainedModel;
    }
}

async function importTrainedModel(file) {
    try {
        const bundle = JSON.parse(await file.text());
        if (!bundle || bundle.format !== MODEL_FILE_FORMAT) throw new Error('This is not a trained-model file exported from this app.');
        const model = await ModelEngine.deserializeModel(bundle.parameters);
        const prep = bundle.preprocessing;
        if (!prep || !Array.isArray(prep.featureCols)) throw new Error('The file has no preprocessing information.');
        AppState.trainedModel = model;
        AppState.prepMeta = prep;
        AppState.evalResults = { ...(bundle.evaluation || {}), decisionThreshold: bundle.decisionThreshold ?? (bundle.evaluation || {}).decisionThreshold, isClassification: prep.isClassification };
        AppState.deployedSource = 'imported';
        AppState.deployedModelId = bundle.model && bundle.model.id;
        AppState.importedModelInfo = { name: (bundle.model && bundle.model.name) || 'Imported model', exportedAt: bundle.exportedAt, dataset: bundle.trainingDataset && bundle.trainingDataset.name, version: bundle.software && bundle.software.version };
        if (DOM.toStep5Btn) DOM.toStep5Btn.disabled = false;
        if (DOM.toStep6Btn) DOM.toStep6Btn.disabled = false;
        updateModelExportButton();
        renderSinglePredictionForm();
        updatePredictionProvenanceBar();
        goToStep(5);
        showSeedToast(`✓ Imported "${AppState.importedModelInfo.name}" — ready for single and batch predictions`);
    } catch (e) {
        alert(`The model could not be imported: ${e.message || e}`);
    }
}

function updateModelExportButton() {
    const btn = document.getElementById('model-export-btn');
    if (btn) { btn.disabled = !AppState.trainedModel; btn.title = AppState.trainedModel ? 'Save the active model (with its fitted preprocessing) as a file' : 'Train or deploy a model first'; }
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
        dataset_name: AppState.fileName,
        target_variable: AppState.pipeline.targetCol,
        reproducibility_seed: AppState.pipeline.evalSeed,
        automatic_tuning: AppState.pipeline.tuningReport ? {
            method: `${AppState.pipeline.tuningReport.folds}-fold stratified cross-validation within the training set; SMOTENC and encoding/scaling refitted inside each fold; neural network at most 3 folds; test set not used`,
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
