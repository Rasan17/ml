// shell.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { downloadBatchCSV, executeBatchPrediction, handleBatchFileUpload, loadSampleBatchDataset } from './batch-prediction.js';
import { ALL_MODEL_IDS, AppState, MODEL_INFO } from './core.js';
import { updateSplitMethodVisibility } from './cv-tuning.js';
import { executeImputation } from './imputation.js';
import { loadSampleDataset } from './ingest.js';
import { downloadPipelineProtocolReport, exportTrainedModel, importTrainedModel } from './model-io.js';
import { goToStep, handleFileUpload, handleTestFileUpload, loadSampleTestingCohort, refreshOptionBSummaries, syncTrainTestSplit, toggleTheme } from './navigation.js';
import { goToPipelineSubstep, initPipelineUI, updatePipelineCompleteCaseInfo } from './pipeline.js';
import { executePipelineStep6 } from './pipeline-balance.js';
import { dropUnselectedUnivariateFeatures, filterSignificantUnivariateFeatures, pipelineDropSelectedVIF, renderPipelineStep2, renderPipelineStep3, selectAllUnivariateFeatures } from './pipeline-screening.js';
import { executePipelineStep1 } from './pipeline-step1.js';
import { executePipelineStep7 } from './pipeline-train.js';
import { autoDeployBestPipelineModel, togglePipelineBaselineComparison } from './pipeline-results.js';
import { downloadStandardAnalysisReport } from './reports.js';
import { downloadSettings, loadSettingsFile } from './settings.js';
import { runSinglePrediction } from './single-prediction.js';
import { startModelTraining } from './standard-training.js';
import { rerankExistingResults } from './training-actions.js';
import { setupSeedManagement, updateHyperparamsVisibility } from './training-setup.js';
import { setAllVariablesIncluded } from './variables.js';
import { ExecutionWatchdog } from './watchdog.js';


// DOM Elements
const DOM = {};

function initDOM() {
    DOM.themeToggle = document.getElementById('theme-toggle');
    DOM.stepTabs = document.querySelectorAll('.step-tab');
    DOM.stepContents = document.querySelectorAll('.step-content');
    DOM.fileInput = document.getElementById('csv-file-input');
    DOM.dropzone = document.getElementById('upload-dropzone');
    DOM.sampleCardioBtn = document.getElementById('sample-cardio-btn');
    DOM.sampleCohortBtn = document.getElementById('sample-cohort-btn');
    DOM.sampleRegressionBtn = document.getElementById('sample-regression-btn');
    DOM.dataStatsGrid = document.getElementById('data-stats-grid');
    DOM.dataPreviewContainer = document.getElementById('data-preview-container');
    DOM.toStep2Btn = document.getElementById('to-step-2-btn');
    
    DOM.varConfigGrid = document.getElementById('var-config-grid');
    DOM.toStep3Btn = document.getElementById('to-step-3-btn');
    
    DOM.imputeTableBody = document.getElementById('impute-table-body');
    DOM.applyImputationBtn = document.getElementById('apply-imputation-btn');
    DOM.imputeAuditBox = document.getElementById('impute-audit-box');
    DOM.toStep4Btn = document.getElementById('to-step-4-btn');
    
    // Model Selection & Scope Elements
    DOM.modelCards = document.querySelectorAll('.model-card');
    DOM.modelSelectCheckboxes = document.querySelectorAll('.model-select-checkbox');
    DOM.scopeSingle = document.getElementById('scope-single');
    DOM.scopeMulti = document.getElementById('scope-multi');
    DOM.scopeAll = document.getElementById('scope-all');
    DOM.scopeSingleLabel = document.getElementById('scope-single-label');
    DOM.scopeMultiLabel = document.getElementById('scope-multi-label');
    DOM.scopeAllLabel = document.getElementById('scope-all-label');
    DOM.btnSelectAllModels = document.getElementById('btn-select-all-models');
    DOM.btnSelectTopModels = document.getElementById('btn-select-top-models');
    DOM.btnClearModelSelection = document.getElementById('btn-clear-model-selection');
    DOM.activeHyperparamSelect = document.getElementById('active-hyperparam-select');

    DOM.trainModelBtn = document.getElementById('train-model-btn');
    DOM.trainStatus = document.getElementById('train-status');
    DOM.trainLossChartCanvas = document.getElementById('train-loss-chart');
    DOM.resultsDashboard = document.getElementById('results-dashboard');
    DOM.metricsSummaryRow = document.getElementById('metrics-summary-row');
    DOM.evaluationDetailContainer = document.getElementById('eval-detail-container');
    DOM.retrainStatusBanner = document.getElementById('retrain-status-banner');
    DOM.multiModelLeaderboardContainer = document.getElementById('multi-model-leaderboard-container');
    DOM.multiModelCmContainer = document.getElementById('multi-model-cm-container');
    DOM.dashboardSingleHeading = document.getElementById('dashboard-single-heading');

    DOM.toStep5Btn = document.getElementById('to-step-5-btn');
    DOM.toStep6Btn = document.getElementById('to-step-6-btn');
    
    // Option A vs Option B Protocol Elements
    DOM.valModeA = document.getElementById('val-mode-a');
    DOM.valModeB = document.getElementById('val-mode-b');
    DOM.cardModeA = document.getElementById('protocol-mode-a-label');
    DOM.cardModeB = document.getElementById('protocol-mode-b-label');
    DOM.optionAContainer = document.getElementById('option-a-container');
    DOM.optionBContainer = document.getElementById('option-b-container');
    DOM.retrainYes = document.getElementById('retrain-yes');
    DOM.retrainNo = document.getElementById('retrain-no');
    DOM.infoTrainPct = document.getElementById('info-train-pct');
    DOM.infoTestPct = document.getElementById('info-test-pct');
    DOM.infoTestPct2 = document.getElementById('info-test-pct-2');
    DOM.optBTrainSummary = document.getElementById('opt-b-train-summary');
    DOM.optBTestBadge = document.getElementById('opt-b-test-badge');
    DOM.optBTestSummary = document.getElementById('opt-b-test-summary');
    DOM.testDropzone = document.getElementById('test-dropzone');
    DOM.testFileInput = document.getElementById('test-csv-file-input');
    DOM.loadSampleTestBtn = document.getElementById('load-sample-test-btn');

    DOM.predictFormGrid = document.getElementById('predict-form-grid');
    DOM.runSinglePredictBtn = document.getElementById('run-single-predict-btn');
    DOM.singleResultBanner = document.getElementById('single-result-banner');
    
    DOM.batchFileInput = document.getElementById('batch-file-input');
    DOM.batchDropzone = document.getElementById('batch-dropzone');
    DOM.sampleBatchBtn = document.getElementById('sample-batch-btn');
    DOM.runBatchPredictBtn = document.getElementById('run-batch-predict-btn');
    DOM.batchResultsCard = document.getElementById('batch-results-card');
    DOM.batchTableContainer = document.getElementById('batch-table-container');
    DOM.downloadBatchCsvBtn = document.getElementById('download-batch-csv-btn');

    // 7-Step Clinical ML Pipeline DOM References
    DOM.tabPipeline = document.getElementById('tab-step-pipeline');
    DOM.pipelineDatasetName = document.getElementById('pipeline-dataset-name');
    DOM.pipelineLoadCardioBtn = document.getElementById('pipeline-load-cardio-btn');
    DOM.pipelineLoadShuntBtn = document.getElementById('pipeline-load-shunt-btn');
    DOM.pipelineSubPills = document.querySelectorAll('.pipeline-step-pill');
    DOM.pipelinePanels = document.querySelectorAll('.pipeline-step-panel');
    DOM.pipelineSeedInput = document.getElementById('pipeline-seed-input');
    DOM.pipelineSeedRandomBtn = document.getElementById('pipeline-seed-random-btn');
    DOM.pipelineImputeStrategy = document.getElementById('pipeline-impute-strategy');
    DOM.pipelineStep1RunBtn = document.getElementById('pipeline-step1-run-btn');
    DOM.pipelineStep1Results = document.getElementById('pipeline-step1-results');
    DOM.pipelineTrainCount = document.getElementById('pipeline-train-count');
    DOM.pipelineTestCount = document.getElementById('pipeline-test-count');
    DOM.pipelineMissingImputedCount = document.getElementById('pipeline-missing-imputed-count');
    DOM.pipelineImputerTableContainer = document.getElementById('pipeline-imputer-table-container');
    DOM.pipelineToStep2Btn = document.getElementById('pipeline-to-step2-btn');
    DOM.pipelineNormalityTableContainer = document.getElementById('pipeline-normality-table-container');
    DOM.pipelineCorrPairsContainer = document.getElementById('pipeline-corr-pairs-container');
    DOM.pipelineCorrPairCounter = document.getElementById('pipeline-corr-pair-counter');
    DOM.pipelineRunCorrBtn = document.getElementById('pipeline-run-corr-btn');
    DOM.pipelineToStep3Btn = document.getElementById('pipeline-to-step3-btn');
    DOM.pipelineVifTableContainer = document.getElementById('pipeline-vif-table-container');
    DOM.pipelineDropVifBtn = document.getElementById('pipeline-drop-vif-btn');
    DOM.pipelineRunVifBtn = document.getElementById('pipeline-run-vif-btn');
    DOM.pipelineToStep4Btn = document.getElementById('pipeline-to-step4-btn');
    DOM.pipelineUnivariateTableContainer = document.getElementById('pipeline-univariate-table-container');
    DOM.pipelineFilterSignificantBtn = document.getElementById('pipeline-filter-significant-btn');
    DOM.pipelineSelectAllUnivBtn = document.getElementById('pipeline-select-all-univ-btn');
    DOM.pipelineDropUnselectedUnivBtn = document.getElementById('pipeline-drop-unselected-univ-btn');
    DOM.pipelineUnivSelectedSummary = document.getElementById('pipeline-univ-selected-summary');
    DOM.pipelineToStep5Btn = document.getElementById('pipeline-to-step5-btn');
    DOM.pipelineSmotencContainer = document.getElementById('pipeline-smotenc-container');
    DOM.pipelineToStep6Btn = document.getElementById('pipeline-to-step6-btn');
    DOM.pipelineRunRfScoutBtn = document.getElementById('pipeline-run-rf-scout-btn');
    DOM.pipelineRfImportanceContainer = document.getElementById('pipeline-rf-importance-container');
    DOM.pipelineToStep7Btn = document.getElementById('pipeline-to-step7-btn');
    DOM.pipelineEvalSeedInput = document.getElementById('pipeline-eval-seed-input');
    DOM.pipelineTrainModelsBtn = document.getElementById('pipeline-train-models-btn');
    DOM.pipelineTrainStatus = document.getElementById('pipeline-train-status');
    DOM.pipelineTrainStatusText = document.getElementById('pipeline-train-status-text');
    DOM.pipelineResultsContainer = document.getElementById('pipeline-results-container');
    DOM.pipelineLeaderboardContainer = document.getElementById('pipeline-leaderboard-container');
    DOM.pipelineCmGrid = document.getElementById('pipeline-cm-grid');
    DOM.pipelineDownloadReportBtn = document.getElementById('pipeline-download-report-btn');
    DOM.pipelineDirectLaunchBtn = document.getElementById('pipeline-direct-launch-btn');
    DOM.singlePredictProvenanceBar = document.getElementById('single-predict-provenance-bar');
    DOM.batchPredictProvenanceBar = document.getElementById('batch-predict-provenance-bar');
    DOM.pipelineActiveDeploymentStatus = document.getElementById('pipeline-active-deployment-status');
    DOM.pipelineToSinglePredictBtn = document.getElementById('pipeline-to-single-predict-btn');
    DOM.pipelineToBatchPredictBtn = document.getElementById('pipeline-to-batch-predict-btn');
    DOM.pipelineCompareBaselineBtn = document.getElementById('pipeline-compare-baseline-btn');
    DOM.pipelineVsBaselinePanel = document.getElementById('pipeline-vs-baseline-panel');
    DOM.pipelineVsBaselineContent = document.getElementById('pipeline-vs-baseline-content');

    // Step 7 Pipeline Watchdog
    DOM.pipelineMonitor = {
        container: document.getElementById('pipeline-train-status'),
        badge: document.getElementById('pipeline-monitor-badge'),
        stateText: document.getElementById('pipeline-monitor-state'),
        timerText: document.getElementById('pipeline-monitor-timer'),
        loopStatus: document.getElementById('pipeline-monitor-loop-status'),
        loopText: document.getElementById('pipeline-monitor-loop-text'),
        taskName: document.getElementById('pipeline-train-status-text'),
        stepCounter: document.getElementById('pipeline-monitor-step-counter'),
        progressBar: document.getElementById('pipeline-monitor-progress-bar'),
        threadStatus: document.getElementById('pipeline-monitor-thread-status'),
        heartbeatTick: document.getElementById('pipeline-monitor-heartbeat-tick'),
        stuckHint: document.getElementById('pipeline-monitor-stuck-hint'),
        cancelBtn: document.getElementById('pipeline-monitor-cancel-btn')
    };

    // Step 4 Model Training Watchdog
    DOM.step4Monitor = {
        container: document.getElementById('step4-train-monitor'),
        badge: document.getElementById('step4-monitor-badge'),
        stateText: document.getElementById('step4-monitor-state'),
        timerText: document.getElementById('step4-monitor-timer'),
        loopStatus: document.getElementById('step4-monitor-loop-status'),
        loopText: document.getElementById('step4-monitor-loop-text'),
        taskName: document.getElementById('step4-monitor-task-name'),
        stepCounter: document.getElementById('step4-monitor-step-counter'),
        progressBar: document.getElementById('step4-monitor-progress-bar'),
        threadStatus: document.getElementById('step4-monitor-thread-status'),
        heartbeatTick: document.getElementById('step4-monitor-heartbeat-tick'),
        stuckHint: document.getElementById('step4-monitor-stuck-hint'),
        cancelBtn: document.getElementById('step4-monitor-cancel-btn')
    };

    AppState.watchdogs = {
        pipeline: new ExecutionWatchdog(DOM.pipelineMonitor),
        step4: new ExecutionWatchdog(DOM.step4Monitor)
    };
}

function setupEventListeners() {
    // Theme toggle
    DOM.themeToggle.addEventListener('click', toggleTheme);

    // Stepper Navigation
    DOM.stepTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const step = tab.dataset.step;
            goToStep(step);
        });
    });

    // 7-Step Pipeline Sub-stepper Nav
    if (DOM.pipelineSubPills) {
        DOM.pipelineSubPills.forEach(pill => {
            pill.addEventListener('click', () => {
                const substep = parseInt(pill.dataset.substep, 10);
                goToPipelineSubstep(substep);
            });
        });
    }

    // Pipeline Sample Cohort Quick Loaders
    if (DOM.pipelineLoadCardioBtn) {
        DOM.pipelineLoadCardioBtn.addEventListener('click', async () => {
            await loadSampleDataset('universal_medical_cohort.csv', 'Cardiometabolic Risk Cohort (Binary)');
            initPipelineUI();
        });
    }
    if (DOM.pipelineLoadShuntBtn) {
        DOM.pipelineLoadShuntBtn.addEventListener('click', async () => {
            await loadSampleDataset('medical_shunt_cohort.csv', 'Neurosurgical Shunt Response Cohort');
            initPipelineUI();
        });
    }

    if (DOM.pipelineSeedRandomBtn) {
        DOM.pipelineSeedRandomBtn.addEventListener('click', () => {
            const r = Math.floor(Math.random() * 90000) + 1000;
            DOM.pipelineSeedInput.value = r;
            if (DOM.pipelineEvalSeedInput) DOM.pipelineEvalSeedInput.value = r;
        });
    }

    // Pipeline Step Action Buttons
    if (DOM.pipelineStep1RunBtn) DOM.pipelineStep1RunBtn.addEventListener('click', executePipelineStep1);
    const pipelineCompleteCase = document.getElementById('pipeline-complete-case');
    if (pipelineCompleteCase) pipelineCompleteCase.addEventListener('change', updatePipelineCompleteCaseInfo);
    if (DOM.pipelineToStep2Btn) DOM.pipelineToStep2Btn.addEventListener('click', () => goToPipelineSubstep(2));
    if (DOM.pipelineRunCorrBtn) DOM.pipelineRunCorrBtn.addEventListener('click', renderPipelineStep2);
    if (DOM.pipelineToStep3Btn) DOM.pipelineToStep3Btn.addEventListener('click', () => goToPipelineSubstep(3));
    if (DOM.pipelineRunVifBtn) DOM.pipelineRunVifBtn.addEventListener('click', renderPipelineStep3);
    if (DOM.pipelineDropVifBtn) DOM.pipelineDropVifBtn.addEventListener('click', pipelineDropSelectedVIF);
    if (DOM.pipelineToStep4Btn) DOM.pipelineToStep4Btn.addEventListener('click', () => goToPipelineSubstep(4));
    if (DOM.pipelineFilterSignificantBtn) DOM.pipelineFilterSignificantBtn.addEventListener('click', filterSignificantUnivariateFeatures);
    if (DOM.pipelineSelectAllUnivBtn) DOM.pipelineSelectAllUnivBtn.addEventListener('click', selectAllUnivariateFeatures);
    if (DOM.pipelineDropUnselectedUnivBtn) DOM.pipelineDropUnselectedUnivBtn.addEventListener('click', dropUnselectedUnivariateFeatures);
    if (DOM.pipelineToStep5Btn) DOM.pipelineToStep5Btn.addEventListener('click', () => goToPipelineSubstep(5));
    if (DOM.pipelineToStep6Btn) DOM.pipelineToStep6Btn.addEventListener('click', () => goToPipelineSubstep(6));
    if (DOM.pipelineRunRfScoutBtn) DOM.pipelineRunRfScoutBtn.addEventListener('click', executePipelineStep6);
    if (DOM.pipelineToStep7Btn) DOM.pipelineToStep7Btn.addEventListener('click', () => goToPipelineSubstep(7));
    if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.addEventListener('click', executePipelineStep7);
    if (DOM.pipelineDownloadReportBtn) DOM.pipelineDownloadReportBtn.addEventListener('click', downloadPipelineProtocolReport);
    document.getElementById('standard-download-report-btn')?.addEventListener('click', downloadStandardAnalysisReport);
    document.getElementById('model-export-btn')?.addEventListener('click', exportTrainedModel);
    document.getElementById('model-import-btn')?.addEventListener('click', () => document.getElementById('model-file-input').click());
    document.getElementById('model-file-input')?.addEventListener('change', e => {
        const f = e.target.files && e.target.files[0];
        if (f) importTrainedModel(f);
        e.target.value = '';
    });
    if (DOM.pipelineDirectLaunchBtn) {
        DOM.pipelineDirectLaunchBtn.addEventListener('click', (e) => window.launch7StepPipeline(e));
    }
    if (DOM.pipelineToSinglePredictBtn) {
        DOM.pipelineToSinglePredictBtn.addEventListener('click', () => {
            if (!AppState.trainedModel || AppState.deployedSource !== '7step') {
                autoDeployBestPipelineModel();
            }
            goToStep(5);
        });
    }
    if (DOM.pipelineToBatchPredictBtn) {
        DOM.pipelineToBatchPredictBtn.addEventListener('click', () => {
            if (!AppState.trainedModel || AppState.deployedSource !== '7step') {
                autoDeployBestPipelineModel();
            }
            goToStep(6);
        });
    }
    if (DOM.pipelineCompareBaselineBtn) {
        DOM.pipelineCompareBaselineBtn.addEventListener('click', togglePipelineBaselineComparison);
    }

    // Next Step Buttons
    DOM.toStep2Btn.addEventListener('click', () => {
        if (!AppState.parsedData || !AppState.parsedData.rows || AppState.parsedData.rows.length === 0) {
            alert("Please select a sample medical cohort or upload a clinical CSV first.");
            return;
        }
        goToStep(2);
    });
    DOM.toStep3Btn.addEventListener('click', () => goToStep(3));
    ['split-method', 'pipeline-split-method'].forEach(id => document.getElementById(id)?.addEventListener('change', updateSplitMethodVisibility));
    document.getElementById('var-include-all-btn').addEventListener('click', () => { if (AppState.parsedData) setAllVariablesIncluded(true); });
    document.getElementById('var-exclude-all-btn').addEventListener('click', () => { if (AppState.parsedData) setAllVariablesIncluded(false); });
    DOM.toStep4Btn.addEventListener('click', () => goToStep(4));
    DOM.toStep5Btn.addEventListener('click', () => goToStep(5));
    DOM.toStep6Btn.addEventListener('click', () => goToStep(6));

    // CSV Drag and Drop
    DOM.dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        DOM.dropzone.classList.add('drag-active');
    });
    DOM.dropzone.addEventListener('dragleave', () => DOM.dropzone.classList.remove('drag-active'));
    DOM.dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        DOM.dropzone.classList.remove('drag-active');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    DOM.dropzone.addEventListener('click', () => DOM.fileInput.click());
    DOM.fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Sample Data Loaders
    document.getElementById('sample-survival-btn')?.addEventListener('click', async () => {
        await loadSampleDataset('medical_shunt_revision_survival.csv', 'Shunt Revision Time-to-Event Cohort (simulated, 320 patients)');
        if (AppState.parsedData) goToStep('survival');
    });
    document.getElementById('sample-breast-btn')?.addEventListener('click', () => loadSampleDataset('breast_cancer_wisconsin.csv', 'Breast Cancer Wisconsin Diagnostic (public UCI data, 569 patients)'));
    if (DOM.sampleCardioBtn) {
        DOM.sampleCardioBtn.addEventListener('click', () => loadSampleDataset('universal_medical_cohort.csv', 'Cardiometabolic Risk Cohort (Binary)'));
    }
    DOM.sampleCohortBtn.addEventListener('click', () => loadSampleDataset('medical_shunt_cohort.csv', 'Neurosurgical Shunt Response Cohort'));
    DOM.sampleRegressionBtn.addEventListener('click', () => loadSampleDataset('medical_recovery_score.csv', 'Functional Recovery Score (Continuous Regression)'));

    // Imputation execution
    DOM.applyImputationBtn.addEventListener('click', () => {
        // Complete data: still build the validated dataset Step 4 trains from, then go straight on
        if (DOM.applyImputationBtn.dataset.mode === 'complete') {
            executeImputation();
            if (AppState.imputedData) goToStep(4);
        } else {
            executeImputation();
        }
    });

    // =========================================================================
    // MODEL TESTING SCOPE & SELECTION (Single, Multi, All Models)
    // =========================================================================
    function setModelScope(scope) {
        AppState.modelScope = scope;
        if (DOM.scopeSingle) DOM.scopeSingle.checked = (scope === 'single');
        if (DOM.scopeMulti) DOM.scopeMulti.checked = (scope === 'multi');
        if (DOM.scopeAll) DOM.scopeAll.checked = (scope === 'all');

        if (DOM.scopeSingleLabel) DOM.scopeSingleLabel.classList.toggle('active-single', scope === 'single');
        if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.classList.toggle('active-multi', scope === 'multi');
        if (DOM.scopeAllLabel) DOM.scopeAllLabel.classList.toggle('active-all', scope === 'all');

        if (scope === 'single') {
            const singleId = AppState.selectedModelId || AppState.selectedModelIds[0] || 'nn';
            AppState.selectedModelIds = [singleId];
            AppState.selectedModelId = singleId;
            AppState.activeHyperparamModelId = singleId;
        } else if (scope === 'all') {
            AppState.selectedModelIds = ALL_MODEL_IDS.slice();
        } else if (scope === 'multi') {
            if (AppState.selectedModelIds.length <= 1) {
                AppState.selectedModelIds = ['ensemble', 'rf', 'nn'];
            }
        }
        syncModelCardSelections();
        updateTrainButtonLabel();
    }

    function syncModelCardSelections() {
        DOM.modelCards.forEach(card => {
            const mId = card.dataset.modelId;
            const isSelected = AppState.selectedModelIds.includes(mId);
            card.classList.toggle('selected', isSelected);
            const chk = card.querySelector('.model-select-checkbox');
            if (chk) chk.checked = isSelected;
        });

        if (DOM.activeHyperparamSelect) {
            if (AppState.selectedModelIds.includes(AppState.activeHyperparamModelId)) {
                DOM.activeHyperparamSelect.value = AppState.activeHyperparamModelId;
            } else if (AppState.selectedModelIds.length > 0) {
                AppState.activeHyperparamModelId = AppState.selectedModelIds[0];
                DOM.activeHyperparamSelect.value = AppState.activeHyperparamModelId;
            }
        }
        updateHyperparamsVisibility(AppState.activeHyperparamModelId);
    }

    function toggleModelSelection(mId) {
        if (AppState.modelScope === 'single') {
            AppState.selectedModelId = mId;
            AppState.selectedModelIds = [mId];
            AppState.activeHyperparamModelId = mId;
        } else {
            const idx = AppState.selectedModelIds.indexOf(mId);
            if (idx >= 0) {
                if (AppState.selectedModelIds.length > 1) {
                    AppState.selectedModelIds.splice(idx, 1);
                } else {
                    // Prevent deselecting last remaining model
                    return;
                }
            } else {
                AppState.selectedModelIds.push(mId);
            }
            AppState.activeHyperparamModelId = mId;

            if (AppState.selectedModelIds.length === ALL_MODEL_IDS.length) {
                AppState.modelScope = 'all';
            } else if (AppState.selectedModelIds.length === 1) {
                AppState.modelScope = 'single';
                AppState.selectedModelId = AppState.selectedModelIds[0];
            } else {
                AppState.modelScope = 'multi';
            }
            if (DOM.scopeSingle) DOM.scopeSingle.checked = (AppState.modelScope === 'single');
            if (DOM.scopeMulti) DOM.scopeMulti.checked = (AppState.modelScope === 'multi');
            if (DOM.scopeAll) DOM.scopeAll.checked = (AppState.modelScope === 'all');
            if (DOM.scopeSingleLabel) DOM.scopeSingleLabel.classList.toggle('active-single', AppState.modelScope === 'single');
            if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.classList.toggle('active-multi', AppState.modelScope === 'multi');
            if (DOM.scopeAllLabel) DOM.scopeAllLabel.classList.toggle('active-all', AppState.modelScope === 'all');
        }
        syncModelCardSelections();
        updateTrainButtonLabel();
    }

    function updateTrainButtonLabel() {
        if (!DOM.trainModelBtn) return;
        const count = AppState.selectedModelIds.length;
        if (count === 1) {
            const mInfo = MODEL_INFO[AppState.selectedModelIds[0]] || { short: AppState.selectedModelIds[0].toUpperCase() };
            DOM.trainModelBtn.textContent = `🚀 Train & Test 1 Model (${mInfo.short})`;
        } else if (count === ALL_MODEL_IDS.length) {
            DOM.trainModelBtn.textContent = `🏆 Run Full Tournament (Train & Test All ${ALL_MODEL_IDS.length} Models)`;
        } else {
            DOM.trainModelBtn.textContent = `🚀 Train & Benchmark ${count} Selected Models`;
        }
    }

    // Scope Radio Changes & Label Clicks
    if (DOM.scopeSingle) DOM.scopeSingle.addEventListener('change', () => setModelScope('single'));
    if (DOM.scopeMulti) DOM.scopeMulti.addEventListener('change', () => setModelScope('multi'));
    if (DOM.scopeAll) DOM.scopeAll.addEventListener('change', () => setModelScope('all'));
    if (DOM.scopeSingleLabel) DOM.scopeSingleLabel.addEventListener('click', () => setModelScope('single'));
    if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.addEventListener('click', () => setModelScope('multi'));
    if (DOM.scopeAllLabel) DOM.scopeAllLabel.addEventListener('click', () => setModelScope('all'));

    // Preset Quick Action Buttons
    if (DOM.btnSelectAllModels) {
        DOM.btnSelectAllModels.addEventListener('click', () => setModelScope('all'));
    }
    if (DOM.btnSelectTopModels) {
        DOM.btnSelectTopModels.addEventListener('click', () => {
            AppState.modelScope = 'multi';
            AppState.selectedModelIds = ['ensemble', 'rf', 'nn'];
            if (DOM.scopeMulti) DOM.scopeMulti.checked = true;
            if (DOM.scopeSingleLabel) DOM.scopeSingleLabel.classList.remove('active-single');
            if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.classList.add('active-multi');
            if (DOM.scopeAllLabel) DOM.scopeAllLabel.classList.remove('active-all');
            syncModelCardSelections();
            updateTrainButtonLabel();
        });
    }
    if (DOM.btnClearModelSelection) {
        DOM.btnClearModelSelection.addEventListener('click', () => setModelScope('single'));
    }

    // Model Card Clicks & Checkbox Changes
    DOM.modelCards.forEach(card => {
        const mId = card.dataset.modelId;
        const chk = card.querySelector('.model-select-checkbox');

        card.addEventListener('click', (e) => {
            if (e.target.closest('.model-seed-box')) return;
            if (e.target === chk) return; // checkbox change will handle it
            toggleModelSelection(mId);
        });

        if (chk) {
            chk.addEventListener('change', (e) => {
                e.stopPropagation();
                toggleModelSelection(mId);
            });
        }
    });

    // Hyperparameter model selector dropdown
    if (DOM.activeHyperparamSelect) {
        DOM.activeHyperparamSelect.addEventListener('change', (e) => {
            AppState.activeHyperparamModelId = e.target.value;
            updateHyperparamsVisibility(e.target.value);
        });
    }

    // Protocol Mode Toggle (Option A: split, Option B: separate test file, Option C: bootstrap internal validation)
    function setValidationMode(mode) {
        AppState.validationMode = mode;
        const cards = [
            { mode: 'split', radio: 'val-mode-a', card: 'protocol-mode-a-label', box: 'option-a-container', colour: 'var(--accent-cyan)', bg: 'rgba(6, 182, 212, 0.08)' },
            { mode: 'separate', radio: 'val-mode-b', card: 'protocol-mode-b-label', box: 'option-b-container', colour: 'var(--accent-purple)', bg: 'rgba(168, 85, 247, 0.08)' },
            { mode: 'bootstrap', radio: 'val-mode-c', card: 'protocol-mode-c-label', box: 'option-c-container', colour: 'var(--accent-emerald)', bg: 'rgba(16, 185, 129, 0.08)' }
        ];
        cards.forEach(c => {
            const on = c.mode === mode;
            const radio = document.getElementById(c.radio), card = document.getElementById(c.card), box = document.getElementById(c.box);
            if (radio && on) radio.checked = true;
            if (card) { card.style.borderColor = on ? c.colour : 'var(--border-color)'; card.style.background = on ? c.bg : 'var(--bg-surface-subtle)'; card.classList.toggle('selected', on); }
            if (box) box.style.display = on ? 'block' : 'none';
        });
        if (mode === 'separate') refreshOptionBSummaries();
    }

    [['val-mode-a', 'split'], ['val-mode-b', 'separate'], ['val-mode-c', 'bootstrap']].forEach(([id, mode]) => {
        document.getElementById(id)?.addEventListener('change', () => setValidationMode(mode));
    });
    [['protocol-mode-a-label', 'split'], ['protocol-mode-b-label', 'separate'], ['protocol-mode-c-label', 'bootstrap']].forEach(([id, mode]) => {
        document.getElementById(id)?.addEventListener('click', () => setValidationMode(mode));
    });

    // Option B Testing Cohort Upload
    if (DOM.testDropzone && DOM.testFileInput) {
        DOM.testDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            DOM.testDropzone.classList.add('drag-active');
        });
        DOM.testDropzone.addEventListener('dragleave', () => DOM.testDropzone.classList.remove('drag-active'));
        DOM.testDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            DOM.testDropzone.classList.remove('drag-active');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleTestFileUpload(e.dataTransfer.files[0]);
            }
        });
        DOM.testDropzone.addEventListener('click', () => DOM.testFileInput.click());
        DOM.testFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleTestFileUpload(e.target.files[0]);
            }
        });
    }

    if (DOM.loadSampleTestBtn) {
        DOM.loadSampleTestBtn.addEventListener('click', loadSampleTestingCohort);
    }

    // Train / Test Split interactive synchronization (Option A)
    const slider = document.getElementById('train-ratio-slider');
    const trainInput = document.getElementById('train-pct-input');
    const testInput = document.getElementById('test-pct-input');
    if (slider) {
        slider.addEventListener('input', (e) => syncTrainTestSplit(parseFloat(e.target.value)));
    }
    if (trainInput) {
        trainInput.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (!isNaN(val)) syncTrainTestSplit(val);
        });
    }
    if (testInput) {
        testInput.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (!isNaN(val)) syncTrainTestSplit(100 - val);
        });
    }
    document.querySelectorAll('.preset-split-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const trainVal = parseFloat(btn.dataset.train);
            if (!isNaN(trainVal)) syncTrainTestSplit(trainVal);
        });
    });

    // Train Model
    DOM.trainModelBtn.addEventListener('click', startModelTraining);

    // Primary ranking metric (shared by Step 4 and the 7-step pipeline)
    document.querySelectorAll('.ranking-metric-select').forEach(sel => {
        sel.addEventListener('change', (e) => {
            const [kind, key] = e.target.value.split(':');
            if (kind === 'clf') AppState.rankingMetricClf = key;
            else AppState.rankingMetricReg = key;
            // Keep both selectors in sync (each <select> shows one selected option)
            document.querySelectorAll('.ranking-metric-select').forEach(other => { other.value = e.target.value; });
            rerankExistingResults();
        });
    });

    // Reproducibility Seed Management (Default 42 vs Custom Seed)
    setupSeedManagement();

    // Single Prediction
    DOM.runSinglePredictBtn.addEventListener('click', runSinglePrediction);

    // Batch Prediction
    DOM.batchDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        DOM.batchDropzone.classList.add('drag-active');
    });
    DOM.batchDropzone.addEventListener('dragleave', () => DOM.batchDropzone.classList.remove('drag-active'));
    DOM.batchDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        DOM.batchDropzone.classList.remove('drag-active');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleBatchFileUpload(e.dataTransfer.files[0]);
        }
    });
    DOM.batchDropzone.addEventListener('click', () => DOM.batchFileInput.click());
    DOM.batchFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleBatchFileUpload(e.target.files[0]);
        }
    });
    DOM.sampleBatchBtn.addEventListener('click', loadSampleBatchDataset);
    DOM.runBatchPredictBtn.addEventListener('click', executeBatchPrediction);
    DOM.downloadBatchCsvBtn.addEventListener('click', downloadBatchCSV);

    // Initial model-scope sync (setModelScope is local to this function)
    setModelScope('single');

    // Used by "Load settings" to restore the chosen models and scope
    AppState.applyModelSelection = (scope, ids) => {
        const valid = (ids || []).filter(id => ALL_MODEL_IDS.includes(id));
        if (valid.length === 0) return;
        AppState.selectedModelIds = valid.slice();
        AppState.selectedModelId = valid[0];
        AppState.activeHyperparamModelId = valid[0];
        AppState.modelScope = valid.length === ALL_MODEL_IDS.length ? 'all' : (valid.length === 1 ? 'single' : (scope || 'multi'));
        if (DOM.scopeSingle) DOM.scopeSingle.checked = AppState.modelScope === 'single';
        if (DOM.scopeMulti) DOM.scopeMulti.checked = AppState.modelScope === 'multi';
        if (DOM.scopeAll) DOM.scopeAll.checked = AppState.modelScope === 'all';
        if (DOM.scopeSingleLabel) DOM.scopeSingleLabel.classList.toggle('active-single', AppState.modelScope === 'single');
        if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.classList.toggle('active-multi', AppState.modelScope === 'multi');
        if (DOM.scopeAllLabel) DOM.scopeAllLabel.classList.toggle('active-all', AppState.modelScope === 'all');
        syncModelCardSelections();
        updateTrainButtonLabel();
    };

    // Analysis settings bar
    const settingsSaveBtn = document.getElementById('settings-save-btn');
    const settingsLoadBtn = document.getElementById('settings-load-btn');
    const settingsFileInput = document.getElementById('settings-file-input');
    if (settingsSaveBtn) settingsSaveBtn.addEventListener('click', downloadSettings);
    if (settingsLoadBtn && settingsFileInput) {
        settingsLoadBtn.addEventListener('click', () => settingsFileInput.click());
        settingsFileInput.addEventListener('change', (e) => {
            const file = e.target.files && e.target.files[0];
            if (file) loadSettingsFile(file);
            e.target.value = '';
        });
    }
}

export { DOM, initDOM, setupEventListeners };
