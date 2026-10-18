// pipeline-step1.js: part of the Universal Medical ML/NeuralNet Predictor application controller (split from pipeline.js)
import { AppState, escapeHtml, fmtValue, parseSeed } from './core.js';
import { describeSplit, splitRowsForWorkflow } from './cv-tuning.js';
import { updatePipelineActiveDeploymentStatus, updatePredictionProvenanceBar } from './pipeline-results.js';
import { DOM } from './shell.js';
import { disposeModels } from './training-actions.js';
// =========================================================================
// STEP 1: Train/Test Split (80/20) & Strict Leak-Free Imputation
// =========================================================================
function executePipelineStep1() {
    if (!AppState.parsedData || !AppState.parsedData.rows || AppState.parsedData.rows.length === 0) {
        alert("Please upload a CSV dataset or select a sample medical cohort first.");
        return;
    }

    // Identify target column
    let targetCol = Object.keys(AppState.columnConfigs).find(col => AppState.columnConfigs[col].role === 'target');
    if (!targetCol && AppState.parsedData.headers.length > 0) {
        targetCol = AppState.parsedData.headers[AppState.parsedData.headers.length - 1];
        if (AppState.columnConfigs[targetCol]) AppState.columnConfigs[targetCol].role = 'target';
    }

    if (!targetCol) {
        alert("Please select a target variable for prediction.");
        return;
    }

    // Identify feature columns
    const featureCols = Object.keys(AppState.columnConfigs).filter(col =>
        col !== targetCol && AppState.columnConfigs[col].role !== 'meta'
    );

    if (featureCols.length === 0) {
        alert("No predictor feature columns found. Ensure at least one column is configured as a feature.");
        return;
    }

    // Read configuration
    const userSeed = DOM.pipelineSeedInput ? parseSeed(DOM.pipelineSeedInput.value) : 42;
    AppState.pipeline.seed = userSeed === null ? 42 : userSeed;
    if (DOM.pipelineSeedInput) DOM.pipelineSeedInput.value = AppState.pipeline.seed; // show the seed actually used
    AppState.pipeline.evalSeed = AppState.pipeline.seed;
    if (DOM.pipelineEvalSeedInput) DOM.pipelineEvalSeedInput.value = AppState.pipeline.seed;

    const imputeStrategy = DOM.pipelineImputeStrategy ? DOM.pipelineImputeStrategy.value : 'median';
    AppState.pipeline.imputeStrategy = imputeStrategy;

    // Rows with a missing (or, for regression, non-numeric) outcome are excluded: outcomes are never imputed
    const isClassification = AppState.columnConfigs[targetCol]?.type !== 'continuous';
    let usableRows = AppState.parsedData.rows.filter(r => !DataEngine.isMissing(r[targetCol]) &&
        (isClassification || !isNaN(Number(r[targetCol]))));
    const excludedMissingTarget = AppState.parsedData.rows.length - usableRows.length;

    // Optional complete-case analysis: drop every row with an empty predictor cell BEFORE splitting,
    // so the training and testing sets both contain complete rows only (each row is judged on its own
    // values, so this introduces no train/test leakage)
    const completeCaseEl = document.getElementById('pipeline-complete-case');
    const completeCase = Boolean(completeCaseEl && completeCaseEl.checked);
    let excludedIncomplete = 0;
    if (completeCase) {
        const before = usableRows.length;
        usableRows = usableRows.filter(r => featureCols.every(c => !DataEngine.isMissing(r[c])));
        excludedIncomplete = before - usableRows.length;
    }
    if (usableRows.length < 5) {
        alert(completeCase
            ? `Only ${usableRows.length} complete row(s) remain after removing rows with an empty cell. At least 5 are needed for a train/test split; untick the complete-case option to impute instead.`
            : `Only ${usableRows.length} row(s) have a valid outcome for '${targetCol}'. At least 5 are needed for a train/test split.`);
        return;
    }

    // Train/test split at the selected ratio (default 80/20; stratified by outcome class for classification)
    const ratioEl = document.getElementById('pipeline-split-ratio');
    const pipeTrainRatio = ratioEl && !isNaN(parseFloat(ratioEl.value)) ? Math.min(0.9, Math.max(0.6, parseFloat(ratioEl.value))) : 0.8;
    AppState.pipeline.trainRatio = pipeTrainRatio;
    const targetVals = usableRows.map(r => String(r[targetCol]).trim());

    let splitResult;
    try {
        splitResult = splitRowsForWorkflow(usableRows, targetVals, pipeTrainRatio, isClassification, AppState.pipeline.seed, 'pipeline');
    } catch (err) {
        alert(err.message);
        return;
    }
    AppState.pipeline.splitInfo = splitResult.info;

    const trainRaw = splitResult.trainX;
    const testRaw = splitResult.testX;
    AppState.pipeline.testRawRows = testRaw; // raw patient records, used to tell whether the comparison is paired

    // Fit strict imputer ONLY on the Training set (Zero Data Leakage)
    const imputer = DataEngine.fitStrictImputer(
        trainRaw,
        featureCols,
        targetCol,
        AppState.columnConfigs,
        imputeStrategy
    );

    // Transform both Training and Testing sets with the fitted imputer
    const trainTransformed = DataEngine.transformWithStrictImputer(trainRaw, imputer, AppState.columnConfigs);
    const testTransformed = DataEngine.transformWithStrictImputer(testRaw, imputer, AppState.columnConfigs);

    // Save pipeline state
    AppState.pipeline.trainRows = trainTransformed.rows;
    AppState.pipeline.trainRawRows = trainRaw; // same patients and order, before imputation (for leak-free CV folds)
    AppState.pipeline.testRows = testTransformed.rows;
    AppState.pipeline.imputer = imputer;
    AppState.pipeline.featureCols = featureCols;
    AppState.pipeline.targetCol = targetCol;
    AppState.pipeline.columnConfigs = { ...AppState.columnConfigs };
    AppState.pipeline.excludedMissingTarget = excludedMissingTarget;
    AppState.pipeline.completeCase = completeCase;
    AppState.pipeline.excludedIncomplete = excludedIncomplete;
    // Re-running Step 1 invalidates every downstream result
    AppState.pipeline.droppedFeatures = new Set();
    AppState.pipeline.normalityMap = {};
    AppState.pipeline.corrPairs = [];
    AppState.pipeline.vifResults = [];
    AppState.pipeline.univariateResults = [];
    AppState.pipeline.smotencResult = null;
    AppState.pipeline.balancedTrainRows = null;
    AppState.pipeline.rfImportance = [];
    AppState.pipeline.top70Features = [];
    AppState.pipeline.bottom30Features = [];
    AppState.pipeline.reinstatedFeatures = new Set();
    AppState.pipeline.rfTopCount = 0;
    // Old pipeline models are discarded by a new Step 1 (the deployed one is kept until replaced)
    disposeModels((AppState.pipeline.benchmarkResults || []).map(b => b.model));
    AppState.pipeline.benchmarkResults = [];
    if (DOM.pipelineResultsContainer) DOM.pipelineResultsContainer.style.display = 'none';
    if (DOM.pipelineRfImportanceContainer) DOM.pipelineRfImportanceContainer.innerHTML = '';
    if (DOM.pipelineToStep7Btn) DOM.pipelineToStep7Btn.disabled = true;
    if (DOM.pipelineVsBaselinePanel) DOM.pipelineVsBaselinePanel.style.display = 'none';
    if (AppState.deployedSource === '7step') {
        AppState.trainedModel = null;
        AppState.prepMeta = null;
        AppState.evalResults = null;
        AppState.deployedSource = null;
        AppState.deployedPipelineModel = null;
        updatePipelineActiveDeploymentStatus();
        updatePredictionProvenanceBar();
    }

    renderPipelineStep1Results(trainTransformed.missingFilledCount, testTransformed.missingFilledCount);
    const removalNotes = [];
    if (excludedMissingTarget > 0) removalNotes.push(`${excludedMissingTarget} row(s) with a missing outcome ('${targetCol}') were excluded (outcomes are never imputed).`);
    if (excludedIncomplete > 0) removalNotes.push(`${excludedIncomplete} row(s) with one or more empty predictor cells were removed (complete-case analysis).`);
    if (removalNotes.length > 0) {
        alert(removalNotes.join('\n') + `\n${usableRows.length} row(s) remain for the train/test split.`);
    }
}

function renderPipelineStep1Results(trainMissing = 0, testMissing = 0) {
    if (!AppState.pipeline.trainRows || !AppState.pipeline.imputer) return;

    const trainRows = AppState.pipeline.trainRows;
    const testRows = AppState.pipeline.testRows;
    const imputer = AppState.pipeline.imputer;

    if (DOM.pipelineTrainCount) DOM.pipelineTrainCount.innerText = trainRows.length;
    if (DOM.pipelineTestCount) DOM.pipelineTestCount.innerText = testRows.length;
    if (DOM.pipelineMissingImputedCount) DOM.pipelineMissingImputedCount.innerText = trainMissing + testMissing;

    // Render imputer audit table
    let tableHtml = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Role</th>
                    <th>Measurement Scale</th>
                    <th>Fitted Imputation Value (Strictly from Train)</th>
                    <th>Training Summary Stats</th>
                </tr>
            </thead>
            <tbody>
    `;

    Object.keys(imputer.fillValues).forEach(col => {
        const conf = AppState.pipeline.columnConfigs[col] || { type: 'continuous', role: 'feature' };
        const stats = imputer.columnStats[col] || {};
        const fillVal = imputer.fillValues[col];

        let statsStr = '';
        if (conf.type === 'nominal' || stats.mode !== undefined) {
            statsStr = `Mode: ${escapeHtml(stats.mode || 'N/A')} (Distinct: ${stats.distinctCount || 0})`;
        } else {
            statsStr = `Mean: ${stats.mean !== undefined ? stats.mean : 'N/A'}, Median: ${stats.median !== undefined ? stats.median : 'N/A'}, SD (sample, n−1): ${stats.std !== undefined ? stats.std : 'N/A'}`;
        }

        tableHtml += `
            <tr>
                <td><strong>${escapeHtml(col)}</strong></td>
                <td><span class="pill pill-${conf.role || 'feature'}">${conf.role || 'feature'}</span></td>
                <td><span class="pill pill-${conf.type || 'continuous'}">${conf.type || 'continuous'}</span></td>
                <td><strong style="color: var(--accent-emerald);">${escapeHtml(fmtValue(fillVal))}</strong></td>
                <td style="font-size: 12px; color: var(--text-secondary);">${statsStr}</td>
            </tr>
        `;
    });

    tableHtml += `</tbody></table>`;
    if (AppState.pipeline.completeCase) {
        tableHtml += `<p style="font-size: 12px; color: var(--accent-cyan); margin-top: 8px;">
            Complete-case analysis: ${AppState.pipeline.excludedIncomplete} row(s) with one or more empty predictor cells were removed before the split,
            so no predictor values needed imputing (the fitted values above are kept only as a fallback for new cases at prediction time).
        </p>`;
    }
    tableHtml += `<p style="font-size: 12px; color: var(--text-secondary); margin-top: 8px;">
        Split method: <strong>${escapeHtml(describeSplit(AppState.pipeline.splitInfo))}</strong>.
    </p>`;
    tableHtml += `<p style="font-size: 12px; color: var(--text-secondary); margin-top: 8px;">
        Outcome '<strong>${escapeHtml(AppState.pipeline.targetCol)}</strong>' is never imputed${AppState.pipeline.excludedMissingTarget > 0
            ? `: ${AppState.pipeline.excludedMissingTarget} row(s) with a missing outcome were excluded before the split.`
            : '; no rows had a missing outcome.'}
    </p>`;

    if (DOM.pipelineImputerTableContainer) {
        DOM.pipelineImputerTableContainer.innerHTML = tableHtml;
    }

    if (DOM.pipelineStep1Results) {
        DOM.pipelineStep1Results.style.display = 'block';
    }
}

export { executePipelineStep1, renderPipelineStep1Results };
