// pipeline.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { fitModel } from './compute.js';
import { AppState, compareEvalResults, describeNNTraining, displayModelName, escapeHtml, fmtR2, fmtValue, jsArg, parseSeed, rankingMetricLabel, summariseHeldOutSet, trainingMedian } from './core.js';
import { TUNABLE_MODELS, TUNE_MAX_ROWS, bootstrapFor, buildPipelineFolds, ciSpan, cvSummaryFrom, describeSplit, formatCvScore, readDecisionThreshold, renderTuningReport, runTuningOnFolds, splitRowsForWorkflow } from './cv-tuning.js';
import { renderCalibrationPanel, renderExplainPanel } from './panels.js';
import { deployPipelineModel, renderPipelineBenchmarkResults, renderTooGoodToBeTrueCheck, updatePipelineActiveDeploymentStatus, updatePredictionProvenanceBar } from './pipeline-results.js';
import { saveSettingsToBrowser } from './settings.js';
import { DOM } from './shell.js';
import { disposeModels, evalAbortOpts, fitWithYields, syncRankingSelectors } from './standard-training.js';


// =========================================================================
// 7-STEP CLINICAL ML PIPELINE CONTROLLER
// Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
// =========================================================================

// Preview for the pipeline complete-case option: how many rows it would remove
function updatePipelineCompleteCaseInfo() {
    const info = document.getElementById('pipeline-complete-case-info');
    const toggle = document.getElementById('pipeline-complete-case');
    const strategy = document.getElementById('pipeline-impute-strategy');
    if (!info || !toggle) return;
    if (strategy) strategy.disabled = toggle.checked; // nothing is imputed in complete-case mode
    if (!AppState.parsedData || !AppState.parsedData.rows) { info.textContent = ''; return; }
    const cfgs = AppState.columnConfigs;
    const targetCol = Object.keys(cfgs).find(c => cfgs[c].role === 'target');
    const predictors = Object.keys(cfgs).filter(c => c !== targetCol && cfgs[c].role !== 'meta');
    const rows = targetCol ? AppState.parsedData.rows.filter(r => !DataEngine.isMissing(r[targetCol])) : AppState.parsedData.rows;
    const incomplete = rows.filter(r => predictors.some(c => DataEngine.isMissing(r[c]))).length;
    const remaining = rows.length - incomplete;
    if (incomplete === 0) {
        info.textContent = 'No predictor cells are empty in this dataset: nothing would be removed.';
        info.style.color = 'var(--text-muted)';
        return;
    }
    const heavy = incomplete / rows.length > 0.2 || remaining < 30;
    info.textContent = `${incomplete} of ${rows.length} rows (${(100 * incomplete / rows.length).toFixed(1)}%) have an empty predictor cell ` +
        `(Meta / ID columns ignored); ${remaining} complete rows would remain.` +
        (heavy ? ' ⚠️ This removes a large share of the data; complete-case analysis is unbiased only if values are missing completely at random, otherwise imputation is usually preferable.' : '');
    info.style.color = heavy ? 'var(--accent-amber)' : 'var(--text-muted)';
}

function initPipelineUI() {
    updatePipelineCompleteCaseInfo();
    if (DOM.pipelineDatasetName) {
        DOM.pipelineDatasetName.innerText = AppState.fileName || "No dataset loaded yet (select cohort or upload CSV)";
    }
    if (DOM.pipelineSeedInput && !DOM.pipelineSeedInput.value) {
        DOM.pipelineSeedInput.value = AppState.pipeline.seed;
    }
    if (DOM.pipelineEvalSeedInput && !DOM.pipelineEvalSeedInput.value) {
        DOM.pipelineEvalSeedInput.value = AppState.pipeline.evalSeed;
    }
    if (AppState.pipeline.trainRows) {
        renderPipelineStep1Results();
    }
}

function goToPipelineSubstep(substepNum) {
    substepNum = parseInt(substepNum, 10);
    if (substepNum > 1 && !AppState.pipeline.trainRows) {
        alert("Please execute Step 1: Train/Test Split & Strict Imputation first.");
        return;
    }
    if (substepNum === 7 && (!AppState.pipeline.rfImportance || AppState.pipeline.rfImportance.length === 0)) {
        alert("Please execute Step 6: Random Forest Scout & Feature Ranking first.");
        return;
    }

    AppState.pipeline.substep = substepNum;

    if (DOM.pipelineSubPills) {
        DOM.pipelineSubPills.forEach(pill => {
            const s = parseInt(pill.dataset.substep, 10);
            pill.classList.toggle('active', s === substepNum);
            if (s === substepNum) pill.setAttribute('aria-current', 'step'); else pill.removeAttribute('aria-current');
            if (s < substepNum) pill.classList.add('completed');
        });
    }

    if (DOM.pipelinePanels) {
        DOM.pipelinePanels.forEach(panel => {
            const isActive = panel.id === `pipeline-panel-${substepNum}`;
            panel.classList.toggle('active', isActive);
            panel.style.display = isActive ? 'block' : 'none';
        });
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (substepNum === 2) renderPipelineStep2();
    if (substepNum === 3) renderPipelineStep3();
    if (substepNum === 4) renderPipelineStep4();
    if (substepNum === 5) renderPipelineStep5();
    if (substepNum === 6) renderPipelineStep6UI();
}

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

// =========================================================================
// STEP 2: Continuous Variable Correlation Check (Pearson vs. Spearman)
// =========================================================================
function renderPipelineStep2() {
    if (!AppState.pipeline.trainRows) return;

    const contCols = AppState.pipeline.featureCols.filter(col =>
        !AppState.pipeline.droppedFeatures.has(col) &&
        (AppState.pipeline.columnConfigs[col]?.type === 'continuous' || AppState.pipeline.columnConfigs[col]?.type === 'ordinal')
    );

    if (contCols.length < 2) {
        if (DOM.pipelineNormalityTableContainer) {
            DOM.pipelineNormalityTableContainer.innerHTML = `<p style="padding: 16px; color: var(--text-muted);">Less than two continuous features available for pairwise correlation analysis.</p>`;
        }
        if (DOM.pipelineCorrPairsContainer) {
            DOM.pipelineCorrPairsContainer.innerHTML = `<p style="padding: 16px; color: var(--text-muted);">No continuous feature pairs to evaluate.</p>`;
        }
        if (DOM.pipelineCorrPairCounter) DOM.pipelineCorrPairCounter.innerText = `0 High Pairs`;
        return;
    }

    const corrRes = DataEngine.checkPairwiseCorrelations(AppState.pipeline.trainRows, contCols, 0.8, AppState.pipeline.columnConfigs);
    AppState.pipeline.normalityMap = corrRes.normalityMap;
    AppState.pipeline.corrPairs = corrRes.allPairs;

    // 1. Render Normality Table
    let normHtml = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Continuous Feature</th>
                    <th>Sample Mean</th>
                    <th>Std Deviation</th>
                    <th>Skewness (adjusted Fisher–Pearson G₁)</th>
                    <th>Skewness Screen (rule of thumb)</th>
                    <th>Correlation Method</th>
                </tr>
            </thead>
            <tbody>
    `;

    contCols.forEach(col => {
        const norm = corrRes.normalityMap[col] || { mean: 0, std: 1, skewness: 0, isNormal: true };
        normHtml += `
            <tr>
                <td><strong>${escapeHtml(col)}</strong></td>
                <td>${norm.mean}</td>
                <td>${norm.std}</td>
                <td><strong style="color: ${norm.isNormal ? 'var(--accent-emerald)' : 'var(--accent-amber)'};">${norm.skewness}</strong></td>
                <td>
                    <span class="pill" style="${norm.isNormal ? 'background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.4);' : 'background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border-color: rgba(245, 158, 11, 0.4);'}">
                        ${norm.isNormal ? '✓ Approx. symmetric (|G₁| ≤ 1.0)' : '⚠️ Skewed (|G₁| > 1.0)'}
                    </span>
                </td>
                <td style="font-size: 12px; color: var(--text-secondary);">
                    ${norm.isNormal ? 'Pearson if its partner is also approx. symmetric' : 'Spearman rank correlation for its pairs'}
                </td>
            </tr>
        `;
    });
    normHtml += `</tbody></table>`;
    if (DOM.pipelineNormalityTableContainer) DOM.pipelineNormalityTableContainer.innerHTML = normHtml;

    // 2. Render High Correlation Pairs (|r| or |rho| >= 0.8)
    const highPairs = corrRes.highPairs;
    if (DOM.pipelineCorrPairCounter) {
        DOM.pipelineCorrPairCounter.innerText = `${highPairs.length} High Pairs`;
    }

    if (highPairs.length === 0) {
        DOM.pipelineCorrPairsContainer.innerHTML = `
            <div class="pipeline-alert pipeline-alert-success">
                <div style="font-size: 18px;">✅</div>
                <div>
                    <strong>No Excessive Pairwise Collinearity Detected:</strong> All continuous variable pairs have |correlation| &lt; 0.80. No automatic dropping is strictly required; you may proceed to Step 3.
                </div>
            </div>
        `;
    } else {
        let pairsHtml = '';
        highPairs.forEach((p, idx) => {
            const var1Dropped = AppState.pipeline.droppedFeatures.has(p.var1);
            const var2Dropped = AppState.pipeline.droppedFeatures.has(p.var2);

            pairsHtml += `
                <div class="corr-pair-card high-corr ${var1Dropped || var2Dropped ? 'dropped' : ''}">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                            <strong style="font-size: 14px; color: var(--text-primary);">${escapeHtml(p.var1)}</strong>
                            <span style="color: var(--text-muted);">&harr;</span>
                            <strong style="font-size: 14px; color: var(--text-primary);">${escapeHtml(p.var2)}</strong>
                            <span class="vif-badge-danger">${p.method} = ${Number(p.coeff).toFixed(3)}</span>
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary);">
                            <strong>Method:</strong> ${p.bothNormal ? 'Both variables approximately symmetric (|G₁| ≤ 1) &rarr; Pearson product-moment correlation.' : `Skewness detected (${escapeHtml([p.norm1.isNormal ? '' : p.var1, p.norm2.isNormal ? '' : p.var2].filter(Boolean).join(', '))}) &rarr; Spearman rank correlation.`}
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <button class="btn btn-rose btn-sm" onclick="window.pipelineDropFeature(${jsArg(p.var1)})" ${var1Dropped ? 'disabled' : ''}>
                            ${var1Dropped ? `✓ Dropped ${escapeHtml(p.var1)}` : `🗑️ Drop ${escapeHtml(p.var1)}`}
                        </button>
                        <button class="btn btn-rose btn-sm" onclick="window.pipelineDropFeature(${jsArg(p.var2)})" ${var2Dropped ? 'disabled' : ''}>
                            ${var2Dropped ? `✓ Dropped ${escapeHtml(p.var2)}` : `🗑️ Drop ${escapeHtml(p.var2)}`}
                        </button>
                        <button class="btn btn-secondary btn-sm" onclick="window.pipelineKeepBothPairs(${jsArg(p.var1)}, ${jsArg(p.var2)})">
                            🤝 Keep Both
                        </button>
                    </div>
                </div>
            `;
        });
        DOM.pipelineCorrPairsContainer.innerHTML = pairsHtml;
    }
}

window.pipelineDropFeature = function (featureName) {
    AppState.pipeline.droppedFeatures.add(featureName);
    renderPipelineStep2();
    if (AppState.pipeline.substep === 3) renderPipelineStep3();
};

window.pipelineKeepBothPairs = function (var1, var2) {
    alert(`Pair [${var1}] and [${var2}] retained. Record the clinical justification for keeping both in your study protocol.`);
};

// =========================================================================
// STEP 3: Variance Inflation Factor (VIF) Check
// =========================================================================
function renderPipelineStep3() {
    if (!AppState.pipeline.trainRows) return;

    const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
    let vifResults;
    try {
        vifResults = DataEngine.calculateVIF(AppState.pipeline.trainRows, activeCols, AppState.pipeline.columnConfigs);
    } catch (err) {
        if (DOM.pipelineVifTableContainer) {
            DOM.pipelineVifTableContainer.innerHTML = `<div class="pipeline-alert pipeline-alert-warning"><div>⚠️</div><div>${escapeHtml(err.message)}</div></div>`;
        }
        if (DOM.pipelineDropVifBtn) DOM.pipelineDropVifBtn.disabled = true;
        return;
    }
    AppState.pipeline.vifResults = vifResults;

    const hasHighVif = vifResults.some(v => v.highVIF);
    if (DOM.pipelineDropVifBtn) DOM.pipelineDropVifBtn.disabled = !hasHighVif;

    let tableHtml = `
        <table class="data-table">
            <thead>
                <tr>
                    <th style="width: 40px; text-align: center;">Select</th>
                    <th>Feature Name</th>
                    <th>Measurement Scale</th>
                    <th>df (Encoded Columns)</th>
                    <th>Equivalent R²</th>
                    <th>VIF (GVIF<sup>1/df</sup> for df &gt; 1)</th>
                    <th>Collinearity Status</th>
                </tr>
            </thead>
            <tbody>
    `;

    vifResults.forEach(item => {
        const isHigh = item.highVIF;
        const isDropped = AppState.pipeline.droppedFeatures.has(item.feature);

        tableHtml += `
            <tr style="${isHigh ? 'background: rgba(244, 63, 94, 0.04);' : ''}">
                <td style="text-align: center;">
                    <input type="checkbox" class="pipeline-vif-checkbox" data-feature="${escapeHtml(item.feature)}" ${isHigh && !isDropped ? 'checked' : ''} ${isDropped ? 'disabled' : ''}>
                </td>
                <td><strong>${escapeHtml(item.feature)}</strong> ${isDropped ? '<span class="pill pill-meta" style="font-size: 10px;">Dropped</span>' : ''}</td>
                <td><span class="pill pill-${item.type}">${item.type}</span></td>
                <td style="font-size: 12px; color: var(--text-secondary);">${item.dummyCount}</td>
                <td>${item.rSquared}</td>
                <td><strong style="color: ${isHigh ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; font-size: 13.5px;">${item.vif >= 999 ? '≥ 999 (≈ perfect collinearity)' : item.vif}</strong>${item.dummyCount > 1 && item.vif < 999 ? `<div style="font-size: 10.5px; color: var(--text-muted);">GVIF ${item.gvif}; GVIF<sup>1/(2·df)</sup> ${item.adjustedGVIF}</div>` : ''}</td>
                <td>
                    <span class="${isHigh ? 'vif-badge-danger' : 'vif-badge-ok'}">
                        ${isHigh ? '⚠️ VIF > 5.0 (High Multicollinearity)' : '✓ VIF ≤ 5.0 (Acceptable)'}
                    </span>
                </td>
            </tr>
        `;
    });

    tableHtml += `</tbody></table>`;
    if (DOM.pipelineVifTableContainer) DOM.pipelineVifTableContainer.innerHTML = tableHtml;
}

function pipelineDropSelectedVIF() {
    const checkboxes = document.querySelectorAll('.pipeline-vif-checkbox:checked');
    if (checkboxes.length === 0) {
        alert("No variables selected to eliminate.");
        return;
    }

    const droppedNames = [];
    checkboxes.forEach(cb => {
        const f = cb.dataset.feature;
        if (f) {
            AppState.pipeline.droppedFeatures.add(f);
            droppedNames.push(f);
        }
    });

    alert(`Excluded ${droppedNames.length} feature(s) from the pipeline (Train and Test): ${droppedNames.join(', ')}. They can be restored in Step 4.`);
    renderPipelineStep3();
}

// =========================================================================
// STEP 4: Univariate Statistical Testing Check
// =========================================================================
function renderPipelineStep4() {
    if (!AppState.pipeline.trainRows) return;

    // Run univariate test on all pipeline features so user can see rankings and restore any previously dropped variables
    const allFeatures = AppState.pipeline.featureCols;
    const results = DataEngine.runUnivariateTesting(
        AppState.pipeline.trainRows,
        allFeatures,
        AppState.pipeline.targetCol,
        AppState.pipeline.columnConfigs
    );

    AppState.pipeline.univariateResults = results;

    const activeCount = allFeatures.filter(c => !AppState.pipeline.droppedFeatures.has(c)).length;
    const sigCount = results.filter(r => r.significant && !AppState.pipeline.droppedFeatures.has(r.feature)).length;
    const nonSigCount = activeCount - sigCount;

    if (DOM.pipelineUnivSelectedSummary) {
        DOM.pipelineUnivSelectedSummary.innerHTML = `
            Active in Pipeline: <strong>${activeCount} / ${allFeatures.length}</strong> features 
            (${sigCount} significant, ${nonSigCount} non-significant). 
            ${AppState.pipeline.droppedFeatures.size > 0 ? `<span style="color: var(--accent-rose);">(${AppState.pipeline.droppedFeatures.size} currently excluded)</span>` : ''}
        `;
    }

    let tableHtml = `
        <table class="data-table">
            <thead>
                <tr>
                    <th style="width: 40px; text-align: center;">
                        <input type="checkbox" id="pipeline-univ-master-check" title="Toggle all active" checked>
                    </th>
                    <th style="width: 50px;">Rank</th>
                    <th>Feature</th>
                    <th>p-value</th>
                    <th style="width: 110px;">Action</th>
                    <th>Significance</th>
                    <th>Measurement Scale</th>
                    <th>Statistical Test</th>
                    <th>Test Statistic</th>
                    <th>Degrees of Freedom</th>
                </tr>
            </thead>
            <tbody>
    `;

    results.forEach((r, idx) => {
        const isSig = r.significant;
        const isDropped = AppState.pipeline.droppedFeatures.has(r.feature);

        tableHtml += `
            <tr style="${isDropped ? 'opacity: 0.55; background: rgba(244, 63, 94, 0.04);' : (isSig ? 'background: rgba(16, 185, 129, 0.02);' : '')}">
                <td style="text-align: center;">
                    <input type="checkbox" class="pipeline-univ-checkbox" data-feature="${escapeHtml(r.feature)}" data-sig="${isSig}" ${!isDropped ? 'checked' : ''}>
                </td>
                <td style="font-weight: 700; color: var(--accent-cyan);">#${idx + 1}</td>
                <td>
                    <strong style="${isDropped ? 'text-decoration: line-through;' : ''}">${escapeHtml(r.feature)}</strong>
                    ${isDropped ? '<span class="pill pill-meta" style="font-size: 10px; margin-left: 6px;">Excluded</span>' : ''}
                </td>
                <td><strong style="color: ${isSig ? 'var(--accent-emerald)' : 'var(--text-muted)'}; font-size: 13.5px;">${r.pValue < 0.0001 ? '< 0.0001' : r.pValue}</strong></td>
                <td>
                    ${isDropped ? `
                        <button class="btn btn-secondary btn-sm" onclick="window.pipelineRestoreFeatureStep4(${jsArg(r.feature)})" title="Restore variable into modeling pipeline">
                            ↩️ Include
                        </button>
                    ` : `
                        <button class="btn btn-rose btn-sm" onclick="window.pipelineDropFeatureStep4(${jsArg(r.feature)})" title="Exclude this variable from modeling">
                            🗑️ Exclude
                        </button>
                    `}
                </td>
                <td>
                    <span class="pill" style="${isSig ? 'background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.4);' : 'background: rgba(100, 116, 139, 0.15); color: var(--text-muted);'}">
                        ${isSig ? '✓ p < 0.05 (Sig)' : 'p ≥ 0.05 (Non-Sig)'}
                    </span>
                </td>
                <td><span class="pill pill-${r.scale.toLowerCase().includes('nominal') ? 'nominal' : 'continuous'}">${r.scale}</span></td>
                <td style="font-weight: 600;">${escapeHtml(r.testName)}</td>
                <td>${escapeHtml(r.statisticLabel)}: <strong>${r.statisticValue === Infinity ? '∞' : r.statisticValue}</strong></td>
                <td style="font-size: 12px; color: var(--text-secondary);">${r.df}</td>
            </tr>
        `;
    });

    tableHtml += `</tbody></table>`;
    if (DOM.pipelineUnivariateTableContainer) {
        DOM.pipelineUnivariateTableContainer.innerHTML = tableHtml;

        // Master checkbox event
        const masterCheck = document.getElementById('pipeline-univ-master-check');
        if (masterCheck) {
            masterCheck.addEventListener('change', (e) => {
                const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
                cbs.forEach(cb => cb.checked = e.target.checked);
                updateUnivSelectionSummary();
            });
        }

        // Individual checkbox events
        const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
        cbs.forEach(cb => {
            cb.addEventListener('change', updateUnivSelectionSummary);
        });
    }
}

function updateUnivSelectionSummary() {
    const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
    const checkedCount = Array.from(cbs).filter(cb => cb.checked).length;
    if (DOM.pipelineUnivSelectedSummary) {
        DOM.pipelineUnivSelectedSummary.innerHTML = `
            Checked for Inclusion: <strong>${checkedCount} / ${cbs.length}</strong> features. 
            Click <strong>"🗑️ Exclude Unchecked Features"</strong> to apply changes to subsequent pipeline steps.
        `;
    }
}

function filterSignificantUnivariateFeatures() {
    const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
    let uncheckCount = 0;
    cbs.forEach(cb => {
        const isSig = cb.dataset.sig === 'true';
        if (!isSig) {
            cb.checked = false;
            uncheckCount++;
        } else {
            cb.checked = true;
        }
    });
    updateUnivSelectionSummary();
    alert(`Unchecked ${uncheckCount} non-significant feature(s) (p ≥ 0.05). Click "🗑️ Exclude Unchecked Features" to finalize.`);
}

function selectAllUnivariateFeatures() {
    const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
    cbs.forEach(cb => cb.checked = true);
    const masterCheck = document.getElementById('pipeline-univ-master-check');
    if (masterCheck) masterCheck.checked = true;
    updateUnivSelectionSummary();
}

function dropUnselectedUnivariateFeatures() {
    const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
    let newlyDropped = [];
    let newlyRestored = [];

    cbs.forEach(cb => {
        const f = cb.dataset.feature;
        if (!f) return;
        if (!cb.checked) {
            if (!AppState.pipeline.droppedFeatures.has(f)) {
                AppState.pipeline.droppedFeatures.add(f);
                newlyDropped.push(f);
            }
        } else {
            if (AppState.pipeline.droppedFeatures.has(f)) {
                AppState.pipeline.droppedFeatures.delete(f);
                newlyRestored.push(f);
            }
        }
    });

    const activeCount = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c)).length;
    if (activeCount === 0) {
        alert("Warning: At least 1 predictor variable must remain active in the pipeline!");
        return;
    }

    let msg = `Pipeline features updated successfully! (${activeCount} active)`;
    if (newlyDropped.length > 0) msg += `\nExcluded: ${newlyDropped.join(', ')}`;
    if (newlyRestored.length > 0) msg += `\nRestored: ${newlyRestored.join(', ')}`;
    alert(msg);

    renderPipelineStep4();
}

window.pipelineDropFeatureStep4 = function (featureName) {
    AppState.pipeline.droppedFeatures.add(featureName);
    renderPipelineStep4();
};

window.pipelineRestoreFeatureStep4 = function (featureName) {
    AppState.pipeline.droppedFeatures.delete(featureName);
    renderPipelineStep4();
};

// =========================================================================
// STEP 5: Imbalance Handling (SMOTENC)
// =========================================================================
function renderPipelineStep5() {
    if (!AppState.pipeline.trainRows) return;

    const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
    const smotencRes = DataEngine.applySMOTENC(
        AppState.pipeline.trainRows,
        activeCols,
        AppState.pipeline.targetCol,
        AppState.pipeline.columnConfigs,
        AppState.pipeline.seed
    );

    AppState.pipeline.smotencResult = smotencRes;
    AppState.pipeline.balancedTrainRows = smotencRes.balancedRows;

    let contentHtml = '';
    if (smotencRes.applied) {
        contentHtml = `
            <div class="pipeline-alert pipeline-alert-warning" style="margin-bottom: 20px;">
                <div style="font-size: 20px;">⚠️</div>
                <div>
                    <strong>Class Imbalance Detected:</strong><br>
                    ${escapeHtml(smotencRes.reason)}<br>
                    Smallest class <code>${escapeHtml(smotencRes.minorityClass)}</code>: ${smotencRes.minorityPctBefore}% of training rows before &rarr; ${smotencRes.minorityPctAfter}% after resampling.
                    Synthetic rows are used for training only; all test-set metrics are computed on real patients at their true prevalence.
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 16px;">
                <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                    <h5 style="font-size: 13px; font-weight: 700; margin-bottom: 10px; color: var(--text-secondary);">Original Training Class Distribution</h5>
                    ${Object.keys(smotencRes.originalCounts).map(c => `
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
                            <span>Class <strong>${escapeHtml(c)}</strong>:</span>
                            <strong>${smotencRes.originalCounts[c]} cases</strong>
                        </div>
                    `).join('')}
                </div>
                <div class="card" style="padding: 16px; background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.3);">
                    <h5 style="font-size: 13px; font-weight: 700; margin-bottom: 10px; color: var(--accent-emerald);">Post-SMOTENC Training Class Distribution</h5>
                    ${Object.keys(smotencRes.balancedCounts).map(c => `
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
                            <span>Class <strong>${escapeHtml(c)}</strong>:</span>
                            <strong style="color: var(--accent-emerald);">${smotencRes.balancedCounts[c]} cases</strong>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    } else {
        contentHtml = `
            <div class="pipeline-alert pipeline-alert-success" style="margin-bottom: 20px;">
                <div style="font-size: 20px;">✅</div>
                <div>
                    <strong>No Synthetic Resampling Applied:</strong><br>
                    ${escapeHtml(smotencRes.reason || 'No synthetic resampling needed.')}
                </div>
            </div>
            <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                <h5 style="font-size: 13px; font-weight: 700; margin-bottom: 10px; color: var(--text-secondary);">Training Class Distribution</h5>
                ${Object.keys(smotencRes.classCounts || {}).map(c => `
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
                        <span>Class <strong>${escapeHtml(c)}</strong>:</span>
                        <strong>${smotencRes.classCounts[c]} cases</strong>
                    </div>
                `).join('')}
            </div>
        `;
    }

    if (DOM.pipelineSmotencContainer) {
        DOM.pipelineSmotencContainer.innerHTML = contentHtml;
    }
}

// =========================================================================
// STEP 6: Model-Driven Feature Ranking (Random Forest Scout)
// =========================================================================
function renderPipelineStep6UI() {
    if (!AppState.pipeline.rfImportance || AppState.pipeline.rfImportance.length === 0) {
        if (DOM.pipelineRfImportanceContainer) {
            DOM.pipelineRfImportanceContainer.innerHTML = `
                <p style="padding: 24px; color: var(--text-muted); text-align: center;">
                    Click the purple 'Run Random Forest Scout & Feature Ranking' button above to train a Random Forest classifier on the balanced training set and extract feature importances.
                </p>
            `;
        }
    }
}

async function executePipelineStep6() {
    const btn = DOM.pipelineRunRfScoutBtn;
    const label = btn ? btn.innerHTML : '';
    try {
        if (btn) { btn.disabled = true; btn.textContent = 'Training Random Forest scout…'; }
        await runPipelineStep6();
    } catch (err) {
        alert(`Random Forest scout could not run: ${err.message}`);
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = label; }
    }
}

async function runPipelineStep6() {
    // Always recompute Step 5 from the current split & feature set so the scout never uses stale rows
    renderPipelineStep5();

    const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
    if (activeCols.length === 0) {
        alert("No active predictor features remain. Restore at least one feature in Step 4.");
        return;
    }
    const subConfigs = {};
    activeCols.forEach(c => subConfigs[c] = AppState.pipeline.columnConfigs[c]);
    subConfigs[AppState.pipeline.targetCol] = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol];

    const prep = DataEngine.prepareDataset(
        AppState.pipeline.balancedTrainRows,
        subConfigs,
        activeCols,
        AppState.pipeline.targetCol
    );

    // Train Random Forest Scout
    // Scout trained in the background worker when available
    const rf = await fitModel('rf', prep.isClassification, prep.targetClasses.length || 2, { numTrees: 40, maxDepth: 6, minSamplesSplit: 2, featureRatio: 'sqrt', seed: AppState.pipeline.seed }, prep.X, prep.y);

    // Map feature importances back to original clinical features
    const featureImpMap = {};
    activeCols.forEach(col => {
        const meta = prep.featureMetadata[col];
        let sumImp = 0;
        if (meta) {
            for (let k = 0; k < meta.length; k++) {
                sumImp += rf.featureImportances[meta.startIndex + k] || 0;
            }
        }
        featureImpMap[col] = sumImp;
    });

    const totalImp = Object.values(featureImpMap).reduce((a, b) => a + b, 0) || 1;
    const ranked = activeCols.map(col => ({
        name: col,
        importance: Number((featureImpMap[col] / totalImp).toFixed(4)),
        pct: Number(((featureImpMap[col] / totalImp) * 100).toFixed(1))
    }));

    ranked.sort((a, b) => b.importance - a.importance);

    // Bottom 30% cutoff
    const k = Math.max(1, Math.round(activeCols.length * 0.30));
    const topCount = activeCols.length - k;

    AppState.pipeline.rfImportance = ranked;
    AppState.pipeline.rfTopCount = topCount;
    // Keep earlier reinstatements that still fall below the cutoff
    const bottomNames = ranked.slice(topCount).map(f => f.name);
    const kept = Array.from(AppState.pipeline.reinstatedFeatures || []).filter(f => bottomNames.includes(f));
    AppState.pipeline.reinstatedFeatures = new Set(kept);
    applyRfFeatureSelection();
    renderRfImportanceChart();
    if (DOM.pipelineToStep7Btn) DOM.pipelineToStep7Btn.disabled = false;
}

// Top 70% plus any bottom-30% features the user reinstated (in importance order) go to k-NN & Naive Bayes
function applyRfFeatureSelection() {
    const ranked = AppState.pipeline.rfImportance || [];
    const topCount = AppState.pipeline.rfTopCount;
    const reinstated = AppState.pipeline.reinstatedFeatures || new Set();
    AppState.pipeline.top70Features = ranked.filter((f, idx) => idx < topCount || reinstated.has(f.name)).map(f => f.name);
    AppState.pipeline.bottom30Features = ranked.filter((f, idx) => idx >= topCount && !reinstated.has(f.name)).map(f => f.name);
}

window.pipelineToggleReinstateFeature = function (name) {
    const set = AppState.pipeline.reinstatedFeatures || (AppState.pipeline.reinstatedFeatures = new Set());
    if (set.has(name)) set.delete(name); else set.add(name);
    applyRfFeatureSelection();
    renderRfImportanceChart();
};

function renderRfImportanceChart() {
    const ranked = AppState.pipeline.rfImportance || [];
    const topCount = AppState.pipeline.rfTopCount;
    const reinstated = AppState.pipeline.reinstatedFeatures || new Set();
    const hasResults = (AppState.pipeline.benchmarkResults || []).length > 0;
    const maxPct = Math.max(...ranked.map(r => r.pct), 1);
    let chartHtml = `
        <div class="rf-importance-bar-container">
    `;

    ranked.forEach((r, idx) => {
        const isBottomTier = idx >= topCount;
        const isReinstated = isBottomTier && reinstated.has(r.name);
        const isPruned = isBottomTier && !isReinstated;
        const barWidth = Math.max(4, Math.round((r.pct / maxPct) * 100));

        if (idx === topCount) {
            chartHtml += `
                <div class="cutoff-line-notice">
                    <span>✂️</span>
                    <span><strong>Bottom 30% Cutoff:</strong> Features below this line are removed for k-NN and Naive Bayes only in Step 7 (other models use all surviving features). Click <strong>↩️ Reinstate</strong> to keep a feature for k-NN and Naive Bayes anyway (for example, if it is clinically important). MDI importance is computed on the SMOTENC-balanced training set and can favour features with many distinct values.</span>
                </div>
            `;
        }

        const pill = isReinstated
            ? `<span class="pill" style="font-size: 10px; background: rgba(56,189,248,0.15); color: var(--accent-cyan); border-color: rgba(56,189,248,0.4);">Reinstated</span>`
            : `<span class="pill" style="font-size: 10px; ${isPruned ? 'background: rgba(245,158,11,0.15); color: var(--accent-amber); border-color: rgba(245,158,11,0.4);' : 'background: rgba(16,185,129,0.15); color: var(--accent-emerald); border-color: rgba(16,185,129,0.4);'}">
                        ${isPruned ? 'Bottom 30%' : 'Top 70%'}
                    </span>`;
        const action = !isBottomTier ? '' : (isReinstated
            ? `<button type="button" class="btn btn-outline btn-sm" onclick="window.pipelineToggleReinstateFeature(${jsArg(r.name)})" title="Remove this feature again for k-NN and Naive Bayes">✂️ Remove again</button>`
            : `<button type="button" class="btn btn-outline btn-sm" onclick="window.pipelineToggleReinstateFeature(${jsArg(r.name)})" title="Keep this feature for k-NN and Naive Bayes in Step 7">↩️ Reinstate</button>`);

        chartHtml += `
            <div class="rf-importance-row">
                <div>
                    <strong style="color: ${isPruned ? 'var(--accent-amber)' : 'var(--text-primary)'};">#${idx + 1} ${escapeHtml(r.name)}</strong>
                </div>
                <div class="rf-meter-track">
                    <div class="rf-meter-fill ${isPruned ? 'bottom-tier' : 'top-tier'}" style="width: ${barWidth}%;"></div>
                </div>
                <div style="font-weight: 700; color: ${isPruned ? 'var(--accent-amber)' : 'var(--accent-emerald)'}; text-align: right;">
                    ${r.pct}%
                </div>
                <div>${pill}</div>
                <div class="rf-importance-action">${action}</div>
            </div>
        `;
    });

    const nReinstated = ranked.filter((r, idx) => idx >= topCount && reinstated.has(r.name)).length;
    chartHtml += `
            <div class="rf-importance-summary">
                k-NN and Naive Bayes will use <strong>${AppState.pipeline.top70Features.length}</strong> of ${ranked.length} features
                (${topCount} in the top 70%${nReinstated ? ` + ${nReinstated} reinstated` : ''}).
                ${hasResults ? '<span style="color: var(--accent-amber);">Re-run Step 7 to apply any change to the results.</span>' : ''}
            </div>
        </div>`;
    if (DOM.pipelineRfImportanceContainer) DOM.pipelineRfImportanceContainer.innerHTML = chartHtml;
}

// =========================================================================
// STEP 7: Algorithm-Specific Training & Constraints
// =========================================================================
async function executePipelineStep7() {
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
        // Model plan: Naive Bayes is classification-only, so regression runs 4 models, not 5
        const plannedModels = isClassification ? ['knn', 'nb', 'rf', 'nn', 'glm', 'ensemble'] : ['knn', 'rf', 'nn', 'glm', 'ensemble'];
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
        // Training rows, SMOTENC re-applied within each fold) may replace them
        const evalSeed = AppState.pipeline.evalSeed;
        const pHp = {
            knn: { k: 5, metric: 'euclidean', weighting: 'distance', seed: evalSeed },
            nb: { varSmoothing: 1e-4, priorMode: 'empirical', seed: evalSeed },
            rf: { numTrees: 40, maxDepth: 6, minSamplesSplit: 2, featureRatio: 'sqrt', seed: evalSeed },
            nn: { epochs: 25, learningRate: 0.005, hiddenLayers: [48, 24], dropoutRate: 0.15, l2Reg: 0.001, activation: 'relu', optimizer: 'adam', batchSize: 16, seed: evalSeed },
            glm: { lr: 0.05, epochs: 150, lambdaL2: 0.02, solver: 'momentum', seed: evalSeed }
        };
        const pipeTune = Boolean(document.getElementById('pipeline-autotune-check')?.checked);
        const pipeFolds = parseInt(document.getElementById('pipeline-autotune-folds')?.value, 10) || 5;
        const pipeSelectCV = (document.querySelector('input[name="pipeline-selection-mode"]:checked')?.value || 'cv') === 'cv';
        const pipeTuningReport = { folds: pipeFolds, models: [], pipeline: true, ensembleSkipped: pipeTune, selectByCV: pipeSelectCV };
        pHp.ensemble = { voting: 'soft', baseModels: ['nn', 'rf', 'glm'], capacity: 'balanced', seed: evalSeed };
        const cvByModel = {};
        if (pipeTune || pipeSelectCV) {
            const nTrainReal = AppState.pipeline.trainRows.length;
            if (pipeTune && nTrainReal < 100) {
                pipeTuningReport.smallSampleWarning = `Only ${nTrainReal} training patients: each fold scores very few patients, so differences between settings are mostly chance. Tuning is unlikely to help much with a cohort this small; the protocol defaults are a sound choice.`;
            }
            const foldCache = {};
            const foldsFor = async (cols, k, label, stepId, inFold = false) => {
                const key = `${k}|${inFold ? 'sel' : ''}|${cols.join('␟')}`;
                if (!foldCache[key]) {
                    foldCache[key] = await buildPipelineFolds(AppState.pipeline.trainRows, cols, k, evalSeed, (f, nf) => {
                        if (watchdog) watchdog.tick(stepOf(stepId), `🎯 Preparing cross-validation folds for ${label}: fold ${f}/${nf} (SMOTENC, scaling${inFold ? ' and top-70% selection' : ''} inside each fold)...`);
                    }, inFold);
                }
                return foldCache[key];
            };
            const plan = pipeSelectCV ? plannedModels.slice() : plannedModels.filter(id => id !== 'ensemble');
            if (watchdog) watchdog.tick(1, `🎯 ${pipeTune ? 'Automatic tuning' : 'Cross-validation'} of ${plan.length} models on ${Math.min(nTrainReal, TUNE_MAX_ROWS).toLocaleString('en-GB')} training patients: this takes several times longer than training alone (the neural network is slowest). A time estimate appears once each model's first trial has run.`);
            await ModelEngine.yieldToBrowser();
            for (let t = 0; t < plan.length; t++) {
                const id = plan[t];
                const reduced = id === 'knn' || id === 'nb';
                const k = (id === 'nn' || id === 'ensemble') ? Math.min(pipeFolds, 3) : pipeFolds;
                const label = { knn: 'k-NN', nb: 'Naive Bayes', rf: 'Random Forest', nn: 'Neural Network', glm: 'Logistic / Ridge Regression', ensemble: 'Ensemble' }[id];
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

        // C. Deep Neural Network (Strict constraint: L2 Weight Decay + Dropout)
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

        // Cross-validated scores (model selection inside the Training set) when that mode is chosen
        benchmarkList.forEach(b => { if (cvByModel[b.id]) b.eval.cv = cvByModel[b.id]; });
        // Bootstrap 95% confidence intervals for every test metric
        for (const b of benchmarkList) {
            if (watchdog) watchdog.tick(totalModels, `Computing 95% confidence intervals (bootstrap): ${b.name}...`);
            b.eval.ci = await bootstrapFor(b.eval, evalSeed);
        }
        // Store ranked best-first, so benchmarkResults[0] is genuinely the top model everywhere it is used
        benchmarkList.sort((a, b) => compareEvalResults(a.eval, b.eval, isClassification));
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
                        <span class="pill" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); font-weight: 700; border-color: rgba(16, 185, 129, 0.3);">
                            ✓ Active
                        </span>
                    ` : `
                        <button class="btn btn-secondary btn-sm" onclick="window.deployPipelineModel('${item.id}')" title="Deploy this model for live patient inference">
                            🚀 Deploy
                        </button>
                    `}
                </td>
            </tr>
        `;
    });

    tableHtml += `</tbody></table>`;
    const caveats = new Set();
    sorted.forEach(item => {
        summariseHeldOutSet(item.eval).forEach(n => caveats.add(n));
        if (item.eval.fallbackNote) caveats.add(`⚠️ ${escapeHtml(item.eval.fallbackNote)}`);
    });
    if (sorted[0] && sorted[0].eval.cv) {
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
    syncRankingSelectors(isClassification);
}

export { updatePipelineCompleteCaseInfo, initPipelineUI, goToPipelineSubstep, executePipelineStep1, renderPipelineStep1Results, renderPipelineStep2, renderPipelineStep3, pipelineDropSelectedVIF, renderPipelineStep4, updateUnivSelectionSummary, filterSignificantUnivariateFeatures, selectAllUnivariateFeatures, dropUnselectedUnivariateFeatures, renderPipelineStep5, renderPipelineStep6UI, executePipelineStep6, runPipelineStep6, applyRfFeatureSelection, renderRfImportanceChart, executePipelineStep7, renderPipelineLeaderboardOnly };
