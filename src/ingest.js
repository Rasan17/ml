// ingest.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { AppState, escapeHtml } from './core.js';
import { populateSplitColumnSelects } from './cv-tuning.js';
import { goToPipelineSubstep, initPipelineUI } from './pipeline.js';
import { offerStoredSettings } from './settings.js';
import { DOM } from './shell.js';
import { disposeModels } from './training-actions.js';
// Known built-in cohorts and their matching validation (testing) files
const SAMPLE_COHORTS = {
    'universal_medical_cohort.csv': { id: 'cardio', testFile: 'universal_medical_test_cohort.csv', testLabel: 'Cardiometabolic Risk Test Cohort' },
    'medical_shunt_cohort_large.csv': { id: 'shunt-large', testFile: 'medical_shunt_validation_cohort.csv', testLabel: 'iNPH External Validation Cohort (Hospital D, 150 patients)' },
    'medical_shunt_cohort.csv': { id: 'shunt', testFile: 'medical_batch_unseen_cases.csv', testLabel: 'Neurosurgical External Validation Cohort' },
    'medical_recovery_score.csv': { id: 'recovery', testFile: 'medical_recovery_test_cohort.csv', testLabel: 'Functional Recovery Test Cohort' }
};

async function loadSampleDataset(filename, displayName) {
    try {
        const resp = await fetch(filename);
        if (!resp.ok) throw new Error("Could not load sample dataset file.");
        const text = await resp.text();
        processCSVContent(text, displayName);
        // Recorded after processCSVContent (which resets it): identifies the built-in cohort explicitly
        AppState.sampleCohortFile = SAMPLE_COHORTS[filename] ? filename : null;
    } catch (err) {
        alert(`Error loading sample dataset: ${err.message}`);
    }
}

function createEmptyPipelineState(seed = 42) {
    return {
        substep: 1,
        seed,
        evalSeed: seed,
        imputeStrategy: 'median',
        trainRows: null,
        testRows: null,
        balancedTrainRows: null,
        imputer: null,
        featureCols: [],
        targetCol: null,
        columnConfigs: {},
        droppedFeatures: new Set(),
        normalityMap: {},
        corrPairs: [],
        vifResults: [],
        univariateResults: [],
        smotencResult: null,
        rfScout: null,
        rfImportance: [],
        top70Features: [],
        bottom30Features: [],
        reinstatedFeatures: new Set(), // bottom-30% features the user chose to keep for k-NN & Naive Bayes
        rfTopCount: 0,
        benchmarkResults: [],
        excludedMissingTarget: 0,
        completeCase: false,
        excludedIncomplete: 0
    };
}

// Clears every result derived from a previous dataset so nothing stale can be trained or predicted on
function resetDerivedState() {
    // Free GPU memory of every model from the previous dataset (including the deployed one)
    const oldModels = [
        ...((AppState.allTestedResults || []).map(r => r.model)),
        ...(((AppState.pipeline && AppState.pipeline.benchmarkResults) || []).map(b => b.model)),
        AppState.trainedModel
    ];
    AppState.trainedModel = null;
    disposeModels(oldModels);
    AppState.imputedData = null;
    AppState.prepMeta = null;
    AppState.trainTestData = null;
    AppState.trainedModel = null;
    AppState.evalResults = null;
    AppState.allTestedResults = null;
    AppState.separateTestData = null;
    AppState.batchPredictions = null;
    AppState.deployedSource = null;
    AppState.deployedPipelineModel = null;
    AppState.baselinePrepMeta = null;
    AppState.completeCase = false;
    AppState.missingIndicators = false;
    const pipelineCC = document.getElementById('pipeline-complete-case');
    if (pipelineCC) pipelineCC.checked = false;
    if (AppState.charts.loss) {
        AppState.charts.loss.destroy();
        AppState.charts.loss = null;
    }
    AppState.pipeline = createEmptyPipelineState(AppState.pipeline ? AppState.pipeline.seed : 42);

    if (DOM.imputeAuditBox) DOM.imputeAuditBox.innerHTML = '';
    if (DOM.toStep4Btn) DOM.toStep4Btn.disabled = true;
    if (DOM.toStep5Btn) DOM.toStep5Btn.disabled = true;
    if (DOM.toStep6Btn) DOM.toStep6Btn.disabled = true;
    if (DOM.resultsDashboard) DOM.resultsDashboard.style.display = 'none';
    if (DOM.trainStatus) DOM.trainStatus.innerHTML = '';
    if (DOM.singleResultBanner) DOM.singleResultBanner.style.display = 'none';
    if (DOM.predictFormGrid) DOM.predictFormGrid.innerHTML = '';
    if (DOM.batchResultsCard) DOM.batchResultsCard.style.display = 'none';
    if (DOM.downloadBatchCsvBtn) DOM.downloadBatchCsvBtn.disabled = true;
    if (DOM.pipelineStep1Results) DOM.pipelineStep1Results.style.display = 'none';
    if (DOM.pipelineResultsContainer) DOM.pipelineResultsContainer.style.display = 'none';
    if (DOM.pipelineVsBaselinePanel) DOM.pipelineVsBaselinePanel.style.display = 'none';
    if (DOM.pipelineRfImportanceContainer) DOM.pipelineRfImportanceContainer.innerHTML = '';
    if (DOM.pipelineSmotencContainer) DOM.pipelineSmotencContainer.innerHTML = '';
    if (DOM.pipelineToStep7Btn) DOM.pipelineToStep7Btn.disabled = true;
    if (DOM.pipelineSubPills) DOM.pipelineSubPills.forEach(p => p.classList.remove('completed'));
    if (DOM.singlePredictProvenanceBar) DOM.singlePredictProvenanceBar.style.display = 'none';
    if (DOM.batchPredictProvenanceBar) DOM.batchPredictProvenanceBar.style.display = 'none';
    if (DOM.pipelineActiveDeploymentStatus) {
        DOM.pipelineActiveDeploymentStatus.innerHTML = 'Select any model from the leaderboard to deploy for real-time patient predictions.';
    }
    if (DOM.stepTabs) DOM.stepTabs.forEach(t => t.classList.remove('completed'));
}

function processCSVContent(csvText, name) {
    try {
        const parsed = DataEngine.parseCSV(csvText);
        resetDerivedState();
        AppState.sampleCohortFile = null; // set again by loadSampleDataset for built-in cohorts
        AppState.parsedData = parsed;
        AppState.fileName = name;
        if (parsed.renamedHeaders && parsed.renamedHeaders.length > 0) {
            alert(`Duplicate column name(s) were renamed so every column is unique:\n` +
                parsed.renamedHeaders.map(r => `• column ${r.column}: "${r.from}" → "${r.to}"`).join('\n'));
        }
        if (parsed.raggedRowCount > 0) {
            alert(`${parsed.raggedRowCount} row(s) have a different number of cells from the header row. Short rows were padded with blanks (treated as missing) and extra cells were ignored. Please check the file's delimiters and quoting.`);
        }

        const analysis = DataEngine.analyzeColumns(parsed.headers, parsed.rows);
        AppState.columnAnalysis = analysis;

        // Initialize default column configs
        AppState.columnConfigs = {};
        parsed.headers.forEach(col => {
            const a = analysis[col];
            AppState.columnConfigs[col] = {
                role: a.role,
                type: a.type,
                imputationStrategy: a.suggestedImputation,
                customImputeVal: ''
            };
        });

        renderDataOverview();
        DOM.toStep2Btn.disabled = false;
        const saveBtn = document.getElementById('settings-save-btn');
        if (saveBtn) { saveBtn.disabled = false; saveBtn.title = 'Save all choices for this dataset'; }
        populateSplitColumnSelects(parsed.headers);
    offerStoredSettings(parsed.headers);
        if (DOM.pipelineDirectLaunchBtn) {
            DOM.pipelineDirectLaunchBtn.disabled = false;
        }

        // Sync with pipeline tab if currently active or loaded
        if (DOM.pipelineDatasetName) {
            DOM.pipelineDatasetName.innerText = name;
        }
        if (AppState.currentStep === 'pipeline') {
            initPipelineUI();
            goToPipelineSubstep(1);
        }
    } catch (e) {
        alert(`Failed to parse CSV: ${e.message}`);
    }
}

function renderDataOverview() {
    const { headers, rows, totalRows, totalCols } = AppState.parsedData;
    const analysis = AppState.columnAnalysis;

    let totalMissing = 0;
    let colsWithMissing = 0;
    headers.forEach(h => {
        totalMissing += analysis[h].missingCount;
        if (analysis[h].missingCount > 0) colsWithMissing++;
    });


    DOM.dataStatsGrid.innerHTML = `
        <div class="stat-card">
            <span class="stat-label">File Name</span>
            <span class="stat-value" style="font-size: 16px; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(AppState.fileName)}</span>
            <span class="stat-desc">Active dataset</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Total Observations</span>
            <span class="stat-value">${totalRows}</span>
            <span class="stat-desc">Patient records / rows</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Total Variables</span>
            <span class="stat-value">${totalCols}</span>
            <span class="stat-desc">Columns detected</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Missing Cells</span>
            <span class="stat-value" style="color: ${totalMissing > 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)'}">${totalMissing}</span>
            <span class="stat-desc">${colsWithMissing} variable(s) with empty values</span>
        </div>
    `;
    // One-line summary of what will need attention (missing values, columns excluded automatically)
    const incompleteRows = rows.filter(r => headers.some(h => DataEngine.isMissing(r[h]))).length;
    const missingCols = headers.filter(h => analysis[h].missingCount > 0);
    const autoExcluded = headers.filter(h => analysis[h].role === 'meta' || analysis[h].autoMetaReason);
    const summaryParts = [`Cohort loaded: ${totalRows.toLocaleString('en-GB')} rows and ${totalCols} variables.`];
    summaryParts.push(incompleteRows
        ? `${incompleteRows.toLocaleString('en-GB')} row(s) (${(100 * incompleteRows / Math.max(1, totalRows)).toFixed(1)}%) have at least one missing value, in ${missingCols.length} variable(s)${missingCols.length <= 6 ? ` (${missingCols.map(escapeHtml).join(', ')})` : ''}; choose how to handle them in Step 3.`
        : 'No missing values.');
    if (autoExcluded.length) summaryParts.push(`Excluded from the predictors by default (identifier or date): ${autoExcluded.map(escapeHtml).join(', ')}; you can change this in Step 2.`);
    DOM.dataStatsGrid.insertAdjacentHTML('beforeend', `<div class="stat-summary" style="grid-column: 1 / -1; font-size: 12.5px; color: var(--text-secondary);">ℹ️ ${summaryParts.join(' ')}</div>`);

    // Render Data Preview Table (first rows, in a scrollable box with visible scrollbars)
    const PREVIEW_ROWS = 200;
    const previewRows = rows.slice(0, PREVIEW_ROWS);
    const previewNote = document.getElementById('data-preview-note');
    if (previewNote) {
        previewNote.textContent = `${previewRows.length < rows.length ? `Showing the first ${previewRows.length.toLocaleString('en-GB')} of ${rows.length.toLocaleString('en-GB')} rows` : `Showing all ${rows.length.toLocaleString('en-GB')} rows`} and all ${headers.length} columns. Scroll down and to the right inside the table to see more.`;
    }
    let tableHtml = `
        <table class="data-table">
            <thead>
                <tr>
                    ${headers.map(h => `<th>${escapeHtml(h)}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
                ${previewRows.map(r => `
                    <tr>
                        ${headers.map(h => {
                            const val = r[h];
                            const isMiss = DataEngine.isMissing(val);
                            return `<td style="${isMiss ? 'color: var(--accent-rose); font-style: italic;' : ''}">${isMiss ? '∅ (Empty)' : escapeHtml(val)}</td>`;
                        }).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    DOM.dataPreviewContainer.innerHTML = tableHtml;
}

export { SAMPLE_COHORTS, loadSampleDataset, createEmptyPipelineState, resetDerivedState, processCSVContent, renderDataOverview };
