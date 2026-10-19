// navigation.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { SURV, renderSurvivalSinglePrediction } from './survival.js';
import { survivalSpec } from './outcome-type.js';
import { updateImputationEstimate } from './multiple-imputation.js';
import { renderSampleSizePanel } from './panels.js';
import { AppState, escapeHtml } from './core.js';
import { renderImputationTable } from './imputation.js';
import { SAMPLE_COHORTS, loadSampleDataset, processCSVContent } from './ingest.js';
import { goToPipelineSubstep, initPipelineUI } from './pipeline.js';
import { DOM } from './shell.js';
import { renderSinglePredictionForm } from './single-prediction.js';
import { renderVariableConfigGrid } from './variables.js';
import { refreshModelSampleGuide } from './predictor-budget.js';
import { scheduleAnalysisSummary } from './analysis-summary.js';


function syncTrainTestSplit(trainVal) {
    let train = Math.min(95, Math.max(50, Math.round(trainVal)));
    let test = 100 - train;

    const slider = document.getElementById('train-ratio-slider');
    const trainInput = document.getElementById('train-pct-input');
    const testInput = document.getElementById('test-pct-input');
    const dispTrain = document.getElementById('split-display-train');
    const dispTest = document.getElementById('split-display-test');
    const infoTrain = document.getElementById('info-train-pct');
    const infoTest = document.getElementById('info-test-pct');
    const infoTest2 = document.getElementById('info-test-pct-2');

    if (slider) slider.value = train;
    if (trainInput) trainInput.value = train;
    if (testInput) testInput.value = test;
    if (dispTrain) dispTrain.innerText = `${train}% Train`;
    if (dispTest) dispTest.innerText = `${test}% Test`;
    if (infoTrain) infoTrain.innerText = `${train}%`;
    if (infoTest) infoTest.innerText = `${test}%`;
    if (infoTest2) infoTest2.innerText = `${test}%`;
}

// Option B File & Sample Handlers
function handleTestFileUpload(file) {
    if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
        alert("Please upload a valid .csv file for the testing cohort.");
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        processTestingCSVContent(e.target.result, file.name);
    };
    reader.readAsText(file);
}

function processTestingCSVContent(csvText, fileName) {
    try {
        const parsed = DataEngine.parseCSV(csvText);
        DataEngine.parseIssues(parsed).forEach(msg => alert(msg));
        const rows = parsed && parsed.rows ? parsed.rows : null;
        if (!rows || rows.length === 0) {
            alert("The uploaded testing CSV appears to be empty or unparseable.");
            return;
        }
        AppState.separateTestRawRows = rows;
        AppState.separateTestFileName = fileName;
        refreshOptionBSummaries();
        scheduleAnalysisSummary();
    } catch (err) {
        alert("Failed to parse testing CSV: " + err.message);
    }
}

async function loadSampleTestingCohort() {
    try {
        // Choose the validation file from the built-in cohort actually loaded, not from the file name
        const sample = AppState.sampleCohortFile ? SAMPLE_COHORTS[AppState.sampleCohortFile] : null;
        if (!sample) {
            alert("The built-in validation cohort only matches the built-in sample datasets. For your own data, please upload its separate testing CSV.");
            return;
        }
        const filename = sample.testFile;
        const label = sample.testLabel;

        const resp = await fetch(filename);
        if (!resp.ok) throw new Error("Could not load " + filename);
        const text = await resp.text();
        processTestingCSVContent(text, `${label} (${filename})`);
    } catch (err) {
        console.error("Error loading sample test cohort:", err);
        alert("Could not load sample test cohort: " + err.message);
    }
}

function refreshOptionBSummaries() {
    if (!DOM.optBTrainSummary) return;

    // Dataset (i) Summary
    if (AppState.imputedData && AppState.columnConfigs) {
        const targetCol = Object.keys(AppState.columnConfigs).find(c => AppState.columnConfigs[c].role === 'target');
        const targetCfg = targetCol ? AppState.columnConfigs[targetCol] : null;
        // Step 2 types are 'continuous' | 'ordinal' | 'nominal'; only a continuous target is regression
        const isClf = targetCfg ? targetCfg.type !== 'continuous' : true;
        DOM.optBTrainSummary.innerHTML = `
            <div><strong>Source:</strong> ${escapeHtml(AppState.fileName || 'Active Dataset')}</div>
            <div><strong>Training Cases:</strong> ${AppState.imputedData.imputedRows.length} patients (100% used for model parameter fitting)</div>
            <div><strong>Target:</strong> <span style="color: var(--accent-cyan); font-weight: 700;">${escapeHtml(targetCol || 'Unspecified')}</span> (${isClf ? 'Classification' : 'Continuous Regression'})</div>
        `;
    } else {
        DOM.optBTrainSummary.innerHTML = `
            <span style="color: var(--text-muted);">Please complete Steps 1-3 first to finalize the training cohort.</span>
        `;
    }

    // Dataset (ii) Summary
    if (AppState.separateTestRawRows) {
        const testValidCount = AppState.separateTestRawRows.length;
        let targetStatus = "Checking schema...";
        if (AppState.columnConfigs) {
            const targetCol = Object.keys(AppState.columnConfigs).find(c => AppState.columnConfigs[c].role === 'target');
            if (targetCol) {
                // Parsed rows carry every header as a key, so the first row's keys are the file's header
                const testHeaders = Object.keys(AppState.separateTestRawRows[0] || {});
                const hasTarget = testHeaders.includes(targetCol);
                targetStatus = hasTarget 
                    ? `<span style="color: var(--accent-emerald); font-weight: 600;">✓Target '${escapeHtml(targetCol)}' found</span>`
                    : `<span style="color: var(--accent-amber); font-weight: 600;">⚠︎ Target '${escapeHtml(targetCol)}' not in header!</span>`;
                const predictors = Object.keys(AppState.columnConfigs).filter(c => AppState.columnConfigs[c].role === 'feature');
                const missingPredictors = predictors.filter(c => !testHeaders.includes(c));
                if (missingPredictors.length > 0) {
                    targetStatus += `<br><span style="color: var(--accent-amber); font-weight: 600;">⚠︎ Predictor column(s) missing from the testing file (would be filled from training statistics for every row): ${escapeHtml(missingPredictors.join(', '))}</span>`;
                }
            }
        }

        if (DOM.optBTestBadge) {
            DOM.optBTestBadge.className = 'pill pill-target';
            DOM.optBTestBadge.style.background = 'rgba(168, 85, 247, 0.15)';
            DOM.optBTestBadge.style.color = 'var(--accent-purple)';
            DOM.optBTestBadge.style.borderColor = 'rgba(168, 85, 247, 0.4)';
            DOM.optBTestBadge.innerText = `✓Ready (${testValidCount} cases)`;
        }

        if (DOM.optBTestSummary) {
            DOM.optBTestSummary.innerHTML = `
                <div style="background: var(--bg-surface); padding: 10px 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); display: flex; flex-direction: column; gap: 4px;">
                    <div><strong>Testing File:</strong> ${escapeHtml(AppState.separateTestFileName || 'Testing Cohort CSV')}</div>
                    <div><strong>Total Test Cases:</strong> ${testValidCount} external patients</div>
                    <div><strong>Target Verification:</strong> ${targetStatus}</div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                        Test data are encoded and scaled using training-cohort parameters only. Rows with a missing outcome are excluded; missing predictor values are filled from training statistics.
                    </div>
                </div>
            `;
        }
    } else {
        if (DOM.optBTestBadge) {
            DOM.optBTestBadge.className = 'pill pill-meta';
            DOM.optBTestBadge.style.background = '';
            DOM.optBTestBadge.style.color = '';
            DOM.optBTestBadge.style.borderColor = '';
            DOM.optBTestBadge.innerText = 'Pending Upload';
        }
        if (DOM.optBTestSummary) {
            DOM.optBTestSummary.innerHTML = `
                <span style="color: var(--text-muted); font-size: 11.5px;">No separate testing dataset provided yet. Upload a CSV above or click 'Load Built-In Validation Cohort'.</span>
            `;
        }
    }
}

function toggleTheme() {
    AppState.theme = AppState.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', AppState.theme);
    DOM.themeToggle.innerHTML = AppState.theme === 'dark' ? 'Light' : 'Dark';
    // Re-apply the theme palette to the existing chart (colours are set per option, not via CSS)
    if (AppState.charts.loss) {
        const isDark = AppState.theme === 'dark';
        const grid = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
        const tick = isDark ? '#94a3b8' : '#64748b';
        const o = AppState.charts.loss.options;
        ['x', 'y'].forEach(ax => {
            o.scales[ax].grid.color = grid;
            o.scales[ax].ticks.color = tick;
            o.scales[ax].title.color = tick;
        });
        o.plugins.legend.labels.color = isDark ? '#f8fafc' : '#0f172a';
        if (o.plugins.title) o.plugins.title.color = tick;
        AppState.charts.loss.update('none');
    }
}

window.launch7StepPipeline = async function(event) {
    if (event) {
        try {
            if (typeof event.preventDefault === 'function') event.preventDefault();
            if (typeof event.stopPropagation === 'function') event.stopPropagation();
        } catch (err) {}
    }

    // If dataset is not loaded yet, auto-load the default clinical cohort so the user is never blocked
    if (!AppState.parsedData || !AppState.parsedData.rows || AppState.parsedData.rows.length === 0) {
        try {
            await loadSampleDataset('medical_shunt_cohort_large.csv', 'iNPH Shunt Response Cohort (simulated, 420 patients)');
        } catch (err) {
            console.warn("Auto-load shunt cohort error:", err);
        }
    }

    goToStep('pipeline');
    initPipelineUI();
    goToPipelineSubstep(AppState.pipeline && AppState.pipeline.substep ? AppState.pipeline.substep : 1);
};

function goToStep(stepNum) {
    // Enforce prerequisites
    if (stepNum !== 'pipeline') {
        const num = parseInt(stepNum, 10);
        if (num > 1 && !AppState.parsedData && !(num > 4 && AppState.deployedSource === 'imported' && AppState.trainedModel)) {
            alert("Please upload a CSV file or load a sample dataset first.");
            return;
        }
        // Step 4 trains from Step 3's imputed data; Steps 5-6 only need a deployed model (which may
        // come from the 7-step pipeline without the standard imputation step having been run)
        if (num === 4 && !AppState.imputedData) {
            alert("Please complete the Data Imputation step first.");
            return;
        }
        if (num > 4 && !AppState.trainedModel && !(survivalSpec() && SURV.result)) {
            alert("Please train a machine learning model first in Step 4.");
            return;
        }
        // Predictions for an individual patient: confirmed once per session as research use only (software that
        // informs individual patient care can be a regulated medical device)
        if (num === 5 && !AppState.researchUseAcknowledged) {
            if (!confirm('Research and education use only\n\nPredictions from this app are not validated for decisions about an individual patient\'s care and must not be used for them. Software used to inform patient care can be a regulated medical device (in the UK, MHRA). Use predictions only to explore the model, for example with test cases or de-identified research data.\n\nContinue?')) return;
            AppState.researchUseAcknowledged = true;
        }
    }

    AppState.currentStep = stepNum;

    DOM.stepTabs.forEach(tab => {
        const s = tab.dataset.step;
        const isActive = String(s) === String(stepNum);
        tab.classList.toggle('active', isActive);
        if (isActive) tab.setAttribute('aria-current', 'step'); else tab.removeAttribute('aria-current');
        const num = parseInt(s, 10);
        const targetNum = parseInt(stepNum, 10);
        if (!isNaN(num) && !isNaN(targetNum) && num < targetNum) {
            tab.classList.add('completed');
        }
    });

    DOM.stepContents.forEach(content => {
        const s = content.dataset.step;
        const isActive = String(s) === String(stepNum);
        content.classList.toggle('active', isActive);
        content.style.display = isActive ? 'flex' : 'none';
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (stepNum === 2 || stepNum === '2') renderVariableConfigGrid();
    if (stepNum === 3 || stepNum === '3') { renderImputationTable(); updateImputationEstimate(); }
    if (stepNum === 4 || stepNum === '4') { if (AppState.updateTrainButtonLabel) AppState.updateTrainButtonLabel(); refreshOptionBSummaries(); renderSampleSizePanel(); updateImputationEstimate(); refreshModelSampleGuide(); }
    if (stepNum === 5 || stepNum === '5') { if (survivalSpec()) renderSurvivalSinglePrediction('survival-step5'); else { const sb = document.getElementById('survival-step5'); if (sb) sb.innerHTML = ''; renderSinglePredictionForm(); } }
    if (stepNum === 'pipeline') {
        initPipelineUI();
        goToPipelineSubstep(AppState.pipeline && AppState.pipeline.substep ? AppState.pipeline.substep : 1);
    }
}
window.goToStep = goToStep;

function handleFileUpload(file) {
    if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
        alert("Please upload a valid .csv file.");
        return;
    }
    AppState.fileName = file.name;
    const reader = new FileReader();
    reader.onload = (e) => {
        processCSVContent(e.target.result, file.name);
    };
    reader.readAsText(file);
}

export { syncTrainTestSplit, handleTestFileUpload, processTestingCSVContent, loadSampleTestingCohort, refreshOptionBSummaries, toggleTheme, goToStep, handleFileUpload };
