// pipeline.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { AppState } from './core.js';
import { renderPipelineStep5, renderPipelineStep6UI } from './pipeline-balance.js';
import { renderPipelineStep2, renderPipelineStep3, renderPipelineStep4 } from './pipeline-screening.js';
import { renderPipelineStep1Results } from './pipeline-step1.js';
import { DOM } from './shell.js';
import { refreshPipelineModelGuide } from './predictor-budget.js';

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
    if (substepNum === 7) refreshPipelineModelGuide();
}

export { updatePipelineCompleteCaseInfo, initPipelineUI, goToPipelineSubstep };
