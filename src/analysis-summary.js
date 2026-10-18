// analysis-summary.js: a persistent one-line summary of the analysis under the step tabs: dataset, outcome and
// positive class, independent patients, predictors, validation design, missing-data handling, recalibration
// and the active research model. Refreshed after any change on the page, so it always shows the current state.
/* global DataEngine */
import { AppState, displayModelName, escapeHtml } from './core.js';

let summaryPending = false;

// Refreshes at the next frame (several changes in a row cause one update)
function scheduleAnalysisSummary() {
    if (summaryPending) return;
    summaryPending = true;
    (typeof requestAnimationFrame === 'function' ? requestAnimationFrame : setTimeout)(() => { summaryPending = false; renderAnalysisSummary(); });
}

const val = id => document.getElementById(id)?.value || '';

function validationText() {
    const mode = document.querySelector('input[name="validation-mode"]:checked')?.value || 'split';
    const method = val('split-method') || 'random';
    const how = method === 'group' ? `by patient (${val('split-group-col') || 'column not chosen'})` : method === 'time' ? `by date (${val('split-date-col') || 'column not chosen'})` : 'random, stratified';
    if (mode === 'bootstrap') {
        const g = val('bootstrap-group-col');
        return `Option C: bootstrap, ${val('bootstrap-reps') || '?'} resamples${g ? `, by patient (${g})` : ''}`;
    }
    if (mode === 'separate') return `Option B: separate test file${AppState.separateTestFileName ? ` (${AppState.separateTestFileName})` : ' (not loaded yet)'}`;
    const pct = Math.min(95, Math.max(50, parseFloat(val('train-pct-input')) || 80));
    return `Option A: ${pct}/${100 - pct} split, ${how}`;
}

function activeModelText() {
    if (!AppState.trainedModel) return null;
    if (AppState.deployedSource === 'imported' && AppState.importedModelInfo) return `${AppState.importedModelInfo.name} (imported file)`;
    if (AppState.deployedSource === '7step' && AppState.deployedPipelineModel) return `${AppState.deployedPipelineModel.name} (7-step pipeline)`;
    return `${displayModelName(AppState.deployedModelId || AppState.selectedModelId, AppState.trainedModel)} (Step 4)`;
}

function renderAnalysisSummary() {
    const box = document.getElementById('analysis-summary');
    if (!box) return;
    const pd = AppState.parsedData, cfgs = AppState.columnConfigs || {};
    const model = activeModelText();
    if (!pd && !model) { box.hidden = true; box.innerHTML = ''; return; }
    const items = [];
    if (pd) {
        const target = pd.headers.find(h => cfgs[h] && cfgs[h].role === 'target');
        const rows = target ? pd.rows.filter(r => !DataEngine.isMissing(r[target])) : pd.rows;
        const groupCol = val('split-method') === 'group' ? val('split-group-col') : (document.querySelector('input[name="validation-mode"]:checked')?.value === 'bootstrap' ? val('bootstrap-group-col') : '');
        const patients = groupCol ? new Set(rows.map(r => String(r[groupCol] ?? '').trim())).size : null;
        items.push(['📄 Data', `${AppState.fileName || 'dataset'}: ${rows.length.toLocaleString('en-GB')} records${target ? ' with an outcome' : ''}${patients !== null ? ` from ${patients.toLocaleString('en-GB')} patients` : ''}${target && pd.rows.length > rows.length ? ` (${pd.rows.length - rows.length} without outcome excluded)` : ''}`]);
        if (target) {
            const c = cfgs[target];
            let outcome = `${target} (${c.type === 'continuous' ? 'numerical' : 'categories'})`;
            const levels = ((AppState.columnAnalysis || {})[target]?.distinctValues || []).map(String);
            if (c.type !== 'continuous' && levels.length === 2) {
                const positive = c.positiveClass !== undefined && levels.includes(String(c.positiveClass)) ? String(c.positiveClass) : [...levels].sort()[1];
                outcome = `${target}, positive class "${positive}"`;
            }
            items.push(['🎯 Outcome', outcome]);
        } else {
            items.push(['🎯 Outcome', 'not chosen yet (Step 2)']);
        }
        items.push(['🧮 Predictors', String(pd.headers.filter(h => cfgs[h] && cfgs[h].role === 'feature').length)]);
        items.push(['✂️ Validation', validationText()]);
        const mi = val('mi-select');
        items.push(['🧩 Missing data', AppState.completeCase ? 'complete cases only' : mi === 'mice5' ? 'multiple imputation (5)' : mi === 'mice10' ? 'multiple imputation (10)' : 'single imputation (Step 3)']);
        const recal = val('recalibration-select');
        if (recal && recal !== 'none') items.push(['📏 Recalibration', recal === 'platt' ? 'Platt scaling' : 'isotonic regression']);
    }
    items.push(['🤖 Active research model', model || 'none yet']);
    box.hidden = false;
    box.innerHTML = `<span class="analysis-summary-title">Analysis</span>${items.map(([k, v]) => `<span class="analysis-summary-item"><span class="k">${k}:</span> ${escapeHtml(v)}</span>`).join('')}`;
}

// Any change or click on the page may alter the analysis: refresh after it
function initAnalysisSummary() {
    ['change', 'click'].forEach(t => document.addEventListener(t, scheduleAnalysisSummary));
    scheduleAnalysisSummary();
}

export { initAnalysisSummary, renderAnalysisSummary, scheduleAnalysisSummary };
