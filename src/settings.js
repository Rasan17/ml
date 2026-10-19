// settings.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { setMissingIndicators } from './variables.js';
import { applySurvivalRoles } from './outcome-type.js';
import { SURV } from './survival.js';
import { APP_VERSION, AppState, RANKING_METRICS, escapeHtml } from './core.js';
import { executeImputation, renderImputationTable } from './imputation.js';
import { goToStep } from './navigation.js';
import { goToPipelineSubstep, updatePipelineCompleteCaseInfo } from './pipeline.js';
import { executePipelineStep6 } from './pipeline-balance.js';
import { executePipelineStep1 } from './pipeline-step1.js';
import { executePipelineStep7 } from './pipeline-train.js';
import { DOM } from './shell.js';
import { startModelTraining } from './standard-training.js';
import { showSeedToast, syncRankingSelectors } from './training-actions.js';
import { renderVariableConfigGrid } from './variables.js';


// =========================================================================
// ANALYSIS SETTINGS: SAVE / RESTORE / ONE-CLICK RE-RUN
// Settings hold the user's choices only (column names, roles, options, seeds, hyper-parameters),
// never patient data. Stored as a downloadable .json file and, per dataset, in this browser.
// =========================================================================
const SETTINGS_SCHEMA = 1;
const SETTINGS_APP = 'Universal Medical ML/NeuralNet Predictor';
const SETTINGS_KEY_PREFIX = 'ummp-settings:';
const SETTINGS_MAX_STORED = 20;
const VARIABLE_FIELDS = ['role', 'prevRole', 'type', 'imputationStrategy', 'prevImputationStrategy', 'customImputeVal', 'positiveClass', 'ordinalOrder'];

// A settings file is untrusted input: each saved variable choice must be one the app itself can make
const ROLES = ['feature', 'target', 'meta'], TYPES = ['continuous', 'ordinal', 'nominal'];
const STRATEGIES = ['median', 'mean', 'mode', 'constant', 'knn', 'drop'];
function validVariableField(f, v) {
    if (f === 'role' || f === 'prevRole') return ROLES.includes(v);
    if (f === 'type') return TYPES.includes(v);
    if (f === 'imputationStrategy' || f === 'prevImputationStrategy') return STRATEGIES.includes(v);
    if (f === 'customImputeVal' || f === 'positiveClass') return (typeof v === 'string' && v.length <= 500) || (typeof v === 'number' && isFinite(v));
    if (f === 'ordinalOrder') return Array.isArray(v) && v.length <= 1000 && v.every(x => typeof x === 'string' && x.length <= 500);
    return false;
}

// Identifies a dataset by its column headers (FNV-1a hash), so a re-loaded or updated file matches
function datasetSignature(headers) {
    let h = 0x811c9dc5;
    // Derived "(was missing)" indicator columns are not part of the original file's identity
    const str = headers.filter(x => !/ \(was missing\)$/.test(x)).join('\u0001');
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h.toString(16).padStart(8, '0');
}

// Form controls to remember (by id; radio groups by name). Controls whose state lives in AppState,
// or that are rebuilt per run (model cards, scope, ranking metric, results tables), are excluded.
function isRememberedControl(el) {
    if (el.type === 'file' || el.type === 'button' || el.type === 'submit') return false;
    if (el.classList.contains('model-select-checkbox') || el.classList.contains('ranking-metric-select')) return false;
    if (el.type === 'radio') return Boolean(el.name) && el.name !== 'model-scope' && !/^scope-/.test(el.id);
    if (!el.id || /^scope-/.test(el.id) || el.id === 'pipeline-univ-master-check') return false;
    return true;
}

function collectFormState(rootSelector) {
    const root = document.querySelector(rootSelector);
    const state = {};
    if (!root) return state;
    root.querySelectorAll('input, select, textarea').forEach(el => {
        if (!isRememberedControl(el)) return;
        if (el.type === 'radio') {
            if (el.checked) state[`radio:${el.name}`] = el.value;
        } else if (el.type === 'checkbox') {
            state[el.id] = el.checked;
        } else {
            state[el.id] = el.value;
        }
    });
    return state;
}

function applyFormState(state, rootSelector) {
    if (!state || typeof state !== 'object') return;
    // Only controls inside the part of the page the settings were saved from can be restored
    const root = document.querySelector(rootSelector);
    if (!root) return;
    const fire = (el, types) => types.forEach(t => el.dispatchEvent(new Event(t, { bubbles: true })));
    // Radios first: some (seed mode, validation mode) reset related inputs when they change
    Object.keys(state).filter(k => k.startsWith('radio:')).forEach(k => {
        const name = k.slice(6);
        const el = Array.from(root.querySelectorAll('input[type="radio"]')).find(r => r.name === name && r.value === state[k]);
        if (el && !el.checked) { el.checked = true; fire(el, ['change']); }
    });
    Object.keys(state).filter(k => !k.startsWith('radio:')).forEach(id => {
        const el = document.getElementById(id);
        if (!el || !root.contains(el) || !isRememberedControl(el)) return;
        if (el.type === 'checkbox') {
            if (el.checked !== Boolean(state[id])) { el.checked = Boolean(state[id]); fire(el, ['change']); }
        } else if (el.value !== String(state[id])) {
            el.value = state[id];
            fire(el, ['input', 'change']);
        }
    });
}

function collectSettings() {
    const { headers, totalRows } = AppState.parsedData;
    const variables = {};
    headers.forEach(h => {
        const c = AppState.columnConfigs[h] || {};
        variables[h] = {};
        VARIABLE_FIELDS.forEach(f => { if (c[f] !== undefined) variables[h][f] = c[f]; });
    });
    return {
        app: SETTINGS_APP,
        appVersion: APP_VERSION,
        schema: SETTINGS_SCHEMA,
        savedAt: new Date().toISOString(),
        note: 'Analysis choices only; contains no patient data.',
        dataset: { fileName: AppState.fileName, headers: headers.filter(x => !(AppState.columnConfigs[x] && AppState.columnConfigs[x].derivedFrom)), rowCount: totalRows, signature: datasetSignature(headers) },
        variables,
        // Step 2 outcome type: a time-to-event outcome is a time column, an event column, its event value and a horizon
        outcome: AppState.outcomeType === 'survival' && AppState.survival ? { type: 'survival', ...AppState.survival } : { type: 'standard' },
        standard: {
            completeCase: Boolean(AppState.completeCase),
            missingIndicators: Boolean(AppState.missingIndicators),
            modelScope: AppState.modelScope,
            selectedModelIds: (AppState.selectedModelIds || []).slice(),
            rankingMetricClf: AppState.rankingMetricClf || 'macroF1',
            rankingMetricReg: AppState.rankingMetricReg || 'r2',
            form: collectFormState('#step-4-content'),
            // The imputation choice sits in Step 3 (one pane for single and multiple imputation)
            multipleImputation: document.getElementById('mi-select')?.value || 'single',
            miceQuickApproximation: Boolean(document.getElementById('mi-quick-cv')?.checked)
        },
        pipeline: {
            form: collectFormState('#step-pipeline-content'),
            droppedFeatures: Array.from((AppState.pipeline && AppState.pipeline.droppedFeatures) || []),
            // Step 1 resets the evaluation seed to the split seed, so the seed used in Step 7 is kept separately
            evalSeed: DOM.pipelineEvalSeedInput ? DOM.pipelineEvalSeedInput.value : '',
            reinstatedFeatures: Array.from((AppState.pipeline && AppState.pipeline.reinstatedFeatures) || [])
        },
        used: {
            standard: Boolean((AppState.allTestedResults && AppState.allTestedResults.length) || (AppState.outcomeType === 'survival' && SURV.result)),
            pipeline: Boolean(AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length)
        }
    };
}

// Browser memory (per-viewer convenience; may be unavailable or cleared, so always optional)
// Removes every set of settings this app remembered in this browser
function forgetStoredSettings() {
    try {
        Object.keys(localStorage).filter(k => k.startsWith(SETTINGS_KEY_PREFIX)).forEach(k => localStorage.removeItem(k));
    } catch (e) { /* storage unavailable */ }
}

function saveSettingsToBrowser() {
    if (!AppState.parsedData) return;
    try {
        const settings = collectSettings();
        const key = SETTINGS_KEY_PREFIX + settings.dataset.signature;
        localStorage.setItem(key, JSON.stringify(settings));
        const index = JSON.parse(localStorage.getItem(SETTINGS_KEY_PREFIX + 'index') || '[]').filter(k => k !== key);
        index.unshift(key);
        index.slice(SETTINGS_MAX_STORED).forEach(k => localStorage.removeItem(k));
        localStorage.setItem(SETTINGS_KEY_PREFIX + 'index', JSON.stringify(index.slice(0, SETTINGS_MAX_STORED)));
    } catch (e) { /* storage unavailable: the downloadable file still works */ }
}

function readStoredSettings(headers) {
    try {
        const raw = localStorage.getItem(SETTINGS_KEY_PREFIX + datasetSignature(headers));
        return raw ? JSON.parse(raw) : null;
    } catch (e) {
        return null;
    }
}

function downloadSettings() {
    if (!AppState.parsedData) {
        alert('Load a dataset first.');
        return;
    }
    const settings = collectSettings();
    saveSettingsToBrowser();
    const base = String(AppState.fileName || 'dataset').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_').slice(0, 60) || 'dataset';
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${base}_analysis-settings_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showSeedToast('Settings saved (file downloaded and remembered in this browser)');
}

function loadSettingsFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        let settings;
        try {
            settings = JSON.parse(e.target.result);
        } catch (err) {
            alert('This file is not a valid settings file (it could not be read as JSON).');
            return;
        }
        if (!settings || settings.app !== SETTINGS_APP || !settings.variables || !settings.dataset) {
            alert('This file is not a settings file from this app.');
            return;
        }
        if (!AppState.parsedData) {
            alert(`Please load the dataset first (settings were saved for "${settings.dataset.fileName}"), then load the settings file.`);
            return;
        }
        applySettings(settings);
    };
    reader.readAsText(file);
}

function offerStoredSettings(headers) {
    const banner = document.getElementById('settings-banner');
    if (!banner) return;
    banner.innerHTML = '';
    const stored = readStoredSettings(headers);
    if (!stored) return;
    const when = new Date(stored.savedAt).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' });
    banner.innerHTML = `
        <div class="settings-banner-box">
            <div>Settings from a previous session with this dataset are available (saved ${escapeHtml(when)}${stored.dataset.fileName ? `, file "${escapeHtml(stored.dataset.fileName)}"` : ''}).</div>
            <div class="settings-banner-actions">
                <button type="button" class="btn btn-primary btn-sm" id="settings-apply-stored">Apply these settings</button>
                <button type="button" class="btn btn-secondary btn-sm" id="settings-dismiss">Dismiss</button>
            </div>
        </div>`;
    document.getElementById('settings-apply-stored').addEventListener('click', () => applySettings(stored));
    document.getElementById('settings-dismiss').addEventListener('click', () => { banner.innerHTML = ''; });
}

function applySettings(settings) {
    // Derived "was missing" indicators first, so their columns exist when the variable choices are restored
    setMissingIndicators(Boolean((settings.standard || {}).missingIndicators));
    const headers = AppState.parsedData.headers;
    const savedHeaders = settings.dataset.headers || Object.keys(settings.variables);
    const missing = savedHeaders.filter(h => !headers.includes(h));
    const extra = headers.filter(h => !savedHeaders.includes(h));

    // Step 2/3 choices (only for columns present in this file)
    headers.forEach(h => {
        const saved = settings.variables[h];
        if (!saved || !AppState.columnConfigs[h]) return;
        const cfg = AppState.columnConfigs[h];
        // Optional choices absent from the saved file must not linger from the current session
        ['prevRole', 'prevImputationStrategy', 'positiveClass', 'ordinalOrder', 'survivalTime', 'survivalEvent', 'survivalDisplaced', 'survivalPrevRole'].forEach(f => { delete cfg[f]; });
        VARIABLE_FIELDS.forEach(f => { if (saved[f] !== undefined && validVariableField(f, saved[f])) cfg[f] = Array.isArray(saved[f]) ? saved[f].slice() : saved[f]; });
    });
    // Exactly one target
    const targets = headers.filter(h => AppState.columnConfigs[h].role === 'target');
    targets.slice(1).forEach(h => { AppState.columnConfigs[h].role = 'feature'; });
    // Outcome type (Step 2): a time-to-event outcome only if its columns are in this file
    const oc = settings.outcome || {};
    if (oc.type === 'survival' && headers.includes(oc.timeCol) && headers.includes(oc.eventCol)) {
        AppState.outcomeType = 'survival';
        AppState.survival = { timeCol: oc.timeCol, eventCol: oc.eventCol, eventValue: String(oc.eventValue ?? ''), horizon: Number(oc.horizon) > 0 ? Number(oc.horizon) : null };
    } else {
        AppState.outcomeType = 'standard'; AppState.survival = null;
    }
    applySurvivalRoles();
    document.body.dataset.outcome = AppState.outcomeType;

    // Standard workflow (Step 4)
    const std = settings.standard || {};
    AppState.completeCase = Boolean(std.completeCase);
    if (Object.prototype.hasOwnProperty.call(RANKING_METRICS.classification, std.rankingMetricClf)) AppState.rankingMetricClf = std.rankingMetricClf;
    if (Object.prototype.hasOwnProperty.call(RANKING_METRICS.regression, std.rankingMetricReg)) AppState.rankingMetricReg = std.rankingMetricReg;
    if (AppState.applyModelSelection && std.selectedModelIds) AppState.applyModelSelection(std.modelScope, std.selectedModelIds);
    applyFormState(std.form, '#step-4-content');
    // Imputation choice (Step 3); older settings files kept it in the Step 4 form
    const miVal = ['single', 'mice5', 'mice10'].find(v => v === (std.multipleImputation || (std.form || {})['mi-select']));
    const miEl = document.getElementById('mi-select');
    if (miEl && miVal && miEl.value !== miVal) { miEl.value = miVal; miEl.dispatchEvent(new Event('change', { bubbles: true })); }
    const quickEl = document.getElementById('mi-quick-cv');
    if (quickEl) quickEl.checked = std.miceQuickApproximation === true;

    // 7-step pipeline
    const pipe = settings.pipeline || {};
    applyFormState(pipe.form, '#step-pipeline-content');
    AppState.pendingPipelineDropped = (pipe.droppedFeatures || []).filter(f => headers.includes(f));
    AppState.pendingPipelineReinstated = (pipe.reinstatedFeatures || []).filter(f => headers.includes(f));
    AppState.pendingPipelineEvalSeed = pipe.evalSeed !== undefined && pipe.evalSeed !== '' ? String(pipe.evalSeed)
        : (DOM.pipelineEvalSeedInput && DOM.pipelineEvalSeedInput.value ? DOM.pipelineEvalSeedInput.value : null);

    // Restored settings may include expert choices: show every setting so nothing that affects the analysis is hidden
    const lvl = document.getElementById('ui-level-select');
    if (lvl && lvl.value !== 'advanced') { lvl.value = 'advanced'; document.body.dataset.uiLevel = 'advanced'; try { localStorage.setItem('ummp-ui-level', 'advanced'); } catch (e) { /* storage unavailable */ } }

    // Earlier results no longer match the restored choices
    AppState.imputedData = null;
    if (DOM.toStep4Btn) DOM.toStep4Btn.disabled = true;
    if (DOM.imputeAuditBox) DOM.imputeAuditBox.innerHTML = '';
    const targetCol = headers.find(h => AppState.columnConfigs[h].role === 'target');
    syncRankingSelectors(targetCol ? AppState.columnConfigs[targetCol].type !== 'continuous' : true);
    if (String(AppState.currentStep) === '2') renderVariableConfigGrid();
    if (String(AppState.currentStep) === '3') renderImputationTable();
    updatePipelineCompleteCaseInfo();

    // Confirmation with one-click re-run
    const banner = document.getElementById('settings-banner');
    const when = new Date(settings.savedAt).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' });
    const warnings = [];
    if (missing.length) warnings.push(`Saved column(s) not in this file (ignored): ${missing.join(', ')}.`);
    if (extra.length) warnings.push(`New column(s) in this file kept their automatic settings: ${extra.join(', ')}.`);
    const optionB = std.form && std.form['radio:validation-mode'] === 'separate';
    if (optionB) warnings.push('These settings use Option B: load the separate testing file (Dataset ii) in Step 4 before running the standard workflow.');
    if (banner) {
        banner.innerHTML = `
            <div class="settings-banner-box">
                <div>✓Settings applied (saved ${escapeHtml(when)}). Every step is filled in; review anything you like, or reproduce the analysis now:</div>
                ${warnings.map(w => `<div class="settings-warn">⚠︎ ${escapeHtml(w)}</div>`).join('')}
                <div class="settings-banner-actions">
                    <button type="button" class="btn ${settings.used && settings.used.standard ? 'btn-primary' : 'btn-secondary'} btn-sm" id="settings-run-standard">▶ Run standard workflow</button>
                    <button type="button" class="btn ${settings.used && settings.used.pipeline ? 'btn-primary' : 'btn-secondary'} btn-sm" id="settings-run-pipeline">▶ Run 7-step pipeline</button>
                    <button type="button" class="btn btn-secondary btn-sm" id="settings-dismiss">Close</button>
                </div>
            </div>`;
        document.getElementById('settings-run-standard').addEventListener('click', runStandardWithSettings);
        document.getElementById('settings-run-pipeline').addEventListener('click', runPipelineWithSettings);
        document.getElementById('settings-dismiss').addEventListener('click', () => { banner.innerHTML = ''; });
    }
    showSeedToast('Settings applied');
}

async function runStandardWithSettings() {
    goToStep(2);
    goToStep(3);               // renders the imputation table (applies the complete-case choice)
    executeImputation();
    if (!AppState.imputedData) return;
    goToStep(4);
    const optionB = DOM.valModeB && DOM.valModeB.checked;
    if (optionB && (!AppState.separateTestRawRows || AppState.separateTestRawRows.length === 0)) {
        alert('These settings use Option B. Please load the separate testing file (Dataset ii) in this step, then click Train.');
        return;
    }
    await startModelTraining();
}

async function runPipelineWithSettings() {
    goToStep('pipeline');
    goToPipelineSubstep(1);
    executePipelineStep1();
    if (!AppState.pipeline.trainRows) return;
    const dropped = (AppState.pendingPipelineDropped || []).filter(f => AppState.pipeline.featureCols.includes(f));
    AppState.pipeline.droppedFeatures = new Set(dropped);
    if (AppState.pendingPipelineEvalSeed !== null && AppState.pendingPipelineEvalSeed !== undefined && DOM.pipelineEvalSeedInput) {
        DOM.pipelineEvalSeedInput.value = AppState.pendingPipelineEvalSeed;
    }
    for (const n of [2, 3, 4, 5]) goToPipelineSubstep(n);
    goToPipelineSubstep(6);
    AppState.pipeline.reinstatedFeatures = new Set(AppState.pendingPipelineReinstated || []);
    await executePipelineStep6();
    if (!AppState.pipeline.rfImportance || AppState.pipeline.rfImportance.length === 0) return;
    goToPipelineSubstep(7);
    await executePipelineStep7();
}

// Safety net: an unexpected error anywhere (e.g. in a form renderer) is reported to the user and the
// training buttons are re-enabled, instead of leaving the app looking busy or half-updated
function handleUnexpectedError(err) {
    const msg = err && err.message ? err.message : String(err || 'Unknown error');
    if (msg === 'Training aborted by user.' || msg === 'Operation halted by user.') return;
    console.error('Unexpected error:', err);
    if (DOM.trainModelBtn) DOM.trainModelBtn.disabled = false;
    if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.disabled = false;
    try { showSeedToast(`⚠︎ Unexpected error: ${msg}. Please check your data/settings and retry.`); } catch (e) { /* ignore */ }
}

export { forgetStoredSettings, SETTINGS_SCHEMA, SETTINGS_APP, SETTINGS_KEY_PREFIX, SETTINGS_MAX_STORED, VARIABLE_FIELDS, datasetSignature, isRememberedControl, collectFormState, applyFormState, collectSettings, saveSettingsToBrowser, readStoredSettings, downloadSettings, loadSettingsFile, offerStoredSettings, applySettings, runStandardWithSettings, runPipelineWithSettings, handleUnexpectedError };
