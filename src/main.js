// main.js: entry point of the Universal Medical ML/NeuralNet Predictor application controller.
// Loads every module, initialises the shared state and starts the interface once the page is ready.
import { AppState } from './core.js';
import './watchdog.js';
import { cancelAll } from './compute.js';
import './validation-core.js';
import { initDOM, setupEventListeners } from './shell.js';
import './navigation.js';
import { createEmptyPipelineState } from './ingest.js';
import { renderVariableIncludeSummary } from './variables.js';
import { executeImputation, renderImputationTable } from './imputation.js';
import './training-setup.js';
import { renderSampleSizePanel } from './panels.js';
import './subgroups.js';
import { clearImputationCache, loadImputedDataFile, saveImputedData, updateImputationEstimate, updateImputedDataButtons } from './multiple-imputation.js';
import './cv-tuning.js';
import './standard-training.js';
import './training-actions.js';
import './evaluation-dashboard.js';
import './single-prediction.js';
import './batch-prediction.js';
import './pipeline.js';
import './pipeline-step1.js';
import './pipeline-screening.js';
import './pipeline-balance.js';
import './pipeline-train.js';
import './pipeline-results.js';
import './reports.js';
import { updateModelExportButton } from './model-io.js';
import './outcome-type.js';
import './survival.js';
import { refreshModelSampleGuide } from './predictor-budget.js';
import { downloadTripodChecklist } from './tripod.js';
import { downloadHtmlReport, downloadValidationReport } from './html-report.js';
import './external-validation.js';
import { forgetStoredSettings, handleUnexpectedError } from './settings.js';
import { initAnalysisSummary } from './analysis-summary.js';
import './interpretation.js';
import './safeguards.js';
import { downloadModelCard } from './model-card.js';

AppState.pipeline = createEmptyPipelineState(42);

window.addEventListener('error', (e) => handleUnexpectedError(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => handleUnexpectedError(e.reason));

// Delegated handling of controls marked with data-action (see actionAttrs in core.js): the named action is
// called with the stored arguments, then the selected index (for a change) and the event. No inline handlers.
// Only the actions listed here can run, whatever a page element names (never an arbitrary global function).
const ACTIONS = new Set(['copySeedToClipboard', 'deployPipelineModel', 'deployTestedModel', 'goToStep', 'launch7StepPipeline',
    'pipelineDropFeature', 'pipelineDropFeatureStep4', 'pipelineKeepBothPairs', 'pipelineRestoreFeatureStep4', 'pipelineToggleReinstateFeature', 'renderCalibrationChoice', 'runMiceOddsRatios',
    'runPermutationImportance', 'switchToBaselineModel', 'switchToPipelineModel', 'trainStandardBaselineFromPipeline']);
function runAction(e, kind) {
    const el = e.target && e.target.closest ? e.target.closest('[data-action]') : null;
    if (!el || (el.dataset.event || 'click') !== kind) return;
    if (!ACTIONS.has(el.dataset.action)) return;
    const fn = window[el.dataset.action];
    if (typeof fn !== 'function') return;
    let args = [];
    try { args = JSON.parse(el.dataset.args || '[]'); } catch (err) { args = []; }
    if (kind === 'change') args.push(el.selectedIndex);
    fn(...args, e);
}
document.addEventListener('click', e => runAction(e, 'click'));
document.addEventListener('change', e => runAction(e, 'change'));

function start() {
    // Seed boxes inside the model cards: a click there must not also select the card
    document.querySelectorAll('[data-stop-click]').forEach(el => el.addEventListener('click', e => e.stopPropagation()));
    initDOM();
    setupEventListeners();
    document.getElementById('standard-tripod-btn')?.addEventListener('click', () => downloadTripodChecklist('standard'));
    document.getElementById('pipeline-tripod-btn')?.addEventListener('click', () => downloadTripodChecklist('pipeline'));
    document.getElementById('standard-html-report-btn')?.addEventListener('click', () => downloadHtmlReport('standard'));
    window.downloadValidationReport = downloadValidationReport;
    document.getElementById('pipeline-html-report-btn')?.addEventListener('click', () => downloadHtmlReport('pipeline'));
    document.getElementById('mi-select')?.addEventListener('change', () => {
        updateImputationEstimate();
        // The table and the Step 3 result follow the single / multiple imputation choice
        if (AppState.parsedData && AppState.columnAnalysis) {
            renderImputationTable();
            if (AppState.imputedData) executeImputation();
        }
    });
    document.getElementById('mi-save-btn')?.addEventListener('click', saveImputedData);
    document.getElementById('mi-load-btn')?.addEventListener('click', () => document.getElementById('mi-file-input')?.click());
    document.getElementById('mi-file-input')?.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0];
        if (file) loadImputedDataFile(file);
        e.target.value = '';
    });
    updateImputedDataButtons();
    initAnalysisSummary();
    // Firefox restores buttons' enabled state on reload: set the data-dependent ones from the (empty) page state
    updateModelExportButton();
    ['settings-save-btn', 'run-batch-predict-btn'].forEach(id => { const el = document.getElementById(id); if (el) el.disabled = true; });
    // Interface level (remembered in this browser): settings hidden at a level are reset to their defaults, so
    // nothing the user cannot see changes the analysis
    const levelSel = document.getElementById('ui-level-select');
    const applyLevel = (level, reset) => {
        document.body.dataset.uiLevel = level;
        if (!reset) return;
        const hidden = [...document.querySelectorAll('[data-level], .hyperparam-box')].filter(el => getComputedStyle(el).display === 'none' && (el.matches('.hyperparam-box') || (level === 'beginner') || el.dataset.level === 'advanced'));
        hidden.forEach(box => box.querySelectorAll('input, select').forEach(el => {
            let changed = false;
            if (el.type === 'checkbox' || el.type === 'radio') { if (el.checked !== el.defaultChecked) { el.checked = el.defaultChecked; changed = true; } }
            else if (el.tagName === 'SELECT') { const d = [...el.options].find(o => o.defaultSelected) || el.options[0]; if (d && el.value !== d.value) { el.value = d.value; changed = true; } }
            else if (el.value !== el.defaultValue) { el.value = el.defaultValue; changed = true; }
            if (changed) el.dispatchEvent(new Event('change', { bubbles: true }));
        }));
    };
    // First visit: Beginner (recommended defaults); the choice is then remembered in this browser
    let savedLevel = 'beginner';
    try { savedLevel = localStorage.getItem('ummp-ui-level') || 'beginner'; } catch (e) { /* storage unavailable */ }
    if (levelSel) {
        levelSel.value = ['beginner', 'intermediate', 'advanced'].includes(savedLevel) ? savedLevel : 'beginner';
        applyLevel(levelSel.value, levelSel.value !== 'advanced');
        levelSel.addEventListener('change', () => {
            applyLevel(levelSel.value, true);
            try { localStorage.setItem('ummp-ui-level', levelSel.value); } catch (e) { /* storage unavailable */ }
        });
    }
    document.getElementById('model-card-btn')?.addEventListener('click', downloadModelCard);
    // Start page task choice
    document.getElementById('task-develop')?.addEventListener('click', () => {
        const input = document.getElementById('csv-file-input');
        document.getElementById('step-1-content')?.querySelector('.card')?.scrollIntoView({ behavior: 'smooth' });
        if (input) input.click();
    });
    document.getElementById('task-validate')?.addEventListener('click', () => {
        // The file dialog opens straight away (a dialog box first would block it in some browsers); the steps are shown below
        document.getElementById('model-import-btn')?.click();
        const hint = document.getElementById('task-hint');
        if (hint) { hint.style.display = 'block'; hint.textContent = 'Validating a shared model: 1) choose the model file in the window that opened (or click "Import trained model"); 2) in Step 6, upload a CSV of your patients that includes the outcome column. You get discrimination, calibration and clinical usefulness on your patients, with a printable validation report; no patient data leave your computer.'; }
    });
    document.getElementById('task-demo')?.addEventListener('click', () => document.getElementById('sample-shunt-large-btn')?.click());
    // Clear session: stops background work and reloads the page, which discards every dataset, result, model and
    // cached imputation held in memory. Browser storage holds only saved settings (choices and column names).
    document.getElementById('clear-session-btn')?.addEventListener('click', () => {
        if (!confirm('Remove all data from this page?\n\nEvery loaded dataset, result, trained model and cached imputation is discarded (files you saved are not affected).')) return;
        // Settings remembered in this browser hold choices, column names, file names and category labels (e.g.
        // the positive class), not patient rows; on a shared computer they should go too
        if (confirm('Also forget the analysis settings this browser remembered for your datasets?\n\nThey contain no patient rows but do include file names, column names and category labels. Choose OK on a shared computer.')) forgetStoredSettings();
        cancelAll('Session cleared.');
        clearImputationCache();
        location.reload();
    });
    // Keep the sample-size check in step with the chosen split and validation protocol
    ['train-pct-input', 'train-ratio-slider', 'val-mode-a', 'val-mode-b'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.addEventListener('change', renderSampleSizePanel); el.addEventListener('input', renderSampleSizePanel); }
    });
    // ... and the Step 2 guide on how many predictors the Training set supports
    ['train-pct-input', 'train-ratio-slider', 'val-mode-a', 'val-mode-b', 'val-mode-c'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', () => { if (AppState.parsedData) { renderVariableIncludeSummary(); refreshModelSampleGuide(); } });
    });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
else start();
