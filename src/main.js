// main.js: entry point of the Universal Medical ML/NeuralNet Predictor application controller.
// Loads every module, initialises the shared state and starts the interface once the page is ready.
import { AppState } from './core.js';
import './watchdog.js';
import './compute.js';
import './validation-core.js';
import { initDOM, setupEventListeners } from './shell.js';
import './navigation.js';
import { createEmptyPipelineState } from './ingest.js';
import './variables.js';
import './imputation.js';
import './training-setup.js';
import { renderSampleSizePanel } from './panels.js';
import './subgroups.js';
import { loadImputedDataFile, saveImputedData, updateImputationEstimate, updateImputedDataButtons } from './multiple-imputation.js';
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
import './model-io.js';
import './survival.js';
import { downloadTripodChecklist } from './tripod.js';
import { downloadHtmlReport, downloadValidationReport } from './html-report.js';
import './external-validation.js';
import { handleUnexpectedError } from './settings.js';

AppState.pipeline = createEmptyPipelineState(42);

window.addEventListener('error', (e) => handleUnexpectedError(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => handleUnexpectedError(e.reason));

// Delegated handling of controls marked with data-action (see actionAttrs in core.js): window[fn] is called
// with the stored arguments, then the selected index (for a change) and the event. No inline handlers.
function runAction(e, kind) {
    const el = e.target && e.target.closest ? e.target.closest('[data-action]') : null;
    if (!el || (el.dataset.event || 'click') !== kind) return;
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
    document.getElementById('mi-select')?.addEventListener('change', updateImputationEstimate);
    document.getElementById('mi-save-btn')?.addEventListener('click', saveImputedData);
    document.getElementById('mi-load-btn')?.addEventListener('click', () => document.getElementById('mi-file-input')?.click());
    document.getElementById('mi-file-input')?.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0];
        if (file) loadImputedDataFile(file);
        e.target.value = '';
    });
    updateImputedDataButtons();
    // Keep the sample-size check in step with the chosen split and validation protocol
    ['train-pct-input', 'train-ratio-slider', 'val-mode-a', 'val-mode-b'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.addEventListener('change', renderSampleSizePanel); el.addEventListener('input', renderSampleSizePanel); }
    });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
else start();
