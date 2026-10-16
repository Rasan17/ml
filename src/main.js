// main.js: entry point of the Universal Medical ML/NeuralNet Predictor application controller.
// Loads every module, initialises the shared state and starts the interface once the page is ready.
import { AppState } from './core.js';
import './watchdog.js';
import './compute.js';
import { initDOM, setupEventListeners } from './shell.js';
import './navigation.js';
import { createEmptyPipelineState } from './ingest.js';
import './variables.js';
import './imputation.js';
import './training-setup.js';
import { renderSampleSizePanel } from './panels.js';
import './cv-tuning.js';
import './standard-training.js';
import './single-prediction.js';
import './batch-prediction.js';
import './pipeline.js';
import './pipeline-results.js';
import './reports.js';
import './model-io.js';
import './survival.js';
import { handleUnexpectedError } from './settings.js';

AppState.pipeline = createEmptyPipelineState(42);

window.addEventListener('error', (e) => handleUnexpectedError(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => handleUnexpectedError(e.reason));

function start() {
    initDOM();
    setupEventListeners();
    // Keep the sample-size check in step with the chosen split and validation protocol
    ['train-pct-input', 'train-ratio-slider', 'val-mode-a', 'val-mode-b'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.addEventListener('change', renderSampleSizePanel); el.addEventListener('input', renderSampleSizePanel); }
    });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
else start();
