// training-actions.js: part of the Universal Medical ML/NeuralNet Predictor application controller (split from standard-training.js)
import { AppState, compareEvalResults, displayModelName, rankingMetricKey } from './core.js';
import { renderEvaluationDashboard } from './evaluation-dashboard.js';
import { updatePipelineActiveDeploymentStatus, updatePredictionProvenanceBar } from './pipeline-results.js';
import { renderPipelineLeaderboardOnly } from './pipeline-train.js';
import { DOM } from './shell.js';
import { renderSinglePredictionForm } from './single-prediction.js';

// Shows the ranking metric that applies to the current outcome type in both selectors
function syncRankingSelectors(isClassification) {
    const value = `${isClassification ? 'clf' : 'reg'}:${rankingMetricKey(isClassification)}`;
    const otherPrefix = isClassification ? 'reg:' : 'clf:';
    document.querySelectorAll('.ranking-metric-select').forEach(sel => {
        // Metrics for the other outcome type cannot be chosen (selecting one would have no effect
        // and the selector would snap back), so they are disabled while these results are shown
        Array.from(sel.options).forEach(opt => { opt.disabled = opt.value.startsWith(otherPrefix); });
        sel.value = value;
    });
}

// Frees GPU (WebGL) memory held by trained neural networks / ensembles that are no longer reachable.
// The currently deployed model is never disposed.
function disposeModels(models) {
    (models || []).forEach(m => {
        if (m && m !== AppState.trainedModel && typeof m.dispose === 'function') {
            try { m.dispose(); } catch (e) { /* ignore */ }
        }
    });
}

// Fits a non-neural model. Random Forest and logistic/ridge regression yield to the browser between
// trees/iterations so the UI repaints and Force Stop can interrupt them; other models fit synchronously.
async function fitWithYields(model, X, y, watchdog) {
    if (typeof model.fitAsync === 'function') {
        await model.fitAsync(X, y, {
            onYield: () => {
                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");
            }
        });
    } else {
        model.fit(X, y);
    }
}

// Lets Force Stop interrupt long (chunked) predictions during evaluation, e.g. k-NN on large cohorts
function evalAbortOpts(watchdog) {
    return {
        onYield: () => {
            if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");
        }
    };
}

// Re-sorts already-computed results after the ranking metric changes (no retraining, no redeployment)
function rerankExistingResults() {
    if (AppState.allTestedResults && AppState.allTestedResults.length > 0 && AppState.prepMeta) {
        const isClf = AppState.allTestedResults[0].evalRes.isClassification;
        AppState.allTestedResults.sort((a, b) => compareEvalResults(a.evalRes, b.evalRes, isClf));
        if (AppState.evalResults) renderEvaluationDashboard();
    }
    if (AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0) {
        const isClf = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous';
        AppState.pipeline.benchmarkResults.sort((a, b) => compareEvalResults(a.eval, b.eval, isClf));
        renderPipelineLeaderboardOnly();
    }
}

function setupTrainingLossChart() {
    const ctx = DOM.trainLossChartCanvas.getContext('2d');
    if (AppState.charts.loss) {
        AppState.charts.loss.destroy();
    }

    const isDark = AppState.theme === 'dark';
    AppState.charts.loss = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2
                },
                {
                    label: 'Internal Validation Loss (held out from training data)',
                    data: [],
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' },
                    ticks: { color: isDark ? '#94a3b8' : '#64748b' },
                    title: { display: true, text: 'Epoch', color: isDark ? '#94a3b8' : '#64748b' }
                },
                y: {
                    grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' },
                    ticks: { color: isDark ? '#94a3b8' : '#64748b' },
                    title: { display: true, text: 'Loss', color: isDark ? '#94a3b8' : '#64748b' }
                }
            },
            plugins: {
                legend: { labels: { color: isDark ? '#f8fafc' : '#0f172a' } },
                title: {
                    display: true,
                    // The optional 100% retrain is not plotted
                    text: 'Evaluation-phase fit (single neural network run); 100% retrain not plotted',
                    color: isDark ? '#94a3b8' : '#64748b',
                    font: { size: 11, weight: 'normal' }
                }
            }
        }
    });
}

function updateLossChart(epoch, trainLoss, valLoss) {
    if (!AppState.charts.loss) return;
    const chart = AppState.charts.loss;
    chart.data.labels.push(epoch);
    chart.data.datasets[0].data.push(trainLoss);
    if (valLoss !== undefined) {
        chart.data.datasets[1].data.push(valLoss);
    }
    chart.update('none');
}

function deployTestedModel(modelId) {
    if (!AppState.allTestedResults) return;
    const targetItem = AppState.allTestedResults.find(r => r.modelId === modelId);
    if (!targetItem) return;

    AppState.deployedModelId = modelId;
    AppState.trainedModel = targetItem.model;
    AppState.evalResults = targetItem.evalRes;
    AppState.selectedModelId = modelId;
    markBaselineDeployed();

    // Re-render dashboard to refresh badges and active buttons
    renderEvaluationDashboard();

    // Update single prediction form headers if open
    if (AppState.prepMeta) {
        renderSinglePredictionForm();
    }

    showSeedToast(`✓ "${displayModelName(modelId, targetItem.model)}" is now the active research model for Steps 5 & 6`);
}

// A Step 4 (standard workflow) model is now active: record that, and restore the encoding it was
// trained with (a previously deployed pipeline model may have replaced AppState.prepMeta)
function markBaselineDeployed() {
    AppState.deployedSource = 'standard';
    AppState.deployedPipelineModel = null;
    if (AppState.baselinePrepMeta) AppState.prepMeta = AppState.baselinePrepMeta;
    if (typeof updatePipelineActiveDeploymentStatus === 'function') updatePipelineActiveDeploymentStatus();
    if (typeof updatePredictionProvenanceBar === 'function') updatePredictionProvenanceBar();
    if (AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0 && DOM.pipelineLeaderboardContainer) {
        renderPipelineLeaderboardOnly();
    }
}
window.deployTestedModel = deployTestedModel;

function showSeedToast(msg) {
    const existing = document.querySelector('.seed-toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = 'seed-toast';
    toast.setAttribute('role', 'status');
    toast.innerHTML = '<span>📋</span> <span></span>';
    toast.lastElementChild.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2600);
}

function copySeedToClipboard(seed) {
    if (seed === undefined || seed === null) return;
    const seedStr = String(seed);
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(seedStr).then(() => {
            showSeedToast(`Seed "${seedStr}" copied to clipboard! (Keep as reference)`);
        }).catch(() => {
            prompt("Copy reproducibility seed:", seedStr);
        });
    } else {
        prompt("Copy reproducibility seed:", seedStr);
    }
}
window.copySeedToClipboard = copySeedToClipboard;

export { syncRankingSelectors, disposeModels, fitWithYields, evalAbortOpts, rerankExistingResults, setupTrainingLossChart, updateLossChart, deployTestedModel, markBaselineDeployed, showSeedToast, copySeedToClipboard };
