// single-prediction.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { AppState, displayModelName, escapeHtml, probabilityLabel } from './core.js';
import { thresholdedOutput } from './cv-tuning.js';
import { renderPatientExplanation } from './panels.js';
import { DOM } from './shell.js';
import { copySeedToClipboard } from './standard-training.js';


// =========================================================================
// STEP 5: SINGLE PREDICTION (CLINICAL PATIENT PREDICTOR)
// =========================================================================
function renderSinglePredictionForm() {
    if (!AppState.prepMeta) return;

    const { featureCols, featureMetadata } = AppState.prepMeta;
    let formHtml = '';

    featureCols.forEach((col, colIdx) => {
        const meta = featureMetadata[col];
        const fieldId = `predict-field-${colIdx}`;

        if (meta.type === 'continuous') {
            const step = (meta.max - meta.min) > 20 ? '1' : '0.01';
            formHtml += `
                <div class="form-field">
                    <label for="${fieldId}">
                        <span>${escapeHtml(col)}</span>
                        <span class="field-hint">Range: ${meta.min} &ndash; ${meta.max}</span>
                    </label>
                    <input type="number" step="any" id="${fieldId}" class="control-input single-input-field" data-col="${escapeHtml(col)}" value="${meta.mean.toFixed(2)}">
                </div>
            `;
        } else if (meta.type === 'ordinal') {
            formHtml += `
                <div class="form-field">
                    <label for="${fieldId}">
                        <span>${escapeHtml(col)}</span>
                        <span class="field-hint">Ordinal Rank</span>
                    </label>
                    <select id="${fieldId}" class="control-select single-input-field" data-col="${escapeHtml(col)}">
                        ${meta.ranks.map(rk => `<option value="${escapeHtml(rk)}">${escapeHtml(rk)}</option>`).join('')}
                    </select>
                </div>
            `;
        } else {
            // Nominal
            formHtml += `
                <div class="form-field">
                    <label for="${fieldId}">
                        <span>${escapeHtml(col)}</span>
                        <span class="field-hint">Categorical</span>
                    </label>
                    <select id="${fieldId}" class="control-select single-input-field" data-col="${escapeHtml(col)}">
                        ${meta.categories.map(cat => `<option value="${escapeHtml(cat)}">${escapeHtml(cat)}</option>`).join('')}
                    </select>
                </div>
            `;
        }
    });

    DOM.predictFormGrid.innerHTML = formHtml;
}

function runSinglePrediction() {
    if (!AppState.trainedModel || !AppState.prepMeta) {
        alert("Model is not trained yet.");
        return;
    }

    const inputRow = {};
    document.querySelectorAll('.single-input-field').forEach(field => {
        const col = field.dataset.col;
        inputRow[col] = field.value;
    });

    const inputIssues = [];
    const xVector = DataEngine.transformSingleRow(inputRow, AppState.prepMeta, inputIssues);
    const result = thresholdedOutput(AppState.trainedModel.predictSingle(xVector));
    const prep = AppState.prepMeta;
    const inputWarningHtml = inputIssues.length > 0
        ? `<div style="font-size: 11.5px; color: var(--accent-amber); margin-top: 6px; font-weight: 600;">
                ⚠️ Blank input(s) filled as in training (the fitted missing-data method); unrecognised values replaced with the most frequent training category / middle rank: ${escapeHtml(inputIssues.map(i => i.column).join(', '))}
           </div>`
        : '';

    DOM.singleResultBanner.style.display = 'flex';

    let ensembleDetailsHtml = '';
    if (result.ensembleDetails && result.ensembleDetails.length > 0) {
        const weightsStr = result.ensembleDetails.map(d => `${d.id.toUpperCase()}: ${Math.round((d.normalizedWeight !== undefined ? d.normalizedWeight : d.weight) * 100)}%`).join(' &bull; ');
        ensembleDetailsHtml = `
            <div style="font-size: 11px; color: var(--accent-cyan); margin-top: 5px; font-weight: 600;">
                Ensemble Blend: ${weightsStr}
            </div>
        `;
    }

    const activeSeed = (AppState.trainedModel && AppState.trainedModel.seed !== undefined && AppState.trainedModel.seed !== null)
        ? AppState.trainedModel.seed
        : (AppState.evalResults?.seed !== null && AppState.evalResults?.seed !== undefined ? AppState.evalResults.seed : null);

    const seedSnippet = activeSeed !== null
        ? ` &bull; Reproducibility Seed: <code style="color: var(--accent-purple); font-weight: 700; background: rgba(168,85,247,0.12); padding: 1px 6px; border-radius: 4px; cursor: pointer;" onclick="window.copySeedToClipboard(${activeSeed})" title="Click to copy reproducibility seed">🌱 ${activeSeed} 📋</code>`
        : '';

    if (prep.isClassification) {
        const predictedLabel = prep.targetIdxToClass[result.pred] !== undefined ? prep.targetIdxToClass[result.pred] : result.pred;
        const hasConfidence = typeof result.confidence === 'number' && !isNaN(result.confidence);
        const confidencePercent = hasConfidence ? (result.confidence * 100).toFixed(1) : null;

        let probBarsHtml = '';
        if (result.probabilities) {
            probBarsHtml = Object.keys(result.probabilities).map(clsIdx => {
                const clsLabel = prep.targetIdxToClass[clsIdx] !== undefined ? prep.targetIdxToClass[clsIdx] : clsIdx;
                const p = (result.probabilities[clsIdx] * 100).toFixed(1);
                return `
                    <div style="font-size: 12px; display: flex; justify-content: space-between; gap: 10px;">
                        <span>${escapeHtml(clsLabel)}:</span>
                        <strong>${p}%</strong>
                    </div>
                `;
            }).join('');
        }

        DOM.singleResultBanner.innerHTML = `
            <div class="banner-left">
                <h3>Predicted ${escapeHtml(prep.targetCol)}</h3>
                <div class="pred-outcome-val">${escapeHtml(predictedLabel)}</div>
                <div style="font-size: 13px; color: var(--text-secondary); margin-top: 6px;">
                    Active Model: <strong>${escapeHtml(displayModelName(AppState.deployedModelId || AppState.selectedModelId, AppState.trainedModel))}</strong> &bull; Target: <em>${escapeHtml(prep.targetCol)}</em>${seedSnippet}
                </div>
                ${ensembleDetailsHtml}
                ${inputWarningHtml}
                ${renderPatientExplanation(AppState.trainedModel, xVector, prep, inputRow)}
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px;">
                    ${probabilityLabel(result.probabilityType).note}: not a calibrated clinical risk or a measure of certainty (check the calibration panel and Brier score in Step 4 / Step 7). For research and educational use only.${AppState.evalResults && AppState.evalResults.decisionThreshold && Math.abs(AppState.evalResults.decisionThreshold - 0.5) > 1e-9 ? ` Classed with a decision threshold of ${AppState.evalResults.decisionThreshold.toFixed(2)} for the positive outcome.` : ''}
                </div>
            </div>
            <div class="banner-confidence">
                <span style="font-size: 12px; color: var(--text-muted); text-transform: uppercase;">${probabilityLabel(result.probabilityType).title}</span>
                <span style="font-size: 26px; font-weight: 800; color: var(--accent-emerald);">${hasConfidence ? `${confidencePercent}%` : 'N/A'}</span>
                <div class="prob-meter-bar">
                    <div class="prob-meter-fill" style="width: ${hasConfidence ? confidencePercent : 0}%;"></div>
                </div>
                <div style="margin-top: 8px;">
                    ${probBarsHtml}
                </div>
            </div>
        `;
    } else {
        DOM.singleResultBanner.innerHTML = `
            <div class="banner-left">
                <h3>Predicted Continuous ${escapeHtml(prep.targetCol)}</h3>
                <div class="pred-outcome-val">${result.pred}</div>
                <div style="font-size: 13px; color: var(--text-secondary); margin-top: 6px;">
                    Active Model: <strong>${escapeHtml(displayModelName(AppState.deployedModelId || AppState.selectedModelId, AppState.trainedModel))}</strong> &bull; Unit/Scale: <em>${escapeHtml(prep.targetCol)}</em>${seedSnippet}
                </div>
                ${ensembleDetailsHtml}
                ${inputWarningHtml}
                ${renderPatientExplanation(AppState.trainedModel, xVector, prep, inputRow)}
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px;">For research and educational use only; not for decisions about an individual patient's care.</div>
            </div>
        `;
    }
}

export { renderSinglePredictionForm, runSinglePrediction };
