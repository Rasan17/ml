// batch-prediction.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { AppState, displayModelName, escapeHtml, probabilityLabel } from './core.js';
import { thresholdedOutput } from './cv-tuning.js';
import { DOM } from './shell.js';
import { validateOnLabelledCohort } from './external-validation.js';


// =========================================================================
// STEP 6: BATCH PREDICTION FACILITY
// =========================================================================
function handleBatchFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.csv') && file.type !== 'text/csv') {
        alert("Please upload a valid .csv file for batch prediction.");
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        alert("The batch file exceeds 50 MB. Please split it into smaller files (all processing happens in the browser).");
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        processBatchCSV(e.target.result, file.name);
    };
    reader.readAsText(file);
}

async function loadSampleBatchDataset() {
    try {
        const resp = await fetch('medical_batch_unseen_cases.csv');
        if (!resp.ok) throw new Error("Could not load sample batch file.");
        const text = await resp.text();
        processBatchCSV(text, 'Sample Unseen Medical Cohort');
    } catch (e) {
        alert(`Error: ${e.message}`);
    }
}

function processBatchCSV(csvText, name) {
    try {
        const parsed = DataEngine.parseCSV(csvText);
        DataEngine.parseIssues(parsed).forEach(msg => alert(msg));
        AppState.batchData = parsed;
        AppState.batchFileName = name;
        DOM.runBatchPredictBtn.disabled = false;
        alert(`Uploaded batch file "${name}" with ${parsed.totalRows} records.`);
    } catch (e) {
        alert(`Batch CSV parse error: ${e.message}`);
    }
}

async function executeBatchPrediction() {
    if (!AppState.trainedModel || !AppState.prepMeta) {
        alert("Please train a model in Step 4 first.");
        return;
    }
    if (!AppState.batchData) {
        alert("Please upload a batch CSV file first.");
        return;
    }

    const { headers, rows } = AppState.batchData;
    const prep = AppState.prepMeta;
    const results = [];
    const activeSeed = (AppState.trainedModel && AppState.trainedModel.seed !== undefined && AppState.trainedModel.seed !== null)
        ? AppState.trainedModel.seed
        : (AppState.evalResults?.seed !== null && AppState.evalResults?.seed !== undefined ? AppState.evalResults.seed : 'Auto');

    const batchModelName = displayModelName(AppState.deployedModelId || AppState.selectedModelId, AppState.trainedModel);
    // Predictor columns the model needs but the batch file lacks entirely
    const absentCols = prep.featureCols.filter(c => !headers.includes(c));
    let rowsWithImputation = 0;
    let totalImputedCells = 0;

    // Encode every row, then predict in one batched pass (one GPU call for neural networks)
    const rowIssueLists = rows.map(() => []);
    const rowRangeLists = rows.map(() => []);
    const xMatrix = rows.map((r, i) => DataEngine.transformSingleRow(r, prep, rowIssueLists[i], rowRangeLists[i]));
    const rowsOutOfRange = rowRangeLists.filter(l => l.length).length;
    // Asynchronous, chunked prediction keeps the page responsive for large files
    let outputs = [];
    if (rows.length) {
        DOM.runBatchPredictBtn.disabled = true;
        const btnLabel = DOM.runBatchPredictBtn.innerHTML;
        try {
            outputs = await ModelEngine.predictOutputsAsync(AppState.trainedModel, xMatrix, {
                onYield: (done, total) => {
                    DOM.runBatchPredictBtn.textContent = `Predicting… ${Math.round((done / total) * 100)}%`;
                }
            });
        } finally {
            DOM.runBatchPredictBtn.disabled = false;
            DOM.runBatchPredictBtn.innerHTML = btnLabel;
        }
    }

    rows.forEach((r, rowIdx) => {
        const rowIssues = rowIssueLists[rowIdx];
        const predRes = thresholdedOutput(outputs[rowIdx]);

        let predictedLabel = predRes.pred;
        let confidence = 'N/A';
        if (prep.isClassification) {
            predictedLabel = prep.targetIdxToClass[predRes.pred] !== undefined ? prep.targetIdxToClass[predRes.pred] : predRes.pred;
            confidence = (typeof predRes.confidence === 'number' && !isNaN(predRes.confidence))
                ? `${(predRes.confidence * 100).toFixed(1)}%` : 'N/A';
        }
        if (rowIssues.length > 0) {
            rowsWithImputation++;
            totalImputedCells += rowIssues.length;
        }

        const rowCopy = { ...r };
        rowCopy[`Predicted_${prep.targetCol}`] = predictedLabel;
        rowCopy[`Prediction_Probability`] = confidence;
        rowCopy[`Probability_Type`] = prep.isClassification ? probabilityLabel(predRes.probabilityType).title : 'N/A (continuous outcome)';
        rowCopy[`Imputed_Inputs`] = rowIssues.length > 0 ? rowIssues.map(i => `${i.column}:${i.kind}`).join('; ') : '';
        rowCopy[`Outside_Training_Range`] = rowRangeLists[rowIdx].map(o => `${o.column}=${o.value} (training ${o.min}-${o.max})`).join('; ');
        rowCopy[`Model`] = batchModelName;
        rowCopy[`Model_Seed`] = activeSeed;
        results.push(rowCopy);
    });

    if (results.length === 0) {
        alert("The batch file contains no data rows.");
        return;
    }
    if (rowsWithImputation > 0 || absentCols.length > 0 || rowsOutOfRange > 0) {
        alert(`Batch prediction completed with warnings:\n` +
            (absentCols.length > 0 ? `• Columns missing from the file (filled for every row): ${absentCols.join(', ')}\n` : '') +
            (rowsWithImputation > 0 ? `• ${rowsWithImputation} of ${results.length} row(s) had ${totalImputedCells} blank or unrecognised predictor value(s) replaced with training-cohort values (see the 'Imputed_Inputs' column).\n` : '') +
            (rowsOutOfRange > 0 ? `• ${rowsOutOfRange} row(s) have values outside the range of the training patients (see 'Outside_Training_Range'): check for typing errors; otherwise those predictions are extrapolations.\n` : '') +
            `Predictions for these rows are less reliable.`);
    }

    AppState.batchPredictions = results;
    // The file also contains the outcome: validate the model on these patients
    if (headers.includes(prep.targetCol)) await validateOnLabelledCohort(rows, AppState.batchFileName);
    else { const vb = document.getElementById('external-validation-container'); if (vb) vb.innerHTML = ''; }
    DOM.batchResultsCard.style.display = 'flex';
    DOM.downloadBatchCsvBtn.disabled = false;

    // Render preview table
    const resultHeaders = Object.keys(results[0]);
    const previewSlice = results.slice(0, 10);

    let tableHtml = `
        <table class="data-table">
            <thead>
                <tr>
                    ${resultHeaders.map(h => `<th style="${h.startsWith('Predicted_') ? 'color: var(--accent-cyan); background: rgba(6,182,212,0.1);' : ''}">${escapeHtml(h)}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
                ${previewSlice.map(r => `
                    <tr>
                        ${resultHeaders.map(h => `
                            <td style="${h.startsWith('Predicted_') ? 'font-weight: 700; color: var(--accent-cyan); background: rgba(6,182,212,0.05);' : ''}">${escapeHtml(r[h])}</td>
                        `).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    DOM.batchTableContainer.innerHTML = tableHtml;
}

function downloadBatchCSV() {
    if (!AppState.batchPredictions || AppState.batchPredictions.length === 0) return;

    const headers = Object.keys(AppState.batchPredictions[0]);
    // Every cell, headers included: text a spreadsheet would run as a formula gets a leading apostrophe
    const cell = raw => {
        let val = String(raw !== undefined ? raw : '');
        if (/^[=+\-@\t\r]/.test(val) && isNaN(Number(val))) val = `'${val}`;
        return /[",\r\n]/.test(val) ? `"${val.replace(/"/g, '""')}"` : val;
    };
    const lines = [headers.map(cell).join(',')];
    AppState.batchPredictions.forEach(row => { lines.push(headers.map(h => cell(row[h])).join(',')); });

    const csvString = lines.join('\r\n');
    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Universal_Medical_Predictions_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

export { handleBatchFileUpload, loadSampleBatchDataset, processBatchCSV, executeBatchPrediction, downloadBatchCSV };
