// external-validation.js: validation of the active model (trained here or imported from a file) on a labelled
// cohort. When a batch file also contains the model's outcome column, the frozen model is scored on those
// patients: discrimination, calibration, clinical usefulness and subgroups, with bootstrap confidence intervals,
// and a printable validation report. Nothing is refitted: preprocessing and model are used exactly as saved.
/* global DataEngine, ModelEngine */
import { AppState, displayModelName, escapeHtml } from './core.js';
import { bootstrapFor, testPatientGroups } from './cv-tuning.js';
import { renderCalibrationPanel } from './panels.js';
import { renderSubgroupPanel } from './subgroups.js';

const vPct = v => (typeof v === 'number' && isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—');
const vNum = (v, d = 3) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');
const vCi = (ev, k, f) => (ev.ci && ev.ci[k] ? ` <span class="tune-sd">(95% CI ${f(ev.ci[k][0])}–${f(ev.ci[k][1])})</span>` : '');

// Summary figures of a validation (binary, multi-category or numerical outcome)
function validationSummary(ev, prep) {
    const rows = [];
    if (!prep.isClassification) {
        rows.push(['R²', vNum(ev.r2) + vCi(ev, 'r2', vNum)], ['RMSE (lower is better)', vNum(ev.rmse) + vCi(ev, 'rmse', vNum)], ['MAE (lower is better)', vNum(ev.mae) + vCi(ev, 'mae', vNum)]);
        if (Array.isArray(ev.actuals) && Array.isArray(ev.predictions)) {
            const mo = ev.actuals.reduce((a, b) => a + b, 0) / ev.actuals.length, mp = ev.predictions.reduce((a, b) => a + b, 0) / ev.predictions.length;
            rows.push(['Mean observed vs mean predicted', `${vNum(mo, 2)} vs ${vNum(mp, 2)}`]);
        }
        return rows;
    }
    const binary = (prep.targetClasses || []).length === 2 && Array.isArray(ev.probPositive);
    if (binary) {
        const y = ev.actuals.map(v => (v === 1 ? 1 : 0));
        const events = y.reduce((a, b) => a + b, 0), meanP = ev.probPositive.reduce((a, b) => a + b, 0) / y.length;
        const cm = ev.confMatrix || [[0, 0], [0, 0]];
        const sens = (cm[1][0] + cm[1][1]) ? cm[1][1] / (cm[1][0] + cm[1][1]) : null, spec = (cm[0][0] + cm[0][1]) ? cm[0][0] / (cm[0][0] + cm[0][1]) : null;
        const cal = ev.calibration || {};
        rows.push(
            ['Patients with the outcome ("' + escapeHtml(String(prep.targetIdxToClass[1])) + '")', `${events} of ${y.length} (${vPct(events / y.length)})`],
            ['ROC-AUC (discrimination)', vNum(ev.rocAuc) + vCi(ev, 'rocAuc', vNum)],
            ['Observed vs mean predicted risk', `${vPct(events / y.length)} vs ${vPct(meanP)} (O:E ${vNum((events / y.length) / meanP, 2)})`],
            ['Calibration slope (ideal 1)', vNum(cal.slope, 2) + vCi(ev, 'calibrationSlope', v => v.toFixed(2))],
            ['Calibration intercept (ideal 0)', vNum(cal.intercept, 2) + vCi(ev, 'calibrationIntercept', v => v.toFixed(2))],
            ['Brier score (lower is better)', vNum(ev.brierScore) + vCi(ev, 'brierScore', vNum)],
            [`Sensitivity / specificity at the model's threshold (${vPct(ev.decisionThreshold || 0.5)})`, `${vPct(sens)} / ${vPct(spec)}`]);
    } else {
        rows.push(['Accuracy', vPct(ev.accuracy) + vCi(ev, 'accuracy', vPct)], ['Balanced accuracy', vPct(ev.balancedAccuracy) + vCi(ev, 'balancedAccuracy', vPct)], ['Macro F1', vPct(ev.macroF1) + vCi(ev, 'macroF1', vPct)]);
    }
    return rows;
}

/**
 * Validates the active model on labelled rows (raw patient rows that include the outcome column).
 * Renders into #external-validation-container; returns the evaluation (or null).
 */
async function validateOnLabelledCohort(rows, fileName) {
    const box = document.getElementById('external-validation-container');
    const prep = AppState.prepMeta, model = AppState.trainedModel;
    if (!box) return null;
    if (!prep || !model || !rows || !rows.length || !rows.some(r => !DataEngine.isMissing(r[prep.targetCol]))) { box.innerHTML = ''; AppState.externalValidation = null; return null; }
    box.innerHTML = `<div class="cal-panel"><p class="tune-muted">Validating the model on the patients in ${escapeHtml(fileName || 'this file')}…</p></div>`;
    await ModelEngine.yieldToBrowser();
    const test = DataEngine.prepareTestingDataset(rows, prep);
    const name = displayModelName(AppState.deployedModelId || AppState.selectedModelId, model);
    if (test.X.length < 10) {
        box.innerHTML = `<div class="cal-panel"><div class="tune-title">🧪 Validation on a labelled cohort</div><p class="tune-muted">Only ${test.X.length} patient(s) have a usable value of "${escapeHtml(prep.targetCol)}"; at least 10 are needed (and far more for a meaningful validation).</p></div>`;
        return null;
    }
    const threshold = AppState.evalResults && AppState.evalResults.decisionThreshold;
    const ev = await ModelEngine.evaluateAsync(model, test.X, test.y, prep.isClassification, prep.targetClasses,
        prep.isClassification ? { decisionThreshold: threshold } : { regressionThreshold: null });
    if (!ev) { box.innerHTML = ''; return null; }
    ev.ci = await bootstrapFor(ev, 2024, testPatientGroups(test.validRows, 'standard')); // patients resampled when a patient column is set
    ev.validationCohort = { fileName: fileName || null, n: test.X.length, provided: rows.length, missingOutcome: test.missingTargetCount, unknownClass: test.unknownClassCount, imputedCells: test.imputedFeatureCells };
    AppState.externalValidation = { ev, name, fileName, rows: test.validRows, prep };
    const summary = validationSummary(ev, prep);
    const sameAsTraining = AppState.fileName && fileName && AppState.fileName === fileName;
    box.innerHTML = `
        <div class="cal-panel">
            <div class="cal-head">
                <div class="tune-title">🧪 Validation of ${escapeHtml(name)} on ${escapeHtml(fileName || 'the uploaded patients')}</div>
                <button type="button" id="external-validation-report-btn" class="btn btn-secondary btn-sm" title="Readable validation report; print it to PDF from the browser">🖨️ Printable Validation Report</button>
            </div>
            <p class="tune-muted">The file contains the outcome "${escapeHtml(prep.targetCol)}", so the model was scored on these ${test.X.length} patients exactly as saved: nothing was refitted.
                ${ev.validationCohort.missingOutcome ? `${ev.validationCohort.missingOutcome} row(s) without a usable outcome were left out. ` : ''}${ev.validationCohort.unknownClass ? `${ev.validationCohort.unknownClass} row(s) with an outcome category the model does not know were left out. ` : ''}${ev.validationCohort.imputedCells ? `${ev.validationCohort.imputedCells} blank or unrecognised predictor value(s) were filled with the model's own missing-data method. ` : ''}
                This is an <strong>external validation</strong> only if none of these patients were used to develop the model${sameAsTraining ? ' — <strong>caution: this file has the same name as the training data</strong>' : ''}.</p>
            <table class="data-table thr-table"><tbody>${summary.map(([k, v]) => `<tr><th style="width: 45%;">${k}</th><td>${v}</td></tr>`).join('')}</tbody></table>
            <p class="tune-muted">Models often keep their discrimination (AUC) in a new hospital but predict risks that are too high or too low on average (O:E away from 1, intercept away from 0), or too extreme (slope below 1). That is common when case mix or care differs, and is usually handled by updating the model's calibration with local data before use.</p>
        </div>
        <div id="external-validation-calibration"></div>
        <div id="external-validation-subgroups"></div>`;
    renderCalibrationPanel('external-validation-calibration', [{ name, ev }], prep.isClassification);
    renderSubgroupPanel('external-validation-subgroups', [{ name, model, prep, threshold }], test.validRows, prep.isClassification);
    document.getElementById('external-validation-report-btn').addEventListener('click', () => window.downloadValidationReport && window.downloadValidationReport());
    return ev;
}

export { validateOnLabelledCohort, validationSummary };
