// model-card.js: a one-page "model card" for the active model (Mitchell et al., 2019): intended use, the population
// it was developed on, predictors, missing-data handling, validation design, performance with uncertainty, and
// known limitations. Downloaded as a printable HTML file (strict security policy, every value escaped).
import { APP_VERSION, AppState, RESEARCH_USE_NOTICE, displayModelName, escapeHtml } from './core.js';

const cardNum = (v, d = 3) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : 'not available');
const cardPct = v => (typeof v === 'number' && isFinite(v) ? `${(v * 100).toFixed(1)}%` : 'not available');

function modelCardHtml() {
    const ev = AppState.evalResults || {}, prep = AppState.prepMeta, model = AppState.trainedModel;
    const name = AppState.deployedSource === 'imported' && AppState.importedModelInfo ? AppState.importedModelInfo.name
        : AppState.deployedSource === '7step' && AppState.deployedPipelineModel ? AppState.deployedPipelineModel.name
            : displayModelName(AppState.deployedModelId || AppState.selectedModelId, model);
    const binary = prep.isClassification && (prep.targetClasses || []).length === 2;
    const positive = binary ? String(prep.targetIdxToClass ? prep.targetIdxToClass[1] : prep.targetClasses[1]) : null;
    const design = ev.protocol === 'option-c' ? 'Bootstrap internal validation (optimism-corrected; all patients used for development)'
        : ev.protocol === 'option-b' ? 'Validation on a separate cohort (Dataset ii)' : AppState.deployedSource === 'imported' ? 'As recorded in the imported model file'
            : 'Held-out testing set from the same dataset';
    const mi = prep.multipleImputation;
    const missing = mi ? `Multiple imputation by chained equations (${mi.m} completed datasets, predictive mean matching), fitted on the training patients without the outcome; predictions pooled across imputations`
        : prep.imputation && prep.imputation.plan ? 'Single imputation with the per-variable methods chosen in Step 3, fitted on the training patients'
            : AppState.completeCase ? 'Complete-case analysis (patients with missing values excluded)' : 'Single imputation fitted on the training patients';
    const ci = k => (ev.ci && Array.isArray(ev.ci[k]) ? ` (95% CI ${cardNum(ev.ci[k][0])}–${cardNum(ev.ci[k][1])})` : '');
    const cal = ev.calibration || {};
    const rows = [];
    if (prep.isClassification) {
        if (typeof ev.rocAuc === 'number') rows.push(['ROC-AUC (discrimination)', cardNum(ev.rocAuc) + ci('rocAuc')]);
        if (typeof ev.brierScore === 'number') rows.push(['Brier score (lower is better)', cardNum(ev.brierScore) + ci('brierScore')]);
        if (typeof cal.slope === 'number') rows.push(['Calibration slope (ideal 1)', cardNum(cal.slope, 2)]);
        if (typeof cal.intercept === 'number') rows.push(['Calibration intercept (ideal 0)', cardNum(cal.intercept, 2)]);
        if (ev.optimismCorrection && typeof ev.optimismCorrection.calibrationSlope === 'number') rows.push(['Optimism-corrected calibration slope', cardNum(ev.optimismCorrection.calibrationSlope, 2)]);
        rows.push(['Accuracy / balanced accuracy', `${cardPct(ev.accuracy)} / ${cardPct(ev.balancedAccuracy)}`]);
        const cm = ev.confMatrix;
        if (binary && cm && cm[0] && cm[1] && (cm[1][0] + cm[1][1]) && (cm[0][0] + cm[0][1])) {
            rows.push([`Sensitivity / specificity at threshold ${cardPct(ev.decisionThreshold || 0.5)}`, `${cardPct(cm[1][1] / (cm[1][0] + cm[1][1]))} / ${cardPct(cm[0][0] / (cm[0][0] + cm[0][1]))}`]);
        }
    } else {
        rows.push(['R²', cardNum(ev.r2) + ci('r2')], ['RMSE', cardNum(ev.rmse) + ci('rmse')], ['MAE', cardNum(ev.mae) + ci('mae')]);
    }
    const n = ev.totalCount || ev.totalSamples || (Array.isArray(ev.actuals) ? ev.actuals.length : null);
    const limits = [
        'Research and education use only: not validated for decisions about individual patients.',
        ev.protocol === 'option-b' ? 'Validated on one separate cohort; further validation in other settings is advisable.' : 'Internal validation only: external validation (another time period or centre) is needed before any clinical use.',
        'Predictions for patients unlike the development population (values out of range, unusual combinations) are less reliable; the app flags them.',
        mi ? 'Multiple imputation assumes values are missing at random.' : 'Single imputation understates the uncertainty from missing values.',
        model && model.usedFallback ? 'The neural network could not be trained and a linear model was used instead.' : null
    ].filter(Boolean);
    const items = (list) => list.map(x => `<li>${escapeHtml(x)}</li>`).join('');
    const kv = list => list.map(([k, v]) => `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(v)}</td></tr>`).join('');
    return `<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model card: ${escapeHtml(name)}</title>
<style>body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:#0f172a;max-width:860px;margin:24px auto;padding:0 16px}h1{font-size:22px;margin-bottom:2px}h2{font-size:15px;margin:22px 0 6px;color:#0e7490;border-bottom:1px solid #e2e8f0}table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;padding:4px 8px;border-bottom:1px solid #e2e8f0}th{width:38%;font-weight:600}.note{border:2px solid #b45309;padding:8px 12px;font-weight:600}.muted{color:#475569;font-size:12.5px}</style></head><body>
<h1>Model card: ${escapeHtml(name)}</h1>
<p class="muted">Generated ${escapeHtml(new Date().toISOString().slice(0, 10))} by Universal Medical ML/NeuralNet Predictor v${escapeHtml(APP_VERSION)}</p>
<p class="note">${escapeHtml(RESEARCH_USE_NOTICE)}</p>
<h2>Intended use</h2>
<table>${kv([
        ['Predicts', binary ? `The probability that "${prep.targetCol}" is "${positive}"` : prep.isClassification ? `The category of "${prep.targetCol}"` : `The value of "${prep.targetCol}"`],
        ['Intended users and use', 'Researchers and clinicians exploring prediction research; not for individual patient care'],
        ['Time horizon', `As "${prep.targetCol}" was recorded in the development data (state it when reporting)`]
    ])}</table>
<h2>Development data and model</h2>
<table>${kv([
        ['Dataset', (AppState.importedModelInfo && AppState.importedModelInfo.dataset) || AppState.fileName || 'not recorded'],
        ['Patients evaluated', n ? String(n) : 'not recorded'],
        ['Predictors', `${prep.featureCols.length}: ${prep.featureCols.join(', ')}`],
        ['Missing data', missing],
        ['Model', name],
        ['Validation design', design],
        ['Decision threshold', binary ? cardPct(ev.decisionThreshold || 0.5) : 'not applicable'],
        ['Reproducibility seed', typeof ev.seed === 'number' ? String(ev.seed) : 'not recorded']
    ])}</table>
<h2>Performance</h2>
<table>${kv(rows)}</table>
<p class="muted">${escapeHtml(ev.protocol === 'option-c' ? 'Optimism-corrected estimates from bootstrap internal validation.' : 'Estimates on patients not used to fit the model.')} Confidence intervals are bootstrap intervals over the evaluation patients where shown.</p>
<h2>Known limitations and cautions</h2>
<ul>${items(limits)}</ul>
<p class="muted">Model-card structure after Mitchell M et al., "Model cards for model reporting" (FAT* 2019). Report the full study following TRIPOD+AI.</p>
</body></html>`;
}

function downloadModelCard() {
    if (!AppState.trainedModel || !AppState.prepMeta) { alert('Train or import a model first.'); return; }
    const url = URL.createObjectURL(new Blob([modelCardHtml()], { type: 'text/html' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `model-card_${String(AppState.fileName || 'model').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_').slice(0, 50)}_${new Date().toISOString().slice(0, 10)}.html`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

export { downloadModelCard, modelCardHtml };
