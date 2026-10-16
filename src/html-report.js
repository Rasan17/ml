// html-report.js: a readable, printable analysis report (HTML; "Print → Save as PDF" in the browser gives a
// PDF) with the methods, the results of every model with confidence intervals, and the key plots
// (ROC and precision-recall curves, calibration, decision curve, confusion matrix), generated from the
// analysis actually run. Complements the JSON reports, which hold the full machine-readable detail.
import { AppState, APP_VERSION, MODEL_INFO, RESEARCH_USE_NOTICE, escapeHtml } from './core.js';
import { buildChecklist, methodsParagraph } from './tripod.js';
import { calibrationBody, discriminationSection, renderOptimismPanel, residualSection } from './panels.js';

const REPORT_CSS = `
:root { --bg: #fff; --text-primary: #0f172a; --text-secondary: #334155; --text-muted: #64748b; --border-color: #cbd5e1;
  --accent-cyan: #0e7490; --accent-emerald: #047857; --accent-amber: #b45309; --accent-rose: #be123c; --accent-purple: #7e22ce;
  --bg-surface-subtle: #f8fafc; }
body { font: 13.5px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-primary); background: #fff; max-width: 1000px; margin: 24px auto; padding: 0 16px; }
h1 { font-size: 21px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 26px 0 8px; border-bottom: 2px solid var(--border-color); padding-bottom: 4px; }
.notice { border: 2px solid var(--accent-amber); padding: 8px 12px; font-weight: 600; margin: 12px 0; }
.meta { color: var(--text-secondary); font-size: 12.5px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 6px 0 10px; }
th, td { border: 1px solid var(--border-color); padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: #f1f5f9; }
.ci { color: var(--text-muted); font-size: 11px; display: block; }
.tune-muted, .tune-sd { color: var(--text-muted); font-size: 12px; }
.cmp-guide-title, .tune-title { font-weight: 700; margin: 12px 0 6px; }
.cal-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; align-items: start; }
.cal-stats > div { margin: 3px 0; }
.chart-legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11.5px; }
.chart-legend i, .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.explain-box, .cal-panel, .tune-model { border: 1px solid var(--border-color); border-radius: 6px; padding: 8px 10px; margin: 8px 0; background: var(--bg-surface-subtle); }
.tune-model-head { display: flex; justify-content: space-between; }
.tune-warn { color: var(--accent-amber); font-weight: 600; }
svg text { font-family: inherit; }
@media print { body { margin: 0; max-width: none; } h2 { break-after: avoid; } table, svg, .cal-panel, .tune-model { break-inside: avoid; } }
`;

const pct = v => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—');
const n3 = v => (typeof v === 'number' ? v.toFixed(3) : '—');
const ciTxt = (ev, k, f) => (ev && ev.ci && ev.ci[k] ? `<span class="ci">95% CI ${f(ev.ci[k][0])}–${f(ev.ci[k][1])}</span>` : '');

function reportItems(source) {
    if (source === 'pipeline') {
        return (AppState.pipeline.benchmarkResults || []).map(b => ({ name: b.name, ev: b.eval, hp: null, note: b.eval && b.eval.constraintNote }));
    }
    return (AppState.allTestedResults || []).map(r => ({ name: (MODEL_INFO[r.modelId] || { name: r.modelId }).name, ev: r.evalRes, hp: r.hyperparams, result: r }));
}

function resultsTable(items, isClassification, optionC) {
    if (isClassification) {
        return `<table><thead><tr><th>Model</th>${items.some(it => it.ev.cv) ? '<th>Cross-validated score (selection)</th>' : ''}<th>${optionC ? 'Corrected accuracy' : 'Accuracy'}</th><th>ROC-AUC</th><th>Brier ↓</th><th>Balanced accuracy</th><th>Macro F1</th><th>Calibration slope</th></tr></thead><tbody>
            ${items.map(it => { const ev = it.ev; const cv = ev.cv; return `<tr><td><strong>${escapeHtml(it.name)}</strong></td>${items.some(x => x.ev.cv) ? `<td>${cv && typeof cv[cv.metricKey] === 'number' ? (['brierScore', 'rocAuc'].includes(cv.metricKey) ? n3(cv[cv.metricKey]) : pct(cv[cv.metricKey])) : '—'}</td>` : ''}
                <td>${pct(ev.accuracy)}${ciTxt(ev, 'accuracy', pct)}</td><td>${n3(ev.rocAuc)}${ciTxt(ev, 'rocAuc', n3)}</td><td>${n3(ev.brierScore)}${ciTxt(ev, 'brierScore', n3)}</td>
                <td>${pct(ev.balancedAccuracy)}${ciTxt(ev, 'balancedAccuracy', pct)}</td><td>${pct(ev.macroF1)}${ciTxt(ev, 'macroF1', pct)}</td>
                <td>${ev.calibration && typeof ev.calibration.slope === 'number' ? n3(ev.calibration.slope) : (ev.optimismCorrection && typeof ev.optimismCorrection.calibrationSlope === 'number' ? `${n3(ev.optimismCorrection.calibrationSlope)} (corrected)` : '—')}</td></tr>`; }).join('')}
        </tbody></table>`;
    }
    return `<table><thead><tr><th>Model</th><th>R²</th><th>RMSE ↓</th><th>MAE ↓</th></tr></thead><tbody>
        ${items.map(it => `<tr><td><strong>${escapeHtml(it.name)}</strong></td><td>${n3(it.ev.r2)}${ciTxt(it.ev, 'r2', n3)}</td><td>${n3(it.ev.rmse)}${ciTxt(it.ev, 'rmse', n3)}</td><td>${n3(it.ev.mae)}${ciTxt(it.ev, 'mae', n3)}</td></tr>`).join('')}
    </tbody></table>`;
}

function confusionTable(ev, classNames) {
    if (!ev || !Array.isArray(ev.confMatrix)) return '';
    const names = classNames.length === ev.confMatrix.length ? classNames : ev.confMatrix.map((_, i) => String(i));
    return `<table style="width: auto;"><thead><tr><th>Actual ↓ / Predicted →</th>${names.map(n => `<th>${escapeHtml(n)}</th>`).join('')}</tr></thead><tbody>
        ${ev.confMatrix.map((row, i) => `<tr><th>${escapeHtml(names[i])}</th>${row.map(v => `<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

// The Option C panel, rendered off-screen with the app's own code
function optimismHtml(results) {
    const tmp = document.createElement('div');
    tmp.id = `report-optimism-${Date.now()}`;
    tmp.style.display = 'none';
    document.body.appendChild(tmp);
    try { renderOptimismPanel(tmp.id, results); return tmp.innerHTML; } finally { tmp.remove(); }
}

function downloadHtmlReport(source) {
    const items = reportItems(source).filter(it => it.ev);
    if (!items.length) { alert(source === 'pipeline' ? 'Run Step 7 of the pipeline first.' : 'Train at least one model in Step 4 first.'); return; }
    const { d } = buildChecklist(source);
    const top = items[0];
    const isClassification = Boolean(top.ev.isClassification);
    const optionC = top.ev.protocol === 'option-c';
    const prep = source === 'pipeline' ? (AppState.pipeline.benchmarkResults[0] || {}).prep : AppState.prepMeta;
    const classNames = prep && prep.targetIdxToClass ? Object.keys(prep.targetIdxToClass).sort((a, b) => a - b).map(k => String(prep.targetIdxToClass[k])) : [];
    const positiveLabel = top.ev.positiveClass !== null && top.ev.positiveClass !== undefined ? String(top.ev.positiveClass) : 'positive outcome';
    const binary = isClassification && top.ev.probPositive && top.ev.calibration;
    const sections = [];
    sections.push(`<h2>1. Methods</h2><p>${escapeHtml(methodsParagraph(d))}</p>
        <p class="meta">Predictors (${d.predictors.length}): ${d.predictors.map(escapeHtml).join(', ')}.${d.excludedPredictors && d.excludedPredictors.length ? ` Not used: ${d.excludedPredictors.map(escapeHtml).join(', ')}.` : ''}</p>`);
    sections.push(`<h2>2. Results${optionC ? ' (bootstrap internal validation, all patients)' : ' (Testing set)'}</h2>
        <p class="meta">Models are listed in ranked order${top.ev.cv ? ' (chosen by cross-validation inside the Training set)' : ''}. ${optionC ? 'The figures are corrected for optimism by bootstrap internal validation (section 3).' : '95% confidence intervals are percentile bootstrap intervals over the Testing patients, with each model held fixed.'}</p>
        ${resultsTable(items, isClassification, optionC)}`);
    if (optionC && source !== 'pipeline') sections.push(`<h2>3. Bootstrap internal validation</h2>${optimismHtml(items.map(it => it.result))}`);
    if (binary && !optionC) {
        sections.push(`<h2>3. Discrimination</h2>${discriminationSection(items, positiveLabel)}`);
        sections.push(`<h2>4. Calibration, thresholds and clinical usefulness: ${escapeHtml(top.name)}</h2>${calibrationBody(top, positiveLabel)}`);
    }
    if (!isClassification && top.ev.actuals) sections.push(`<h2>3. Prediction errors: ${escapeHtml(top.name)}</h2>${residualSection(top.ev, d.target || 'outcome')}`);
    if (isClassification && !optionC) sections.push(`<h2>${binary ? '5' : '3'}. Confusion matrix: ${escapeHtml(top.name)}</h2>${confusionTable(top.ev, classNames)}`);
    const settings = items.map(it => {
        const hp = it.hp ? Object.entries(it.hp).filter(([, v]) => v !== null && v !== undefined && typeof v !== 'object').map(([k, v]) => `${escapeHtml(k)} = ${escapeHtml(String(v))}`).join('; ') : '';
        return `<tr><td>${escapeHtml(it.name)}</td><td>${it.ev.seed !== undefined && it.ev.seed !== null ? escapeHtml(String(it.ev.seed)) : '—'}</td><td>${hp || escapeHtml(it.note || '—')}</td></tr>`;
    }).join('');
    sections.push(`<h2>Model settings and seeds</h2><table><thead><tr><th>Model</th><th>Seed</th><th>Settings</th></tr></thead><tbody>${settings}</tbody></table>
        <p class="meta">Data split seed: ${escapeHtml(String(d.seed ?? '—'))}. Missing values: ${escapeHtml(d.missing || '—')}.</p>`);
    sections.push(`<h2>Limitations</h2><ul>
        <li>${optionC ? 'Internal validation only (bootstrap); performance in other hospitals or populations may differ.' : 'A single train/test split gives an imprecise estimate, especially with few patients; confirm the model in an independent cohort.'}</li>
        <li>Predicted probabilities can be miscalibrated; check the calibration results before reading them as risks.</li>
        <li>Associations learned by the models are not causal effects.</li></ul>`);
    const title = `Prediction model report: ${d.target || 'outcome'}`;
    const html = `<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title><style>${REPORT_CSS}</style></head><body>
<h1>${escapeHtml(title)}</h1>
<div class="meta">Dataset: ${escapeHtml(AppState.fileName || '—')} · ${source === 'pipeline' ? '7-step pipeline' : 'Standard workflow'} · Universal Medical ML/NeuralNet Predictor ${escapeHtml(APP_VERSION)} · generated ${escapeHtml(new Date().toLocaleString('en-GB'))}</div>
<div class="notice">${escapeHtml(RESEARCH_USE_NOTICE)}</div>
${sections.join('\n')}
<p class="meta">Use your browser's Print command (Save as PDF) for a PDF copy. The JSON report holds the full numerical detail.</p>
</body></html>`;
    const blob = new Blob([html], { type: 'text/html' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${source === 'pipeline' ? 'pipeline' : 'analysis'}-report-${new Date().toISOString().slice(0, 10)}.html`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

export { downloadHtmlReport };
