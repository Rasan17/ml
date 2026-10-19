// panels.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { AppState, escapeHtml, actionAttrs } from './core.js';
import { plus632 } from './validation-core.js';




// =========================================================================
// CALIBRATION & DECISION-THRESHOLD PANEL (binary outcomes, Testing set)
// =========================================================================
const calibrationItems = {};

function calibrationSvg(cal) {
    const W = 240, P = 30, PL = 40, S = W - P - PL; // wider left margin: the axis title clears the tick labels
    const x = v => PL + v * S, y = v => W - P - v * S;
    // Each group's observed proportion with its 95% Wilson interval (vertical bar): wide bars mean few patients
    const wilson = (p, n) => { if (!n) return [0, 1]; const z = 1.959964, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [Math.max(0, c - h), Math.min(1, c + h)]; };
    const pts = (cal.bins || []).map(b => { const ci = wilson(b.observed, b.n); return { cx: x(b.meanPred), cy: y(b.observed), lo: y(ci[0]), hi: y(ci[1]), r: Math.max(3, Math.min(9, 2 + Math.sqrt(b.n))), tip: `${b.n} patients: mean predicted ${(b.meanPred * 100).toFixed(1)}%, observed ${(b.observed * 100).toFixed(1)}% (95% CI ${(ci[0] * 100).toFixed(0)}–${(ci[1] * 100).toFixed(0)}%)` }; });
    const path = pts.map((q, i) => `${i ? 'L' : 'M'}${q.cx.toFixed(1)},${q.cy.toFixed(1)}`).join(' ');
    const ticks = [0, 0.25, 0.5, 0.75, 1];
    return `<svg viewBox="0 0 ${W} ${W}" width="100%" style="max-width: 260px;" role="img" aria-label="Calibration plot: observed against predicted probability">
        <rect x="${PL}" y="${W - P - S}" width="${S}" height="${S}" fill="none" stroke="var(--border-color)"/>
        ${ticks.map(t => `<line x1="${x(t)}" y1="${y(0)}" x2="${x(t)}" y2="${y(0) + 4}" stroke="var(--text-muted)"/><text x="${x(t)}" y="${y(0) + 14}" font-size="9" text-anchor="middle" fill="var(--text-muted)">${t}</text>
            <line x1="${x(0) - 4}" y1="${y(t)}" x2="${x(0)}" y2="${y(t)}" stroke="var(--text-muted)"/><text x="${x(0) - 6}" y="${y(t) + 3}" font-size="9" text-anchor="end" fill="var(--text-muted)">${t}</text>`).join('')}
        <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--text-muted)" stroke-dasharray="4 4"/>
        ${path ? `<path d="${path}" fill="none" stroke="var(--accent-cyan)" stroke-width="2"/>` : ''}
        ${pts.map(q => `<line x1="${q.cx.toFixed(1)}" y1="${q.lo.toFixed(1)}" x2="${q.cx.toFixed(1)}" y2="${q.hi.toFixed(1)}" stroke="var(--accent-cyan)" stroke-opacity="0.6" stroke-width="1.5"/>`).join('')}
        ${pts.map(q => `<circle cx="${q.cx.toFixed(1)}" cy="${q.cy.toFixed(1)}" r="${q.r.toFixed(1)}" fill="var(--accent-cyan)" fill-opacity="0.75"><title>${escapeHtml(q.tip)}</title></circle>`).join('')}
        <text x="${PL + S / 2}" y="${W - 4}" font-size="10" text-anchor="middle" fill="var(--text-secondary)">Predicted probability</text>
        <text x="10" y="${W - P - S / 2}" font-size="10" text-anchor="middle" fill="var(--text-secondary)" transform="rotate(-90 10 ${W - P - S / 2})">Observed proportion</text>
    </svg>`;
}

function thresholdTable(ev) {
    const p = ev.probPositive, yv = ev.actuals;
    if (!p || !yv) return '';
    const cur = ev.decisionThreshold || 0.5;
    const ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    if (!ths.some(t => Math.abs(t - cur) < 1e-9)) ths.push(cur);
    ths.sort((a, b) => a - b);
    const pct = (a, b) => (b > 0 ? `${((a / b) * 100).toFixed(0)}%` : '–');
    const rows = ths.map(t => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        p.forEach((v, i) => { const pos = v >= t; const act = yv[i] === 1; if (pos && act) tp++; else if (pos) fp++; else if (act) fn++; else tn++; });
        const isCur = Math.abs(t - cur) < 1e-9;
        return `<tr${isCur ? ' class="thr-current"' : ''}><td>${t.toFixed(2)}${isCur ? ' (current)' : ''}</td><td>${pct(tp, tp + fn)}</td><td>${pct(tn, tn + fp)}</td><td>${pct(tp, tp + fp)}</td><td>${pct(tn, tn + fn)}</td><td>${pct(tp + fp, p.length)}</td></tr>`;
    }).join('');
    return `<table class="data-table thr-table"><thead><tr><th>Threshold</th><th>Sensitivity</th><th>Specificity</th><th>PPV</th><th>NPV</th><th>Classed positive</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// Calibration-in-the-large: observed event rate against the mean predicted risk (ratio O:E, ideal 1)
function citlLine(ev) {
    if (!Array.isArray(ev.probPositive) || !Array.isArray(ev.actuals) || !ev.actuals.length) return '';
    const n = ev.actuals.length, obs = ev.actuals.filter(v => v === 1).length / n, pred = ev.probPositive.reduce((a, b) => a + b, 0) / n;
    if (!(pred > 0)) return '';
    return `<div><strong>Calibration-in-the-large</strong>: observed ${(obs * 100).toFixed(1)}% vs mean predicted ${(pred * 100).toFixed(1)}% (O:E ${(obs / pred).toFixed(2)}; ideal 1)</div>`;
}

// Reference Brier score: everyone given the observed event rate; scaled Brier = share of that error removed
function brierReference(ev) {
    if (typeof ev.brierScore !== 'number' || !Array.isArray(ev.actuals) || !ev.actuals.length) return '';
    const prev = ev.actuals.filter(v => v === 1).length / ev.actuals.length, ref = prev * (1 - prev);
    if (!(ref > 0)) return '';
    return ` <span class="tune-sd">(reference ${ref.toFixed(3)} for giving everyone the observed rate of ${(prev * 100).toFixed(1)}%; scaled Brier ${((1 - ev.brierScore / ref) * 100).toFixed(1)}%, where 0% = no better than that and 100% = perfect)</span>`;
}

function calibrationBody(item, positiveLabel) {
    const ev = item.ev;
    const cal = ev.calibration || { bins: [], n: 0, events: 0 };
    const slopeTxt = cal.slope === null || cal.slope === undefined ? 'not estimable'
        : `${cal.slope.toFixed(2)} &mdash; ${cal.slope < 0.8 ? 'predictions are <strong>too extreme</strong> (too close to 0% and 100%), typical of overfitting' : cal.slope > 1.2 ? 'predictions are <strong>too modest</strong> (bunched towards the middle)' : 'close to the ideal of 1'}`;
    const intTxt = cal.intercept === null || cal.intercept === undefined ? 'not estimable'
        : `${cal.intercept.toFixed(2)} &mdash; ${cal.intercept > 0.1 ? 'on average the model <strong>under-predicts</strong> the risk' : cal.intercept < -0.1 ? 'on average the model <strong>over-predicts</strong> the risk' : 'on average the predicted risk is about right'}`;
    const small = cal.events < 100 || cal.n - cal.events < 100;
    const rc = ev.recalibration;
    const f2 = v => (typeof v === 'number' ? v.toFixed(2) : 'n/a');
    const recalNote = rc
        ? `<div class="explain-box" style="margin-top: 8px;"><strong>Recalibrated</strong> with ${rc.method === 'prior-shift' ? 'a prior-shift correction (undoing the SMOTENC balancing) based on' : rc.method === 'platt' ? 'Platt scaling fitted on' : 'isotonic regression fitted on'} ${escapeHtml(rc.fittedOn)}. Before recalibration on these test patients: slope ${f2(rc.before.slope)}, intercept ${f2(rc.before.intercept)}, Brier ${typeof rc.before.brierScore === 'number' ? rc.before.brierScore.toFixed(3) : 'n/a'}${typeof rc.before.rocAuc === 'number' ? `, AUC ${rc.before.rocAuc.toFixed(3)}` : ''}. The values above are after recalibration.</div>`
        : (ev.recalibrationSkipped ? `<div class="tune-muted" style="margin-top: 6px;">${escapeHtml(ev.recalibrationSkipped)}</div>`
            // Miscalibrated (the slope's 95% CI excludes 1) and not recalibrated: say how to correct it
            : (ev.ci && ev.ci.calibrationSlope && (ev.ci.calibrationSlope[0] > 1 || ev.ci.calibrationSlope[1] < 1)
                ? `<div class="tune-warn" style="margin-top: 6px;">⚠︎ The calibration slope differs from 1 beyond chance (95% CI ${f2(ev.ci.calibrationSlope[0])}–${f2(ev.ci.calibrationSlope[1])}), so the predicted percentages should not be read as patients' risks. Recalibration (the "Recalibrate two-outcome probabilities" option before training: Platt scaling or isotonic regression, fitted on the Training set only) can correct this; confirm the result on new patients.</div>` : ''));
    const probNote = ev.probabilityType && ev.probabilityType !== 'model' && ev.probabilityType !== 'recalibrated' ? ' This model\'s "probabilities" are vote or neighbour shares, which are often poorly calibrated.' : '';
    return `
        <div class="cal-grid">
            <div>${cal.bins && cal.bins.length ? calibrationSvg(cal) : '<p class="tune-muted">Too few test patients to draw a calibration plot (at least 10 are needed).</p>'}</div>
            <div class="cal-stats">
                <div><strong>Calibration slope</strong> (ideal 1): ${slopeTxt}${ev.ci && ev.ci.calibrationSlope && cal.slope !== null ? ` <span class="tune-sd">(95% CI ${ev.ci.calibrationSlope[0].toFixed(2)}–${ev.ci.calibrationSlope[1].toFixed(2)})</span>` : ''}</div>
                <div><strong>Calibration intercept</strong> (ideal 0): ${intTxt}</div>
                ${citlLine(ev)}
                <div><strong>Brier score</strong> (lower is better): ${typeof ev.brierScore === 'number' ? ev.brierScore.toFixed(3) : 'N/A'}${brierReference(ev)}</div>
                <div class="tune-muted">Based on ${cal.n} test patients, ${cal.events} with "${escapeHtml(positiveLabel)}". ${cal.note ? escapeHtml(cal.note) : ''}</div>
                <p class="tune-muted" style="margin-top: 6px;"><strong>Brier score</strong> is the mean squared difference between predicted probabilities and outcomes (0 or 1). It mixes calibration and discrimination and depends on how common the outcome is, so compare it with the reference (everyone given the observed rate), not across datasets.</p>
                <p class="tune-muted" style="margin-top: 6px;"><strong>How to read the plot:</strong> each vertical bar is the 95% confidence interval of a group's observed proportion; patients are grouped by their predicted probability. Each dot compares the average predicted probability in a group with the proportion who actually had "${escapeHtml(positiveLabel)}". Dots on the dashed diagonal mean that a predicted 30% really corresponds to about 30% of patients. Dots below the line mean the model over-estimates the risk; dots above it mean it under-estimates.</p>
            </div>
        </div>
        ${recalNote}
        ${small ? `<div class="tune-warn">⚠︎ Calibration needs many patients: with fewer than about 100 patients with and 100 without the outcome in the Testing set, these values are very imprecise.${probNote}</div>` : (probNote ? `<div class="tune-warn">⚠︎${probNote}</div>` : '')}
        <div class="cal-thr">
            <div class="cmp-guide-title" style="font-size: 13px;">Sensitivity and specificity at different thresholds (Testing set)</div>
            ${thresholdTable(ev)}
            ${ev.youden ? `<p class="tune-muted">ℹ️ <strong>Youden's J</strong> on the cross-validated Training-set predictions (${ev.youden.n} patients, not the Testing set) would put the threshold at <strong>${(ev.youden.threshold * 100).toFixed(1)}%</strong> (sensitivity ${(ev.youden.sensitivity * 100).toFixed(0)}%, specificity ${(ev.youden.specificity * 100).toFixed(0)}%). This treats a missed case and a false alarm as equally bad, which is rarely true clinically; use it as a starting point, not a substitute for a clinical choice (the decision curve below helps).</p>` : ''}
            <p class="tune-muted">This table shows the trade-off: a lower threshold catches more true cases (higher sensitivity) at the cost of more false alarms (lower specificity). Use it to understand the trade-off and choose the threshold on <strong>clinical grounds</strong>. Picking the "best" row here and reporting its results would make the Testing-set results optimistic; confirm any chosen threshold on new patients.</p>
        </div>
        ${decisionCurveSection(ev, positiveLabel)}`;
}

const CURVE_COLOURS = ['var(--accent-cyan)', 'var(--accent-emerald)', 'var(--accent-purple)', 'var(--accent-amber)', 'var(--accent-rose)', '#94a3b8'];

// Simple SVG line chart: series of {name, colour, points:[{x,y}], dashed}
function lineChartSvg({ series, xLabel, yLabel, xMin = 0, xMax = 1, yMin = 0, yMax = 1, diagonal = false, aria }) {
    const W = 300, H = 240, L = 40, R = 10, T = 10, B = 34;
    const sx = v => L + ((v - xMin) / (xMax - xMin || 1)) * (W - L - R);
    const sy = v => H - B - ((v - yMin) / (yMax - yMin || 1)) * (H - T - B);
    const ticks = n => Array.from({ length: n + 1 }, (_, i) => i / n);
    const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2).replace(/\.?0+$/, '') || '0');
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width: 340px;" role="img" aria-label="${escapeHtml(aria || yLabel + ' against ' + xLabel)}">
        <rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="none" stroke="var(--border-color)"/>
        ${ticks(4).map(f => { const xv = xMin + f * (xMax - xMin), yv = yMin + f * (yMax - yMin); return `
            <text x="${sx(xv)}" y="${H - B + 12}" font-size="9" text-anchor="middle" fill="var(--text-muted)">${fmt(xv)}</text>
            <text x="${L - 4}" y="${sy(yv) + 3}" font-size="9" text-anchor="end" fill="var(--text-muted)">${fmt(yv)}</text>`; }).join('')}
        ${yMin < 0 && yMax > 0 ? `<line x1="${L}" y1="${sy(0)}" x2="${W - R}" y2="${sy(0)}" stroke="var(--text-muted)" stroke-width="0.6"/>` : ''}
        ${diagonal ? `<line x1="${sx(xMin)}" y1="${sy(yMin)}" x2="${sx(xMax)}" y2="${sy(yMax)}" stroke="var(--text-muted)" stroke-dasharray="4 4"/>` : ''}
        ${series.map(se => {
            const pts = se.points.filter(p => isFinite(p.x) && isFinite(p.y));
            // Hover read-out: each point's values (named after the axes), on up to 150 points per line
            const tip = p => `${se.name ? `${se.name}: ` : ''}${xLabel} ${fmt(p.x)}, ${yLabel} ${fmt(p.y)}`;
            if (se.scatter) return pts.map(p => `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(Math.max(yMin, Math.min(yMax, p.y))).toFixed(1)}" r="2.2" fill="${se.colour}" fill-opacity="0.6"><title>${escapeHtml(tip(p))}</title></circle>`).join('');
            const d = pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(Math.max(yMin, Math.min(yMax, p.y))).toFixed(1)}`).join(' ');
            const step = Math.max(1, Math.ceil(pts.length / 150));
            const hover = se.noHover ? '' : pts.filter((_, i) => i % step === 0 || i === pts.length - 1).map(p => `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(Math.max(yMin, Math.min(yMax, p.y))).toFixed(1)}" r="4" fill="transparent" class="chart-hover"><title>${escapeHtml(tip(p))}</title></circle>`).join('');
            return `<path d="${d}" fill="none" stroke="${se.colour}" stroke-width="${se.width || 1.8}" ${se.dashed ? 'stroke-dasharray="5 4"' : ''}/>${hover}`;
        }).join('')}
        <text x="${(L + W - R) / 2}" y="${H - 4}" font-size="10" text-anchor="middle" fill="var(--text-secondary)">${escapeHtml(xLabel)}</text>
        <text x="11" y="${(T + H - B) / 2}" font-size="10" text-anchor="middle" fill="var(--text-secondary)" transform="rotate(-90 11 ${(T + H - B) / 2})">${escapeHtml(yLabel)}</text>
    </svg>`;
}

function chartLegend(entries) {
    return `<div class="chart-legend">${entries.map(e => `<span><i style="background: ${e.colour}; ${e.dashed ? 'opacity: 0.6;' : ''}"></i>${e.label}</span>`).join('')}</div>`;
}

// ROC and precision-recall curves for every model, with AUC CIs and DeLong tests against the top model
function discriminationSection(items, positiveLabel) {
    const usable = items.filter(it => it.ev && it.ev.probPositive && it.ev.actuals);
    if (!usable.length) return '';
    const y = usable[0].ev.actuals.map(v => (v === 1 ? 1 : 0));
    const roc = usable.map((it, i) => ({ name: it.name, colour: CURVE_COLOURS[i % CURVE_COLOURS.length], points: ModelEngine.rocCurve(y, it.ev.probPositive).map(p => ({ x: p.fpr, y: p.tpr })) }));
    const prs = usable.map((it, i) => { const r = ModelEngine.prCurve(y, it.ev.probPositive); return { name: it.name, colour: CURVE_COLOURS[i % CURVE_COLOURS.length], ap: r.averagePrecision, points: [{ x: 0, y: r.points.length ? r.points[0].precision : 1 }, ...r.points.map(p => ({ x: p.recall, y: p.precision }))] }; });
    const prev = y.reduce((a, b) => a + b, 0) / y.length;
    const top = usable[0];
    const rows = usable.map((it, i) => {
        const d = i === 0 ? null : ModelEngine.delongTest(y, top.ev.probPositive, it.ev.probPositive);
        const auc = typeof it.ev.rocAuc === 'number' ? it.ev.rocAuc.toFixed(3) : 'N/A';
        const ci = it.ev.ci && it.ev.ci.rocAuc ? ` (${it.ev.ci.rocAuc[0].toFixed(3)}–${it.ev.ci.rocAuc[1].toFixed(3)})` : '';
        const pTxt = i === 0 ? 'reference (top-ranked)' : (d && d.p !== null ? `${d.p < 0.001 ? '< 0.001' : d.p.toFixed(3)}${d.p < 0.05 ? ' ✱' : ''}` : 'n/a');
        return `<tr><td><i class="legend-dot" style="background: ${CURVE_COLOURS[i % CURVE_COLOURS.length]}"></i>${escapeHtml(it.name)}</td><td>${auc}${ci}</td><td>${prs[i].ap !== null ? prs[i].ap.toFixed(3) : 'N/A'}</td><td>${pTxt}</td></tr>`;
    }).join('');
    return `
        <div class="cmp-guide-title" style="font-size: 13px;">Discrimination: how well each model separates patients with and without "${escapeHtml(positiveLabel)}" (Testing set)</div>
        <div class="cal-grid" style="grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));">
            <div>${lineChartSvg({ series: roc, xLabel: '1 − specificity (false-positive rate)', yLabel: 'Sensitivity', diagonal: true, aria: 'ROC curves' })}</div>
            <div>${lineChartSvg({ series: [...prs, { name: 'Prevalence (no skill)', colour: 'var(--text-muted)', dashed: true, width: 1, points: [{ x: 0, y: prev }, { x: 1, y: prev }] }], xLabel: 'Recall (sensitivity)', yLabel: 'Precision (PPV)', aria: 'Precision-recall curves' })}</div>
        </div>
        <table class="data-table thr-table"><thead><tr><th>Model</th><th>ROC-AUC (95% CI)</th><th>Average precision</th><th>AUC difference vs top model: DeLong p</th></tr></thead><tbody>${rows}</tbody></table>
        <p class="tune-muted"><strong>ROC curve</strong>: the closer to the top-left corner, the better (the diagonal is chance). <strong>Precision-recall curve</strong>: more informative when the outcome is uncommon; the dashed line is the prevalence (${(prev * 100).toFixed(0)}%), which is what a useless model would achieve. The <strong>DeLong test</strong> asks whether a model's AUC differs from the top-ranked model's on these same patients; p ≥ 0.05 means the difference could easily be chance (✱marks p &lt; 0.05). With small test sets most differences are not significant.</p>`;
}

// Decision curve analysis for one model
function decisionCurveSection(ev, positiveLabel) {
    if (!ev.probPositive || !ev.actuals) return '';
    const y = ev.actuals.map(v => (v === 1 ? 1 : 0));
    const dc = ModelEngine.decisionCurve(y, ev.probPositive);
    const all = dc.map(d => d.treatAll);
    const yMax = Math.max(0.05, ...dc.map(d => d.model), ...all);
    const yMin = Math.max(-0.05, Math.min(-0.02, ...dc.map(d => d.model)));
    const useful = dc.filter(d => d.model > Math.max(0, d.treatAll) + 1e-9).map(d => d.threshold);
    const range = useful.length ? `${Math.round(useful[0] * 100)}%–${Math.round(useful[useful.length - 1] * 100)}%` : null;
    return `
        <div class="cal-thr">
            <div class="cmp-guide-title" style="font-size: 13px;">Decision curve: is using the model better than treating everyone or no one? (Testing set)</div>
            <div class="cal-grid">
                <div>${lineChartSvg({ series: [
                    { name: 'Model', colour: 'var(--accent-cyan)', points: dc.map(d => ({ x: d.threshold, y: d.model })) },
                    { name: 'Treat all', colour: 'var(--accent-amber)', dashed: true, points: dc.map(d => ({ x: d.threshold, y: d.treatAll })) },
                    { name: 'Treat none', colour: 'var(--text-muted)', width: 1, noHover: true, points: dc.map(d => ({ x: d.threshold, y: 0 })) }
                ], xLabel: 'Threshold probability', yLabel: 'Net benefit', xMin: 0, xMax: 1, yMin, yMax, aria: 'Decision curve' })}
                ${chartLegend([{ colour: 'var(--accent-cyan)', label: 'Using the model' }, { colour: 'var(--accent-amber)', label: 'Treat / investigate all', dashed: true }, { colour: 'var(--text-muted)', label: 'Treat none' }])}</div>
                <p class="tune-muted"><strong>How to read it:</strong> the threshold probability is the risk at which you would act (e.g. operate or investigate), reflecting how you weigh a missed case against an unnecessary intervention. <strong>Net benefit</strong> counts true positives minus false positives weighted by that trade-off. Where the model's line is above both "treat all" and "treat none", using the model would lead to better decisions on these patients${range ? ` (here, for thresholds of about <strong>${range}</strong>)` : ' (here: at no threshold)'}. With few test patients the curve is noisy.</p>
            </div>
        </div>`;
}

// Residuals against fitted values (numerical outcomes)
function residualSection(ev, targetName) {
    if (!ev || !ev.actuals || !ev.predictions) return '';
    const pts = ev.predictions.map((f, i) => ({ x: f, y: ev.actuals[i] - f }));
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const lim = Math.max(...ys.map(Math.abs), 1e-9);
    const ci = k => (ev.ci && ev.ci[k] ? ` (95% CI ${ev.ci[k][0].toFixed(3)}–${ev.ci[k][1].toFixed(3)})` : '');
    return `
        <div class="cal-grid">
            <div>${lineChartSvg({ series: [{ name: '', colour: 'var(--accent-cyan)', scatter: true, points: pts }], xLabel: `Predicted ${targetName}`, yLabel: 'Residual (actual − predicted)', xMin, xMax: xMax === xMin ? xMin + 1 : xMax, yMin: -lim * 1.05, yMax: lim * 1.05, aria: 'Residuals against fitted values' })}</div>
            <div class="cal-stats">
                <div><strong>R²</strong>: ${typeof ev.r2 === 'number' ? ev.r2.toFixed(3) : 'N/A'}${ci('r2')}</div>
                <div><strong>RMSE</strong>: ${typeof ev.rmse === 'number' ? ev.rmse.toFixed(3) : 'N/A'}${ci('rmse')}</div>
                <div><strong>MAE</strong>: ${typeof ev.mae === 'number' ? ev.mae.toFixed(3) : 'N/A'}${ci('mae')}</div>
                <p class="tune-muted" style="margin-top: 6px;"><strong>How to read the plot:</strong> each dot is a test patient. Dots should scatter evenly around the zero line with no pattern. A funnel shape means errors grow with the predicted value; a curve means the model misses a non-linear relationship; isolated far-away dots are patients predicted badly.</p>
            </div>
        </div>`;
}

window.renderCalibrationChoice = function (containerId, idx) {
    const st = calibrationItems[containerId];
    const body = document.getElementById(`${containerId}-body`);
    if (!st || !body || !st.items[idx]) return;
    body.innerHTML = st.regression ? residualSection(st.items[idx].ev, st.target) : calibrationBody(st.items[idx], st.positiveLabel);
};

function renderCalibrationPanel(containerId, items, isClassification) {
    const box = document.getElementById(containerId);
    if (!box) return;
    const all = (items || []).filter(it => it.ev);
    if (!all.length) { box.innerHTML = ''; return; }
    const select = () => `<label class="autotune-folds-label">Model:
                    <select class="control-select" style="width: auto; display: inline-block;" aria-label="Model to show" data-event="change" ${actionAttrs('renderCalibrationChoice', containerId)}>
                        ${all.map((it, i) => `<option${i === 0 ? ' selected' : ''}>${escapeHtml(it.name)}</option>`).join('')}
                    </select></label>`;
    if (!isClassification) {
        const target = escapeHtml(AppState.columnConfigs ? (Object.keys(AppState.columnConfigs).find(c => AppState.columnConfigs[c].role === 'target') || 'outcome') : 'outcome');
        calibrationItems[containerId] = { items: all, regression: true, target };
        box.innerHTML = `
            <div class="cal-panel">
                <div class="cal-head"><div class="tune-title">Prediction errors (Testing set)</div>${select()}</div>
                <div id="${containerId}-body">${residualSection(all[0].ev, target)}</div>
            </div>`;
        return;
    }
    const usable = all.filter(it => it.ev.calibration);
    if (!usable.length) { box.innerHTML = ''; return; }
    const positiveLabel = usable[0].ev.positiveClass !== null && usable[0].ev.positiveClass !== undefined ? String(usable[0].ev.positiveClass) : 'positive outcome';
    calibrationItems[containerId] = { items: usable, positiveLabel };
    box.innerHTML = `
        <div class="cal-panel">
            <div class="tune-title">Discrimination, calibration &amp; clinical usefulness (Testing set)</div>
            ${discriminationSection(usable, positiveLabel)}
            <div class="cal-head" style="margin-top: 14px;">
                <div class="cmp-guide-title" style="font-size: 13px;">Calibration, thresholds and decision curve for one model</div>
                ${select()}
            </div>
            <p class="tune-muted">Calibration asks whether the predicted percentages can be taken at face value: if the model says "30% chance", do about 30 in 100 such patients have the outcome? A model can rank patients well (good AUC) and still give percentages that are too high or too low.</p>
            <div id="${containerId}-body">${calibrationBody(usable[0], positiveLabel)}</div>
        </div>`;
}


// =========================================================================
// EXPLAINABILITY: permutation importance (Testing set) and per-patient explanations
// =========================================================================
const explainItems = {};

function referenceEncoding(meta) {
    // Encoded value of a "typical" patient for one predictor: training mean, most frequent category, middle rank
    if (meta.type === 'nominal') {
        const v = new Array(meta.length).fill(0);
        const i = meta.categories.indexOf(meta.mode !== undefined ? meta.mode : meta.categories[0]);
        if (i >= 0) v[i] = 1;
        return v;
    }
    if (meta.type === 'ordinal') return [DataEngine.scaleOrdinalRank(meta.defaultRank !== undefined ? meta.defaultRank : 0, meta)];
    return [0]; // continuous: standardised training mean
}

function typicalValueText(meta) {
    if (meta.type === 'nominal') return String(meta.mode !== undefined ? meta.mode : meta.categories[0]);
    if (meta.type === 'ordinal') {
        const lbl = Object.keys(meta.rankMap || {}).find(k => meta.rankMap[k] === meta.defaultRank);
        return lbl !== undefined ? lbl : String(meta.defaultRank);
    }
    return Number(meta.mean).toFixed(Math.abs(meta.mean) >= 100 ? 0 : 2);
}

// Model output used for explanations: probability of the positive class (binary), of the predicted
// class (multiclass), or the predicted value (regression)
function explainOutput(model, x, prep, classIdx) {
    const o = model.predictSingle(x);
    if (!prep.isClassification) return Number(o.pred);
    const k = classIdx !== undefined ? classIdx : ((prep.targetClasses || []).length === 2 ? 1 : o.pred);
    return o.probabilities ? Number(o.probabilities[k]) || 0 : (o.pred === k ? 1 : 0);
}

// One-at-a-time contributions: change in the model output when each predictor is set to a typical value
function explainPatient(model, x, prep) {
    const binary = prep.isClassification && (prep.targetClasses || []).length === 2;
    const base = model.predictSingle(x);
    const classIdx = prep.isClassification ? (binary ? 1 : base.pred) : undefined;
    const f0 = explainOutput(model, x, prep, classIdx);
    const rows = prep.featureCols.map(col => {
        const meta = prep.featureMetadata[col];
        const ref = referenceEncoding(meta);
        const x2 = x.slice();
        for (let j = 0; j < meta.length; j++) x2[meta.startIndex + j] = ref[j];
        if (x.__mi) Object.defineProperty(x2, '__mi', { value: x.__mi.map(v => { const w = v.slice(); for (let j = 0; j < meta.length; j++) w[meta.startIndex + j] = ref[j]; return w; }), enumerable: false });
        const same = ref.every((v, j) => Math.abs(v - x[meta.startIndex + j]) < 1e-9);
        return { col, meta, delta: same ? 0 : f0 - explainOutput(model, x2, prep, classIdx) };
    });
    return { f0, classIdx, binary, rows: rows.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)) };
}

// Shapley-value explanation against a background of up to 100 training patients: exact for tree models
// (interventional TreeSHAP), estimated from random orderings of the predictors for other models.
// Returns null when no training data are available (e.g. a model loaded from a file).
function shapleyExplanation(model, x, prep, classIdx) {
    if (!Array.isArray(prep.X) || prep.X.length < 5 || !prep.featureCols) return null;
    const rng = DataEngine.createPRNG(20240611);
    const idx = prep.X.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const background = idx.slice(0, 100).map(i => prep.X[i]);
    const groups = [];
    prep.featureCols.forEach((col, g) => { const m = prep.featureMetadata[col]; for (let j = 0; j < m.length; j++) groups[m.startIndex + j] = g; });
    const P = prep.featureCols.length;
    const calibrated = Boolean(model && model.calibrator);
    // Exact TreeSHAP needs one version of the patient; a multiply imputed patient uses the sampled method
    let r = calibrated || x.__mi ? null : ModelEngine.treeShap(model, x, background, groups, P, classIdx);
    if (r && r.scale === 'log-odds' && !r.multiclass) {
        // Boosted trees: exact on the log-odds scale; shown in percentage points by sharing out the change in probability
        const sig = v => 1 / (1 + Math.exp(-v));
        const total = r.phi.reduce((a, b) => a + b, 0);
        const dp = sig(r.value) - sig(r.base);
        const f = Math.abs(total) > 1e-12 ? dp / total : 0;
        r = { ...r, phi: r.phi.map(v => v * f), base: sig(r.base), value: sig(r.value), scale: 'probability', converted: true };
    }
    if (!r) {
        const slow = model && (model.constructor && /Neural|Ensemble/i.test(model.constructor.name));
        const outputs = rows => (typeof model.predictOutputs === 'function' ? model.predictOutputs(rows) : rows.map(z => model.predictSingle(z)));
        const value = o => (!prep.isClassification ? Number(o.pred) : o.probabilities ? Number(o.probabilities[classIdx]) || 0 : (o.pred === classIdx ? 1 : 0));
        r = ModelEngine.sampledShapley(rows => outputs(rows).map(value), x, background, groups, P, slow ? 64 : 200, 7);
        r.scale = prep.isClassification ? 'probability' : 'outcome';
    }
    return { ...r, background: background.length, rows: prep.featureCols.map((col, g) => ({ col, meta: prep.featureMetadata[col], delta: r.phi[g] })).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)) };
}

function renderPatientExplanation(model, x, prep, inputRow) {
    try {
        const ex = explainPatient(model, x, prep);
        const cls = prep.isClassification ? String((prep.targetIdxToClass || {})[ex.classIdx] ?? ex.classIdx) : null;
        const logOdds = sh => sh && sh.scale === 'log-odds';
        const unitOf = sh => (logOdds(sh) ? 'log-odds units' : prep.isClassification ? 'percentage points' : 'units');
        const fmtOf = sh => d => (logOdds(sh) ? Math.abs(d).toFixed(2) : prep.isClassification ? (Math.abs(d) * 100).toFixed(1) : Math.abs(d).toFixed(2));
        const valueOf = r => (inputRow && inputRow[r.col] !== undefined && String(inputRow[r.col]).trim() !== '' ? String(inputRow[r.col]) : '(blank)');
        const what = prep.isClassification ? `probability of "${escapeHtml(cls)}"` : 'predicted value';
        // Quick view: each predictor changed on its own to a typical value
        const minEffect = prep.isClassification ? 0.0005 : Math.max(1e-6, Math.abs(ex.f0) * 1e-3);
        const quick = ex.rows.filter(r => Math.abs(r.delta) >= minEffect).slice(0, 6);
        const fq = fmtOf(null);
        const quickHtml = quick.length
            ? `<ul>${quick.map(r => `<li><span class="${r.delta > 0 ? 'ex-up' : 'ex-down'}">${r.delta > 0 ? '▲' : '▼'}</span> <strong>${escapeHtml(r.col)} = ${escapeHtml(valueOf(r))}</strong> ${r.delta > 0 ? 'raises' : 'lowers'} the ${what} by <strong>${fq(r.delta)} ${unitOf(null)}</strong> compared with a typical value (${escapeHtml(typicalValueText(r.meta))}).</li>`).join('')}</ul>`
            : '<p class="tune-muted">No single predictor changes the prediction when set to a typical value on its own.</p>';
        const quickNote = '<div class="tune-muted">Each predictor is changed on its own to a typical training value (average, most common category or middle rank) while the others are kept as entered; interactions between predictors are not captured.</div>';
        let sh = null;
        try { sh = shapleyExplanation(model, x, prep, ex.classIdx); } catch (e) { sh = null; }
        if (!sh) {
            return `<div class="explain-box"><strong>Why this prediction?</strong> The predictors that moved this patient's result most (quick view):${quickHtml}${quickNote}
                <div class="tune-muted">Shapley values need the training data, which are not available for a model loaded from a file. The explanation describes how the model behaves, not what causes the outcome.</div></div>`;
        }
        const fs = fmtOf(sh);
        const minSh = logOdds(sh) ? 0.005 : prep.isClassification ? 0.0005 : Math.max(1e-6, Math.abs(sh.value) * 1e-3);
        const top = sh.rows.filter(r => Math.abs(r.delta) >= minSh).slice(0, 6);
        const level = v => (logOdds(sh) ? `${v.toFixed(2)} (log-odds)` : prep.isClassification ? `${(v * 100).toFixed(1)}%` : v.toFixed(2));
        const items = top.length ? `<ul>${top.map(r => `<li><span class="${r.delta > 0 ? 'ex-up' : 'ex-down'}">${r.delta > 0 ? '▲' : '▼'}</span> <strong>${escapeHtml(r.col)} = ${escapeHtml(valueOf(r))}</strong> ${r.delta > 0 ? 'raises' : 'lowers'} it by <strong>${fs(r.delta)} ${unitOf(sh)}</strong></li>`).join('')}</ul>` : '<p class="tune-muted">No predictor moves this prediction noticeably away from the average.</p>';
        const method = sh.exact
            ? `Exact Shapley values for the trees (interventional TreeSHAP)${sh.converted ? ', calculated on the model\'s log-odds scale and shown in percentage points' : ''}`
            : `Shapley values estimated from ${sh.orderings} random orderings of the predictors (small random error)`;
        return `<div class="explain-box">
            <strong>Why this prediction?</strong> Starting from the average ${what} for ${sh.background} training patients (<strong>${level(sh.base)}</strong>), these predictors moved this patient to <strong>${level(sh.value)}</strong>:
            ${items}
            <div class="tune-muted">${method}. The contributions of all predictors add up to the difference between this patient and the average, and interactions are shared fairly between the predictors involved. Each predictor's effect is judged by replacing it with values from real training patients, which can create unlikely combinations when predictors are strongly correlated. It describes how the model behaves, not what causes the outcome.</div>
            <details class="explain-quick"><summary>Quick view: one predictor at a time</summary>${quickHtml}${quickNote}</details>
        </div>`;
    } catch (e) {
        return '';
    }
}

// Odds ratios from a logistic regression fitted to convergence (IRLS) on the training data: an explanatory
// companion to the predictive models, with one reference category per categorical predictor
// Design matrix for the explanatory logistic regression: one reference category (the most frequent in the
// fitted encoding, `refs`) per categorical predictor; continuous predictors per unit (or per SD if SD < 1)
function orDesign(prep, refs) {
    const cols = [];
    prep.featureCols.forEach(col => {
        const m = prep.featureMetadata[col];
        if (m.type === 'nominal') {
            const ref = refs && refs[col] !== undefined ? refs[col] : (m.mode !== undefined ? m.mode : m.categories[0]);
            m.categories.forEach((cat, k) => { if (cat !== ref) cols.push({ idx: m.startIndex + k, name: `${col}: ${cat}`, note: `vs ${ref}`, scale: 1 }); });
        } else {
            const sd = m.std > 0 ? m.std : 1;
            const perUnit = m.type === 'ordinal' || sd >= 1;
            cols.push({ idx: m.startIndex, name: col, note: m.type === 'ordinal' ? 'per rank step' : perUnit ? 'per unit' : `per SD (${sd.toPrecision(2)} units)`, scale: perUnit ? sd : 1, perSD: perUnit });
        }
    });
    return cols;
}

function oddsRatioRows(cols, estimates, pooled) {
    const f2 = v => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
    return estimates.map((r, i) => {
        const c = cols[i];
        const k = c.perSD ? 1 / c.scale : 1; // coefficient is per SD of the standardised predictor
        const or = Math.exp(r.coef * k), lo = Math.exp(r.low * k), hi = Math.exp(r.high * k);
        const pTxt = r.p === null ? 'n/a' : r.p < 0.001 ? '< 0.001' : r.p.toFixed(3);
        return `<tr><td>${escapeHtml(c.name)}</td><td>${f2(or)} (${f2(lo)}–${f2(hi)})</td><td>${escapeHtml(c.note)}${c.perSD ? ` <span class="tune-sd">(per SD: ${f2(Math.exp(r.coef))})</span>` : ''}</td><td>${pTxt}</td>${pooled ? `<td>${(r.fmi * 100).toFixed(0)}%</td>` : ''}</tr>`;
    }).join('');
}

// Odds ratios from a logistic regression fitted to convergence (IRLS) on the training data: an explanatory
// companion to the predictive models. With missing predictor values, they can be re-estimated by multiple
// imputation (MICE with predictive mean matching) pooled with Rubin's rules.
function renderOddsRatioPanel(containerId, prep) {
    const box = document.getElementById(containerId);
    if (!box) return;
    if (!prep || !prep.isClassification || (prep.targetClasses || []).length !== 2 || !Array.isArray(prep.X) || !prep.X.length) { box.innerHTML = ''; return; }
    const cols = orDesign(prep);
    const posLabel = String(prep.targetClasses[1]);
    let body;
    if (cols.length > 60 || prep.X.length > 50000) {
        body = `<p class="tune-muted">Not shown: too many predictor columns (${cols.length}) or patients for an exact fit in the browser.</p>`;
    } else {
        const X = prep.X.map(r => cols.map(c => r[c.idx]));
        const y = prep.y.map(v => (v === 1 ? 1 : 0));
        const fit = ModelEngine.logisticIRLS(X, y, cols.map(c => c.name));
        if (!fit || !fit.converged) {
            body = `<p class="tune-muted">${escapeHtml((fit && fit.note) || 'The logistic regression could not be fitted.')}</p>`;
        } else {
            const est = fit.rows.slice(1).map(r => ({ coef: r.coef, low: r.coef - 1.959964 * r.se, high: r.coef + 1.959964 * r.se, p: r.p }));
            const epv = Math.min(fit.events, fit.n - fit.events) / cols.length;
            const raw = AppState.orRawRows || [];
            const nMissing = raw.filter(r => prep.featureCols.some(c => DataEngine.isMissing(r[c]))).length;
            body = `<table class="data-table thr-table"><thead><tr><th>Predictor</th><th>Odds ratio (95% CI)</th><th>Scale</th><th>p-value</th></tr></thead><tbody>${oddsRatioRows(cols, est, false)}</tbody></table>
                <p class="tune-muted">Odds ratios for "${escapeHtml(posLabel)}" from an <strong>unpenalised logistic regression fitted to convergence</strong> on the ${fit.n} training patients (${fit.events} with the outcome), each adjusted for all the other predictors. An odds ratio above 1 means higher odds of "${escapeHtml(posLabel)}"; a confidence interval that includes 1 is compatible with no association. Categorical predictors are compared with their most common category. This table explains associations in the training data; it is separate from the ridge-penalised model used for prediction and does not show causation.${epv < 10 ? ` <strong>Caution:</strong> only ${epv.toFixed(1)} patients in the smaller outcome group per predictor, so these estimates are unstable.` : ''}</p>
                ${nMissing > 0 ? `<div class="explain-box"><strong>Missing data:</strong> ${nMissing} of ${raw.length} training patients had at least one missing predictor value, filled once (single imputation), which makes the confidence intervals too narrow. <button type="button" class="btn btn-outline btn-sm" id="${containerId}-mice-btn" ${actionAttrs('runMiceOddsRatios', containerId)}>Re-estimate with multiple imputation (MICE)</button><div id="${containerId}-mice"></div></div>` : ''}`;
        }
    }
    box.innerHTML = `<div class="cal-panel"><div class="tune-title">Odds ratios (logistic regression, training data)</div>${body}</div>`;
}

window.runMiceOddsRatios = async function (containerId) {
    const prep = AppState.prepMeta;
    const out = document.getElementById(`${containerId}-mice`);
    const btn = document.getElementById(`${containerId}-mice-btn`);
    const raw = (AppState.orRawRows || []).filter(r => !DataEngine.isMissing(r[prep.targetCol]));
    if (!out || !prep || !raw.length) return;
    btn.disabled = true;
    out.innerHTML = '<p class="tune-muted">Creating 10 imputed datasets (MICE, 10 cycles each)…</p>';
    await ModelEngine.yieldToBrowser();
    try {
        const refs = {};
        prep.featureCols.forEach(c => { const m = prep.featureMetadata[c]; if (m.type === 'nominal') refs[c] = m.mode !== undefined ? m.mode : m.categories[0]; });
        const mi = DataEngine.miceImpute(raw, AppState.columnConfigs, { m: 10, iterations: 10, seed: 2024 });
        const estimates = [];
        let names = null, colsRef = null;
        for (const data of mi.datasets) {
            const p = DataEngine.prepareFeaturesAndTarget(data, AppState.columnConfigs);
            const cols = orDesign(p, refs);
            const X = p.X.map(r => cols.map(c => r[c.idx]));
            const fit = ModelEngine.logisticIRLS(X, p.y.map(v => (v === 1 ? 1 : 0)), cols.map(c => c.name));
            if (!fit || !fit.converged) throw new Error((fit && fit.note) || 'a fit did not converge');
            const nm = cols.map(c => c.name).join('|');
            if (names === null) { names = nm; colsRef = cols; } else if (nm !== names) throw new Error('the imputed datasets have different categories');
            estimates.push({ coef: fit.rows.slice(1).map(r => r.coef), se: fit.rows.slice(1).map(r => r.se) });
            await ModelEngine.yieldToBrowser();
        }
        const pooled = ModelEngine.poolRubin(estimates, raw.length - colsRef.length - 1);
        out.innerHTML = `<table class="data-table thr-table" style="margin-top: 8px;"><thead><tr><th>Predictor</th><th>Pooled odds ratio (95% CI)</th><th>Scale</th><th>p-value</th><th>Missing information</th></tr></thead><tbody>${oddsRatioRows(colsRef, pooled, true)}</tbody></table>
            <p class="tune-muted"><strong>Multiple imputation:</strong> missing predictor values were imputed 10 times by chained equations with predictive mean matching (each imputed value is a value actually observed in a similar patient; the outcome is used as a predictor, as recommended). The logistic regression was fitted to each completed dataset and the results pooled with <strong>Rubin's rules</strong>, so the confidence intervals include the uncertainty due to the missing values. "Missing information" is the fraction of each estimate's uncertainty that comes from the missing data. Valid if values are missing at random given the other variables.</p>`;
    } catch (e) {
        out.innerHTML = `<p class="tune-muted">Multiple imputation could not be completed: ${escapeHtml(e.message || String(e))}</p>`;
    } finally {
        btn.disabled = false;
    }
};

async function permutationImportance(model, prep, rows, isClassification, seed, onProgress) {
    const data = DataEngine.prepareTestingDataset(rows, prep);
    let X = data.X, y = data.y;
    if (X.length > 2000) { const keep = X.map((_, i) => i).filter(i => i % Math.ceil(X.length / 2000) === 0); X = keep.map(i => X[i]); y = keep.map(i => y[i]); }
    const binary = isClassification && (prep.targetClasses || []).length === 2;
    const key = !isClassification ? 'r2' : binary ? 'rocAuc' : 'balancedAccuracy';
    const label = !isClassification ? 'R²' : binary ? 'ROC-AUC' : 'balanced accuracy';
    const score = async Xs => { const ev = await ModelEngine.evaluateAsync(model, Xs, y, isClassification, prep.targetClasses, isClassification ? {} : { regressionThreshold: null }); return ev ? ev[key] : null; };
    const base = await score(X);
    if (typeof base !== 'number') return { error: `The ${label} cannot be computed on these test patients.` };
    const rng = DataEngine.createPRNG(seed);
    const repeats = 5;
    const out = [];
    for (let c = 0; c < prep.featureCols.length; c++) {
        const col = prep.featureCols[c];
        const meta = prep.featureMetadata[col];
        const drops = [];
        for (let r = 0; r < repeats; r++) {
            const perm = X.map((_, i) => i);
            for (let i = perm.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
            // Multiply imputed patients carry one version per imputation (x.__mi): shuffle the column in each
            const swap = (row, src) => { const nr = row.slice(); for (let j = 0; j < meta.length; j++) nr[meta.startIndex + j] = src[meta.startIndex + j]; return nr; };
            const Xp = X.map((row, i) => {
                const nr = swap(row, X[perm[i]]);
                if (row.__mi) Object.defineProperty(nr, '__mi', { value: row.__mi.map((v, l) => swap(v, X[perm[i]].__mi ? X[perm[i]].__mi[l] : X[perm[i]])), enumerable: false });
                return nr;
            });
            const sc = await score(Xp);
            if (typeof sc === 'number') drops.push(base - sc);
        }
        const mean = drops.reduce((a, b) => a + b, 0) / (drops.length || 1);
        const sd = drops.length > 1 ? Math.sqrt(drops.reduce((a, b) => a + (b - mean) ** 2, 0) / (drops.length - 1)) : 0;
        out.push({ col, mean, sd });
        if (onProgress) onProgress(c + 1, prep.featureCols.length);
        await ModelEngine.yieldToBrowser();
    }
    return { base, label, n: X.length, rows: out.sort((a, b) => b.mean - a.mean) };
}

window.runPermutationImportance = async function (containerId) {
    const st = explainItems[containerId];
    const sel = document.getElementById(`${containerId}-select`);
    const out = document.getElementById(`${containerId}-out`);
    const btn = document.getElementById(`${containerId}-btn`);
    if (!st || !sel || !out) return;
    const it = st.items[sel.selectedIndex];
    btn.disabled = true;
    try {
        const res = await permutationImportance(it.model, it.prep, st.rows, st.isClassification, 42, (d, t) => { out.innerHTML = `<p class="tune-muted">Computing… predictor ${d} of ${t}</p>`; });
        if (res.error) { out.innerHTML = `<p class="tune-muted">${escapeHtml(res.error)}</p>`; return; }
        const max = Math.max(...res.rows.map(r => Math.abs(r.mean)), 1e-9);
        out.innerHTML = `
            <p class="tune-muted">${escapeHtml(it.name)}: ${res.label} on ${res.n} test patients = ${res.base.toFixed(3)}. Bars show how much it <strong>drops</strong> when one predictor's values are shuffled between patients (mean ± SD over 5 shuffles).</p>
            <div class="perm-list">${res.rows.map(r => `
                <div class="perm-row"><span class="perm-name">${escapeHtml(r.col)}</span>
                    <span class="perm-track"><span class="perm-bar ${r.mean < 0 ? 'neg' : ''}" style="width: ${Math.max(1, Math.round(Math.abs(r.mean) / max * 100))}%"></span></span>
                    <span class="perm-val">${r.mean >= 0 ? '' : '−'}${Math.abs(r.mean).toFixed(3)} <span class="tune-sd">± ${r.sd.toFixed(3)}</span></span></div>`).join('')}
            </div>
            ${it.inSample ? '<div class="tune-warn">⚠︎ This model was retrained on all patients (including these test patients), so this importance is computed on data the model has seen; the ranking is still informative, but the sizes are optimistic.</div>' : ''}`;
    } catch (e) {
        out.innerHTML = `<p class="tune-muted">Could not compute permutation importance: ${escapeHtml(e.message || String(e))}</p>`;
    } finally {
        btn.disabled = false;
    }
};

function renderExplainPanel(containerId, items, rows, isClassification) {
    const box = document.getElementById(containerId);
    if (!box) return;
    const usable = (items || []).filter(it => it.model && it.prep && typeof it.model.predictSingle === 'function');
    if (!usable.length || !rows || !rows.length) { box.innerHTML = ''; return; }
    explainItems[containerId] = { items: usable, rows, isClassification };
    box.innerHTML = `
        <div class="cal-panel">
            <div class="cal-head">
                <div class="tune-title">Which predictors matter? (permutation importance, Testing set)</div>
                <span>
                    <select id="${containerId}-select" class="control-select" aria-label="Model for permutation importance" style="width: auto; display: inline-block;">${usable.map((it, i) => `<option${i === 0 ? ' selected' : ''}>${escapeHtml(it.name)}</option>`).join('')}</select>
                    <button type="button" id="${containerId}-btn" class="btn btn-outline btn-sm" ${actionAttrs('runPermutationImportance', containerId)}>Compute</button>
                </span>
            </div>
            <p class="tune-muted">Model-agnostic importance: each predictor's values are shuffled between test patients, breaking its link with the outcome, and the drop in performance is measured. A large drop means the model relies on that predictor; around zero means it does not; a negative value means shuffling did not hurt (noise). Predictors that carry the same information share their importance. This shows what the model uses, not what causes the outcome.</p>
            <div id="${containerId}-out"></div>
        </div>`;
}

// =========================================================================
// MICE CONVERGENCE: mean imputed value of each numerical predictor after every round, one line per imputation.
// Converged chains level off and overlap; a line still trending at the last rounds suggests more rounds.
// =========================================================================
// Gelman–Rubin potential scale reduction after a burn-in of 2 rounds (chains = imputations); about 1 when the
// chains have mixed, larger when they are still apart. With 8 rounds per chain it is approximate: on converged
// chains (30 runs of the demo cohort) it exceeded 1.2 in about 1% of checks and never 1.3, hence the 1.3 threshold.
// null with fewer than 2 chains or 4 rounds after burn-in.
function miceRhat(chains, burn = 2) {
    const ok = chains.filter(v => v.length - burn >= 4);
    if (ok.length < 2) return null;
    const n = Math.min(...ok.map(v => v.length)) - burn, tails = ok.map(v => v.slice(v.length - n));
    const means = tails.map(v => v.reduce((a, b) => a + b, 0) / n), grand = means.reduce((a, b) => a + b, 0) / means.length;
    const W = tails.reduce((a, v, j) => a + v.reduce((x, y) => x + (y - means[j]) ** 2, 0) / (n - 1), 0) / tails.length;
    const B = n * means.reduce((a, mu) => a + (mu - grand) ** 2, 0) / (means.length - 1);
    if (!(W > 0)) return B > 0 ? Infinity : 1;
    return Math.sqrt((((n - 1) / n) * W + B / n) / W);
}

function renderMiceDiagnostics(containerId, mi) {
    const box = document.getElementById(containerId);
    if (!box) return;
    const emptyNote = mi && Array.isArray(mi.allMissing) && mi.allMissing.length
        ? `<div class="cal-panel"><div class="tune-warn">⚠︎ No observed values among the training patients for: ${mi.allMissing.map(c => `"${escapeHtml(String(c))}"`).join(', ')}. Multiple imputation cannot impute a predictor that is never recorded, so it was filled with a constant and carries no information. Exclude it in Step 2.</div></div>` : '';
    if (!mi || !Array.isArray(mi.trace) || !mi.trace.length) { box.innerHTML = emptyNote; return; }
    const cols = Object.keys(mi.trace[0] || {}).slice(0, 6);
    if (!cols.length) { box.innerHTML = emptyNote; return; }
    const drifting = [], rhats = [];
    const charts = cols.map(c => {
        const series = mi.trace.map((t, l) => ({ name: `Imputation ${l + 1}`, colour: CURVE_COLOURS[l % CURVE_COLOURS.length], points: (Array.isArray(t && t[c]) ? t[c] : []).map((v, it) => ({ x: it + 1, y: Number(v) })).filter(pt => isFinite(pt.y)) }));
        const all = series.flatMap(se => se.points.map(p => p.y));
        const sd = (mi.traceScale && mi.traceScale[c] && mi.traceScale[c].sd) || 1;
        const rhat = miceRhat(series.map(se => se.points.map(pt => pt.y)));
        if (rhat !== null && rhat > 1.3) drifting.push(`${c} (R̂ ${rhat.toFixed(2)})`);
        rhats.push(rhat);
        const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.15 || sd * 0.1 || 1;
        return `<div><div class="tune-muted" style="font-weight: 600;">${escapeHtml(c)}</div>${lineChartSvg({ series, xLabel: 'Round', yLabel: 'Mean imputed', xMin: 0, xMax: (mi.trace[0][c] || []).length || 10, yMin: lo - pad, yMax: hi + pad, aria: `MICE convergence for ${c}` })}</div>`;
    });
    box.innerHTML = emptyNote + `
        <div class="cal-panel">
            <div class="tune-title">Did the multiple imputation converge? (MICE diagnostics)</div>
            <p class="tune-muted">For each numerical predictor with missing values, the mean of its imputed values after each of the ${(mi.trace[0][cols[0]] || []).length} rounds, one line per imputation (${mi.trace.length}). Converged imputations level off after the first few rounds and mix around the same level; lines that keep rising or falling at the end suggest the chains needed more rounds.${Object.keys(mi.trace[0]).length > cols.length ? ` First ${cols.length} predictors shown.` : ''}</p>
            <div class="mice-diag-grid">${charts.join('')}</div>
            ${mi.trace.length < 2 ? '' : drifting.length
                ? `<div class="tune-warn">⚠︎ Possible non-convergence: ${drifting.map(escapeHtml).join(', ')}. The imputations differ more between chains than within them after the first 2 rounds (Gelman–Rubin R̂ above 1.3; approximate with 10 rounds). Treat the imputations of these predictors with caution; consider more rounds, or whether the variable can be predicted from the others.</div>`
                : `<div class="explain-box">✓No sign of non-convergence: after the first 2 rounds the chains vary as much within as between them (Gelman–Rubin R̂ ${rhats.filter(v => v !== null).length ? `${Math.min(...rhats.filter(v => v !== null)).toFixed(2)}–${Math.max(...rhats.filter(v => v !== null)).toFixed(2)}` : 'n/a'}; below 1.3 for every predictor shown; approximate with 10 rounds). Round-to-round wiggles are expected: each round draws new donor values.</div>`}
            ${chartLegend(mi.trace.map((_, l) => ({ label: `Imputation ${l + 1}`, colour: CURVE_COLOURS[l % CURVE_COLOURS.length] })))}
        </div>`;
}

export { miceRhat, renderMiceDiagnostics, calibrationItems, calibrationSvg, thresholdTable, calibrationBody, CURVE_COLOURS, lineChartSvg, chartLegend, discriminationSection, decisionCurveSection, residualSection, renderCalibrationPanel, explainItems, referenceEncoding, typicalValueText, explainOutput, explainPatient, shapleyExplanation, renderPatientExplanation, renderOddsRatioPanel, permutationImportance, renderExplainPanel };

// =========================================================================
// MINIMUM SAMPLE SIZE (Riley et al. criteria, as in pmsampsize) for the current outcome and predictors
// =========================================================================
function renderSampleSizePanel() {
    const box = document.getElementById('sample-size-container');
    if (!box) return;
    if (!AppState.imputedData || !AppState.imputedData.imputedRows || !AppState.imputedData.imputedRows.length) { box.innerHTML = ''; return; }
    let prep;
    try { prep = DataEngine.prepareFeaturesAndTarget(AppState.imputedData.imputedRows, AppState.columnConfigs); } catch (e) { box.innerHTML = ''; return; }
    if (!prep || !prep.X || !prep.X.length) { box.innerHTML = ''; return; }
    const binary = prep.isClassification && (prep.targetClasses || []).length === 2;
    if (prep.isClassification && !binary) {
        box.innerHTML = `<div class="cal-panel"><div class="tune-title">Is the sample large enough?</div><p class="tune-muted">Formal minimum sample-size criteria (Riley et al.) are available for two-outcome and numerical outcomes; for ${prep.targetClasses.length} outcome groups, aim for at least 10–20 patients in the smallest group per predictor parameter.</p></div>`;
        return;
    }
    // Parameters: one per numerical or ordered predictor, (categories − 1) per unordered categorical one
    const p = Math.max(1, prep.X[0].length - Object.values(prep.featureMetadata || {}).filter(m => m.type === 'nominal' && m.length > 1).length);
    const prev = binary ? prep.y.filter(v => v === 1).length / prep.y.length : null;
    const r2Input = parseFloat(document.getElementById('sample-size-r2')?.value);
    const res = ModelEngine.rileySampleSize({ type: binary ? 'binary' : 'continuous', p, prevalence: prev, r2: isFinite(r2Input) ? r2Input : null });
    if (!res) { box.innerHTML = ''; return; }
    const isOptionA = document.getElementById('val-mode-a') ? document.getElementById('val-mode-a').checked : true;
    const pct = parseFloat(document.getElementById('train-pct-input')?.value) || 80;
    const nTrain = isOptionA ? Math.round(prep.X.length * Math.min(0.95, Math.max(0.5, pct / 100))) : prep.X.length;
    const enough = nTrain >= res.n;
    const posLabel = binary ? escapeHtml(String(prep.targetClasses[1])) : '';
    box.innerHTML = `
        <div class="cal-panel">
            <div class="cal-head">
                <div class="tune-title">Is the sample large enough? (minimum sample size, Riley et al.)</div>
                <label class="autotune-folds-label" title="The R² you expect a model to achieve; leave blank for the conservative default">Anticipated ${binary ? 'Cox-Snell ' : ''}R²:
                    <input type="number" id="sample-size-r2" class="control-input" style="width: 90px; display: inline-block;" min="0.01" max="0.99" step="0.01" placeholder="${res.r2.toFixed(3)}" value="${isFinite(r2Input) ? r2Input : ''}">
                </label>
            </div>
            <div class="${enough ? 'explain-box' : 'tune-warn'}" style="margin-top: 6px;">
                ${enough ? '✓' : '⚠︎'} The Training set has <strong>${nTrain}</strong> patients${binary ? ` (about ${Math.round(nTrain * prev)} with "${posLabel}")` : ''}; the criteria below require at least <strong>${res.n}</strong>${binary ? ` (${res.events} with "${posLabel}")` : ''} for ${p} predictor parameter(s).
                ${enough ? 'The sample meets these criteria.' : 'The sample is <strong>smaller than recommended</strong>: expect overfitting and unstable estimates; consider fewer predictors (or grouping categories), or more patients, and treat results as exploratory.'}
            </div>
            <table class="data-table thr-table"><thead><tr><th>Criterion</th><th>Patients needed</th></tr></thead>
                <tbody>${res.criteria.map(c => `<tr><td>${escapeHtml(c.label)}</td><td>${c.n}</td></tr>`).join('')}</tbody></table>
            <p class="tune-muted">Criteria of Riley et al. (BMJ 2020; Stat Med 2019), as in the pmsampsize package, with ${p} candidate parameter(s) (1 per numerical or ordered predictor, categories − 1 per unordered categorical one)${binary ? `, outcome prevalence ${(prev * 100).toFixed(1)}%` : ''} and an anticipated ${binary ? 'Cox-Snell ' : ''}R² of ${res.r2.toFixed(3)}${res.r2Assumed ? (binary ? ' (conservative default: 15% of the maximum possible; enter a value from previous studies if available)' : ' (default; enter a value from previous studies if available)') : ''}. These replace the older "10 events per variable" rule of thumb. Flexible models (random forests, neural networks) typically need considerably more.</p>
        </div>`;
    document.getElementById('sample-size-r2')?.addEventListener('change', renderSampleSizePanel);
}

// Option C: apparent, optimism and optimism-corrected performance for each model
function renderOptimismPanel(containerId, results) {
    const box = document.getElementById(containerId);
    if (!box) return;
    const items = (results || []).filter(r => r.evalRes && r.evalRes.optimismCorrection);
    if (!items.length) { box.innerHTML = ''; return; }
    const labels = { accuracy: 'Accuracy', balancedAccuracy: 'Balanced accuracy', macroF1: 'Macro F1', rocAuc: 'ROC-AUC', brierScore: 'Brier ↓', r2: 'R²', rmse: 'RMSE ↓', mae: 'MAE ↓' };
    const pctKeys = new Set(['accuracy', 'balancedAccuracy', 'macroF1']);
    const fmt = (k, v) => (typeof v !== 'number' ? '—' : pctKeys.has(k) ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));
    const fmtOpt = (k, v) => (typeof v !== 'number' ? '—' : pctKeys.has(k) ? `${(v * 100).toFixed(1)} pts` : v.toFixed(3));
    const nameOf = r => (r.info && r.info.name) || r.modelId;
    const tuningNote = oc => {
        if (oc.tuning === 'repeated') {
            const share = typeof oc.sameSettingsShare === 'number' ? ` The resamples chose the same settings as the full data in ${Math.round(oc.sameSettingsShare * 100)}% of cases${oc.sameSettingsShare < 0.5 ? ', so the tuned settings are not very stable' : ''}.` : '';
            return `<p class="tune-muted">✓The automatic tuning was repeated inside every resample (${oc.innerFolds}-fold cross-validation), so the corrected figures include the optimism that tuning adds.${share}</p>`;
        }
        if (oc.tuning === 'once') return `<p class="tune-muted">⚠︎ The settings were tuned once on all patients and not again inside each resample (too slow for this model), so the corrected figures leave out the optimism that tuning adds and are somewhat too high.</p>`;
        return '';
    };
    const blocks = items.map(r => {
        const oc = r.evalRes.optimismCorrection;
        const keys = Object.keys(oc.apparent);
        const appSlope = typeof oc.apparentSlope === 'number' ? oc.apparentSlope.toFixed(3) : '—';
        const h = oc.harrell || oc.corrected, e = oc.est632 || {};
        const is632 = oc.method === '632plus';
        const b = (on, v) => (on ? `<strong>${v}</strong>` : v);
        return `<div class="tune-model"><div class="tune-model-head"><strong>${escapeHtml(nameOf(r))}</strong><span class="tune-sd">${oc.B} resamples</span></div>
            ${tuningNote(oc)}
            <p class="tune-muted">${is632
                ? 'Reported: the <strong>.632+</strong> estimate. This model can memorise its training patients, so Harrell\'s correction (scored partly on patients the bootstrap model was trained on) would be too small.'
                : 'Reported: <strong>Harrell\'s</strong> optimism-corrected estimate, the standard for regression-type models; the .632+ estimate is shown for comparison.'}</p>
            <table class="data-table thr-table"><thead><tr><th>Measure</th><th>Apparent (trained and scored on all patients)</th><th>Average optimism</th><th>Harrell corrected</th><th>.632+ estimate</th></tr></thead>
            <tbody>${keys.map(k => `<tr><td>${labels[k] || k}</td><td>${fmt(k, oc.apparent[k])}</td><td>${fmtOpt(k, oc.optimism[k])}</td><td>${b(!is632, fmt(k, h[k]))}</td><td>${b(is632, fmt(k, e[k]))}</td></tr>`).join('')}
            ${oc.calibrationSlope !== null && oc.calibrationSlope !== undefined ? `<tr><td>Calibration slope (ideal 1)</td><td>${appSlope}</td><td>${typeof oc.calibrationSlopeOptimism === 'number' && !is632 ? fmtOpt('calibrationSlope', oc.calibrationSlopeOptimism) : '—'}</td><td colspan="2"><strong>${oc.calibrationSlope.toFixed(3)}</strong> <span class="tune-muted">(${is632 ? 'bootstrap models on their out-of-bag patients' : 'apparent slope minus its optimism (Harrell)'})</span></td></tr>` : ''}</tbody></table></div>`;
    }).join('');
    // Choosing the best model: the winner's curse
    const sel = items[0].evalRes.optimismCorrection.selection;
    let selBlock = '';
    if (sel && sel.repeated && sel.B > 0) {
        const top = items[0];
        const oc = top.evalRes.optimismCorrection;
        const keys = Object.keys(oc.apparent);
        const share = Object.keys(sel.chosenShare || {}).sort((a, b) => sel.chosenShare[b] - sel.chosenShare[a])
            .map(id => { const m = items.find(x => x.modelId === id); return `${escapeHtml(m ? nameOf(m) : id)} ${Math.round(sel.chosenShare[id] * 100)}%`; }).join(', ');
        selBlock = `<div class="tune-model"><div class="tune-model-head"><strong>Choosing the best of ${sel.nModels} models</strong><span class="tune-sd">${sel.B} resamples</span></div>
            <p class="tune-muted">Picking the model with the best cross-validated score makes the winner look a little better than it is (the winner's curse). Inside each resample the same choice was made again, and the performance of whichever model won there was recorded. Applying this to the apparent performance of the top-ranked model (${escapeHtml(nameOf(top))}) gives an estimate for the whole choose-then-fit process. Models chosen across the resamples: ${share}.</p>
            <table class="data-table thr-table"><thead><tr><th>Measure</th><th>Apparent (${escapeHtml(nameOf(top))})</th><th>${oc.method === '632plus' ? 'Chosen models on out-of-bag patients' : 'Optimism of choosing and fitting'}</th><th>Corrected for the choice (${oc.method === '632plus' ? '.632+' : 'Harrell'})</th></tr></thead>
            <tbody>${keys.map(k => {
                const o = sel.optimism[k];
                let v = typeof oc.apparent[k] === 'number' && typeof o === 'number' ? oc.apparent[k] - o : null;
                if (oc.method === '632plus') { const e = plus632(k, oc.apparent[k], sel.oobMean && sel.oobMean[k], oc.noInfo && oc.noInfo[k]); v = e ? e.value : null; }
                return `<tr><td>${labels[k] || k}</td><td>${fmt(k, oc.apparent[k])}</td><td>${oc.method === '632plus' ? `out-of-bag ${fmt(k, sel.oobMean && sel.oobMean[k])}` : fmtOpt(k, o)}</td><td><strong>${fmt(k, v)}</strong></td></tr>`;
            }).join('')}</tbody></table></div>`;
    } else if (sel && !sel.repeated) {
        selBlock = `<p class="tune-muted">⚠︎ ${sel.nModels} models were compared. Choosing the best one adds a small optimistic bias (the winner's curse) that is not corrected here${sel.reason === 'slow' ? ', because the neural network and the ensemble are too slow to cross-validate inside every resample' : ', because the choice is not made by cross-validation (choose "Cross-validation" for model selection to correct it)'}. Treat small differences between models as ties.</p>`;
    }
    const anyRepeated = items.some(r => r.evalRes.optimismCorrection.tuning === 'repeated');
    box.innerHTML = `<div class="cal-panel">
        <div class="tune-title">Bootstrap internal validation (Option C)</div>
        <p class="tune-muted"><strong>Apparent</strong> performance is measured on the same patients the model learned from, so it is too optimistic. Each bootstrap resample draws patients with replacement and repeats the modelling steps on them: imputation (MICE when multiple imputation is chosen), encoding and scaling are refitted${anyRepeated ? ', and the automatic tuning is run again' : ''}. The refitted model is scored on its resample and on the original patients, and the average difference is the <strong>optimism</strong>. Subtracting it gives the <strong>optimism-corrected</strong> estimate of how the model would perform on new patients from the same population. Models that can memorise their training patients (Random Forest, boosted trees, k-NN, neural networks) are instead reported with the <strong>.632+</strong> estimate, which scores each bootstrap model only on the patients it did not see (Efron &amp; Tibshirani, 1997); Harrell's correction is too small for them. Choices you made by hand (which predictors to include, how to encode them) are not repeated. The corrected calibration slope estimates how much the predictions should be shrunk (values below 1 mean overfitting).</p>
        ${blocks}
        ${selBlock}
    </div>`;
}

export { renderSampleSizePanel, renderOptimismPanel };

