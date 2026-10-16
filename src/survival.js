// survival.js: time-to-event analysis with a Cox proportional-hazards model (Efron ties).
// Split first (stratified by event), imputation and encoding fitted on the training patients only; hazard
// ratios; Harrell's C-index, time-dependent AUC and Brier score (censoring-weighted) on the test patients; calibration at a chosen horizon
// against Kaplan-Meier observed risk; Kaplan-Meier curves by predicted-risk group; single-patient prediction.
/* global DataEngine, ModelEngine */
import { AppState, escapeHtml, APP_VERSION } from './core.js';
import { lineChartSvg, chartLegend, CURVE_COLOURS } from './panels.js';

const SURV = { result: null };

function numericColumns(rows, headers) {
    return headers.filter(h => {
        const vals = rows.map(r => r[h]).filter(v => !DataEngine.isMissing(v));
        return vals.length > 0 && vals.every(v => !isNaN(Number(v))) && vals.some(v => Number(v) > 0);
    });
}
function twoValueColumns(rows, headers) {
    return headers.filter(h => new Set(rows.map(r => r[h]).filter(v => !DataEngine.isMissing(v)).map(v => String(v).trim())).size === 2);
}

function renderSurvivalSetup() {
    const box = document.getElementById('survival-setup');
    if (!box || !AppState.parsedData) return;
    const { headers, rows } = AppState.parsedData;
    const timeCols = numericColumns(rows, headers);
    const eventCols = twoValueColumns(rows, headers);
    const guessTime = timeCols.find(h => /time|month|day|year|follow|surv|duration/i.test(h)) || timeCols[0] || '';
    const guessEvent = eventCols.find(h => /event|death|died|revision|status|outcome|fail|recur/i.test(h)) || eventCols[0] || '';
    const opts = (list, sel) => list.map(h => `<option value="${escapeHtml(h)}"${h === sel ? ' selected' : ''}>${escapeHtml(h)}</option>`).join('');
    box.innerHTML = `
        <div class="surv-grid">
            <label class="form-group"><span class="form-label">Follow-up time column</span>
                <select id="surv-time-col" class="form-select">${opts(timeCols, guessTime)}</select></label>
            <label class="form-group"><span class="form-label">Event column (did the event happen?)</span>
                <select id="surv-event-col" class="form-select">${opts(eventCols, guessEvent)}</select></label>
            <label class="form-group"><span class="form-label">Value meaning "event occurred"</span>
                <select id="surv-event-value" class="form-select"></select></label>
            <label class="form-group"><span class="form-label">Training / testing split</span>
                <select id="surv-split" class="form-select"><option value="0.7">70 / 30</option><option value="0.75">75 / 25</option><option value="0.8" selected>80 / 20</option></select></label>
            <label class="form-group"><span class="form-label">Split seed</span>
                <input type="number" id="surv-seed" class="form-input" value="42"></label>
            <label class="form-group"><span class="form-label">Prediction horizon (same units as time)</span>
                <input type="number" id="surv-horizon" class="form-input" min="0" step="any"></label>
        </div>
        <div class="form-label" style="margin-top: 10px;">Predictors</div>
        <div id="surv-predictors" class="surv-predictors"></div>
        <p class="tune-muted">The time column is the time from the start of follow-up to the event, or to the last follow-up for patients without the event (censored). Missing predictor values are filled with the methods chosen in Step 3, fitted on the training patients only.</p>
        <button type="button" id="surv-run-btn" class="btn btn-primary">⏱️ Fit Cox Model &amp; Evaluate on Test Patients</button>
        <div id="surv-status" class="tune-muted" aria-live="polite" style="margin-top: 8px;"></div>`;
    const refresh = () => {
        const ec = document.getElementById('surv-event-col').value;
        const tc = document.getElementById('surv-time-col').value;
        const vals = [...new Set(rows.map(r => r[ec]).filter(v => !DataEngine.isMissing(v)).map(v => String(v).trim()))].sort();
        const evGuess = vals.find(v => /^(yes|1|true|dead|died|event|revised|revision)$/i.test(v)) || vals[vals.length - 1];
        document.getElementById('surv-event-value').innerHTML = vals.map(v => `<option${v === evGuess ? ' selected' : ''}>${escapeHtml(v)}</option>`).join('');
        const times = rows.map(r => Number(r[tc])).filter(v => isFinite(v) && v > 0).sort((a, b) => a - b);
        const h = document.getElementById('surv-horizon');
        if (times.length && !h.dataset.user) h.value = Number(times[Math.floor(times.length / 2)].toPrecision(3));
        const cfgs = AppState.columnConfigs || {};
        document.getElementById('surv-predictors').innerHTML = headers.filter(c => c !== tc && c !== ec && cfgs[c] && !/ \(was missing\)$/.test(c)).map(c => {
            const on = cfgs[c].role === 'feature';
            return `<label class="surv-pred"><input type="checkbox" class="surv-pred-check" value="${escapeHtml(c)}"${on ? ' checked' : ''}> ${escapeHtml(c)} <span class="tune-sd">(${escapeHtml(cfgs[c].type)})</span></label>`;
        }).join('');
    };
    document.getElementById('surv-time-col').addEventListener('change', refresh);
    document.getElementById('surv-event-col').addEventListener('change', refresh);
    document.getElementById('surv-horizon').addEventListener('input', e => { e.target.dataset.user = '1'; });
    document.getElementById('surv-run-btn').addEventListener('click', () => { runSurvival().catch(e => { document.getElementById('surv-status').textContent = `⚠️ ${e.message || e}`; }); });
    refresh();
}

// Columns of the encoded matrix used by the Cox model: one reference category per categorical predictor
function coxDesign(prep) {
    const cols = [];
    prep.featureCols.forEach(col => {
        const m = prep.featureMetadata[col];
        if (m.type === 'nominal') {
            const ref = m.mode !== undefined ? m.mode : m.categories[0];
            m.categories.forEach((cat, k) => { if (cat !== ref) cols.push({ idx: m.startIndex + k, name: `${col}: ${cat}`, note: `vs ${ref}`, scale: 1 }); });
        } else {
            const sd = m.std > 0 ? m.std : 1;
            const perUnit = m.type === 'ordinal' || sd >= 1;
            cols.push({ idx: m.startIndex, name: col, note: m.type === 'ordinal' ? 'per rank step' : perUnit ? 'per unit' : `per SD (${sd.toPrecision(2)} units)`, scale: perUnit ? sd : 1, perSD: perUnit });
        }
    });
    return cols;
}

async function runSurvival() {
    const status = document.getElementById('surv-status');
    const tc = document.getElementById('surv-time-col').value;
    const ec = document.getElementById('surv-event-col').value;
    const ev = document.getElementById('surv-event-value').value;
    const preds = [...document.querySelectorAll('.surv-pred-check')].filter(c => c.checked).map(c => c.value);
    const horizon = parseFloat(document.getElementById('surv-horizon').value);
    const ratio = parseFloat(document.getElementById('surv-split').value) || 0.8;
    const seed = parseInt(document.getElementById('surv-seed').value, 10) || 42;
    if (!tc || !ec) throw new Error('Choose the time and event columns.');
    if (!preds.length) throw new Error('Choose at least one predictor.');
    if (!(horizon > 0)) throw new Error('Enter a prediction horizon greater than 0.');
    status.textContent = 'Fitting…';
    await ModelEngine.yieldToBrowser();

    const rows = AppState.parsedData.rows.filter(r => !DataEngine.isMissing(r[tc]) && Number(r[tc]) > 0 && !DataEngine.isMissing(r[ec]));
    const excluded = AppState.parsedData.rows.length - rows.length;
    const cfg = {};
    preds.forEach(c => { cfg[c] = { ...(AppState.columnConfigs[c] || { type: 'continuous', imputationStrategy: 'median' }), role: 'feature' }; });
    cfg[ec] = { role: 'target', type: 'nominal' };
    const evLabel = r => (String(r[ec]).trim() === ev ? 1 : 0);
    const split = DataEngine.trainTestSplit(rows, rows.map(r => String(evLabel(r))), ratio, true, seed);
    const trImp = DataEngine.imputeDataset(split.trainX, cfg, AppState.columnAnalysis);
    const teImp = DataEngine.imputeDataset(split.testX, cfg, AppState.columnAnalysis, trImp.plan);
    const prep = DataEngine.prepareFeaturesAndTarget(trImp.imputedRows, cfg);
    prep.imputation = { configs: JSON.parse(JSON.stringify(cfg)), plan: trImp.plan };
    const te = DataEngine.prepareTestingDataset(teImp.imputedRows, prep);
    const cols = coxDesign(prep);
    const Xtr = prep.X.map(x => cols.map(c => x[c.idx]));
    const Xte = te.X.map(x => cols.map(c => x[c.idx]));
    const tTr = trImp.imputedRows.map(r => Number(r[tc])), eTr = trImp.imputedRows.map(evLabel);
    const tTe = te.validRows.map(r => Number(r[tc])), eTe = te.validRows.map(evLabel);
    const events = eTr.reduce((a, b) => a + b, 0);
    if (events < 5) throw new Error(`Only ${events} events in the training patients; at least 5 are needed.`);

    let cox = ModelEngine.createCox().fit(Xtr, tTr, eTr);
    let penalised = false;
    if (!cox.converged) { cox = ModelEngine.createCox({ penalty: 0.1 }).fit(Xtr, tTr, eTr); penalised = true; }
    const lpTe = Xte.map(x => cox.linearPredictor(x));
    const cIndex = ModelEngine.concordanceIndex(tTe, eTe, lpTe);
    // bootstrap CI for the C-index (test patients resampled)
    const rng = DataEngine.createPRNG(seed);
    // Time-dependent measures at the horizon, weighted for censoring (censoring distribution from the training patients)
    const censKM = ModelEngine.censoringKM(tTr, eTr);
    const survTe = Xte.map(x => cox.survivalAt(x, horizon));
    const tdAuc = ModelEngine.timeDependentAUC(tTe, eTe, lpTe, horizon, censKM);
    const brier = ModelEngine.ipcwBrier(tTe, eTe, survTe, horizon, censKM);
    const sNull = ModelEngine.kmAt(ModelEngine.kaplanMeier(tTr, eTr), horizon); // the same risk for everyone
    const brierNull = ModelEngine.ipcwBrier(tTe, eTe, tTe.map(() => sNull), horizon, censKM);
    const scaledBrier = brier !== null && brierNull > 0 ? 1 - brier / brierNull : null;
    const boots = [], bootsTd = [];
    for (let b = 0; b < 500; b++) {
        const idx = tTe.map(() => Math.floor(rng() * tTe.length));
        const bt = idx.map(i => tTe[i]), be = idx.map(i => eTe[i]), bl = idx.map(i => lpTe[i]);
        const c = ModelEngine.concordanceIndex(bt, be, bl);
        if (c !== null) boots.push(c);
        const a = ModelEngine.timeDependentAUC(bt, be, bl, horizon, censKM);
        if (a !== null) bootsTd.push(a);
        if (b % 100 === 99) await ModelEngine.yieldToBrowser();
    }
    const pctCI = arr => { arr.sort((a, b) => a - b); return arr.length >= 50 ? [arr[Math.floor(0.025 * (arr.length - 1))], arr[Math.floor(0.975 * (arr.length - 1))]] : null; };
    const cCI = pctCI(boots);
    const tdCI = pctCI(bootsTd);
    // calibration at the horizon: groups of predicted risk vs Kaplan-Meier observed risk
    const risk = Xte.map(x => 1 - cox.survivalAt(x, horizon));
    const nG = tTe.length >= 100 ? 5 : tTe.length >= 30 ? 3 : 2;
    const order = risk.map((r, i) => i).sort((a, b) => risk[a] - risk[b]);
    const groups = Array.from({ length: nG }, (_, g) => order.slice(Math.round(g * order.length / nG), Math.round((g + 1) * order.length / nG)));
    const calib = groups.map(g => {
        const km = ModelEngine.kaplanMeier(g.map(i => tTe[i]), g.map(i => eTe[i]));
        const st = ModelEngine.kmStepAt(km, horizon);
        return { predicted: g.reduce((a, i) => a + risk[i], 0) / g.length, observed: 1 - st.survival, observedCI: st.lower !== null ? [1 - st.upper, 1 - st.lower] : null, n: g.length };
    });
    const kmAll = ModelEngine.kaplanMeier(tTe, eTe);
    const meanPred = risk.reduce((a, b) => a + b, 0) / (risk.length || 1);
    const obsAll = 1 - ModelEngine.kmAt(kmAll, horizon);
    const stAll = ModelEngine.kmStepAt(kmAll, horizon);
    const obsAllCI = stAll.lower !== null ? [1 - stAll.upper, 1 - stAll.lower] : null;
    // Kaplan-Meier curves by risk tertile (test patients)
    const tert = [0, 1, 2].map(g => order.slice(Math.round(g * order.length / 3), Math.round((g + 1) * order.length / 3)));
    const kmGroups = tert.map(g => ModelEngine.kaplanMeier(g.map(i => tTe[i]), g.map(i => eTe[i])));
    const maxFollow = Math.max(...tTr, ...tTe);
    SURV.result = { tc, ec, ev, preds, horizon, ratio, seed, cols, prep, cox, penalised, cIndex, cCI, tdAuc, tdCI, brier, brierNull, scaledBrier, calib, obsAllCI, meanPred, obsAll, kmGroups, maxFollow,
        nTrain: Xtr.length, nTest: Xte.length, eventsTrain: events, eventsTest: eTe.reduce((a, b) => a + b, 0), excluded };
    renderSurvivalResults();
    status.textContent = '';
}

function stepPoints(km) {
    const pts = [];
    km.forEach((k, i) => { if (i > 0) pts.push({ x: k.time, y: km[i - 1].survival }); pts.push({ x: k.time, y: k.survival }); });
    return pts;
}

function renderSurvivalResults() {
    const r = SURV.result;
    const box = document.getElementById('survival-results');
    if (!r || !box) return;
    const f2 = v => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
    const summary = r.cox.summary(r.cols.map(c => c.name));
    const hrRows = summary.map((s, i) => {
        const c = r.cols[i];
        const k = c.perSD ? 1 / c.scale : 1;
        const hr = Math.exp(s.coef * k), lo = Math.exp((s.coef - 1.959964 * s.se) * k), hi = Math.exp((s.coef + 1.959964 * s.se) * k);
        return `<tr><td>${escapeHtml(c.name)}</td><td>${f2(hr)} (${f2(lo)}–${f2(hi)})</td><td>${escapeHtml(c.note)}</td><td>${s.p === null ? 'n/a' : s.p < 0.001 ? '< 0.001' : s.p.toFixed(3)}</td></tr>`;
    }).join('');
    const pctTxt = v => `${(v * 100).toFixed(1)}%`;
    const calMax = Math.min(1, Math.max(0.1, ...r.calib.map(c => Math.max(c.predicted, c.observed))) * 1.1);
    const tertNames = ['Low predicted risk', 'Medium predicted risk', 'High predicted risk'];
    const tertCols = ['var(--accent-emerald)', 'var(--accent-amber)', 'var(--accent-rose)'];
    const epp = r.eventsTrain / Math.max(1, r.cols.length);
    box.innerHTML = `
        <div class="cal-panel">
            <div class="tune-title">⏱️ Cox proportional-hazards model: ${escapeHtml(r.ec)} = "${escapeHtml(r.ev)}" over ${escapeHtml(r.tc)}</div>
            <p class="tune-muted">${r.nTrain} training patients (${r.eventsTrain} events) and ${r.nTest} test patients (${r.eventsTest} events), split ${Math.round(r.ratio * 100)}/${Math.round((1 - r.ratio) * 100)} (stratified by event, seed ${r.seed})${r.excluded ? `; ${r.excluded} row(s) without a usable time or event were excluded` : ''}.${r.penalised ? ' <strong>The unpenalised model did not converge, so a small ridge penalty (0.1) was applied.</strong>' : ''}${epp < 10 ? ` <strong>Caution:</strong> only ${epp.toFixed(1)} events per model parameter; the estimates are unstable.` : ''}</p>
            <div class="cmp-guide-title" style="font-size: 13px;">Hazard ratios (training patients)</div>
            <table class="data-table thr-table"><thead><tr><th>Predictor</th><th>Hazard ratio (95% CI)</th><th>Scale</th><th>p-value</th></tr></thead><tbody>${hrRows}</tbody></table>
            <p class="tune-muted">A hazard ratio above 1 means the event tends to happen sooner (higher instantaneous risk); below 1, later. Each is adjusted for the other predictors; categorical predictors are compared with their most common category. The Cox model assumes the hazard ratios stay constant over follow-up (proportional hazards); check this clinically, e.g. whether the Kaplan-Meier curves below cross. Associations, not causation.</p>
        </div>
        <div class="cal-panel">
            <div class="tune-title">📊 Performance on the test patients</div>
            <div class="cal-stats"><div><strong>Harrell's C-index</strong>: ${r.cIndex !== null ? r.cIndex.toFixed(3) : 'N/A'}${r.cCI ? ` <span class="tune-sd">(95% CI ${r.cCI[0].toFixed(3)}–${r.cCI[1].toFixed(3)})</span>` : ''} &mdash; the probability that, of two comparable patients, the one with the event first had the higher predicted risk (0.5 = chance, 1 = perfect).</div>
            <div><strong>Time-dependent AUC at ${escapeHtml(String(r.horizon))}</strong>: ${typeof r.tdAuc === 'number' ? r.tdAuc.toFixed(3) : 'N/A'}${r.tdCI ? ` <span class="tune-sd">(95% CI ${r.tdCI[0].toFixed(3)}–${r.tdCI[1].toFixed(3)})</span>` : ''} &mdash; of a patient who had the event by ${escapeHtml(String(r.horizon))} and one still event-free after it, how often the first had the higher predicted risk.</div>
            <div><strong>Brier score at ${escapeHtml(String(r.horizon))}</strong>: ${typeof r.brier === 'number' ? r.brier.toFixed(3) : 'N/A'} (lower is better; giving everyone the overall Kaplan-Meier risk scores ${typeof r.brierNull === 'number' ? r.brierNull.toFixed(3) : 'N/A'})${typeof r.scaledBrier === 'number' ? `; <strong>scaled Brier</strong> ${(r.scaledBrier * 100).toFixed(1)}% (share of that error the model removes; 0% = no better than the overall risk)` : ''}.</div>
            <div><strong>Overall risk by ${escapeHtml(String(r.horizon))}</strong>: predicted ${pctTxt(r.meanPred)} vs observed ${pctTxt(r.obsAll)}${r.obsAllCI ? ` <span class="tune-sd">(95% CI ${pctTxt(r.obsAllCI[0])}–${pctTxt(r.obsAllCI[1])})</span>` : ''} (Kaplan-Meier).</div>
            <div class="tune-muted">The time-dependent AUC and Brier score weight each patient for censoring (inverse probability of censoring weights, with the censoring pattern estimated from the training patients), so patients lost to follow-up before ${escapeHtml(String(r.horizon))} do not bias them.</div></div>
            <div class="cal-grid" style="grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));">
                <div>${lineChartSvg({ series: [{ name: 'cal', colour: 'var(--accent-cyan)', points: r.calib.map(c => ({ x: c.predicted, y: c.observed })) }, { name: 'calpts', colour: 'var(--accent-cyan)', scatter: true, points: r.calib.map(c => ({ x: c.predicted, y: c.observed })) }],
                    xLabel: `Predicted risk by ${r.horizon}`, yLabel: 'Observed risk (Kaplan-Meier)', xMax: calMax, yMax: calMax, diagonal: true, aria: 'Survival calibration at the horizon' })}
                    <p class="tune-muted">Calibration at ${escapeHtml(String(r.horizon))}: test patients are grouped by predicted risk; each point compares the group's average predicted risk with its Kaplan-Meier observed risk. Points near the diagonal mean the predicted risks can be taken at face value.</p>
                    <table class="data-table thr-table"><thead><tr><th>Risk group</th><th>Patients</th><th>Mean predicted</th><th>Observed (95% CI)</th></tr></thead><tbody>${r.calib.map((c, i) => `<tr><td>${i + 1} of ${r.calib.length}</td><td>${c.n}</td><td>${pctTxt(c.predicted)}</td><td>${pctTxt(c.observed)}${c.observedCI ? ` <span class="tune-sd">(${pctTxt(c.observedCI[0])}–${pctTxt(c.observedCI[1])})</span>` : ''}</td></tr>`).join('')}</tbody></table>
                    <p class="tune-muted">Observed risks come with 95% confidence intervals (Greenwood's formula on the log(−log) scale), so a group whose interval includes its predicted risk is consistent with good calibration.</p></div>
                <div>${lineChartSvg({ series: r.kmGroups.map((km, g) => ({ name: tertNames[g], colour: tertCols[g], points: stepPoints(km) })), xLabel: `${r.tc}`, yLabel: 'Event-free (Kaplan-Meier)', xMax: r.maxFollow, yMax: 1, aria: 'Kaplan-Meier curves by predicted risk group' })}
                    ${chartLegend(tertNames.map((n, g) => ({ colour: tertCols[g], label: n })))}
                    <p class="tune-muted">Test patients split into thirds by predicted risk. Well-separated curves show the model distinguishes patients with early and late events.</p></div>
            </div>
            ${r.nTest < 100 || r.eventsTest < 50 ? '<div class="tune-warn">⚠️ Few test patients or events: these performance estimates are imprecise.</div>' : ''}
            <button type="button" class="btn btn-secondary btn-sm" id="surv-report-btn">📄 Download Survival Analysis Report (.JSON)</button>
        </div>
        <div class="cal-panel">
            <div class="tune-title">🔮 Predicted event-free probability for one patient</div>
            <div class="predict-form-grid" id="surv-predict-form">${r.prep.featureCols.map((col, i) => {
                const m = r.prep.featureMetadata[col];
                if (m.type === 'nominal') return `<label class="form-field"><span>${escapeHtml(col)}</span><select class="control-select surv-in" data-col="${escapeHtml(col)}">${m.categories.map(c => `<option${c === m.mode ? ' selected' : ''}>${escapeHtml(c)}</option>`).join('')}</select></label>`;
                if (m.type === 'ordinal') return `<label class="form-field"><span>${escapeHtml(col)}</span><select class="control-select surv-in" data-col="${escapeHtml(col)}">${m.ranks.map(c => `<option>${escapeHtml(c)}</option>`).join('')}</select></label>`;
                return `<label class="form-field"><span>${escapeHtml(col)}</span><input type="number" step="any" class="control-input surv-in" data-col="${escapeHtml(col)}" value="${Number(m.mean).toFixed(2)}"></label>`;
            }).join('')}</div>
            <button type="button" class="btn btn-primary btn-sm" id="surv-predict-btn" style="margin-top: 8px;">Predict</button>
            <div id="surv-predict-out" aria-live="polite"></div>
            <p class="tune-muted">For research and education only; not for decisions about an individual patient's care.</p>
        </div>`;
    document.getElementById('surv-report-btn').addEventListener('click', downloadSurvivalReport);
    document.getElementById('surv-predict-btn').addEventListener('click', predictOne);
}

function predictOne() {
    const r = SURV.result;
    const input = {};
    document.querySelectorAll('.surv-in').forEach(el => { input[el.dataset.col] = el.value; });
    const x = DataEngine.transformSingleRow(input, r.prep, []);
    const xd = r.cols.map(c => x[c.idx]);
    const times = [0.25, 0.5, 0.75, 1].map(f => Number((r.horizon * f * 2 / 2).toPrecision(3)));
    times[3] = r.horizon;
    const rows = times.map(t => `<tr><td>${t}</td><td>${(r.cox.survivalAt(xd, t) * 100).toFixed(1)}%</td></tr>`).join('');
    document.getElementById('surv-predict-out').innerHTML = `<table class="data-table thr-table" style="margin-top: 8px;"><thead><tr><th>${escapeHtml(r.tc)}</th><th>Predicted probability of no event so far</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function downloadSurvivalReport() {
    const r = SURV.result;
    if (!r) return;
    const summary = r.cox.summary(r.cols.map(c => c.name));
    const report = {
        title: 'Survival (Cox proportional-hazards) Analysis Report', software: { name: 'Universal Medical ML/NeuralNet Predictor', version: APP_VERSION },
        generated_at: new Date().toISOString(), dataset: AppState.fileName, time_column: r.tc, event_column: r.ec, event_value: r.ev, predictors: r.preds,
        design: { split: `${Math.round(r.ratio * 100)}/${Math.round((1 - r.ratio) * 100)} stratified by event`, seed: r.seed, train_patients: r.nTrain, train_events: r.eventsTrain, test_patients: r.nTest, test_events: r.eventsTest, excluded_rows: r.excluded, ties: 'Efron', penalty: r.penalised ? 0.1 : 0 },
        hazard_ratios: summary.map((s, i) => ({ term: s.name, scale: r.cols[i].note, coef: s.coef, se: s.se, hr: s.hr, hr_95ci: [s.hrLow, s.hrHigh], p: s.p })),
        test_performance: { c_index: r.cIndex, c_index_95ci_bootstrap: r.cCI, horizon: r.horizon, time_dependent_auc: r.tdAuc, time_dependent_auc_95ci_bootstrap: r.tdCI, brier_score_ipcw: r.brier, brier_score_null_km: r.brierNull, scaled_brier: r.scaledBrier, mean_predicted_risk: r.meanPred, observed_risk_kaplan_meier: r.obsAll, observed_risk_95ci: r.obsAllCI, calibration_groups: r.calib }
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Survival_Analysis_Report_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

export { renderSurvivalSetup, runSurvival };
