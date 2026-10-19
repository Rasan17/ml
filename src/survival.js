// survival.js: Step 4 for a time-to-event outcome (chosen in Step 2): a Cox proportional-hazards model (Efron ties).
// Imputation (single or MICE) and encoding are fitted on the training patients only; validated on a held-out
// testing set (Option A), a separate cohort (Option B) or by bootstrap optimism correction (Option C). Hazard
// ratios; Harrell's C-index, time-dependent AUC and Brier score (censoring-weighted); calibration at the horizon
// against Kaplan-Meier observed risk; Kaplan-Meier curves by predicted-risk group; single-patient prediction.
/* global DataEngine, ModelEngine */
import { AppState, escapeHtml, APP_VERSION } from './core.js';
import { lineChartSvg, chartLegend } from './panels.js';
import { prepareMultipleImputation } from './multiple-imputation.js';
import { bootstrapResamples, bootstrapResamplesGrouped } from './validation-core.js';
import { plainFillConfigs } from './cv-tuning.js';

const SURV = { result: null };

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

// Cox models fitted on each multiply imputed training dataset, used together: hazard ratios pooled by Rubin's
// rules, predictions (linear predictor, survival probability) averaged over the imputations. Patients carry one
// encoded version per imputation chain (x.__mi), as for the other models.
function pooledCox(models) {
    const each = (x, f) => { const xs = x.__mi || [x]; return models.reduce((a, m, l) => a + f(m, xs[Math.min(l, xs.length - 1)]), 0) / models.length; };
    return {
        pooled: models.length,
        converged: models.every(m => m.converged),
        linearPredictor: x => each(x, (m, xv) => m.linearPredictor(xv)),
        survivalAt: (x, t) => each(x, (m, xv) => m.survivalAt(xv, t)),
        summary: names => {
            const per = models.map(m => m.summary(names));
            const est = per.map(sm => ({ coef: sm.map(r => r.coef), se: sm.map(r => r.se) }));
            const pool = ModelEngine.poolRubin(est);
            return names.map((nm, j) => ({ name: nm, coef: pool[j].coef, se: pool[j].se, p: pool[j].p, hr: Math.exp(pool[j].coef), hrLow: Math.exp(pool[j].low), hrHigh: Math.exp(pool[j].high) }));
        }
    };
}

/**
 * Time-to-event analysis (Cox model) for Step 4. o: { tc, ec, ev, preds, horizon,
 * ratio, seed, status, boxId, miM }: time and event columns, event value, predictors, horizon, training share,
 * split seed, status element, results container, and the number of multiple imputations (0 = single imputation
 * with the Step 3 methods).
 */
// Fits the whole process on the given patients: missing-value filling (single imputation with the Step 3 methods,
// or MICE with one Cox model per completed dataset), encoding, and the Cox model (penalised if it does not converge).
// Returns the model and a function that encodes other patients the same way.
async function fitSurvivalProcess(rows, cfg, tc, evLabel, miM, seed, quickPlain) {
    let prep, cols, cox, penalised = false, miInfo = null, Xtr, tTr, eTr;
    const fitCox = (X, t, e) => { let m = ModelEngine.createCox().fit(X, t, e); if (!m.converged) { m = ModelEngine.createCox({ penalty: 0.1 }).fit(X, t, e); penalised = true; } return m; };
    const pick = x => cols.map(c => x[c.idx]);
    if (miM && !quickPlain) {
        // MICE on these patients only (the time and event are never used); one Cox model per completed dataset
        const mi = await prepareMultipleImputation(rows, cfg, miM, seed);
        prep = mi.prep;
        cols = coxDesign(prep);
        tTr = rows.map(r => Number(r[tc])); eTr = rows.map(evLabel);
        if (eTr.reduce((a, b) => a + b, 0) < 5) return { tooFewEvents: eTr.reduce((a, b) => a + b, 0) };
        cox = pooledCox(mi.sets.map(st => fitCox(st.X.map(pick), tTr, eTr)));
        Xtr = mi.sets[0].X.map(pick);
        miInfo = { m: miM, imputedColumns: mi.imputedColumns, trace: prep.multipleImputation && prep.multipleImputation.trace, traceScale: prep.multipleImputation && prep.multipleImputation.traceScale };
    } else {
        // quickPlain: the plain fill (median / most frequent) used inside resamples when "quick approximation" is on
        const fillCfg = quickPlain ? plainFillConfigs(cfg) : cfg;
        const imp = DataEngine.imputeDataset(rows, fillCfg, AppState.columnAnalysis);
        prep = DataEngine.prepareFeaturesAndTarget(imp.imputedRows, fillCfg);
        prep.imputation = { configs: JSON.parse(JSON.stringify(fillCfg)), plan: imp.plan };
        cols = coxDesign(prep);
        Xtr = prep.X.map(pick);
        tTr = imp.imputedRows.map(r => Number(r[tc])); eTr = imp.imputedRows.map(evLabel);
        if (eTr.reduce((a, b) => a + b, 0) < 5) return { tooFewEvents: eTr.reduce((a, b) => a + b, 0) };
        cox = fitCox(Xtr, tTr, eTr);
    }
    // Encodes other patients exactly as the training patients (blank values filled by the same fitted imputation)
    const encode = other => {
        const te = miInfo ? DataEngine.prepareTestingDataset(other, prep)
            : DataEngine.prepareTestingDataset(DataEngine.imputeDataset(other, prep.imputation.configs, AppState.columnAnalysis, prep.imputation.plan).imputedRows, prep);
        const X = te.X.map(x => { const d = pick(x); if (x.__mi) Object.defineProperty(d, '__mi', { value: x.__mi.map(pick), enumerable: false }); return d; });
        return { X, rows: te.validRows };
    };
    return { prep, cols, cox, penalised, miInfo, Xtr, tTr, eTr, encode };
}

/**
 * Time-to-event analysis (Cox model) for Step 4. o: { tc, ec, ev, preds, horizon,
 * ratio, seed, status, boxId, miM, completeCase, validation, B, groupCol, quickPlain, externalRows }.
 * validation: 'split' (held-out testing set, Option A), 'separate' (fit on all patients, test on externalRows,
 * Option B) or 'bootstrap' (fit on all patients; Harrell's optimism correction over B resamples, Option C).
 */
async function runSurvivalAnalysis(o) {
    const { tc, ec, ev, horizon, ratio, seed, status, boxId, miM = 0, validation = 'split' } = o;
    if (!tc || !ec) throw new Error('Choose the time and event columns.');
    // The follow-up time and event status are the outcome: never predictors
    const preds = (o.preds || []).filter(c => c !== tc && c !== ec);
    if (!preds.length) throw new Error('Choose at least one predictor.');
    if (!(horizon > 0)) throw new Error('Enter a prediction horizon greater than 0.');
    if (status) status.textContent = 'Fitting…';
    await ModelEngine.yieldToBrowser();

    // Patients need a follow-up time and an event status; complete-case analysis (Step 3) also drops blank predictors
    const usable = list => list.filter(r => !DataEngine.isMissing(r[tc]) && Number(r[tc]) > 0 && !DataEngine.isMissing(r[ec])
        && (!o.completeCase || preds.every(c => !DataEngine.isMissing(r[c]))));
    const rows = usable(AppState.parsedData.rows);
    const excluded = AppState.parsedData.rows.length - rows.length;
    const cfg = {};
    preds.forEach(c => { cfg[c] = { ...(AppState.columnConfigs[c] || { type: 'continuous', imputationStrategy: 'median' }), role: 'feature' }; });
    cfg[ec] = { role: 'target', type: 'nominal' };
    const evLabel = r => (String(r[ec]).trim() === ev ? 1 : 0);
    let trainRows = rows, testRows, externalExcluded = 0, splitInfo = null;
    if (validation === 'split') {
        // The split method chosen in Step 4 (random balanced by event, by patient, or by date)
        const split = o.splitFn ? o.splitFn(rows, rows.map(r => String(evLabel(r))), ratio, seed) : DataEngine.trainTestSplit(rows, rows.map(r => String(evLabel(r))), ratio, true, seed);
        trainRows = split.trainX; testRows = split.testX; splitInfo = split.info || null;
    } else if (validation === 'separate') {
        const ext = o.externalRows || [];
        if (!ext.length) throw new Error('Load the validation cohort (Dataset ii) for Option B first.');
        if (!(tc in ext[0]) || !(ec in ext[0])) throw new Error(`The validation cohort needs the columns "${tc}" and "${ec}".`);
        testRows = usable(ext); externalExcluded = ext.length - testRows.length;
        if (!testRows.length) throw new Error('No patient in the validation cohort has a usable follow-up time and event status.');
    } else {
        testRows = rows; // apparent performance, corrected for optimism below
    }
    const fit = await fitSurvivalProcess(trainRows, cfg, tc, evLabel, miM, seed, false);
    if (fit.tooFewEvents !== undefined) throw new Error(`Only ${fit.tooFewEvents} events in the training patients; at least 5 are needed.`);
    const { prep, cols, cox, penalised, miInfo, Xtr, tTr, eTr } = fit;
    const enc = fit.encode(testRows);
    const Xte = enc.X;
    const tTe = enc.rows.map(r => Number(r[tc])), eTe = enc.rows.map(evLabel);
    const droppedBlank = testRows.length - enc.rows.length; // removed by a "drop rows" missing-data choice
    const events = eTr.reduce((a, b) => a + b, 0);
    const lpTe = Xte.map(x => cox.linearPredictor(x));
    const cIndex = ModelEngine.concordanceIndex(tTe, eTe, lpTe);
    // bootstrap CI for the C-index (test patients resampled)
    const rng = DataEngine.createPRNG(seed);
    // Time-dependent measures at the horizon, weighted for censoring. The censoring distribution comes from the
    // training patients, except for a separate cohort (Option B), whose follow-up can differ: its own is used
    const censKM = validation === 'separate' ? ModelEngine.censoringKM(tTe, eTe) : ModelEngine.censoringKM(tTr, eTr);
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
    const lastEvent = Math.max(0, ...tTr.filter((_, i) => eTr[i] === 1));
    // Option C: Harrell's bootstrap optimism correction. Each resample repeats the whole process (imputation, encoding,
    // Cox fit); the model is scored on its resample and on the original patients, and the mean difference (optimism)
    // is subtracted from the apparent performance. The calibration slope is that of the resample models' linear
    // predictor on the original patients (ideal 1; below 1 means predictions too extreme, i.e. overfitting).
    let optimism = null;
    if (validation === 'bootstrap') {
        const B = o.B || 100;
        const keys = o.groupCol ? rows.map(r => r[o.groupCol]) : null;
        const idxs = keys ? bootstrapResamplesGrouped(keys, B, seed) : bootstrapResamples(rows.length, B, seed);
        const dc = [], dtd = [], slopes = [];
        let skipped = 0;
        const t0 = Date.now();
        for (let b = 0; b < idxs.length; b++) {
            if (status) status.textContent = `Bootstrap internal validation: resample ${b + 1} of ${idxs.length}${b > 2 ? ` (about ${Math.ceil((Date.now() - t0) / b * (idxs.length - b) / 1000)} s left)` : ''}…`;
            await ModelEngine.yieldToBrowser();
            const bRows = idxs[b].map(i => rows[i]);
            const f = await fitSurvivalProcess(bRows, cfg, tc, evLabel, miM, seed + b + 1, Boolean(o.quickPlain));
            if (f.tooFewEvents !== undefined) { skipped++; continue; }
            const eb = f.encode(bRows), eo = f.encode(rows);
            const tB = eb.rows.map(r => Number(r[tc])), eB = eb.rows.map(evLabel), lpB = eb.X.map(x => f.cox.linearPredictor(x));
            const tO = eo.rows.map(r => Number(r[tc])), eO = eo.rows.map(evLabel), lpO = eo.X.map(x => f.cox.linearPredictor(x));
            const op = ModelEngine.coxResampleOptimism({ t: tB, e: eB, lp: lpB }, { t: tO, e: eO, lp: lpO }, horizon, censKM);
            if (op.c === null) { skipped++; continue; }
            dc.push(op.c);
            if (op.td !== null) dtd.push(op.td);
            if (op.slope !== null) slopes.push(op.slope);
        }
        const mean = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
        const oc = mean(dc), otd = mean(dtd), slope = mean(slopes);
        optimism = { B: idxs.length, used: dc.length, skipped, grouped: Boolean(keys), groupCol: o.groupCol || null, quickPlain: Boolean(o.quickPlain && miM),
            cOptimism: oc, cCorrected: oc !== null && cIndex !== null ? cIndex - oc : null,
            tdOptimism: otd, tdCorrected: otd !== null && typeof tdAuc === 'number' ? tdAuc - otd : null,
            slope, slopeCI: slopes.length >= 20 ? (() => { const a = slopes.slice().sort((x, y) => x - y); return [a[Math.floor(0.025 * (a.length - 1))], a[Math.floor(0.975 * (a.length - 1))]]; })() : null,
            seconds: Math.round((Date.now() - t0) / 1000) };
    }
    SURV.result = { droppedBlank, splitInfo, splitText: o.describeSplit && splitInfo ? o.describeSplit(splitInfo) : 'random split, balanced by event', censKM, validation, optimism, externalExcluded, boxId: boxId || 'survival-step4-results', miInfo, completeCase: Boolean(o.completeCase), lastEvent, tc, ec, ev, preds, horizon, ratio, seed, cols, prep, cox, penalised, cIndex, cCI, tdAuc, tdCI, brier, brierNull, scaledBrier, calib, obsAllCI, meanPred, obsAll, kmGroups, maxFollow,
        nTrain: Xtr.length, nTest: Xte.length, eventsTrain: events, eventsTest: eTe.reduce((a, b) => a + b, 0), excluded };
    renderSurvivalResults();
    if (status) status.textContent = '';
    return SURV.result;
}

function stepPoints(km) {
    const pts = [];
    km.forEach((k, i) => { if (i > 0) pts.push({ x: k.time, y: km[i - 1].survival }); pts.push({ x: k.time, y: k.survival }); });
    return pts;
}

// Hazard ratio and 95% CI on the reported scale (per unit, rank step or SD): the model is fitted on standardised
// predictors, so coefficients are rescaled; the interval is the model's own (Rubin's t interval when pooled)
function hrOnScale(s, c) {
    const k = c.perSD ? 1 / c.scale : 1;
    return { hr: Math.exp(s.coef * k), lo: Math.exp(Math.log(s.hrLow) * k), hi: Math.exp(Math.log(s.hrHigh) * k), coef: s.coef * k, se: s.se * k };
}

function renderSurvivalResults() {
    const r = SURV.result;
    const box = document.getElementById(r ? r.boxId : 'survival-step4-results');
    if (!r || !box) return;
    const f2 = v => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
    const summary = r.cox.summary(r.cols.map(c => c.name));
    const hrRows = summary.map((s, i) => {
        const c = r.cols[i];
        const { hr, lo, hi } = hrOnScale(s, c);
        return `<tr><td>${escapeHtml(c.name)}</td><td>${f2(hr)} (${f2(lo)}–${f2(hi)})</td><td>${escapeHtml(c.note)}</td><td>${s.p === null ? 'n/a' : s.p < 0.001 ? '< 0.001' : s.p.toFixed(3)}</td></tr>`;
    }).join('');
    const pctTxt = v => `${(v * 100).toFixed(1)}%`;
    const calMax = Math.min(1, Math.max(0.1, ...r.calib.map(c => Math.max(c.predicted, c.observed))) * 1.1);
    const tertNames = ['Low predicted risk', 'Medium predicted risk', 'High predicted risk'];
    const tertCols = ['var(--accent-emerald)', 'var(--accent-amber)', 'var(--accent-rose)'];
    const epp = r.eventsTrain / Math.max(1, r.cols.length);
    const who = r.validation === 'bootstrap' ? 'all patients' : r.validation === 'separate' ? 'validation-cohort patients' : 'test patients';
    box.innerHTML = `
        <div class="cal-panel">
            <div class="tune-title">⏱️ Cox proportional-hazards model: ${escapeHtml(r.ec)} = "${escapeHtml(r.ev)}" over ${escapeHtml(r.tc)}</div>
            <p class="tune-muted">${r.validation === 'bootstrap' ? `Fitted on all ${r.nTrain} patients (${r.eventsTrain} events); validated by bootstrap internal validation (Option C, ${r.optimism.B} resamples)`
                : r.validation === 'separate' ? `Fitted on all ${r.nTrain} patients of Dataset (i) (${r.eventsTrain} events); validated on ${r.nTest} patients of the separate cohort, Dataset (ii) (${r.eventsTest} events)${r.externalExcluded ? `, after excluding ${r.externalExcluded} of its row(s) without a usable time or event` : ''}`
                : `${r.nTrain} training patients (${r.eventsTrain} events) and ${r.nTest} test patients (${r.eventsTest} events), ${Math.round(r.ratio * 100)}/${Math.round((1 - r.ratio) * 100)} ${escapeHtml(r.splitText)} (seed ${r.seed})`}${r.excluded ? `; ${r.excluded} row(s) without a usable time or event were excluded` : ''}${r.droppedBlank ? `; ${r.droppedBlank} evaluated patient(s) were removed by the "drop rows" missing-data choice` : ''}.${r.penalised ? ' <strong>The unpenalised model did not converge, so a small ridge penalty (0.1) was applied.</strong>' : ''}${epp < 10 ? ` <strong>Caution:</strong> only ${epp.toFixed(1)} events per model parameter; the estimates are unstable.` : ''}
            Missing predictor values: ${r.miInfo ? `multiple imputation (MICE, ${r.miInfo.m} datasets, fitted on the training patients without the time or event); one Cox model per dataset, hazard ratios pooled by Rubin's rules and predictions averaged` : r.completeCase ? 'complete-case analysis (patients with a blank predictor excluded)' : 'single imputation with the Step 3 methods, fitted on the training patients'}.</p>
            <div class="cmp-guide-title" style="font-size: 13px;">Hazard ratios (training patients)</div>
            <table class="data-table thr-table"><thead><tr><th>Predictor</th><th>Hazard ratio (95% CI)</th><th>Scale</th><th>p-value</th></tr></thead><tbody>${hrRows}</tbody></table>
            <p class="tune-muted">A hazard ratio above 1 means the event tends to happen sooner (higher instantaneous risk); below 1, later. Each is adjusted for the other predictors; categorical predictors are compared with their most common category. The Cox model assumes the hazard ratios stay constant over follow-up (proportional hazards); check this clinically, e.g. whether the Kaplan-Meier curves below cross. Associations, not causation.</p>
        </div>
        <div class="cal-panel">
            ${r.optimism ? `<div class="tune-title">🔁 Bootstrap internal validation (optimism-corrected)</div>
            <table class="data-table thr-table"><thead><tr><th>Measure</th><th>Apparent (all patients)</th><th>Optimism</th><th>Corrected</th></tr></thead><tbody>
                <tr><td>Harrell's C-index</td><td>${r.cIndex !== null ? r.cIndex.toFixed(3) : 'N/A'}</td><td>${r.optimism.cOptimism !== null ? r.optimism.cOptimism.toFixed(3) : 'N/A'}</td><td><strong>${r.optimism.cCorrected !== null ? r.optimism.cCorrected.toFixed(3) : 'N/A'}</strong></td></tr>
                <tr><td>Time-dependent AUC at ${escapeHtml(String(r.horizon))}</td><td>${typeof r.tdAuc === 'number' ? r.tdAuc.toFixed(3) : 'N/A'}</td><td>${r.optimism.tdOptimism !== null ? r.optimism.tdOptimism.toFixed(3) : 'N/A'}</td><td><strong>${r.optimism.tdCorrected !== null ? r.optimism.tdCorrected.toFixed(3) : 'N/A'}</strong></td></tr>
                <tr><td>Calibration slope (ideal 1)</td><td>— (reference: 1)</td><td>${r.optimism.slope !== null ? (1 - r.optimism.slope).toFixed(3) : 'N/A'}</td><td><strong>${r.optimism.slope !== null ? r.optimism.slope.toFixed(3) : 'N/A'}</strong>${r.optimism.slopeCI ? ` <span class="tune-sd">(middle 95% of resamples ${r.optimism.slopeCI[0].toFixed(2)}–${r.optimism.slopeCI[1].toFixed(2)})</span>` : ''}</td></tr>
            </tbody></table>
            <p class="tune-muted">Each of the ${r.optimism.used} usable resamples${r.optimism.skipped ? ` (${r.optimism.skipped} skipped: too few events or no comparable pairs)` : ''} repeated the whole process (${r.optimism.quickPlain ? 'with the quick approximation: a plain median / most-frequent fill instead of MICE inside the resamples' : r.miInfo ? 'MICE' : 'single imputation'}, encoding and the Cox fit) on patients drawn with replacement${r.optimism.grouped ? ` (whole patients by "${escapeHtml(r.optimism.groupCol)}")` : ''}. The optimism is the average amount by which a model scores better on its own resample than on the original patients; it is subtracted from the apparent performance. A corrected calibration slope below 1 means the predictions are too extreme and would need shrinking (${r.optimism.seconds} s).${r.optimism.used < 50 ? ' <strong>Few usable resamples: the correction is imprecise.</strong>' : ''}</p>
            <div class="tune-title" style="margin-top: 12px;">📊 Apparent performance on all patients (before correction)</div>
            <p class="tune-muted">The panels below score the final model on the same patients it was fitted to, so they are optimistic; use the corrected values above when reporting.</p>` : `<div class="tune-title">📊 Performance on the ${r.validation === 'separate' ? 'validation cohort (Dataset ii)' : 'test patients'}</div>`}
            <div class="cal-stats"><div><strong>Harrell's C-index</strong>: ${r.cIndex !== null ? r.cIndex.toFixed(3) : 'N/A'}${r.cCI && r.validation !== 'bootstrap' ? ` <span class="tune-sd">(95% CI ${r.cCI[0].toFixed(3)}–${r.cCI[1].toFixed(3)})</span>` : ''} &mdash; the probability that, of two comparable patients, the one with the event first had the higher predicted risk (0.5 = chance, 1 = perfect).</div>
            <div><strong>Time-dependent AUC at ${escapeHtml(String(r.horizon))}</strong>: ${typeof r.tdAuc === 'number' ? r.tdAuc.toFixed(3) : 'N/A'}${r.tdCI && r.validation !== 'bootstrap' ? ` <span class="tune-sd">(95% CI ${r.tdCI[0].toFixed(3)}–${r.tdCI[1].toFixed(3)})</span>` : ''} &mdash; of a patient who had the event by ${escapeHtml(String(r.horizon))} and one still event-free after it, how often the first had the higher predicted risk.</div>
            <div><strong>Brier score at ${escapeHtml(String(r.horizon))}</strong>: ${typeof r.brier === 'number' ? r.brier.toFixed(3) : 'not estimable (too little follow-up to weight the patients at this horizon; choose an earlier horizon)'} (lower is better; giving everyone the overall Kaplan-Meier risk scores ${typeof r.brierNull === 'number' ? r.brierNull.toFixed(3) : 'N/A'})${typeof r.scaledBrier === 'number' ? `; <strong>scaled Brier</strong> ${(r.scaledBrier * 100).toFixed(1)}% (share of that error the model removes; 0% = no better than the overall risk)` : ''}.</div>
            ${r.horizon > r.lastEvent ? `<div class="tune-warn">⚠️ The horizon (${escapeHtml(String(r.horizon))}) is beyond the last event in the training patients (${escapeHtml(String(r.lastEvent))}): the model cannot learn risk after that time, so predicted risks at this horizon are underestimated. Choose a horizon within the follow-up of the training data.</div>` : ''}
            <div><strong>Overall risk by ${escapeHtml(String(r.horizon))}</strong>: predicted ${pctTxt(r.meanPred)} vs observed ${pctTxt(r.obsAll)}${r.obsAllCI ? ` <span class="tune-sd">(95% CI ${pctTxt(r.obsAllCI[0])}–${pctTxt(r.obsAllCI[1])})</span>` : ''} (Kaplan-Meier).</div>
            <div class="tune-muted">The time-dependent AUC and Brier score weight each patient for censoring (inverse probability of censoring weights, with the censoring pattern estimated from the training patients), so patients lost to follow-up before ${escapeHtml(String(r.horizon))} do not bias them.</div></div>
            <div class="cal-grid" style="grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));">
                <div>${lineChartSvg({ series: [{ name: 'Risk group', colour: 'var(--accent-cyan)', noHover: true, points: r.calib.map(c => ({ x: c.predicted, y: c.observed })) }, { name: 'Risk group', colour: 'var(--accent-cyan)', scatter: true, points: r.calib.map(c => ({ x: c.predicted, y: c.observed })) }],
                    xLabel: `Predicted risk by ${r.horizon}`, yLabel: 'Observed risk (Kaplan-Meier)', xMax: calMax, yMax: calMax, diagonal: true, aria: 'Survival calibration at the horizon' })}
                    <p class="tune-muted">Calibration at ${escapeHtml(String(r.horizon))}: ${who} are grouped by predicted risk; each point compares the group's average predicted risk with its Kaplan-Meier observed risk. Points near the diagonal mean the predicted risks can be taken at face value.</p>
                    <table class="data-table thr-table"><thead><tr><th>Risk group</th><th>Patients</th><th>Mean predicted</th><th>Observed (95% CI)</th></tr></thead><tbody>${r.calib.map((c, i) => `<tr><td>${i + 1} of ${r.calib.length}</td><td>${c.n}</td><td>${pctTxt(c.predicted)}</td><td>${pctTxt(c.observed)}${c.observedCI ? ` <span class="tune-sd">(${pctTxt(c.observedCI[0])}–${pctTxt(c.observedCI[1])})</span>` : ''}</td></tr>`).join('')}</tbody></table>
                    <p class="tune-muted">Observed risks come with 95% confidence intervals (Greenwood's formula on the log(−log) scale), so a group whose interval includes its predicted risk is consistent with good calibration.</p></div>
                <div>${lineChartSvg({ series: r.kmGroups.map((km, g) => ({ name: tertNames[g], colour: tertCols[g], points: stepPoints(km) })), xLabel: `${r.tc}`, yLabel: 'Event-free (Kaplan-Meier)', xMax: r.maxFollow, yMax: 1, aria: 'Kaplan-Meier curves by predicted risk group' })}
                    ${chartLegend(tertNames.map((n, g) => ({ colour: tertCols[g], label: n })))}
                    <p class="tune-muted">${who.charAt(0).toUpperCase() + who.slice(1)} split into thirds by predicted risk. Well-separated curves show the model distinguishes patients with early and late events.</p></div>
            </div>
            ${r.nTest < 100 || r.eventsTest < 50 ? `<div class="tune-warn">⚠️ Few ${who} or events: these performance estimates are imprecise.</div>` : ''}
            <button type="button" class="btn btn-secondary btn-sm" id="surv-report-btn">📄 Download Survival Analysis Report (.JSON)</button>
        </div>
        <p class="tune-muted">Predicted event-free probabilities for one patient: Step 5. For a file of patients (and validation on it if it contains the follow-up time and event): Step 6.</p>`;
    document.getElementById('surv-report-btn').addEventListener('click', downloadSurvivalReport);
}

// Step 5 for a time-to-event outcome: predicted event-free probability for one patient
function renderSurvivalSinglePrediction(boxId) {
    const box = document.getElementById(boxId), r = SURV.result;
    if (!box) return;
    if (!r) { box.innerHTML = '<div class="cal-panel"><p class="tune-muted">Fit the Cox model in Step 4 first.</p></div>'; return; }
    box.innerHTML = `
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
    document.getElementById('surv-predict-btn').addEventListener('click', predictOne);
}

// Cox design columns of an encoded patient (and of each of its multiply imputed versions)
function designOf(x, cols) {
    const d = cols.map(c => x[c.idx]);
    if (x.__mi) Object.defineProperty(d, '__mi', { value: x.__mi.map(v => cols.map(c => v[c.idx])), enumerable: false });
    return d;
}

// Step 6 for a time-to-event outcome: predictions for a file of patients and, if the file also has the follow-up
// time and event columns, validation on those patients (C-index, time-dependent AUC, Brier score, overall calibration)
function survivalBatch(rows, headers) {
    const r = SURV.result, H = r.horizon;
    const lps = [], survs = [];
    const results = rows.map(row => {
        const issues = [], ranges = [];
        const xd = designOf(DataEngine.transformSingleRow(row, r.prep, issues, ranges), r.cols);
        const lp = r.cox.linearPredictor(xd), sv = r.cox.survivalAt(xd, H);
        lps.push(lp); survs.push(sv);
        return { ...row, [`Predicted_event_free_at_${H}`]: `${(sv * 100).toFixed(1)}%`, [`Predicted_risk_by_${H}`]: `${((1 - sv) * 100).toFixed(1)}%`, Linear_Predictor: lp.toFixed(4),
            Imputed_Inputs: issues.map(i => `${i.column}:${i.kind}`).join('; '), Outside_Training_Range: ranges.map(o => `${o.column}=${o.value} (training ${o.min}-${o.max})`).join('; '),
            Model: `Cox proportional-hazards${r.miInfo ? ` (pooled over ${r.miInfo.m} imputations)` : ''}` };
    });
    let validation = null;
    if (headers.includes(r.tc) && headers.includes(r.ec)) {
        const ok = rows.map((row, i) => i).filter(i => !DataEngine.isMissing(rows[i][r.tc]) && Number(rows[i][r.tc]) > 0 && !DataEngine.isMissing(rows[i][r.ec]));
        const t = ok.map(i => Number(rows[i][r.tc])), e = ok.map(i => (String(rows[i][r.ec]).trim() === r.ev ? 1 : 0));
        const lp = ok.map(i => lps[i]), sv = ok.map(i => survs[i]);
        if (ok.length >= 10 && e.some(v => v)) {
            const km = ModelEngine.kaplanMeier(t, e);
            const cw = ModelEngine.censoringKM(t, e); // this file's own censoring pattern
            const brier = ModelEngine.ipcwBrier(t, e, sv, H, cw);
            const brierNull = ModelEngine.ipcwBrier(t, e, t.map(() => ModelEngine.kmAt(km, H)), H, cw);
            validation = { n: ok.length, events: e.reduce((a, b) => a + b, 0), excluded: rows.length - ok.length, cIndex: ModelEngine.concordanceIndex(t, e, lp),
                tdAuc: ModelEngine.timeDependentAUC(t, e, lp, H, cw), brier, brierNull, meanPred: sv.reduce((a, b) => a + (1 - b), 0) / sv.length, observed: 1 - ModelEngine.kmAt(km, H) };
        }
    }
    return { results, validation };
}

function survivalValidationHtml(v, fileName) {
    const r = SURV.result, f = x => (typeof x === 'number' ? x.toFixed(3) : 'N/A'), pc = x => `${(x * 100).toFixed(1)}%`;
    return `<div class="cal-panel"><div class="tune-title">✅ Validation of the Cox model on "${escapeHtml(fileName || 'the uploaded file')}"</div>
        <p class="tune-muted">${v.n} patients with a usable follow-up time and event status (${v.events} events)${v.excluded ? `; ${v.excluded} row(s) excluded` : ''}. Horizon ${escapeHtml(String(r.horizon))}.</p>
        <div class="cal-stats"><div><strong>Harrell's C-index</strong>: ${f(v.cIndex)}</div><div><strong>Time-dependent AUC at ${escapeHtml(String(r.horizon))}</strong>: ${f(v.tdAuc)}</div>
        <div><strong>Brier score</strong>: ${typeof v.brier === 'number' ? f(v.brier) : 'not estimable at this horizon'} (overall Kaplan-Meier risk for everyone: ${f(v.brierNull)})</div>
        <div><strong>Overall risk by ${escapeHtml(String(r.horizon))}</strong>: predicted ${pc(v.meanPred)} vs observed ${pc(v.observed)} (Kaplan-Meier)</div></div>
        ${v.n < 100 || v.events < 50 ? '<div class="tune-warn">⚠️ Fewer than 100 patients or 50 events: these estimates are imprecise.</div>' : ''}</div>`;
}

function predictOne() {
    const r = SURV.result;
    const input = {};
    document.querySelectorAll('.surv-in').forEach(el => { input[el.dataset.col] = el.value; });
    const x = DataEngine.transformSingleRow(input, r.prep, []);
    const xd = designOf(x, r.cols); // carries each multiply imputed version
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
        design: { validation: r.validation === 'bootstrap' ? `Option C: bootstrap internal validation, ${r.optimism.B} resamples (Harrell's optimism correction)` : r.validation === 'separate' ? 'Option B: fitted on Dataset (i), validated on a separate cohort, Dataset (ii)' : `Option A: ${Math.round(r.ratio * 100)}/${Math.round((1 - r.ratio) * 100)} ${r.splitText}`,
            missing_data: r.miInfo ? `MICE, ${r.miInfo.m} imputations, Rubin's rules` : r.completeCase ? 'complete cases' : 'single imputation (Step 3 methods)', seed: r.seed, train_patients: r.nTrain, train_events: r.eventsTrain, test_patients: r.nTest, test_events: r.eventsTest, excluded_rows: r.excluded, ties: 'Efron', penalty: r.penalised ? 0.1 : 0 },
        hazard_ratios: summary.map((s, i) => { const h = hrOnScale(s, r.cols[i]); return { term: s.name, scale: r.cols[i].note, coef: h.coef, se: h.se, hr: h.hr, hr_95ci: [h.lo, h.hi], p: s.p }; }),
        optimism_corrected: r.optimism ? { resamples: r.optimism.B, usable_resamples: r.optimism.used, c_index_optimism: r.optimism.cOptimism, c_index_corrected: r.optimism.cCorrected, time_dependent_auc_optimism: r.optimism.tdOptimism, time_dependent_auc_corrected: r.optimism.tdCorrected, calibration_slope_corrected: r.optimism.slope, calibration_slope_middle95_of_resamples: r.optimism.slopeCI, quick_approximation_in_resamples: r.optimism.quickPlain } : null,
        [r.validation === 'bootstrap' ? 'apparent_performance_all_patients' : r.validation === 'separate' ? 'validation_cohort_performance' : 'test_performance']: { c_index: r.cIndex, c_index_95ci_bootstrap: r.validation === 'bootstrap' ? null : r.cCI, horizon: r.horizon, time_dependent_auc: r.tdAuc, time_dependent_auc_95ci_bootstrap: r.validation === 'bootstrap' ? null : r.tdCI, brier_score_ipcw: r.brier, brier_score_null_km: r.brierNull, scaled_brier: r.scaledBrier, mean_predicted_risk: r.meanPred, observed_risk_kaplan_meier: r.obsAll, observed_risk_95ci: r.obsAllCI, calibration_groups: r.calib }
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Survival_Analysis_Report_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

export { SURV, renderSurvivalSinglePrediction, runSurvivalAnalysis, survivalBatch, survivalValidationHtml };
