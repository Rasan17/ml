// survival.js: Step 4 for a time-to-event outcome (chosen in Step 2): a Cox proportional-hazards model (Efron ties).
// Imputation (single or MICE) and encoding are fitted on the training patients only; validated on a held-out
// testing set (Option A), a separate cohort (Option B) or by bootstrap optimism correction (Option C). Hazard
// ratios; Harrell's C-index, time-dependent AUC and Brier score (censoring-weighted); calibration at the horizon
// against Kaplan-Meier observed risk; Kaplan-Meier curves by predicted-risk group; single-patient prediction.
// Competing risks (an optional competing event, e.g. death before revision): cause-specific Cox models for the
// event and the competing event, combined into the cumulative incidence of the event; evaluated with Wolbers'
// C-index, the competing-risk time-dependent AUC and Brier score, and Aalen-Johansen observed incidence.
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

// The fitted model, used the same way with or without a competing event. score(x): ranks patients by risk (the
// linear predictor; with a competing event, the cumulative incidence by the horizon); riskAt(x, t): predicted
// probability of the event by t; competingAt(x, t): of the competing event (competing risks only). Multiply imputed
// patients carry one encoded version per imputation (x.__mi); predictions are averaged over the imputations.
function survivalModel(cox1s, cox2s, horizon) {
    const version = (x, l) => (x.__mi ? x.__mi[Math.min(l, x.__mi.length - 1)] : x);
    const avg = f => cox1s.reduce((a, m, l) => a + f(m, l), 0) / cox1s.length;
    const cox = cox1s.length > 1 ? pooledCox(cox1s) : cox1s[0];
    if (!cox2s) {
        return { competing: false, cox, cox2: null, score: x => cox.linearPredictor(x), riskAt: (x, t) => 1 - cox.survivalAt(x, t), survivalAt: (x, t) => cox.survivalAt(x, t) };
    }
    const cif = (x, t, k) => avg((m, l) => (k === 1 ? ModelEngine.competingCIF(m, cox2s[l], version(x, l), version(x, l), t) : ModelEngine.competingCIF(cox2s[l], m, version(x, l), version(x, l), t)));
    return { competing: true, cox, cox2: cox2s.length > 1 ? pooledCox(cox2s) : cox2s[0], score: x => cif(x, horizon, 1), riskAt: (x, t) => cif(x, t, 1), competingAt: (x, t) => cif(x, t, 2) };
}

// Fits the whole process on the given patients: missing-value filling (single imputation with the Step 3 methods,
// or MICE with one model per completed dataset), encoding, and the Cox model(s) (penalised if one does not
// converge). statusOf(row): 0 censored, 1 event, 2 competing event (with competing risks a second, cause-specific
// Cox model is fitted for the competing event). Returns the model and a function that encodes other patients.
async function fitSurvivalProcess(rows, cfg, tc, statusOf, miM, seed, quickPlain, competing, horizon) {
    let prep, cols, penalised = false, miInfo = null, Xtr, tTr, sTr, sets;
    const fitCox = (X, t, e) => { let m = ModelEngine.createCox().fit(X, t, e); if (!m.converged) { m = ModelEngine.createCox({ penalty: 0.1 }).fit(X, t, e); penalised = true; } return m; };
    const pick = x => cols.map(c => x[c.idx]);
    if (miM && !quickPlain) {
        // MICE on these patients only (the time and event are never used); one model per completed dataset
        const mi = await prepareMultipleImputation(rows, cfg, miM, seed);
        prep = mi.prep;
        cols = coxDesign(prep);
        tTr = rows.map(r => Number(r[tc])); sTr = rows.map(statusOf);
        sets = mi.sets.map(st => st.X.map(pick));
        Xtr = sets[0];
        miInfo = { m: miM, imputedColumns: mi.imputedColumns, trace: prep.multipleImputation && prep.multipleImputation.trace, traceScale: prep.multipleImputation && prep.multipleImputation.traceScale };
    } else {
        // quickPlain: the plain fill (median / most frequent) used inside resamples when "quick approximation" is on
        const fillCfg = quickPlain ? plainFillConfigs(cfg) : cfg;
        const imp = DataEngine.imputeDataset(rows, fillCfg, AppState.columnAnalysis);
        prep = DataEngine.prepareFeaturesAndTarget(imp.imputedRows, fillCfg);
        prep.imputation = { configs: JSON.parse(JSON.stringify(fillCfg)), plan: imp.plan };
        cols = coxDesign(prep);
        Xtr = prep.X.map(pick);
        sets = [Xtr];
        tTr = imp.imputedRows.map(r => Number(r[tc])); sTr = imp.imputedRows.map(statusOf);
    }
    const nEv = sTr.filter(v => v === 1).length, nComp = sTr.filter(v => v === 2).length;
    if (nEv < 5) return { tooFewEvents: nEv };
    if (competing && nComp < 5) return { tooFewCompeting: nComp };
    const e1 = sTr.map(v => (v === 1 ? 1 : 0)), e2 = sTr.map(v => (v === 2 ? 1 : 0));
    const model = survivalModel(sets.map(X => fitCox(X, tTr, e1)), competing ? sets.map(X => fitCox(X, tTr, e2)) : null, horizon);
    // Encodes other patients exactly as the training patients (blank values filled by the same fitted imputation)
    const encode = other => {
        const te = miInfo ? DataEngine.prepareTestingDataset(other, prep)
            : DataEngine.prepareTestingDataset(DataEngine.imputeDataset(other, prep.imputation.configs, AppState.columnAnalysis, prep.imputation.plan).imputedRows, prep);
        const X = te.X.map(x => { const d = pick(x); if (x.__mi) Object.defineProperty(d, '__mi', { value: x.__mi.map(pick), enumerable: false }); return d; });
        return { X, rows: te.validRows };
    };
    return { prep, cols, model, cox: model.cox, penalised, miInfo, Xtr, tTr, sTr, eTr: e1, encode };
}

/**
 * Time-to-event analysis (Cox model) for Step 4. o: { tc, ec, ev, preds, horizon,
 * ratio, seed, status, boxId, miM, completeCase, validation, B, groupCol, quickPlain, externalRows }.
 * validation: 'split' (held-out testing set, Option A), 'separate' (fit on all patients, test on externalRows,
 * Option B) or 'bootstrap' (fit on all patients; Harrell's optimism correction over B resamples, Option C).
 */
async function runSurvivalAnalysis(o) {
    const { tc, ec, ev, horizon, ratio, seed, status, boxId, miM = 0, validation = 'split' } = o;
    const cv = o.competingValue !== undefined && o.competingValue !== null && o.competingValue !== '' && o.competingValue !== ev ? String(o.competingValue) : null;
    const competing = Boolean(cv);
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
    // 1 = the event, 2 = the competing event (competing risks only), 0 = censored (any other value)
    const statusOf = r => { const v = String(r[ec]).trim(); return v === ev ? 1 : competing && v === cv ? 2 : 0; };
    let trainRows = rows, testRows, externalExcluded = 0, splitInfo = null;
    if (validation === 'split') {
        // The split method chosen in Step 4 (random balanced by outcome status, by patient, or by date)
        const lab = rows.map(r => String(statusOf(r)));
        const split = o.splitFn ? o.splitFn(rows, lab, ratio, seed) : DataEngine.trainTestSplit(rows, lab, ratio, true, seed);
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
    const fit = await fitSurvivalProcess(trainRows, cfg, tc, statusOf, miM, seed, false, competing, horizon);
    if (fit.tooFewEvents !== undefined) throw new Error(`Only ${fit.tooFewEvents} events in the training patients; at least 5 are needed.`);
    if (fit.tooFewCompeting !== undefined) throw new Error(`Only ${fit.tooFewCompeting} competing events ("${cv}") in the training patients; at least 5 are needed to model them. Choose "None" for the competing event, or use more patients.`);
    const { prep, cols, model, cox, penalised, miInfo, Xtr, tTr, sTr, eTr } = fit;
    const enc = fit.encode(testRows);
    const Xte = enc.X;
    const tTe = enc.rows.map(r => Number(r[tc])), sTe = enc.rows.map(statusOf), eTe = sTe.map(v => (v === 1 ? 1 : 0));
    const droppedBlank = testRows.length - enc.rows.length; // removed by a "drop rows" missing-data choice
    const events = eTr.reduce((a, b) => a + b, 0);
    const scoreTe = Xte.map(x => model.score(x));
    const risk = Xte.map(x => model.riskAt(x, horizon)); // predicted probability of the event by the horizon
    // Measures with and without a competing event (each reduces to the other when nobody has the competing event)
    const cIndexOf = (t, st, sc) => (competing ? ModelEngine.concordanceIndexCR(t, st, sc) : ModelEngine.concordanceIndex(t, st.map(v => (v === 1 ? 1 : 0)), sc));
    const aucOf = (t, st, sc, ck) => (competing ? ModelEngine.timeDependentAUCcr(t, st, sc, horizon, ck) : ModelEngine.timeDependentAUC(t, st.map(v => (v === 1 ? 1 : 0)), sc, horizon, ck));
    const brierOf = (t, st, rk, ck) => (competing ? ModelEngine.ipcwBrierCR(t, st, rk, horizon, ck) : ModelEngine.ipcwBrier(t, st.map(v => (v === 1 ? 1 : 0)), rk.map(v => 1 - v), horizon, ck));
    const censOf = (t, st) => ModelEngine.censoringKM(t, st.map(v => (v > 0 ? 1 : 0))); // any event ends follow-up
    const cIndex = cIndexOf(tTe, sTe, scoreTe);
    const rng = DataEngine.createPRNG(seed);
    // Time-dependent measures at the horizon, weighted for censoring. The censoring distribution comes from the
    // training patients, except for a separate cohort (Option B), whose follow-up can differ: its own is used
    const censKM = validation === 'separate' ? censOf(tTe, sTe) : censOf(tTr, sTr);
    const tdAuc = aucOf(tTe, sTe, scoreTe, censKM);
    const brier = brierOf(tTe, sTe, risk, censKM);
    // The same risk for everyone: the training patients' observed risk by the horizon (Aalen-Johansen with competing risks)
    const nullRisk = competing ? ModelEngine.cifAt(ModelEngine.aalenJohansen(tTr, sTr), horizon) : 1 - ModelEngine.kmAt(ModelEngine.kaplanMeier(tTr, eTr), horizon);
    const brierNull = brierOf(tTe, sTe, tTe.map(() => nullRisk), censKM);
    const scaledBrier = brier !== null && brierNull > 0 ? 1 - brier / brierNull : null;
    const boots = [], bootsTd = [];
    // Patients with several records are resampled whole (the patient column of a by-patient split, or the
    // patient column chosen for bootstrap validation), so the interval reflects patients, not records
    const ciGroupCol = splitInfo && splitInfo.method === 'group' ? splitInfo.column : (o.groupCol || null);
    const ciGroups = ciGroupCol ? (() => { const m = new Map(); enc.rows.forEach((r, i) => { const k = String(r[ciGroupCol] ?? ''); if (!m.has(k)) m.set(k, []); m.get(k).push(i); }); return [...m.values()]; })() : null;
    for (let b = 0; b < 500; b++) {
        const idx = ciGroups ? Array.from({ length: ciGroups.length }, () => ciGroups[Math.floor(rng() * ciGroups.length)]).flat() : tTe.map(() => Math.floor(rng() * tTe.length));
        const bt = idx.map(i => tTe[i]), bs = idx.map(i => sTe[i]), bl = idx.map(i => scoreTe[i]);
        const c = cIndexOf(bt, bs, bl);
        if (c !== null) boots.push(c);
        const a = aucOf(bt, bs, bl, censKM);
        if (a !== null) bootsTd.push(a);
        if (b % 100 === 99) await ModelEngine.yieldToBrowser();
    }
    const pctCI = arr => { arr.sort((a, b) => a - b); return arr.length >= 50 ? [arr[Math.floor(0.025 * (arr.length - 1))], arr[Math.floor(0.975 * (arr.length - 1))]] : null; };
    const cCI = pctCI(boots);
    const tdCI = pctCI(bootsTd);
    // Observed risk by the horizon in a group of patients: Kaplan-Meier (with a Greenwood interval), or with
    // competing risks the Aalen-Johansen cumulative incidence (interval: percentile bootstrap within the group)
    const observedIn = g => {
        const gt = g.map(i => tTe[i]), gs = g.map(i => sTe[i]);
        if (!competing) {
            const st = ModelEngine.kmStepAt(ModelEngine.kaplanMeier(gt, gs.map(v => (v === 1 ? 1 : 0))), horizon);
            return { observed: 1 - st.survival, observedCI: st.lower !== null ? [1 - st.upper, 1 - st.lower] : null };
        }
        const obs = ModelEngine.cifAt(ModelEngine.aalenJohansen(gt, gs), horizon);
        const bs = [];
        for (let b = 0; b < 200; b++) { const idx = gt.map(() => Math.floor(rng() * gt.length)); bs.push(ModelEngine.cifAt(ModelEngine.aalenJohansen(idx.map(i => gt[i]), idx.map(i => gs[i])), horizon)); }
        return { observed: obs, observedCI: pctCI(bs.length >= 50 ? bs : []) };
    };
    // calibration at the horizon: groups of predicted risk vs observed risk
    const nG = tTe.length >= 100 ? 5 : tTe.length >= 30 ? 3 : 2;
    const order = risk.map((r, i) => i).sort((a, b) => risk[a] - risk[b]);
    const groups = Array.from({ length: nG }, (_, g) => order.slice(Math.round(g * order.length / nG), Math.round((g + 1) * order.length / nG)));
    const calib = groups.map(g => ({ predicted: g.reduce((a, i) => a + risk[i], 0) / g.length, ...observedIn(g), n: g.length }));
    const all = observedIn(tTe.map((_, i) => i));
    const meanPred = risk.reduce((a, b) => a + b, 0) / (risk.length || 1);
    const obsAll = all.observed, obsAllCI = all.observedCI;
    // With competing risks: the observed risk if the competing event were (wrongly) treated as censoring, and the
    // predicted risk of the competing event
    const naiveObs = competing ? 1 - ModelEngine.kmAt(ModelEngine.kaplanMeier(tTe, eTe), horizon) : null;
    const meanCompeting = competing ? Xte.reduce((a, x) => a + model.competingAt(x, horizon), 0) / (Xte.length || 1) : null;
    const obsCompeting = competing ? ModelEngine.cifAt(ModelEngine.aalenJohansen(tTe, sTe.map(v => (v === 2 ? 1 : v === 1 ? 2 : 0))), horizon) : null;
    // Curves by risk tertile: Kaplan-Meier event-free probability, or the cumulative incidence of the event
    const tert = [0, 1, 2].map(g => order.slice(Math.round(g * order.length / 3), Math.round((g + 1) * order.length / 3)));
    const curveGroups = tert.map(g => (competing ? ModelEngine.aalenJohansen(g.map(i => tTe[i]), g.map(i => sTe[i])) : ModelEngine.kaplanMeier(g.map(i => tTe[i]), g.map(i => eTe[i]))));
    const maxFollow = Math.max(...tTr, ...tTe);
    const lastEvent = Math.max(0, ...tTr.filter((_, i) => eTr[i] === 1));
    // Option C: Harrell's bootstrap optimism correction. Each resample repeats the whole process (imputation, encoding,
    // model fit); the model is scored on its resample and on the original patients, and the mean difference (optimism)
    // is subtracted from the apparent performance. The calibration slope is that of the resample models' linear
    // predictor on the original patients (ideal 1; below 1 means predictions too extreme, i.e. overfitting); it is not
    // defined for the cumulative incidence with competing risks.
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
            const f = await fitSurvivalProcess(bRows, cfg, tc, statusOf, miM, seed + b + 1, Boolean(o.quickPlain), competing, horizon);
            if (f.tooFewEvents !== undefined || f.tooFewCompeting !== undefined) { skipped++; continue; }
            const eb = f.encode(bRows), eo = f.encode(rows);
            const tB = eb.rows.map(r => Number(r[tc])), sB = eb.rows.map(statusOf), tO = eo.rows.map(r => Number(r[tc])), sO = eo.rows.map(statusOf);
            if (!competing) {
                const lpB = eb.X.map(x => f.model.score(x)), lpO = eo.X.map(x => f.model.score(x));
                const op = ModelEngine.coxResampleOptimism({ t: tB, e: sB.map(v => (v === 1 ? 1 : 0)), lp: lpB }, { t: tO, e: sO.map(v => (v === 1 ? 1 : 0)), lp: lpO }, horizon, censKM);
                if (op.c === null) { skipped++; continue; }
                dc.push(op.c);
                if (op.td !== null) dtd.push(op.td);
                if (op.slope !== null) slopes.push(op.slope);
            } else {
                const scB = eb.X.map(x => f.model.score(x)), scO = eo.X.map(x => f.model.score(x));
                const cB = cIndexOf(tB, sB, scB), cO = cIndexOf(tO, sO, scO);
                if (cB === null || cO === null) { skipped++; continue; }
                dc.push(cB - cO);
                const aB = aucOf(tB, sB, scB, censOf(tB, sB)), aO = aucOf(tO, sO, scO, censKM);
                if (aB !== null && aO !== null) dtd.push(aB - aO);
            }
        }
        const mean = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
        const oc = mean(dc), otd = mean(dtd), slope = mean(slopes);
        optimism = { B: idxs.length, used: dc.length, skipped, grouped: Boolean(keys), groupCol: o.groupCol || null, quickPlain: Boolean(o.quickPlain && miM),
            cOptimism: oc, cCorrected: oc !== null && cIndex !== null ? cIndex - oc : null,
            tdOptimism: otd, tdCorrected: otd !== null && typeof tdAuc === 'number' ? tdAuc - otd : null,
            slope, slopeCI: slopes.length >= 20 ? (() => { const a = slopes.slice().sort((x, y) => x - y); return [a[Math.floor(0.025 * (a.length - 1))], a[Math.floor(0.975 * (a.length - 1))]]; })() : null,
            seconds: Math.round((Date.now() - t0) / 1000) };
    }
    SURV.result = { competing, cv, model, naiveObs, meanCompeting, obsCompeting, competingTrain: sTr.filter(v => v === 2).length, competingTest: sTe.filter(v => v === 2).length,
        ciGroupCol, droppedBlank, splitInfo, splitText: o.describeSplit && splitInfo ? o.describeSplit(splitInfo) : 'random split, balanced by outcome', censKM, validation, optimism, externalExcluded, boxId: boxId || 'survival-step4-results', miInfo, completeCase: Boolean(o.completeCase), lastEvent, tc, ec, ev, preds, horizon, ratio, seed, cols, prep, cox, penalised, cIndex, cCI, tdAuc, tdCI, brier, brierNull, scaledBrier, calib, obsAllCI, meanPred, obsAll, curveGroups, maxFollow,
        nTrain: Xtr.length, nTest: Xte.length, eventsTrain: events, eventsTest: eTe.reduce((a, b) => a + b, 0), excluded };
    renderSurvivalResults();
    if (status) status.textContent = '';
    return SURV.result;
}

function stepPoints(km, key = 'survival') {
    const pts = [];
    km.forEach((k, i) => { if (i > 0) pts.push({ x: k.time, y: km[i - 1][key] }); pts.push({ x: k.time, y: k[key] }); });
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
    const hrTable = coxModel => coxModel.summary(r.cols.map(c => c.name)).map((s, i) => {
        const c = r.cols[i];
        const { hr, lo, hi } = hrOnScale(s, c);
        return `<tr><td>${escapeHtml(c.name)}</td><td>${f2(hr)} (${f2(lo)}–${f2(hi)})</td><td>${escapeHtml(c.note)}</td><td>${s.p === null ? 'n/a' : s.p < 0.001 ? '< 0.001' : s.p.toFixed(3)}</td></tr>`;
    }).join('');
    const hrRows = hrTable(r.cox);
    const cr = r.competing, obsName = cr ? 'Aalen-Johansen' : 'Kaplan-Meier', cName = cr ? "Wolbers' C-index" : "Harrell's C-index";
    const pctTxt = v => `${(v * 100).toFixed(1)}%`;
    const calMax = Math.min(1, Math.max(0.1, ...r.calib.map(c => Math.max(c.predicted, c.observed))) * 1.1);
    const tertNames = ['Low predicted risk', 'Medium predicted risk', 'High predicted risk'];
    const tertCols = ['var(--accent-emerald)', 'var(--accent-amber)', 'var(--accent-rose)'];
    const epp = r.eventsTrain / Math.max(1, r.cols.length);
    const who = r.validation === 'bootstrap' ? 'all patients' : r.validation === 'separate' ? 'validation-cohort patients' : 'test patients';
    box.innerHTML = `
        <div class="cal-panel">
            <div class="tune-title">${cr ? `Competing-risk model (cause-specific Cox): ${escapeHtml(r.ec)} = "${escapeHtml(r.ev)}", with "${escapeHtml(r.cv)}" as a competing event, over ${escapeHtml(r.tc)}` : `Cox proportional-hazards model: ${escapeHtml(r.ec)} = "${escapeHtml(r.ev)}" over ${escapeHtml(r.tc)}`}</div>
            <p class="tune-muted">${r.validation === 'bootstrap' ? `Fitted on all ${r.nTrain} patients (${r.eventsTrain} events); validated by bootstrap internal validation (Option C, ${r.optimism.B} resamples)`
                : r.validation === 'separate' ? `Fitted on all ${r.nTrain} patients of Dataset (i) (${r.eventsTrain} events); validated on ${r.nTest} patients of the separate cohort, Dataset (ii) (${r.eventsTest} events)${r.externalExcluded ? `, after excluding ${r.externalExcluded} of its row(s) without a usable time or event` : ''}`
                : `${r.nTrain} training patients (${r.eventsTrain} events) and ${r.nTest} test patients (${r.eventsTest} events), ${Math.round(r.ratio * 100)}/${Math.round((1 - r.ratio) * 100)} ${escapeHtml(r.splitText)} (seed ${r.seed})`}${cr ? `; competing events ("${escapeHtml(r.cv)}"): ${r.competingTrain} in the patients used to fit${r.validation === 'bootstrap' ? '' : `, ${r.competingTest} in the evaluated patients`}` : ''}${r.excluded ? `; ${r.excluded} row(s) without a usable time or event were excluded` : ''}${r.droppedBlank ? `; ${r.droppedBlank} evaluated patient(s) were removed by the "drop rows" missing-data choice` : ''}.${r.penalised ? ' <strong>The unpenalised model did not converge, so a small ridge penalty (0.1) was applied.</strong>' : ''}${epp < 10 ? ` <strong>Caution:</strong> only ${epp.toFixed(1)} events per model parameter; the estimates are unstable.` : ''}
            Missing predictor values: ${r.miInfo ? `multiple imputation (MICE, ${r.miInfo.m} datasets, fitted on the training patients without the time or event); one Cox model per dataset, hazard ratios pooled by Rubin's rules and predictions averaged` : r.completeCase ? 'complete-case analysis (patients with a blank predictor excluded)' : 'single imputation with the Step 3 methods, fitted on the training patients'}.</p>
            <div class="cmp-guide-title" style="font-size: 13px;">${cr ? `Cause-specific hazard ratios for "${escapeHtml(r.ev)}"` : 'Hazard ratios'} (training patients)</div>
            <table class="data-table thr-table"><thead><tr><th>Predictor</th><th>Hazard ratio (95% CI)</th><th>Scale</th><th>p-value</th></tr></thead><tbody>${hrRows}</tbody></table>
            ${cr ? `<div class="cmp-guide-title" style="font-size: 13px;">Cause-specific hazard ratios for the competing event "${escapeHtml(r.cv)}"</div>
            <table class="data-table thr-table"><thead><tr><th>Predictor</th><th>Hazard ratio (95% CI)</th><th>Scale</th><th>p-value</th></tr></thead><tbody>${hrTable(r.model.cox2)}</tbody></table>
            <p class="tune-muted">Each table describes the rate of one kind of event among patients who have had neither yet. A predictor can lower the risk of "${escapeHtml(r.ev)}" over time simply by making "${escapeHtml(r.cv)}" happen sooner, so the predicted risks below combine both models (the cumulative incidence) rather than reading either table alone.</p>` : ''}
            <p class="tune-muted">A hazard ratio above 1 means the event tends to happen sooner (higher instantaneous risk); below 1, later. Each is adjusted for the other predictors; categorical predictors are compared with their most common category. The Cox model assumes the hazard ratios stay constant over follow-up (proportional hazards); check this clinically, e.g. whether the Kaplan-Meier curves below cross. Associations, not causation.</p>
        </div>
        <div class="cal-panel">
            ${r.optimism ? `<div class="tune-title">Bootstrap internal validation (optimism-corrected)</div>
            <table class="data-table thr-table"><thead><tr><th>Measure</th><th>Apparent (all patients)</th><th>Optimism</th><th>Corrected</th></tr></thead><tbody>
                <tr><td>${cName}</td><td>${r.cIndex !== null ? r.cIndex.toFixed(3) : 'N/A'}</td><td>${r.optimism.cOptimism !== null ? r.optimism.cOptimism.toFixed(3) : 'N/A'}</td><td><strong>${r.optimism.cCorrected !== null ? r.optimism.cCorrected.toFixed(3) : 'N/A'}</strong></td></tr>
                <tr><td>Time-dependent AUC at ${escapeHtml(String(r.horizon))}</td><td>${typeof r.tdAuc === 'number' ? r.tdAuc.toFixed(3) : 'N/A'}</td><td>${r.optimism.tdOptimism !== null ? r.optimism.tdOptimism.toFixed(3) : 'N/A'}</td><td><strong>${r.optimism.tdCorrected !== null ? r.optimism.tdCorrected.toFixed(3) : 'N/A'}</strong></td></tr>
                ${cr ? '<tr><td>Calibration slope</td><td colspan="3">Not defined for the cumulative incidence with competing risks; see the calibration plot below</td></tr>' : `<tr><td>Calibration slope (ideal 1)</td><td>— (reference: 1)</td><td>${r.optimism.slope !== null ? (1 - r.optimism.slope).toFixed(3) : 'N/A'}</td><td><strong>${r.optimism.slope !== null ? r.optimism.slope.toFixed(3) : 'N/A'}</strong>${r.optimism.slopeCI ? ` <span class="tune-sd">(middle 95% of resamples ${r.optimism.slopeCI[0].toFixed(2)}–${r.optimism.slopeCI[1].toFixed(2)})</span>` : ''}</td></tr>`}
            </tbody></table>
            <p class="tune-muted">Each of the ${r.optimism.used} usable resamples${r.optimism.skipped ? ` (${r.optimism.skipped} skipped: too few events or no comparable pairs)` : ''} repeated the whole process (${r.optimism.quickPlain ? 'with the quick approximation: a plain median / most-frequent fill instead of MICE inside the resamples' : r.miInfo ? 'MICE' : 'single imputation'}, encoding and the Cox fit${cr ? 's for both events' : ''}) on patients drawn with replacement${r.optimism.grouped ? ` (whole patients by "${escapeHtml(r.optimism.groupCol)}")` : ''}. The optimism is the average amount by which a model scores better on its own resample than on the original patients; it is subtracted from the apparent performance. A corrected calibration slope below 1 means the predictions are too extreme and would need shrinking (${r.optimism.seconds} s).${r.optimism.used < 50 ? ' <strong>Few usable resamples: the correction is imprecise.</strong>' : ''}</p>
            <div class="tune-title" style="margin-top: 12px;">Apparent performance on all patients (before correction)</div>
            <p class="tune-muted">The panels below score the final model on the same patients it was fitted to, so they are optimistic; use the corrected values above when reporting.</p>` : `<div class="tune-title">Performance on the ${r.validation === 'separate' ? 'validation cohort (Dataset ii)' : 'test patients'}</div>`}
            <div class="cal-stats"><div><strong>${cName}</strong>: ${r.cIndex !== null ? r.cIndex.toFixed(3) : 'N/A'}${r.cCI && r.validation !== 'bootstrap' ? ` <span class="tune-sd">(95% CI ${r.cCI[0].toFixed(3)}–${r.cCI[1].toFixed(3)}${r.ciGroupCol ? `, patients resampled whole by "${escapeHtml(r.ciGroupCol)}"` : ''})</span>` : ''} &mdash; the probability that, of two comparable patients, the one with the event first had the higher predicted risk (0.5 = chance, 1 = perfect)${cr ? `; a patient who had "${escapeHtml(r.cv)}" can no longer have the event, so counts as having a lower risk than anyone who had the event` : ''}.</div>
            <div><strong>Time-dependent AUC at ${escapeHtml(String(r.horizon))}</strong>: ${typeof r.tdAuc === 'number' ? r.tdAuc.toFixed(3) : 'N/A'}${r.tdCI && r.validation !== 'bootstrap' ? ` <span class="tune-sd">(95% CI ${r.tdCI[0].toFixed(3)}–${r.tdCI[1].toFixed(3)})</span>` : ''} &mdash; of a patient who had the event by ${escapeHtml(String(r.horizon))} and one ${cr ? `who did not (still free of any event, or "${escapeHtml(r.cv)}" first)` : 'still event-free after it'}, how often the first had the higher predicted risk.</div>
            <div><strong>Brier score at ${escapeHtml(String(r.horizon))}</strong>: ${typeof r.brier === 'number' ? r.brier.toFixed(3) : 'not estimable (too little follow-up to weight the patients at this horizon; choose an earlier horizon)'} (lower is better; giving everyone the overall ${obsName} risk scores ${typeof r.brierNull === 'number' ? r.brierNull.toFixed(3) : 'N/A'})${typeof r.scaledBrier === 'number' ? `; <strong>scaled Brier</strong> ${(r.scaledBrier * 100).toFixed(1)}% (share of that error the model removes; 0% = no better than the overall risk)` : ''}.</div>
            ${r.horizon > r.lastEvent ? `<div class="tune-warn">⚠︎ The horizon (${escapeHtml(String(r.horizon))}) is beyond the last event in the training patients (${escapeHtml(String(r.lastEvent))}): the model cannot learn risk after that time, so predicted risks at this horizon are underestimated. Choose a horizon within the follow-up of the training data.</div>` : ''}
            <div><strong>Overall risk by ${escapeHtml(String(r.horizon))}</strong>: predicted ${pctTxt(r.meanPred)} vs observed ${pctTxt(r.obsAll)}${r.obsAllCI ? ` <span class="tune-sd">(95% CI ${pctTxt(r.obsAllCI[0])}–${pctTxt(r.obsAllCI[1])})</span>` : ''} (${obsName}).</div>
            ${cr ? `<div><strong>Competing event "${escapeHtml(r.cv)}" by ${escapeHtml(String(r.horizon))}</strong>: predicted ${pctTxt(r.meanCompeting)} vs observed ${pctTxt(r.obsCompeting)} (Aalen-Johansen).</div>
            <div class="tune-warn">Why competing risks matter here: treating "${escapeHtml(r.cv)}" as censoring (an ordinary Kaplan-Meier analysis) would put the observed risk of "${escapeHtml(r.ev)}" by ${escapeHtml(String(r.horizon))} at ${pctTxt(r.naiveObs)} instead of ${pctTxt(r.obsAll)}, because it assumes that patients who had "${escapeHtml(r.cv)}" could still go on to have the event.</div>` : ''}
            <div class="tune-muted">The time-dependent AUC and Brier score weight each patient for censoring (inverse probability of censoring weights, with the censoring pattern estimated from the training patients), so patients lost to follow-up before ${escapeHtml(String(r.horizon))} do not bias them.</div></div>
            <div class="cal-grid" style="grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));">
                <div>${lineChartSvg({ series: [{ name: 'Risk group', colour: 'var(--accent-cyan)', noHover: true, points: r.calib.map(c => ({ x: c.predicted, y: c.observed })) }, { name: 'Risk group', colour: 'var(--accent-cyan)', scatter: true, points: r.calib.map(c => ({ x: c.predicted, y: c.observed })) }],
                    xLabel: `Predicted risk by ${r.horizon}`, yLabel: `Observed risk (${obsName})`, xMax: calMax, yMax: calMax, diagonal: true, aria: 'Survival calibration at the horizon' })}
                    <p class="tune-muted">Calibration at ${escapeHtml(String(r.horizon))}: ${who} are grouped by predicted risk; each point compares the group's average predicted risk with its ${obsName} observed risk. Points near the diagonal mean the predicted risks can be taken at face value.</p>
                    <table class="data-table thr-table"><thead><tr><th>Risk group</th><th>Patients</th><th>Mean predicted</th><th>Observed (95% CI)</th></tr></thead><tbody>${r.calib.map((c, i) => `<tr><td>${i + 1} of ${r.calib.length}</td><td>${c.n}</td><td>${pctTxt(c.predicted)}</td><td>${pctTxt(c.observed)}${c.observedCI ? ` <span class="tune-sd">(${pctTxt(c.observedCI[0])}–${pctTxt(c.observedCI[1])})</span>` : ''}</td></tr>`).join('')}</tbody></table>
                    <p class="tune-muted">Observed risks come with 95% confidence intervals (${cr ? 'percentile bootstrap within each group' : "Greenwood's formula on the log(−log) scale"}), so a group whose interval includes its predicted risk is consistent with good calibration.</p></div>
                <div>${lineChartSvg({ series: r.curveGroups.map((km, g) => ({ name: tertNames[g], colour: tertCols[g], points: stepPoints(km, cr ? 'cif' : 'survival') })), xLabel: `${r.tc}`, yLabel: cr ? `Cumulative incidence of "${r.ev}"` : 'Event-free (Kaplan-Meier)', xMax: r.maxFollow, yMax: cr ? Math.min(1, Math.max(0.1, ...r.curveGroups.flatMap(a => a.map(k => k.cif))) * 1.1) : 1, aria: cr ? 'Cumulative incidence curves by predicted risk group' : 'Kaplan-Meier curves by predicted risk group' })}
                    ${chartLegend(tertNames.map((n, g) => ({ colour: tertCols[g], label: n })))}
                    <p class="tune-muted">${who.charAt(0).toUpperCase() + who.slice(1)} split into thirds by predicted risk. Well-separated curves show the model distinguishes patients with early and late events${cr ? ' (Aalen-Johansen cumulative incidence, which accounts for the competing event)' : ''}.</p></div>
            </div>
            ${r.nTest < 100 || r.eventsTest < 50 ? `<div class="tune-warn">⚠︎ Few ${who} or events: these performance estimates are imprecise.</div>` : ''}
            <button type="button" class="btn btn-secondary btn-sm" id="surv-report-btn">Download Survival Analysis Report (.JSON)</button>
        </div>
        <p class="tune-muted">Predicted ${cr ? 'risks' : 'event-free probabilities'} for one patient: Step 5. For a file of patients (and validation on it if it contains the follow-up time and event): Step 6.</p>`;
    document.getElementById('surv-report-btn').addEventListener('click', downloadSurvivalReport);
}

// Step 5 for a time-to-event outcome: predicted event-free probability for one patient
function renderSurvivalSinglePrediction(boxId) {
    const box = document.getElementById(boxId), r = SURV.result;
    if (!box) return;
    if (!r) { box.innerHTML = '<div class="cal-panel"><p class="tune-muted">Fit the Cox model in Step 4 first.</p></div>'; return; }
    box.innerHTML = `
        <div class="cal-panel">
            <div class="tune-title">${r.competing ? `Predicted risk of "${escapeHtml(r.ev)}" and of the competing event "${escapeHtml(r.cv)}" for one patient` : 'Predicted event-free probability for one patient'}</div>
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
    const scores = [], risks = [];
    const results = rows.map(row => {
        const issues = [], ranges = [];
        const xd = designOf(DataEngine.transformSingleRow(row, r.prep, issues, ranges), r.cols);
        const sc = r.model.score(xd), rk = r.model.riskAt(xd, H);
        scores.push(sc); risks.push(rk);
        const pred = r.competing
            ? { [`Predicted_risk_by_${H}`]: `${(rk * 100).toFixed(1)}%`, [`Predicted_competing_risk_by_${H}`]: `${(r.model.competingAt(xd, H) * 100).toFixed(1)}%` }
            : { [`Predicted_event_free_at_${H}`]: `${((1 - rk) * 100).toFixed(1)}%`, [`Predicted_risk_by_${H}`]: `${(rk * 100).toFixed(1)}%`, Linear_Predictor: sc.toFixed(4) };
        return { ...row, ...pred,
            Imputed_Inputs: issues.map(i => `${i.column}:${i.kind}`).join('; '), Outside_Training_Range: ranges.map(o => `${o.column}=${o.value} (training ${o.min}-${o.max})`).join('; '),
            Model: `${r.competing ? 'Competing-risk cause-specific Cox' : 'Cox proportional-hazards'}${r.miInfo ? ` (pooled over ${r.miInfo.m} imputations)` : ''}` };
    });
    let validation = null;
    if (headers.includes(r.tc) && headers.includes(r.ec)) {
        const ok = rows.map((row, i) => i).filter(i => !DataEngine.isMissing(rows[i][r.tc]) && Number(rows[i][r.tc]) > 0 && !DataEngine.isMissing(rows[i][r.ec]));
        const st = ok.map(i => { const v = String(rows[i][r.ec]).trim(); return v === r.ev ? 1 : r.competing && v === r.cv ? 2 : 0; });
        const t = ok.map(i => Number(rows[i][r.tc])), e = st.map(v => (v === 1 ? 1 : 0));
        const sc = ok.map(i => scores[i]), rk = ok.map(i => risks[i]);
        if (ok.length >= 10 && e.some(v => v)) {
            const cw = ModelEngine.censoringKM(t, st.map(v => (v > 0 ? 1 : 0))); // this file's own censoring pattern
            const observed = r.competing ? ModelEngine.cifAt(ModelEngine.aalenJohansen(t, st), H) : 1 - ModelEngine.kmAt(ModelEngine.kaplanMeier(t, e), H);
            const brier = r.competing ? ModelEngine.ipcwBrierCR(t, st, rk, H, cw) : ModelEngine.ipcwBrier(t, e, rk.map(v => 1 - v), H, cw);
            const brierNull = r.competing ? ModelEngine.ipcwBrierCR(t, st, t.map(() => observed), H, cw) : ModelEngine.ipcwBrier(t, e, t.map(() => 1 - observed), H, cw);
            validation = { n: ok.length, events: e.reduce((a, b) => a + b, 0), competingEvents: st.filter(v => v === 2).length, excluded: rows.length - ok.length,
                cIndex: r.competing ? ModelEngine.concordanceIndexCR(t, st, sc) : ModelEngine.concordanceIndex(t, e, sc),
                tdAuc: r.competing ? ModelEngine.timeDependentAUCcr(t, st, sc, H, cw) : ModelEngine.timeDependentAUC(t, e, sc, H, cw),
                brier, brierNull, meanPred: rk.reduce((a, b) => a + b, 0) / rk.length, observed };
        }
    }
    return { results, validation };
}

function survivalValidationHtml(v, fileName) {
    const r = SURV.result, f = x => (typeof x === 'number' ? x.toFixed(3) : 'N/A'), pc = x => `${(x * 100).toFixed(1)}%`;
    const obsName = r.competing ? 'Aalen-Johansen' : 'Kaplan-Meier';
    return `<div class="cal-panel"><div class="tune-title">Validation of the ${r.competing ? 'competing-risk model' : 'Cox model'} on "${escapeHtml(fileName || 'the uploaded file')}"</div>
        <p class="tune-muted">${v.n} patients with a usable follow-up time and event status (${v.events} events${r.competing ? `, ${v.competingEvents} competing events` : ''})${v.excluded ? `; ${v.excluded} row(s) excluded` : ''}. Horizon ${escapeHtml(String(r.horizon))}.</p>
        <div class="cal-stats"><div><strong>${r.competing ? "Wolbers' C-index" : "Harrell's C-index"}</strong>: ${f(v.cIndex)}</div><div><strong>Time-dependent AUC at ${escapeHtml(String(r.horizon))}</strong>: ${f(v.tdAuc)}</div>
        <div><strong>Brier score</strong>: ${typeof v.brier === 'number' ? f(v.brier) : 'not estimable at this horizon'} (overall ${obsName} risk for everyone: ${f(v.brierNull)})</div>
        <div><strong>Overall risk by ${escapeHtml(String(r.horizon))}</strong>: predicted ${pc(v.meanPred)} vs observed ${pc(v.observed)} (${obsName})</div></div>
        ${v.n < 100 || v.events < 50 ? '<div class="tune-warn">⚠︎ Fewer than 100 patients or 50 events: these estimates are imprecise.</div>' : ''}</div>`;
}

function predictOne() {
    const r = SURV.result;
    const input = {};
    document.querySelectorAll('.surv-in').forEach(el => { input[el.dataset.col] = el.value; });
    const x = DataEngine.transformSingleRow(input, r.prep, []);
    const xd = designOf(x, r.cols); // carries each multiply imputed version
    const times = [0.25, 0.5, 0.75, 1].map(f => Number((r.horizon * f * 2 / 2).toPrecision(3)));
    times[3] = r.horizon;
    const pc = v => `${(v * 100).toFixed(1)}%`;
    if (r.competing) {
        const rows = times.map(t => { const a = r.model.riskAt(xd, t), b = r.model.competingAt(xd, t); return `<tr><td>${t}</td><td>${pc(a)}</td><td>${pc(b)}</td><td>${pc(Math.max(0, 1 - a - b))}</td></tr>`; }).join('');
        document.getElementById('surv-predict-out').innerHTML = `<table class="data-table thr-table" style="margin-top: 8px;"><thead><tr><th>${escapeHtml(r.tc)}</th><th>Risk of "${escapeHtml(r.ev)}" by then</th><th>Risk of "${escapeHtml(r.cv)}" first</th><th>Neither yet</th></tr></thead><tbody>${rows}</tbody></table>`;
        return;
    }
    const rows = times.map(t => `<tr><td>${t}</td><td>${pc(r.cox.survivalAt(xd, t))}</td></tr>`).join('');
    document.getElementById('surv-predict-out').innerHTML = `<table class="data-table thr-table" style="margin-top: 8px;"><thead><tr><th>${escapeHtml(r.tc)}</th><th>Predicted probability of no event so far</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function downloadSurvivalReport() {
    const r = SURV.result;
    if (!r) return;
    const summary = r.cox.summary(r.cols.map(c => c.name));
    const report = {
        title: r.competing ? 'Competing-risk (cause-specific Cox) Analysis Report' : 'Survival (Cox proportional-hazards) Analysis Report', software: { name: 'Universal Medical ML/NeuralNet Predictor', version: APP_VERSION },
        generated_at: new Date().toISOString(), dataset: AppState.fileName, time_column: r.tc, event_column: r.ec, event_value: r.ev, competing_event_value: r.competing ? r.cv : null, predictors: r.preds,
        design: { validation: r.validation === 'bootstrap' ? `Option C: bootstrap internal validation, ${r.optimism.B} resamples (Harrell's optimism correction)` : r.validation === 'separate' ? 'Option B: fitted on Dataset (i), validated on a separate cohort, Dataset (ii)' : `Option A: ${Math.round(r.ratio * 100)}/${Math.round((1 - r.ratio) * 100)} ${r.splitText}`,
            missing_data: r.miInfo ? `MICE, ${r.miInfo.m} imputations, Rubin's rules` : r.completeCase ? 'complete cases' : 'single imputation (Step 3 methods)', seed: r.seed, train_patients: r.nTrain, train_events: r.eventsTrain, test_patients: r.nTest, test_events: r.eventsTest, excluded_rows: r.excluded, ties: 'Efron', penalty: r.penalised ? 0.1 : 0 },
        hazard_ratios: summary.map((s, i) => { const h = hrOnScale(s, r.cols[i]); return { term: s.name, scale: r.cols[i].note, coef: h.coef, se: h.se, hr: h.hr, hr_95ci: [h.lo, h.hi], p: s.p }; }),
        competing_event_cause_specific_hazard_ratios: r.competing ? r.model.cox2.summary(r.cols.map(c => c.name)).map((s, i) => { const h = hrOnScale(s, r.cols[i]); return { term: s.name, scale: r.cols[i].note, coef: h.coef, se: h.se, hr: h.hr, hr_95ci: [h.lo, h.hi], p: s.p }; }) : null,
        competing_risks: r.competing ? { method: 'cause-specific Cox models combined into the cumulative incidence (Aalen-Johansen observed risk; Wolbers C-index; competing-risk time-dependent AUC and Brier score)', observed_risk_aalen_johansen: r.obsAll, observed_risk_if_competing_event_treated_as_censoring: r.naiveObs, mean_predicted_competing_risk: r.meanCompeting, observed_competing_risk: r.obsCompeting } : null,
        optimism_corrected: r.optimism ? { resamples: r.optimism.B, usable_resamples: r.optimism.used, c_index_optimism: r.optimism.cOptimism, c_index_corrected: r.optimism.cCorrected, time_dependent_auc_optimism: r.optimism.tdOptimism, time_dependent_auc_corrected: r.optimism.tdCorrected, calibration_slope_corrected: r.optimism.slope, calibration_slope_middle95_of_resamples: r.optimism.slopeCI, quick_approximation_in_resamples: r.optimism.quickPlain } : null,
        [r.validation === 'bootstrap' ? 'apparent_performance_all_patients' : r.validation === 'separate' ? 'validation_cohort_performance' : 'test_performance']: { c_index: r.cIndex, c_index_95ci_bootstrap: r.validation === 'bootstrap' ? null : r.cCI, horizon: r.horizon, time_dependent_auc: r.tdAuc, time_dependent_auc_95ci_bootstrap: r.validation === 'bootstrap' ? null : r.tdCI, brier_score_ipcw: r.brier, brier_score_null_km: r.brierNull, scaled_brier: r.scaledBrier, mean_predicted_risk: r.meanPred, [r.competing ? 'observed_risk_aalen_johansen' : 'observed_risk_kaplan_meier']: r.obsAll, observed_risk_95ci: r.obsAllCI, calibration_groups: r.calib }
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
