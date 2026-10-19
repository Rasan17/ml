// subgroups.js: performance of a trained model within subgroups of the Testing set (e.g. by sex, age band
// or site), as TRIPOD+AI asks for an assessment of fairness: discrimination, calibration-in-the-large and
// error rates at the decision threshold, group by group.
/* global DataEngine, ModelEngine */
import { escapeHtml } from './core.js';

const subgroupState = {};
const SUBGROUP_MAX_LEVELS = 8;           // larger categorical columns: the 7 commonest categories and "Other"
const SUBGROUP_MIN_N = 30;           // smaller groups are shown but flagged as too small to judge
const SUBGROUP_MIN_EVENTS = 10;

// Columns worth offering, most likely fairness variables first
function subgroupColumns(rows, targetCol) {
    const cols = Object.keys(rows[0] || {}).filter(c => c !== targetCol && !/ \(was missing\)$/.test(c));
    const levels = c => new Set(rows.map(r => String(r[c] ?? '').trim()).filter(v => v !== '')).size;
    const usable = cols.filter(c => levels(c) >= 2 && !(levels(c) === rows.length && !isNumericSubgroupColumn(rows, c))); // not a patient ID
    const score = c => (/\b(sex|gender)\b/i.test(c) ? 0 : /age/i.test(c) ? 1 : /(site|centre|center|hospital|ethnic|race|region|country)/i.test(c) ? 2 : 3);
    return usable.sort((a, b) => score(a) - score(b));
}

function isNumericSubgroupColumn(rows, c) {
    const vals = rows.map(r => String(r[c] ?? '').trim()).filter(v => v !== '');
    return vals.length > 0 && vals.every(v => !isNaN(Number(v))) && new Set(vals).size > 6;
}

// Group label for every row: categories as they are, numbers in thirds of the Testing set (tertiles)
function groupLabels(rows, col) {
    if (isNumericSubgroupColumn(rows, col)) {
        // Blank cells are "(missing)", never 0 (Number('') would be 0 and fall in the lowest third)
        const nums = rows.map(r => { const t = String(r[col] ?? '').trim(); const v = Number(t); return t === '' || DataEngine.isMissing(t) || !isFinite(v) ? null : v; });
        const sorted = nums.filter(v => v !== null).sort((a, b) => a - b);
        const q = p => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
        const c1 = q(1 / 3), c2 = q(2 / 3);
        const f = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Number(v.toPrecision(3)).toString());
        return nums.map(v => (v === null ? '(missing)' : v <= c1 ? `≤ ${f(c1)}` : v <= c2 ? `${f(c1)} to ${f(c2)}` : `> ${f(c2)}`));
    }
    const raw = rows.map(r => { const v = String(r[col] ?? '').trim(); return v === '' ? '(missing)' : v; });
    const counts = {};
    raw.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    const keep = new Set(Object.keys(counts).sort((a, b) => counts[b] - counts[a]).slice(0, SUBGROUP_MAX_LEVELS - 1));
    return Object.keys(counts).length > SUBGROUP_MAX_LEVELS ? raw.map(v => (keep.has(v) ? v : 'Other')) : raw;
}

function subgroupAuc(y, p) {
    if (!y.some(v => v === 1) || !y.some(v => v === 0)) return null;
    try { return ModelEngine._calculateBinaryAUC(y, p); } catch (e) { return null; }
}

// Metrics for one set of patients
function groupMetrics(y, out, isClassification, binary, threshold) {
    const n = y.length;
    if (!isClassification) {
        const pred = out.map(o => Number(o.pred));
        const my = y.reduce((a, v) => a + v, 0) / n;
        const sse = y.reduce((a, v, i) => a + (v - pred[i]) ** 2, 0);
        const sst = y.reduce((a, v) => a + (v - my) ** 2, 0);
        return { n, meanObs: my, meanPred: pred.reduce((a, v) => a + v, 0) / n, rmse: Math.sqrt(sse / n), r2: sst > 0 ? 1 - sse / sst : null };
    }
    if (!binary) {
        const correct = out.filter((o, i) => o.pred === y[i]).length;
        return { n, accuracy: correct / n };
    }
    const p = out.map(o => (o.probabilities ? Number(o.probabilities[1]) || 0 : (o.pred === 1 ? 1 : 0)));
    const yb = y.map(v => (v === 1 ? 1 : 0));
    const events = yb.reduce((a, v) => a + v, 0);
    let tp = 0, fp = 0, tn = 0, fn = 0;
    p.forEach((v, i) => { const pos = v >= threshold; if (pos && yb[i]) tp++; else if (pos) fp++; else if (yb[i]) fn++; else tn++; });
    const meanPred = p.reduce((a, v) => a + v, 0) / n;
    return {
        n, events, observed: events / n, meanPred, oe: meanPred > 0 ? (events / n) / meanPred : null,
        auc: subgroupAuc(yb, p), brier: p.reduce((a, v, i) => a + (v - yb[i]) ** 2, 0) / n,
        sens: tp + fn ? tp / (tp + fn) : null, spec: tn + fp ? tn / (tn + fp) : null,
        netBenefit: tp / n - (fp / n) * (threshold / (1 - threshold))
    };
}

/**
 * items: [{ name, model, prep, threshold }]; rows: the Testing-set rows the models were scored on
 * (any preprocessing state the prep accepts); labelRows: the same patients' original values for grouping
 * (defaults to rows).
 */
function renderSubgroupPanel(containerId, items, rows, isClassification, labelRows = null) {
    const box = document.getElementById(containerId);
    if (!box) return;
    const usable = (items || []).filter(it => it.model && it.prep && typeof it.model.predictSingle === 'function');
    if (!usable.length || !rows || !rows.length) { box.innerHTML = ''; return; }
    if (rows.length < 20) {
        box.innerHTML = `<div class="cal-panel"><div class="tune-title">Does the model work equally well in subgroups?</div><p class="tune-muted">Only ${rows.length} Testing patients: too few to compare subgroups. Use a larger cohort or an external validation cohort (Option B).</p></div>`;
        return;
    }
    const targetCol = usable[0].prep.targetCol;
    const cols = subgroupColumns(labelRows && labelRows.length === rows.length ? labelRows : rows, targetCol);
    if (!cols.length) { box.innerHTML = ''; return; }
    subgroupState[containerId] = { items: usable, rows, labelRows: labelRows && labelRows.length === rows.length ? labelRows : rows, isClassification };
    box.innerHTML = `
        <div class="cal-panel">
            <div class="cal-head">
                <div class="tune-title">Does the model work equally well in subgroups? (Testing set)</div>
                <span>
                    <select id="${containerId}-model" class="control-select" aria-label="Model for the subgroup analysis" style="width: auto; display: inline-block;">${usable.map((it, i) => `<option value="${i}"${i === 0 ? ' selected' : ''}>${escapeHtml(it.name)}</option>`).join('')}</select>
                    <select id="${containerId}-col" class="control-select" aria-label="Column that defines the subgroups" style="width: auto; display: inline-block;">${cols.map((c, i) => `<option value="${escapeHtml(c)}"${i === 0 ? ' selected' : ''}>${escapeHtml(c)}</option>`).join('')}</select>
                    <button type="button" id="${containerId}-btn" class="btn btn-outline btn-sm">Compare subgroups</button>
                </span>
            </div>
            <p class="tune-muted">A model can perform well overall and still serve some patients badly. Choose a column (for example sex, age or site; numbers are split into thirds) to see discrimination, calibration and errors in each group. The column does not need to be a predictor.</p>
            <div id="${containerId}-out"></div>
        </div>`;
    document.getElementById(`${containerId}-btn`).addEventListener('click', () => runSubgroupAnalysis(containerId));
}

function runSubgroupAnalysis(containerId) {
    const st = subgroupState[containerId];
    const out = document.getElementById(`${containerId}-out`);
    if (!st || !out) return;
    try {
        const it = st.items[Number(document.getElementById(`${containerId}-model`).value) || 0];
        const col = document.getElementById(`${containerId}-col`).value;
        const binary = st.isClassification && (it.prep.targetClasses || []).length === 2;
        const threshold = typeof it.threshold === 'number' && it.threshold > 0 && it.threshold < 1 ? it.threshold : 0.5;
        const test = DataEngine.prepareTestingDataset(st.rows, it.prep);
        const pos = new Map(st.rows.map((r, i) => [r, i]));
        const kept = (test.validRows || []).map(r => pos.get(r));
        if (!test.X.length || kept.some(i => i === undefined)) { out.innerHTML = '<p class="tune-muted">The Testing set could not be matched to the model.</p>'; return; }
        const labels = groupLabels(kept.map(i => st.labelRows[i]), col);
        const outputs = typeof it.model.predictOutputs === 'function' ? it.model.predictOutputs(test.X) : test.X.map(x => it.model.predictSingle(x));
        const groups = {};
        labels.forEach((g, i) => { (groups[g] = groups[g] || []).push(i); });
        // Thirds in order (lowest first); categories by size; missing values last
        const numeric = isNumericSubgroupColumn(kept.map(i => st.labelRows[i]), col);
        const rank = g => (g === '(missing)' ? 3 : !numeric ? 1 : g.startsWith('≤') ? 0 : g.startsWith('>') ? 2 : 1);
        const order = Object.keys(groups).sort((a, b) => rank(a) - rank(b) || groups[b].length - groups[a].length);
        const all = groupMetrics(test.y, outputs, st.isClassification, binary, threshold);
        const res = order.map(g => ({ g, m: groupMetrics(groups[g].map(i => test.y[i]), groups[g].map(i => outputs[i]), st.isClassification, binary, threshold) }));
        const pct = v => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—');
        const num = (v, d = 3) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');
        const small = m => m.n < SUBGROUP_MIN_N || (binary && (m.events < SUBGROUP_MIN_EVENTS || m.n - m.events < SUBGROUP_MIN_EVENTS));
        const flag = m => {
            if (small(m) || !binary) return false;
            return (typeof m.auc === 'number' && typeof all.auc === 'number' && Math.abs(m.auc - all.auc) > 0.1) || (typeof m.oe === 'number' && (m.oe < 0.8 || m.oe > 1.25));
        };
        let head, row;
        if (binary) {
            head = '<th>Group</th><th>Patients</th><th>Events</th><th>Observed rate</th><th>Mean predicted</th><th title="Observed ÷ expected: 1 = calibrated on average; above 1 = risk underestimated">O:E</th><th>AUC</th><th>Brier ↓</th><th>Sensitivity</th><th>Specificity</th><th title="Net benefit at the decision threshold">Net benefit</th>';
            row = (g, m) => `<td>${escapeHtml(g)}</td><td>${m.n}</td><td>${m.events}</td><td>${pct(m.observed)}</td><td>${pct(m.meanPred)}</td><td>${num(m.oe, 2)}</td><td>${num(m.auc)}</td><td>${num(m.brier)}</td><td>${pct(m.sens)}</td><td>${pct(m.spec)}</td><td>${num(m.netBenefit)}</td>`;
        } else if (st.isClassification) {
            head = '<th>Group</th><th>Patients</th><th>Accuracy</th>';
            row = (g, m) => `<td>${escapeHtml(g)}</td><td>${m.n}</td><td>${pct(m.accuracy)}</td>`;
        } else {
            head = '<th>Group</th><th>Patients</th><th>Mean observed</th><th>Mean predicted</th><th>RMSE ↓</th><th>R²</th>';
            row = (g, m) => `<td>${escapeHtml(g)}</td><td>${m.n}</td><td>${num(m.meanObs, 2)}</td><td>${num(m.meanPred, 2)}</td><td>${num(m.rmse, 2)}</td><td>${num(m.r2)}</td>`;
        }
        const anyFlag = res.some(r => flag(r.m)), anySmall = res.some(r => small(r.m));
        out.innerHTML = `
            ${it.inSample ? '<div class="tune-warn">⚠︎ This model was retrained on all patients, including these Testing patients, so its figures here are optimistic. Compare the groups with each other rather than reading the values at face value; for an honest check, train without retraining (Option A) or use an external cohort (Option B).</div>' : ''}
            <div class="table-container" tabindex="0" role="region" aria-label="Subgroup results table">
                <table class="data-table thr-table">
                <thead><tr>${head}</tr></thead>
                <tbody>
                    <tr style="font-weight: 600;">${row('All Testing patients', all)}</tr>
                    ${res.map(r => `<tr${flag(r.m) ? ' class="subgroup-flag"' : small(r.m) ? ' class="subgroup-small"' : ''}>${row(r.g, r.m)}</tr>`).join('')}
                </tbody>
            </table></div>
            <p class="tune-muted">${binary ? `Sensitivity, specificity and net benefit use the decision threshold of ${(threshold * 100).toFixed(0)}%. ` : ''}${anyFlag ? '⚠︎ Highlighted groups differ noticeably from the whole Testing set (AUC by more than 0.10, or O:E outside 0.8–1.25). ' : ''}${anySmall ? `Greyed groups have fewer than ${SUBGROUP_MIN_N} patients${binary ? ` or fewer than ${SUBGROUP_MIN_EVENTS} patients with or without the outcome` : ''}: their figures are too uncertain to judge. ` : ''}Differences between subgroups can arise by chance, especially in small Testing sets; treat them as a prompt to look further (for example in an external cohort), not as proof of unfairness.</p>`;
    } catch (e) {
        out.innerHTML = `<p class="tune-muted">Subgroup analysis failed: ${escapeHtml(e.message || String(e))}</p>`;
    }
}

export { renderSubgroupPanel, runSubgroupAnalysis, groupLabels, groupMetrics };
