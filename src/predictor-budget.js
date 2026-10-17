// predictor-budget.js: "How many predictors?" guidance where predictors are chosen (Step 2 of the standard
// workflow, Step 4 of the 7-step pipeline). From the number of training patients and the size of the smaller
// outcome group it gives the largest number of predictor parameters supported by the Riley et al. minimum
// sample-size criteria (pmsampsize), by 10 and 20 events per parameter, and the much larger samples flexible
// machine-learning models need, and compares them with the current selection.
/* global DataEngine, ModelEngine */
import { escapeHtml } from './core.js';

// Parameters each predictor adds to a regression model: 1 for a number or an ordered scale, (categories − 1)
// for an unordered categorical predictor
function predictorParameters(rows, featureCols, columnConfigs) {
    const perCol = featureCols.map(col => {
        const cfg = columnConfigs[col] || {};
        if (cfg.type !== 'nominal') return { col, df: 1, levels: null };
        const set = new Set();
        for (const r of rows) if (!DataEngine.isMissing(r[col])) set.add(String(r[col]).trim());
        return { col, df: Math.max(1, set.size - 1), levels: set.size };
    });
    return { total: perCol.reduce((a, c) => a + c.df, 0), perCol };
}

// Largest number of parameters for which the Riley criteria are met with nTrain patients (0 if none)
function rileyMaxParameters(type, nTrain, prevalence) {
    let best = 0;
    for (let p = 1; p <= 300; p++) {
        const r = ModelEngine.rileySampleSize({ type, p, prevalence });
        if (!r || r.n > nTrain) break;
        best = p;
    }
    return best;
}

/**
 * Renders the guidance into containerId. rows: patient records (raw or imputed) with the outcome; featureCols:
 * the predictors currently selected; trainFraction: share of rows that will train the model; trainLabel
 * describes it; candidateCols (optional): every predictor considered so far, when some were dropped by
 * looking at the data (counts towards overfitting even if not kept).
 */
function renderPredictorBudget(containerId, { rows, featureCols, targetCol, columnConfigs, trainFraction = 1, trainLabel = '', candidateCols = null }) {
    const box = document.getElementById(containerId);
    if (!box) return;
    if (!rows || !rows.length || !targetCol || !columnConfigs[targetCol]) {
        box.innerHTML = '<div class="cal-panel"><div class="tune-title">📐 How many predictors?</div><p class="tune-muted">Choose the outcome (Target) to see how many predictors the sample can support.</p></div>';
        return;
    }
    const tcfg = columnConfigs[targetCol];
    const withOutcome = rows.filter(r => !DataEngine.isMissing(r[targetCol]));
    const n = withOutcome.length;
    const nTrain = Math.round(n * trainFraction);
    const { total: P, perCol } = predictorParameters(withOutcome, featureCols, columnConfigs);
    const isClass = tcfg.type === 'nominal' || tcfg.type === 'ordinal';
    const fmt = v => v.toLocaleString('en-GB');
    const rows4 = []; // [guide, max parameters, note]
    let headline = '', minority = null, minorityLabel = '', status = 'ok';

    if (isClass) {
        const counts = {};
        withOutcome.forEach(r => { const k = String(r[targetCol]).trim(); counts[k] = (counts[k] || 0) + 1; });
        const groups = Object.entries(counts).sort((a, b) => a[1] - b[1]);
        if (groups.length < 2) { box.innerHTML = ''; return; }
        [minorityLabel, minority] = groups[0];
        const E = Math.round(minority * trainFraction);
        const binary = groups.length === 2;
        headline = `<strong>${fmt(nTrain)}</strong> training patients${trainLabel ? ` (${escapeHtml(trainLabel)})` : ''}, of whom about <strong>${fmt(E)}</strong> are in the smaller outcome group ("${escapeHtml(minorityLabel)}", ${(minority / n * 100).toFixed(1)}%)${binary ? '' : ` of ${groups.length} groups`}.`;
        if (binary) {
            const riley = rileyMaxParameters('binary', nTrain, minority / n);
            rows4.push(['Minimum sample-size criteria (Riley et al.; logistic regression)', riley,
                riley ? 'the recommended standard: limits overfitting and estimates the overall risk precisely' : `not met for any number of predictors: ${fmt(ModelEngine.rileySampleSize({ type: 'binary', p: 1, prevalence: minority / n }).n)} patients are needed just to estimate the overall risk precisely`]);
        }
        rows4.push(['10 patients in the smaller group per parameter', Math.floor(E / 10), 'traditional minimum; often too lenient'],
            ['20 patients in the smaller group per parameter', Math.floor(E / 20), 'more cautious; logistic regression reached stable performance at about 20–50'],
            ['Flexible models (random forest, boosted trees, neural network, k-NN): 200 per parameter', Math.floor(E / 200), 'still unstable at this level in simulations (over 10 times what logistic regression needs); below it, treat these models as exploratory']);
        const main = binary ? Math.min(rows4[0][1], rows4[1][1]) : rows4[0][1];
        status = P <= main ? 'ok' : P <= Math.floor(E / 10) ? 'caution' : 'over';
    } else {
        headline = `<strong>${fmt(nTrain)}</strong> training patients${trainLabel ? ` (${escapeHtml(trainLabel)})` : ''}; numerical outcome.`;
        const riley = rileyMaxParameters('continuous', nTrain, null);
        rows4.push(['Minimum sample-size criteria (Riley et al.; linear regression, anticipated R² 0.2)', riley, riley ? 'the recommended standard' : 'not met for any number of predictors'],
            ['Flexible models (random forest, boosted trees, neural network, k-NN)', Math.floor(nTrain / 200), 'rough guide; flexible models need much more data than regression']);
        status = P <= riley ? 'ok' : P <= riley * 1.5 ? 'caution' : 'over';
    }

    const multi = perCol.filter(c => c.levels && c.levels > 2).sort((a, b) => b.df - a.df).slice(0, 3);
    const candidates = candidateCols ? predictorParameters(withOutcome, candidateCols, columnConfigs).total : null;
    const statusText = {
        ok: '✓ Your selection is within the recommended number of predictor parameters.',
        caution: '⚠️ Your selection is above the recommended number but not grossly so: expect some overfitting; penalised (ridge) regression and cross-validated tuning help.',
        over: '⚠️ <strong>Too many predictor parameters for this sample</strong>: models are likely to overfit (good results in these patients, worse in new ones). Reduce the number of predictors, group rare categories, or add patients.'
    }[status];

    box.innerHTML = `
        <div class="cal-panel">
            <div class="tune-title">📐 How many predictors can this sample support?</div>
            <p class="tune-muted" style="margin-top: 4px;">${headline} Your selection: <strong>${featureCols.length}</strong> predictor(s) = <strong>${P}</strong> parameter(s)${candidates !== null && candidates > P ? `, from <strong>${candidates}</strong> candidate parameters considered` : ''}.</p>
            <div class="${status === 'ok' ? 'explain-box' : 'tune-warn'}" style="margin-top: 6px;">${statusText}</div>
            <table class="data-table thr-table"><thead><tr><th>Guide</th><th>Largest number of parameters</th><th>Comment</th></tr></thead>
                <tbody>${rows4.map(([g, v, c]) => `<tr><td>${escapeHtml(g)}</td><td><strong>${v}</strong>${P > v ? ' <span class="tune-sd">(you have ' + P + ')</span>' : ''}</td><td>${escapeHtml(c)}</td></tr>`).join('')}</tbody></table>
            ${multi.length ? `<p class="tune-muted">Categorical predictors use one parameter per category beyond the first: ${multi.map(c => `"${escapeHtml(c.col)}" ${c.levels} categories = ${c.df}`).join('; ')}. Grouping rare categories saves parameters.</p>` : ''}
            <p class="tune-muted">A parameter is one number the model estimates: 1 for each numerical or ordered predictor, and (categories − 1) for an unordered categorical one. Choose predictors from clinical knowledge and earlier studies before looking at their association with the outcome: selecting them by p-value in the same patients makes the model look better than it is, and every predictor considered counts towards overfitting, not only those kept. SMOTENC and other balancing add no real patients. Riley et al. (BMJ 2020) with the default anticipated R² (15% of the maximum, as in pmsampsize; Step 4 lets you change it); 20–50 and 200 per parameter: van der Ploeg, Austin and Steyerberg, “Modern modelling techniques are data hungry” (BMC Med Res Methodol 2014;14:137).</p>
        </div>`;
}

export { predictorParameters, renderPredictorBudget, rileyMaxParameters };
