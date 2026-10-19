// predictor-budget.js: "How many predictors?" guidance where predictors are chosen (Step 2 of the standard
// workflow, Step 4 of the 7-step pipeline). From the number of training patients and the size of the smaller
// outcome group it gives the largest number of predictor parameters supported by the Riley et al. minimum
// sample-size criteria (pmsampsize), by 10 and 20 events per parameter, and the much larger samples flexible
// machine-learning models need, and compares them with the current selection.
/* global DataEngine, ModelEngine */
import { ALL_MODEL_IDS, AppState, MODEL_INFO, escapeHtml } from './core.js';
import { pipelineGroupFunction, splitGroupFunction } from './cv-tuning.js';

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

// Outcome counts in independent units: distinct patients when a patient-ID column is set (a patient's
// records are not independent), otherwise records, which are then assumed to be one per patient
function outcomeUnits(rows, targetCol, groupOf) {
    const all = new Set(), by = {};
    rows.forEach((r, i) => {
        const u = groupOf ? groupOf(r) : i, k = String(r[targetCol]).trim();
        all.add(u); (by[k] = by[k] || new Set()).add(u);
    });
    return { n: all.size, counts: Object.fromEntries(Object.entries(by).map(([k, v]) => [k, v.size])), unit: groupOf ? 'patients' : 'records' };
}
const UNIT_NOTE = 'Counts assume one record per patient; if patients have several records, choose "keep each patient\'s records together" (Step 4) so that patients, not records, are counted.';

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
function renderPredictorBudget(containerId, { rows, featureCols, targetCol, columnConfigs, trainFraction = 1, trainLabel = '', candidateCols = null, groupOf = null, eventValue = null }) {
    const box = document.getElementById(containerId);
    if (!box) return;
    if (!rows || !rows.length || !targetCol || !columnConfigs[targetCol]) {
        box.innerHTML = '<div class="cal-panel"><div class="tune-title">📐 How many predictors?</div><p class="tune-muted">Choose the outcome (Target) to see how many predictors the sample can support.</p></div>';
        return;
    }
    const tcfg = columnConfigs[targetCol];
    const withOutcome = rows.filter(r => !DataEngine.isMissing(r[targetCol]));
    const units = outcomeUnits(withOutcome, targetCol, groupOf);
    const n = units.n;
    const nTrain = Math.round(n * trainFraction);
    const { total: P, perCol } = predictorParameters(withOutcome, featureCols, columnConfigs);
    // Predictors dropped after looking at the data still count towards overfitting: the verdict uses them all
    const candidates = candidateCols ? predictorParameters(withOutcome, candidateCols, columnConfigs).total : null;
    const Pv = Math.max(P, candidates || 0);
    const isClass = tcfg.type === 'nominal' || tcfg.type === 'ordinal';
    const fmt = v => v.toLocaleString('en-GB');
    const rows4 = []; // [guide, max parameters, note]
    let headline = '', minority = null, minorityLabel = '', status = 'ok';

    if (isClass) {
        const groups = Object.entries(units.counts).sort((a, b) => a[1] - b[1]);
        if (groups.length < 2) { box.innerHTML = ''; return; }
        [minorityLabel, minority] = groups[0];
        // Time-to-event outcome: what limits a Cox model is the number of events (not the smaller group)
        if (eventValue !== null) { const g = groups.find(([k]) => k === String(eventValue)); if (g) [minorityLabel, minority] = g; }
        const E = Math.round(minority * trainFraction);
        const binary = groups.length === 2 && eventValue === null;
        headline = `<strong>${fmt(nTrain)}</strong> training ${units.unit}${trainLabel ? ` (${escapeHtml(trainLabel)})` : ''}, of whom about <strong>${fmt(E)}</strong> ${eventValue !== null ? `had the event ("${escapeHtml(minorityLabel)}", ${(minority / n * 100).toFixed(1)}%; time-to-event outcome, Cox model: events per parameter)` : `are in the smaller outcome group ("${escapeHtml(minorityLabel)}", ${(minority / n * 100).toFixed(1)}%)${binary ? '' : ` of ${groups.length} groups`}`}.`;
        if (binary) {
            const riley = rileyMaxParameters('binary', nTrain, minority / n);
            rows4.push(['Minimum sample-size criteria (Riley et al.; logistic regression)', riley,
                riley ? 'the recommended standard: limits overfitting and estimates the overall risk precisely' : `not met for any number of predictors: ${fmt(ModelEngine.rileySampleSize({ type: 'binary', p: 1, prevalence: minority / n }).n)} patients are needed just to estimate the overall risk precisely`]);
        }
        rows4.push(['10 patients in the smaller group per parameter', Math.floor(E / 10), 'traditional minimum; often too lenient'],
            ['20 patients in the smaller group per parameter', Math.floor(E / 20), 'more cautious; logistic regression reached stable performance at about 20–50'],
            ['Flexible models (random forest, boosted trees, neural network, k-NN): 200 per parameter', Math.floor(E / 200), 'still unstable at this level in simulations (over 10 times what logistic regression needs); below it, treat these models as exploratory']);
        const main = binary ? Math.min(rows4[0][1], rows4[1][1]) : rows4[0][1];
        status = Pv <= main ? 'ok' : Pv <= Math.floor(E / 10) ? 'caution' : 'over';
    } else {
        headline = `<strong>${fmt(nTrain)}</strong> training ${units.unit}${trainLabel ? ` (${escapeHtml(trainLabel)})` : ''}; numerical outcome.`;
        const riley = rileyMaxParameters('continuous', nTrain, null);
        rows4.push(['Minimum sample-size criteria (Riley et al.; linear regression, anticipated R² 0.2)', riley, riley ? 'the recommended standard' : 'not met for any number of predictors'],
            ['Flexible models (random forest, boosted trees, neural network, k-NN)', Math.floor(nTrain / 200), 'rough guide; flexible models need much more data than regression']);
        status = Pv <= riley ? 'ok' : Pv <= riley * 1.5 ? 'caution' : 'over';
    }

    const multi = perCol.filter(c => c.levels && c.levels > 2).sort((a, b) => b.df - a.df).slice(0, 3);
    // A categorical predictor with nearly as many categories as records is probably an identifier: it would
    // overfit badly, and its values (e.g. patient IDs) would be stored in exported model files
    const idLike = perCol.filter(c => c.levels && c.levels > 20 && c.levels >= 0.5 * withOutcome.length);
    const counted = candidates !== null && candidates > P ? `the ${candidates} candidate parameters considered` : 'your selection';
    const statusText = {
        ok: `✓ ${counted === 'your selection' ? 'Your selection meets' : `The ${candidates} candidate parameters considered meet`} these planning criteria for regression models, under the stated assumptions (anticipated R², one record per patient unless patients are grouped). This is a planning guide, not proof that a model will perform well.`,
        caution: `⚠️ ${counted === 'your selection' ? 'Your selection is' : `The ${candidates} candidate parameters considered are`} above the recommended number but not grossly so: expect some overfitting; penalised (ridge) regression and cross-validated tuning help.`,
        over: '⚠️ <strong>Too many predictor parameters for this sample</strong>: models are likely to overfit (good results in these patients, worse in new ones). Reduce the number of predictors, group rare categories, or add patients.'
    }[status];

    box.innerHTML = `
        <div class="cal-panel">
            <div class="tune-title">📐 How many predictors can this sample support?</div>
            <p class="tune-muted" style="margin-top: 4px;">${headline} Your selection: <strong>${featureCols.length}</strong> predictor(s) = <strong>${P}</strong> parameter(s)${candidates !== null && candidates > P ? `, from <strong>${candidates}</strong> candidate parameters considered` : ''}.</p>
            <div class="${status === 'ok' ? 'explain-box' : 'tune-warn'}" style="margin-top: 6px;">${statusText}</div>
            <table class="data-table thr-table budget-table"><thead><tr><th>Guide</th><th>Largest number of parameters</th><th>Comment</th></tr></thead>
                <tbody>${rows4.map(([g, v, c]) => `<tr><td>${escapeHtml(g)}</td><td><strong>${v}</strong>${Pv > v ? ' <span class="tune-sd">(you have ' + Pv + ')</span>' : ''}</td><td>${escapeHtml(c)}</td></tr>`).join('')}</tbody></table>
            ${idLike.length ? `<div class="tune-warn">⚠️ <strong>Looks like an identifier:</strong> ${idLike.map(c => `"${escapeHtml(c.col)}" (${c.levels} categories in ${withOutcome.length} records)`).join(', ')}. Untick it (or set it to Meta / ID): identifiers cannot predict new patients, and their values would be stored in exported model files.</div>` : ''}
            ${multi.length ? `<p class="tune-muted">Categorical predictors use one parameter per category beyond the first: ${multi.map(c => `"${escapeHtml(c.col)}" ${c.levels} categories = ${c.df}`).join('; ')}. Grouping rare categories saves parameters.</p>` : ''}
            ${groupOf ? '' : `<p class="tune-muted">${escapeHtml(UNIT_NOTE)}</p>`}
            <p class="tune-muted">A parameter is one number the model estimates: 1 for each numerical or ordered predictor, and (categories − 1) for an unordered categorical one. Choose predictors from clinical knowledge and earlier studies before looking at their association with the outcome: selecting them by p-value in the same patients makes the model look better than it is, and every predictor considered counts towards overfitting, not only those kept. SMOTENC and other balancing add no real patients. Riley et al. (BMJ 2020) with the default anticipated R² (15% of the maximum, as in pmsampsize; Step 4 lets you change it); 20–50 and 200 per parameter: van der Ploeg, Austin and Steyerberg, “Modern modelling techniques are data hungry” (BMC Med Res Methodol 2014;14:137).</p>
        </div>`;
}


// How much data each model needs, in patients of the smaller outcome group (events) per predictor parameter.
// Logistic regression: Riley criteria / 10-20 per parameter, stable at about 20-50 (van der Ploeg et al. 2014).
// Random forest and neural network: still unstable above 200 per parameter in that study; k-NN, boosted trees
// and the ensemble were not studied there and are treated like them, as flexible models.
const MODEL_NEEDS = {
    glm: { per: 10, better: 20, kind: 'regression', evidence: 'Riley criteria; stable at about 20–50 per parameter in simulations' },
    nb: { per: 10, better: 20, kind: 'simple', evidence: 'simple model with few estimates, like logistic regression (little direct evidence)' },
    rf: { per: 200, kind: 'flexible', evidence: 'still unstable above 200 per parameter in simulations' },
    nn: { per: 200, kind: 'flexible', evidence: 'still unstable above 200 per parameter in simulations' },
    gbm: { per: 200, kind: 'flexible', evidence: 'flexible like random forests (not studied directly)' },
    knn: { per: 200, kind: 'flexible', evidence: 'flexible, and weakens quickly as predictors are added (not studied directly)' },
    ensemble: { per: 200, kind: 'flexible', evidence: 'needs as much as its most data-hungry member' }
};

/**
 * For each selected model: the data it needs for the current predictors, what the Training set has, and a
 * verdict (adequate / borderline / exploratory only).
 */
function renderModelSampleGuide(containerId, { rows, featureCols, targetCol, columnConfigs, modelIds, trainFraction = 1, trainLabel = '', candidateCols = null, groupOf = null }) {
    const box = document.getElementById(containerId);
    if (!box) return;
    if (!rows || !rows.length || !targetCol || !columnConfigs[targetCol] || !modelIds || !modelIds.length) { box.innerHTML = ''; return; }
    const tcfg = columnConfigs[targetCol];
    const withOutcome = rows.filter(r => !DataEngine.isMissing(r[targetCol]));
    const units = outcomeUnits(withOutcome, targetCol, groupOf);
    const n = units.n, nTrain = Math.round(n * trainFraction);
    const P = Math.max(1, predictorParameters(withOutcome, featureCols, columnConfigs).total,
        candidateCols ? predictorParameters(withOutcome, candidateCols, columnConfigs).total : 0);
    const isClass = tcfg.type === 'nominal' || tcfg.type === 'ordinal';
    const fmt = v => Math.round(v).toLocaleString('en-GB');
    let E = nTrain, share = 1, unit = 'patients', binary = false;
    if (isClass) {
        const groups = Object.values(units.counts).sort((a, b) => a - b);
        if (groups.length < 2) { box.innerHTML = ''; return; }
        share = groups[0] / n;
        E = Math.round(groups[0] * trainFraction);
        unit = 'in the smaller outcome group';
        binary = groups.length === 2;
    }
    const riley = binary ? rileyMaxParameters('binary', nTrain, share) : (!isClass ? rileyMaxParameters('continuous', nTrain, null) : null);
    const rowsHtml = modelIds.filter(id => MODEL_NEEDS[id] && (isClass || id !== 'nb')).map(id => {
        const need = MODEL_NEEDS[id];
        const needed = need.per * P, better = need.better ? need.better * P : null;
        let verdict, cls;
        if (need.kind === 'flexible') {
            if (E >= needed) { verdict = '✓ Meets this guide (stability still not assured)'; cls = 'ok'; }
            else if (E >= needed / 4) { verdict = '⚠️ Exploratory only: expect overfitting'; cls = 'warn'; }
            else { verdict = '✗ Far too little data: results unreliable'; cls = 'bad'; }
        } else {
            const rileyOk = id === 'glm' && riley !== null ? P <= riley : true;
            if (E >= (better || needed) && rileyOk) { verdict = '✓ Meets these planning criteria'; cls = 'ok'; }
            else if (E >= needed) { verdict = '⚠️ Borderline: fewer predictors would be safer'; cls = 'warn'; }
            else { verdict = '✗ Too little data for this many predictors'; cls = 'bad'; }
        }
        const approx = v => { const d = Math.pow(10, Math.max(0, Math.floor(Math.log10(Math.max(1, v))) - 1)); return fmt(Math.round(v / d) * d); }; // 2 significant figures
        const patients = isClass ? ` (≈ ${approx(needed / share)}${better ? `–${approx(better / share)}` : ''} patients)` : '';
        const supports = id === 'glm' && riley !== null ? Math.min(riley, Math.floor(E / need.per)) : Math.floor(E / need.per);
        return `<tr><td>${MODEL_INFO[id] ? MODEL_INFO[id].icon + ' ' + escapeHtml(MODEL_INFO[id].short) : id}</td>
            <td><strong>${fmt(needed)}${need.better ? `–${fmt(better)}` : ''}</strong> ${isClass ? 'events' : 'patients'}${patients}<br><span class="tune-sd">${need.better ? `${need.per}–${need.better}` : `${need.per}+`} per parameter: ${escapeHtml(need.evidence)}</span></td>
            <td class="budget-${cls}"><strong>${verdict}</strong><br><span class="tune-sd">supports about ${supports} parameter(s)</span></td></tr>`;
    }).join('');
    if (!rowsHtml) { box.innerHTML = ''; return; }
    box.innerHTML = `
        <div class="cal-panel">
            <div class="tune-title">📏 Enough data for each model?</div>
            <p class="tune-muted" style="margin-top: 4px;">Training set${trainLabel ? ` (${escapeHtml(trainLabel)})` : ''}: <strong>${fmt(nTrain)}</strong> ${units.unit}, <strong>${fmt(E)}</strong> ${isClass ? 'in the smaller outcome group' : 'in all'}; <strong>${P}</strong> predictor parameter(s)${candidateCols ? ' (all candidates considered)' : ''}.${riley !== null ? ` The Riley criteria allow logistic regression up to <strong>${riley}</strong> parameter(s) here.` : ''}</p>
            <table class="data-table thr-table budget-table"><thead><tr><th>Model</th><th>Needed for your ${P} parameter(s)</th><th>Your Training set</th></tr></thead>
                <tbody>${rowsHtml}</tbody></table>
            ${groupOf ? '' : `<p class="tune-muted">${escapeHtml(UNIT_NOTE)}</p>`}
            <p class="tune-muted">These are planning guides under stated assumptions, not guarantees. An "event" is a patient in the smaller outcome group. Flexible models (random forest, boosted trees, neural network, k-NN, ensemble) need roughly ten times the data of logistic regression; with typical clinical cohorts of a few hundred patients they rarely beat penalised logistic regression and overfit more. Training them anyway is fine as a comparison, but judge them by validated results (Option C), not by apparent performance.${isClass ? '' : ' Numerical outcomes: rough guide in patients rather than events.'} Sources: Riley et al. (BMJ 2020); van der Ploeg, Austin and Steyerberg (BMC Med Res Methodol 2014;14:137).</p>
        </div>`;
}

// Step 4 of the standard workflow: the models currently selected, for the share of patients that will train them
function refreshModelSampleGuide() {
    if (!AppState.parsedData || !AppState.columnConfigs) return;
    const cfgs = AppState.columnConfigs, headers = AppState.parsedData.headers;
    const mode = document.querySelector('input[name="validation-mode"]:checked')?.value || 'split';
    const pct = Math.min(95, Math.max(50, parseFloat(document.getElementById('train-pct-input')?.value) || 80));
    renderModelSampleGuide('model-sample-guide', {
        rows: AppState.parsedData.rows, featureCols: headers.filter(h => cfgs[h] && cfgs[h].role === 'feature'),
        targetCol: headers.find(h => cfgs[h] && cfgs[h].role === 'target'), columnConfigs: cfgs,
        modelIds: AppState.selectedModelIds && AppState.selectedModelIds.length ? AppState.selectedModelIds : [AppState.selectedModelId || 'nn'],
        trainFraction: mode === 'split' ? pct / 100 : 1,
        trainLabel: mode === 'split' ? `Option A, ${pct}% of patients` : (mode === 'bootstrap' ? 'Option C, all patients' : 'Option B, all loaded patients'),
        groupOf: splitGroupFunction()
    });
}

// Step 7 of the pipeline: every model, on the pipeline Training set and the predictors still in use
function refreshPipelineModelGuide() {
    const P = AppState.pipeline;
    if (!P || !P.trainRows) return;
    renderModelSampleGuide('pipeline-model-sample-guide', {
        rows: P.trainRows, featureCols: P.featureCols.filter(c => !P.droppedFeatures.has(c)), targetCol: P.targetCol,
        columnConfigs: P.columnConfigs, modelIds: ALL_MODEL_IDS, trainLabel: '7-step pipeline',
        candidateCols: P.featureCols, groupOf: pipelineGroupFunction()
    });
}

export { outcomeUnits, refreshModelSampleGuide, refreshPipelineModelGuide, MODEL_NEEDS, predictorParameters, renderModelSampleGuide, renderPredictorBudget, rileyMaxParameters };
