// safeguards.js: a check run before every training, whatever route led there. A single predictor that predicts the
// outcome almost perfectly is the usual sign of leakage (a value recorded after the outcome, or derived from it, or
// another copy of the outcome). It may be genuine, so the user is asked to confirm rather than stopped.
// (Blocking safeguards live in the data engine: no outcome column as a predictor, no patient on both sides of a split.)
/* global DataEngine, ModelEngine */

const NEAR_PERFECT = 0.99;     // univariate AUC (binary) or |Spearman rho| (numerical)
const NEAR_PERFECT_C = 0.97;   // univariate C-index (time-to-event)

// Predictors that alone predict the outcome almost perfectly: [{ col, measure, value }]
// outcome: { type: 'binary', col, positive } | { type: 'numeric', col } | { type: 'survival', timeCol, eventCol, eventValue }
function nearPerfectPredictors(rows, featureCols, configs, outcome) {
    const flagged = [];
    featureCols.forEach(col => {
        const cfg = configs[col] || {};
        const use = rows.filter(r => !DataEngine.isMissing(r[col]) && (outcome.type === 'survival'
            ? !DataEngine.isMissing(r[outcome.timeCol]) && Number(r[outcome.timeCol]) > 0 && !DataEngine.isMissing(r[outcome.eventCol])
            : !DataEngine.isMissing(r[outcome.col])));
        if (use.length < 20) return;
        // A score per patient: the value itself if numeric, otherwise the category (at most 10 categories) scored by
        // its outcome rate, so a category that always goes with one outcome shows up
        const numeric = cfg.type !== 'nominal' && use.every(r => isFinite(Number(r[col])));
        let score;
        if (numeric) score = use.map(r => Number(r[col]));
        else {
            const cats = [...new Set(use.map(r => String(r[col]).trim()))];
            if (cats.length > 10 || cats.length < 2) return;
            const y = outcome.type === 'binary' ? use.map(r => (String(r[outcome.col]).trim() === String(outcome.positive) ? 1 : 0))
                : outcome.type === 'survival' ? use.map(r => (String(r[outcome.eventCol]).trim() === String(outcome.eventValue) ? 1 : 0)) : use.map(r => Number(r[outcome.col]));
            const mean = {};
            cats.forEach(c => { const v = y.filter((_, i) => String(use[i][col]).trim() === c); mean[c] = v.reduce((a, b) => a + b, 0) / v.length; });
            score = use.map(r => mean[String(r[col]).trim()]);
        }
        if (new Set(score).size < 2) return;
        if (outcome.type === 'binary') {
            const y = use.map(r => (String(r[outcome.col]).trim() === String(outcome.positive) ? 1 : 0));
            if (!y.some(v => v) || y.every(v => v)) return;
            const auc = ModelEngine._calculateBinaryAUC(y, score);
            const a = Math.max(auc, 1 - auc);
            if (a >= NEAR_PERFECT) flagged.push({ col, measure: 'AUC on its own', value: a });
        } else if (outcome.type === 'numeric') {
            const res = DataEngine.calculateSpearman(score, use.map(r => Number(r[outcome.col])));
            const rho = typeof res === 'number' ? res : res && typeof res.rho === 'number' ? res.rho : res && typeof res.r === 'number' ? res.r : null;
            if (rho !== null && Math.abs(rho) >= NEAR_PERFECT) flagged.push({ col, measure: 'rank correlation with the outcome', value: Math.abs(rho) });
        } else {
            const c = ModelEngine.concordanceIndex(use.map(r => Number(r[outcome.timeCol])), use.map(r => (String(r[outcome.eventCol]).trim() === String(outcome.eventValue) ? 1 : 0)), score);
            if (c !== null && Math.max(c, 1 - c) >= NEAR_PERFECT_C) flagged.push({ col, measure: 'C-index on its own', value: Math.max(c, 1 - c) });
        }
    });
    return flagged;
}

// Asks the user to confirm when a predictor looks like leakage; returns false if the user stops the analysis
function confirmNoLeakage(rows, featureCols, configs, outcome) {
    const flagged = nearPerfectPredictors(rows, featureCols, configs, outcome);
    if (!flagged.length) return true;
    const list = flagged.map(f => `• ${f.col}: ${f.measure} ${f.value.toFixed(3)}`).join('\n');
    return confirm(`Possible leakage\n\nThese predictors on their own predict the outcome almost perfectly:\n${list}\n\nThis usually means the value was recorded after the outcome, is derived from it, or is another copy of it. A model using it would look excellent but be useless for new patients.\n\nCheck these columns in Step 2 (exclude them if they would not be known at the time of prediction). Continue anyway?`);
}

// The same check for a standard outcome column: two values (binary) or numerical; other outcomes are not checked
function confirmNoLeakageForTarget(rows, featureCols, configs, targetCol) {
    if (!targetCol) return true;
    const vals = [...new Set(rows.map(r => r[targetCol]).filter(v => !DataEngine.isMissing(v)).map(v => String(v).trim()))];
    if ((configs[targetCol] || {}).type === 'continuous') return confirmNoLeakage(rows, featureCols, configs, { type: 'numeric', col: targetCol });
    if (vals.length === 2) return confirmNoLeakage(rows, featureCols, configs, { type: 'binary', col: targetCol, positive: vals[1] });
    return true;
}

export { confirmNoLeakage, confirmNoLeakageForTarget, nearPerfectPredictors };
