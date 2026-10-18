// interpretation.js: a plain-language reading of the active model's validated results, for clinicians: what is
// predicted and for whom, what the probability means, how it compares with a simple baseline, what errors to
// expect at the chosen threshold, and the limits of the validation behind these numbers.
import { escapeHtml } from './core.js';

const interpPct = v => `${(v * 100).toFixed(0)}%`;

function aucWords(a) {
    return a >= 0.9 ? 'excellent' : a >= 0.8 ? 'good' : a >= 0.7 ? 'acceptable' : a >= 0.6 ? 'weak' : 'little better than chance';
}

/**
 * ev: evaluation of the active model; prep: its preprocessing; name: model name; design: 'A' | 'B' | 'C';
 * dataset: training dataset name. Renders into containerId (nothing for unsupported cases).
 */
function renderInterpretation(containerId, { ev, prep, name, design, dataset }) {
    const box = document.getElementById(containerId);
    if (!box) return;
    if (!ev || !prep) { box.innerHTML = ''; return; }
    const items = [];
    const target = escapeHtml(prep.targetCol);
    const binary = prep.isClassification && (prep.targetClasses || []).length === 2;
    const positive = binary ? escapeHtml(String(prep.targetIdxToClass ? prep.targetIdxToClass[1] : prep.targetClasses[1])) : '';
    const designText = design === 'B' ? 'a separate validation cohort (Dataset ii)' : design === 'C' ? 'bootstrap internal validation (optimism-corrected; no patients were held out)' : 'held-out test patients from the same dataset';

    // 1. What is predicted, for whom
    items.push(['What it predicts', binary
        ? `The probability that "${target}" is "${positive}", for patients like those in ${escapeHtml(dataset || 'the training data')}. The time horizon is whatever period "${target}" was recorded over in your data: state it explicitly when you report or use the model.`
        : prep.isClassification ? `Which category of "${target}" a patient falls into, for patients like those in ${escapeHtml(dataset || 'the training data')}.`
            : `The value of "${target}", for patients like those in ${escapeHtml(dataset || 'the training data')}.`]);

    if (binary) {
        // 2. What the probability means
        const type = ev.probabilityType;
        const slope = ev.calibration && typeof ev.calibration.slope === 'number' ? ev.calibration.slope : null;
        let meaning = type && type !== 'model' && type !== 'recalibrated'
            ? 'This model\'s "probability" is a vote or neighbour share, not a calibrated risk: read it as a ranking score unless the calibration below is good.'
            : 'A predicted 30% is meant to correspond to about 30 in 100 similar patients having the outcome.';
        if (slope !== null) meaning += slope < 0.8 ? ` Here the calibration slope is ${slope.toFixed(2)}: predictions are too extreme, so the percentages overstate certainty.` : slope > 1.2 ? ` Here the calibration slope is ${slope.toFixed(2)}: predictions are too cautious.` : ` The calibration slope (${slope.toFixed(2)}) suggests the percentages are about right on these patients.`;
        items.push(['What the probability means', meaning]);
        // 3. Discrimination and comparison with a simple baseline
        if (typeof ev.rocAuc === 'number') {
            items.push(['How well it separates patients', `AUC ${ev.rocAuc.toFixed(2)} (${aucWords(ev.rocAuc)}): given one patient with "${positive}" and one without, the model gives the higher risk to the right one about ${Math.round(ev.rocAuc * 100)} times in 100 (50 = coin toss).`]);
        }
        if (typeof ev.brierScore === 'number' && Array.isArray(ev.actuals) && ev.actuals.length) {
            const prev = ev.actuals.filter(v => v === 1).length / ev.actuals.length, ref = prev * (1 - prev);
            if (ref > 0) {
                const scaled = 1 - ev.brierScore / ref;
                items.push(['Compared with a simple baseline', scaled <= 0
                    ? `No better than giving every patient the average risk (${interpPct(prev)}): the model adds no useful information about individual patients here.`
                    : `It removes ${interpPct(scaled)} of the prediction error made by giving every patient the average risk (${interpPct(prev)}) (scaled Brier score). Also compare it with simple logistic regression in the leaderboard: a more complex model should earn its place.`]);
            }
        }
        // 4. Errors at the chosen threshold
        const cm = ev.confMatrix;
        if (cm && cm[1] && cm[0]) {
            const pos = cm[1][0] + cm[1][1], neg = cm[0][0] + cm[0][1];
            if (pos && neg) {
                const thr = ev.decisionThreshold || 0.5;
                items.push([`Errors at the ${interpPct(thr)} threshold`, `Of every 100 patients with "${positive}", the model would miss about ${Math.round(cm[1][0] / pos * 100)} (false negatives); of every 100 without, it would wrongly flag about ${Math.round(cm[0][1] / neg * 100)} (false positives). Which error matters more is a clinical judgement: choose the threshold from the consequences of each (the decision curve helps), not from whichever value maximises a statistic.`]);
            }
        }
    } else if (prep.isClassification && Array.isArray(ev.actuals) && ev.actuals.length && typeof ev.accuracy === 'number') {
        const counts = {};
        ev.actuals.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
        const majority = Math.max(...Object.values(counts)) / ev.actuals.length;
        items.push(['Compared with a simple baseline', `Accuracy ${interpPct(ev.accuracy)} against ${interpPct(majority)} for always predicting the most common category${typeof ev.balancedAccuracy === 'number' ? `; balanced accuracy ${interpPct(ev.balancedAccuracy)} (chance level ${interpPct(1 / Object.keys(counts).length)})` : ''}.`]);
    } else if (!prep.isClassification && typeof ev.r2 === 'number') {
        items.push(['Compared with a simple baseline', `R² ${ev.r2.toFixed(2)}: the model explains ${interpPct(Math.max(0, ev.r2))} of the variation that predicting the average for everyone leaves unexplained${ev.r2 <= 0 ? ' (no better than the average)' : ''}. Typical error (MAE): ${typeof ev.mae === 'number' ? ev.mae.toFixed(2) : 'n/a'} in the units of "${target}".`]);
    }

    // 5. Limits of the evidence
    const n = Array.isArray(ev.actuals) ? ev.actuals.length : null;
    items.push(['Limits of this evidence', `These results come from ${designText}${n ? ` (${n} patients)` : ''}. ${design === 'B' ? 'This is the strongest evidence the app can give, if Dataset (ii) is truly independent (another time period or centre).' : 'They show how the model performs on patients like the training data, not in another hospital or period: an independent (external) validation is needed before any clinical use.'} The model is for research; predictions for patients unlike the training patients (the app warns about these) are less reliable.`]);

    box.innerHTML = `
        <div class="cal-panel interpretation-panel">
            <div class="tune-title">🩺 In plain language: ${escapeHtml(name || 'the active model')}</div>
            <dl class="interp-list">${items.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${v}</dd>`).join('')}</dl>
        </div>`;
}

export { renderInterpretation };
