// training-time.js: how long training is likely to take, shown under the Train button before it is pressed.
// The estimate counts the model fits the current settings imply (validation option, bootstrap resamples, the
// cross-validation used to choose between models, tuning, recalibration, multiple imputation) and multiplies them
// by a cost per fit for each model. The costs were measured in Chrome on an Apple M2 Max (420 patients, 14
// predictors; one bootstrap resample of Option C, including its data preparation and scoring), scaled by the
// number of patients and by the speed of this computer (a short benchmark). Checked against complete runs, it was
// within about 15%; it is a guide, and the status line counts down once training starts.
/* global DataEngine */
import { AppState } from './core.js';
import { INNER_TUNE_MAX_FOLDS, TUNABLE_MODELS, tuningGrid } from './validation-core.js';
import { estimateImputationSeconds, readMultipleImputation } from './multiple-imputation.js';
import { miceQuickApproximation, readRecalibration } from './cv-tuning.js';

// Seconds per Option C resample at 420 patients on the reference computer, and how the cost grows with patients
const COST = {
    glm: { s: 0.15, a: 0.45 }, nb: { s: 0.04, a: 1 }, knn: { s: 0.03, a: 1.5 }, rf: { s: 0.26, a: 0.55 },
    gbm: { s: 0.21, a: 0.5 }, nn: { s: 0.97, a: 0.9 }, ensemble: { s: 1.9, a: 0.85 }
};
const N_REF = 420, BENCH_REF_MS = 10.3;
const SPLIT_FACTOR = 0.55;  // a fit outside the bootstrap (no out-of-bag scoring or resampled preparation)
const INNER_FACTOR = 0.7;   // an inner cross-validation fold (two thirds of the resample, scored on the rest)
const OVERHEAD_S = 0.4;     // preparation and reporting per model
const CALIBRATION = 1.08;   // complete runs took on average 8% longer than the sum of the parts

let speed = null;
// Relative speed of this computer (1 = reference; 2 = half as fast), from a fixed arithmetic task
function computerSpeed() {
    if (speed !== null) return speed;
    const run = () => {
        const t = performance.now(); let s = 0; const a = new Float64Array(4096);
        for (let i = 0; i < a.length; i++) a[i] = Math.sin(i);
        for (let r = 0; r < 400; r++) for (let i = 0; i < a.length; i++) s += Math.exp(-a[i] * a[i]) * a[(i * 7) & 4095];
        return s > -1 ? performance.now() - t : 0;
    };
    run();
    const ms = [run(), run(), run()].sort((x, y) => x - y)[1];
    speed = Math.min(6, Math.max(0.5, ms / BENCH_REF_MS));
    return speed;
}

const fmtTime = s => (s < 10 ? 'under 10 seconds' : s < 50 ? `about ${Math.round(s / 5) * 5} seconds` : s < 90 ? 'about a minute' : s < 3300 ? `about ${Math.round(s / 60)} minutes` : `about ${(s / 3600).toFixed(1).replace(/\.0$/, '')} hours`);

/** Estimated seconds and model fits for the current Step 4 settings (null when it cannot be estimated) */
function estimateTraining() {
    const pd = AppState.parsedData, cfgs = AppState.columnConfigs;
    if (!pd || !cfgs || AppState.outcomeType === 'survival') return null;
    const target = Object.keys(cfgs).find(c => cfgs[c].role === 'target');
    const feats = Object.keys(cfgs).filter(c => cfgs[c].role === 'feature');
    const models = (AppState.selectedModelIds && AppState.selectedModelIds.length ? AppState.selectedModelIds : [AppState.selectedModelId]).filter(id => COST[id]);
    if (!target || !feats.length || !models.length) return null;
    const n = pd.rows.filter(r => !DataEngine.isMissing(r[target])).length;
    if (!n) return null;
    const val = id => document.getElementById(id);
    const mode = document.querySelector('input[name="validation-mode"]:checked')?.value || 'split';
    const pct = Math.min(95, Math.max(50, parseFloat(val('train-pct-input')?.value) || 80)) / 100;
    const B = parseInt(val('bootstrap-reps')?.value, 10) || 100;
    const autoTune = Boolean(val('autotune-check')?.checked);
    const tuneFolds = parseInt(val('autotune-folds')?.value, 10) || 5;
    const selectByCV = models.length > 1 && (document.querySelector('input[name="selection-mode"]:checked')?.value || 'cv') === 'cv';
    const recal = cfgs[target].type !== 'continuous' && readRecalibration('recalibration-select') !== 'none';
    const retrain = val('retrain-yes') ? val('retrain-yes').checked : true;
    const miM = AppState.completeCase ? 0 : readMultipleImputation();
    const slow = id => id === 'nn' || id === 'ensemble';
    const repeatSelection = selectByCV && models.length > 1 && !models.some(slow);
    const nFit = mode === 'split' ? Math.round(n * pct) : n;
    let seconds = 0, fits = 0;
    const resamples = new Set();
    models.forEach(id => {
        const unit = COST[id].s * Math.pow(nFit / N_REF, COST[id].a);
        const tunable = autoTune && TUNABLE_MODELS.has(id);
        const G = tunable ? tuningGrid(id, {}, Math.round(nFit * 0.8)).candidates.length : 1;
        const kOuter = slow(id) ? Math.min(tuneFolds, 3) : tuneFolds;
        const cvFits = tunable || selectByCV || recal ? G * kOuter : 0;
        if (mode === 'bootstrap') {
            const Bm = slow(id) ? Math.min(B, 50) : B;
            resamples.add(Bm);
            const tuneInside = tunable && !slow(id);
            const inner = tuneInside || repeatSelection || recal ? INNER_TUNE_MAX_FOLDS * (tuneInside ? G : 1) : 0;
            const finals = Math.max(1, miM);
            fits += cvFits + finals + Bm * (1 + inner);
            seconds += unit * (SPLIT_FACTOR * (cvFits + finals) + Bm * (1 + INNER_FACTOR * inner));
            if (miM && !miceQuickApproximation()) seconds += Bm * (1 + (inner ? INNER_TUNE_MAX_FOLDS : 0)) * estimateImputationSeconds(n, feats.length, 1);
        } else {
            const m = Math.max(1, miM);
            const k = (cvFits + 1 + (mode === 'split' && retrain ? 1 : 0)) * m;
            fits += k;
            seconds += unit * SPLIT_FACTOR * k;
        }
        seconds += OVERHEAD_S;
    });
    if (miM) seconds += estimateImputationSeconds(n, feats.length, miM) * (mode === 'split' && retrain ? 2 : 1);
    return { seconds: seconds * computerSpeed() * CALIBRATION, fits, models: models.length, mode, resamples: [...resamples].sort((x, y) => y - x), n };
}

/** Writes the estimate under the Train button */
function updateTrainingEstimate() {
    const box = document.getElementById('train-estimate');
    if (!box) return;
    const e = estimateTraining();
    if (!e) { box.style.display = 'none'; return; }
    box.style.display = 'block';
    box.innerHTML = `Estimated training time on this computer: <strong>${fmtTime(e.seconds)}</strong> (${e.models} model${e.models > 1 ? 's' : ''}, about ${e.fits.toLocaleString('en-GB')} model fits${e.mode === 'bootstrap' ? ` with ${e.resamples.length > 1 ? `${e.resamples[0]} bootstrap resamples (${e.resamples[1]} for the neural network and ensemble)` : `${e.resamples[0]} bootstrap resamples`}` : ''}). A rough guide: once training starts, the status line counts down.`;
}

export { estimateTraining, updateTrainingEstimate };
