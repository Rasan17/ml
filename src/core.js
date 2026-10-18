// core.js: part of the Universal Medical ML/NeuralNet Predictor application controller


// Application State
const AppState = {
    currentStep: 1,
    theme: 'dark',
    fileName: '',
    parsedData: null,
    columnAnalysis: null,
    columnConfigs: {},
    imputedData: null,
    prepMeta: null,
    trainTestData: null,
    selectedModelId: 'nn',
    trainedModel: null,
    evalResults: null,
    charts: {},
    batchData: null,
    batchPredictions: null,
    // Validation Protocol (Option A vs Option B)
    validationMode: 'split', // 'split' (Option A) or 'separate' (Option B)
    separateTestRawRows: null,
    separateTestFileName: '',
    separateTestData: null,
    // Model Testing Scope (One Model, Multiple Models, All Models)
    modelScope: 'single', // 'single', 'multi', 'all'
    selectedModelIds: ['nn'],
    activeHyperparamModelId: 'nn',
    allTestedResults: null,
    deployedModelId: 'nn',
    // 7-Step Clinical ML Pipeline State
    pipeline: null // initialised below via createEmptyPipelineState()
};


// Model Metadata & Lookup
const MODEL_INFO = {
    nn: { name: 'Neural Network (MLP)', short: 'Neural Network', tag: 'Neural Network', pillClass: 'pill-target', icon: '🧠' },
    rf: { name: 'Random Forest', short: 'Random Forest', tag: 'Ensemble Trees', pillClass: 'pill-feature', icon: '🌲' },
    glm: { name: 'Logistic / Ridge Regression', short: 'GLM / Regression', tag: 'Generalized Linear', pillClass: 'pill-continuous', icon: '📈' },
    knn: { name: 'k-Nearest Neighbors (k-NN)', short: 'k-NN', tag: 'Instance-Based', pillClass: 'pill-ordinal', icon: '🎯' },
    nb: { name: 'Naive Bayes (Gaussian + Bernoulli)', short: 'Naive Bayes', tag: 'Probabilistic', pillClass: 'pill-nominal', icon: '🎲' },
    gbm: { name: 'Gradient-Boosted Trees', short: 'Gradient Boosting', tag: 'Boosting', pillClass: 'pill-continuous', icon: '🌿' },
    ensemble: { name: 'Voting / Weighted Ensemble', short: 'Ensemble', tag: 'Voting Ensemble', pillClass: 'pill-target', icon: '⚡' }
};
const ALL_MODEL_IDS = ['nn', 'rf', 'gbm', 'glm', 'knn', 'nb', 'ensemble'];

// Escapes text for safe insertion into HTML (column names and cell values come from user CSVs)
function escapeHtml(value) {
    return String(value === undefined || value === null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// Attributes for a control handled by the page's delegated listener (main.js): the global function
// window[fn] is called with these arguments (no inline event handlers, so the CSP can forbid them)
function actionAttrs(fn, ...args) {
    return `data-action="${escapeHtml(fn)}" data-args="${escapeHtml(JSON.stringify(args))}"`;
}

// Median of training outcomes: the exploratory 2×2 cutoff for continuous targets is fixed from
// TRAINING data so the test outcomes never influence the threshold
function trainingMedian(values) {
    const v = values.filter(x => typeof x === 'number' && isFinite(x)).sort((a, b) => a - b);
    if (v.length === 0) return null;
    const mid = Math.floor(v.length / 2);
    return v.length % 2 === 0 ? (v[mid - 1] + v[mid]) / 2 : v[mid];
}

// Exploratory 2×2 table for a continuous outcome. The cutoff is fixed from TRAINING data (never
// the test outcomes); it is not a validated clinical decision threshold, so it is labelled as such.
function renderExploratoryCutoffTable(ev, compact) {
    if (!ev || !ev.confMatrix || typeof ev.threshold !== 'number') return '';
    const cm = ev.confMatrix;
    const thresh = ev.threshold;
    const labels = [`< ${thresh}`, `≥ ${thresh}`];
    const pct = v => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : 'N/A');
    const pad = compact ? 'padding: 6px 8px; font-size: 10.5px;' : '';
    return `
        <div class="table-container" style="margin-top: 10px; ${compact ? '' : 'max-width: 500px;'}">
            <div style="font-size: ${compact ? 11 : 13}px; font-weight: 700; color: var(--text-secondary); margin-bottom: 5px;">
                Exploratory split at ${thresh} (${escapeHtml(ev.thresholdSource || 'training median')})
            </div>
            <table class="cm-table" style="${compact ? 'font-size: 11.5px; width: 100%;' : ''}">
                <thead>
                    <tr>
                        <th style="${pad}">Actual \\ Pred</th>
                        ${labels.map(c => `<th style="${pad}">${escapeHtml(c)}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${labels.map((actual, r) => `
                        <tr>
                            <th style="text-align: left; ${pad}">${escapeHtml(actual)}</th>
                            ${labels.map((_, c) => `<td class="${r === c ? 'cm-cell-hit' : 'cm-cell-miss'}" style="${pad} font-weight: 700;">${cm[r][c]}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
            <div style="display: flex; justify-content: space-between; gap: 8px; margin-top: 6px; font-size: 11px; color: var(--text-secondary); padding: 0 4px;">
                <span>Agreement: <strong>${pct(ev.thresholdAccuracy)}</strong></span>
                <span>"Sens" (≥ cutoff): <strong>${pct(ev.sensitivity)}</strong></span>
                <span>"Spec" (&lt; cutoff): <strong>${pct(ev.specificity)}</strong></span>
            </div>
            <div style="margin-top: 6px; padding: 6px 8px; background: var(--bg-surface-subtle); border-radius: var(--radius-sm); font-size: 10.5px; color: var(--text-muted); line-height: 1.35;">
                ⚠️ <em>Exploratory only: the cutoff is the training-data median, not a prespecified clinical threshold, so these are not diagnostic accuracy measures.</em>
            </div>
        </div>
    `;
}

// What a displayed class "probability" actually is, by model type
const PROBABILITY_LABELS = {
    model: { title: 'Model Probability', note: 'Model-estimated class probability' },
    blend: { title: 'Averaged Probability', note: 'Average of the base models\' class probabilities/vote shares' },
    vote: { title: 'Vote Share', note: 'Proportion of trees/models voting for this class' },
    neighbours: { title: 'Neighbour Vote Share', note: 'Share of the k nearest training cases (distance-weighted if selected) in this class' },
    recalibrated: { title: 'Recalibrated Probability', note: 'Model probability recalibrated on out-of-fold Training-set predictions' }
};

function probabilityLabel(type) {
    return PROBABILITY_LABELS[type] || PROBABILITY_LABELS.model;
}

// Display name for a trained model, flagging a neural network that fell back to a linear model
function displayModelName(modelId, model) {
    const base = (MODEL_INFO[modelId] && MODEL_INFO[modelId].name) || String(modelId).toUpperCase();
    return model && model.usedFallback ? `${base} ⚠️ linear fallback (not a neural network)` : base;
}

// Strict integer seed parsing: parseInt would read "1e6" as 1 and "42.5" as 42, silently using a
// different seed from the one typed. Returns null for anything that is not a plain integer.
function parseSeed(raw) {
    const t = String(raw === undefined || raw === null ? '' : raw).trim();
    return /^-?\d+$/.test(t) ? parseInt(t, 10) : null;
}

// Displays a stored numeric value compactly without fixed-decimal rounding (6 significant figures)
function fmtValue(v) {
    return typeof v === 'number' && isFinite(v) ? String(Number(v.toPrecision(6))) : String(v);
}

function fmtR2(v) {
    return typeof v === 'number' && isFinite(v) ? v.toFixed(3) : 'N/A';
}

// Plain-language summary of how a neural network was trained (validation split / early stopping)
function describeNNTraining(info) {
    if (!info) return '';
    const batchNote = info.batchSize && info.requestedBatchSize && info.batchSize > info.requestedBatchSize
        ? ` Mini-batch raised from ${info.requestedBatchSize} to ${info.batchSize} for ${info.fitCases.toLocaleString('en-GB')} cases to keep training time practical.`
        : '';
    return describeNNTrainingCore(info) + batchNote;
}

function describeNNTrainingCore(info) {
    if (!info.earlyStopping) {
        return `trained on all ${info.fitCases} training cases for ${info.maxEpochs} epochs (too few cases for an internal validation split; no early stopping).`;
    }
    return `${info.validationCases} training cases held out for validation; ` +
        (info.earlyStopped
            ? `early stopping at epoch ${info.stoppedEpoch}/${info.maxEpochs}, weights restored from epoch ${info.bestEpoch}.`
            : `ran all ${info.maxEpochs} epochs, best weights from epoch ${info.bestEpoch}.`) +
        ' The test set was not used during training.';
}

function cssEscape(value) {
    return (window.CSS && typeof window.CSS.escape === 'function') ? window.CSS.escape(value) : String(value).replace(/["\\]/g, '\\$&');
}

// Software version recorded in exported reports and settings files
const APP_VERSION = '2026.10.18';
// Stamped on every exported report, model file and checklist
const RESEARCH_USE_NOTICE = 'Investigational research and education use only. Not a certified medical device; not validated for individual clinical decisions.';

// Ranking policy (one policy for Step 4 and the 7-step pipeline; selectable by the user so the
// primary metric can match the prespecified study objective)
const RANKING_METRICS = {
    classification: {
        macroF1: { label: 'Macro F1', higherIsBetter: true },
        balancedAccuracy: { label: 'Balanced accuracy', higherIsBetter: true },
        rocAuc: { label: 'ROC-AUC (binary only)', higherIsBetter: true },
        brierScore: { label: 'Brier score (lower is better)', higherIsBetter: false },
        accuracy: { label: 'Accuracy', higherIsBetter: true }
    },
    regression: {
        r2: { label: 'R²', higherIsBetter: true },
        rmse: { label: 'RMSE (lower is better)', higherIsBetter: false },
        mae: { label: 'MAE (lower is better)', higherIsBetter: false }
    }
};

function rankingMetricKey(isClassification) {
    return isClassification ? (AppState.rankingMetricClf || 'macroF1') : (AppState.rankingMetricReg || 'r2');
}

function rankingMetricLabel(isClassification) {
    const group = isClassification ? RANKING_METRICS.classification : RANKING_METRICS.regression;
    return group[rankingMetricKey(isClassification)].label;
}

// Sorts best-first on the selected metric; missing values rank last; ties broken by macro-F1/AUC
// (classification) or R²/RMSE (regression).
function compareEvalResults(a, b, isClassification) {
    const key = rankingMetricKey(isClassification);
    const def = (isClassification ? RANKING_METRICS.classification : RANKING_METRICS.regression)[key];
    // Model selection by cross-validation inside the Training set: the Testing set is not used to rank
    if (a && b && a.cv && b.cv) {
        const cvv = (ev, k) => (typeof ev.cv[k] === 'number' && isFinite(ev.cv[k]) ? ev.cv[k] : null);
        const order = [key, ...(isClassification ? ['macroF1', 'rocAuc', 'accuracy'] : ['r2', 'rmse'])];
        for (const k2 of order) {
            const d2 = (isClassification ? RANKING_METRICS.classification : RANKING_METRICS.regression)[k2];
            const x = cvv(a, k2), y = cvv(b, k2);
            if (x === null && y !== null) return 1;
            if (y === null && x !== null) return -1;
            if (x !== null && y !== null && Math.abs(x - y) > 1e-12) return d2.higherIsBetter ? y - x : x - y;
        }
        return 0; // tie: keep the fixed model order (no peeking at the Testing set)
    }
    const val = ev => (typeof ev[key] === 'number' && isFinite(ev[key]) ? ev[key] : null);
    const va = val(a);
    const vb = val(b);
    if (va === null && vb !== null) return 1;
    if (vb === null && va !== null) return -1;
    if (va !== null && vb !== null && va !== vb) return def.higherIsBetter ? vb - va : va - vb;
    if (isClassification) {
        return ((b.macroF1 || 0) - (a.macroF1 || 0)) || ((b.rocAuc || 0) - (a.rocAuc || 0)) || ((b.accuracy || 0) - (a.accuracy || 0));
    }
    // Regression fallback: higher R² first; if R² is equal or undefined for both (constant test
    // outcomes), fall back to lower RMSE rather than declaring a tie
    const r2a = typeof a.r2 === 'number' && isFinite(a.r2) ? a.r2 : null;
    const r2b = typeof b.r2 === 'number' && isFinite(b.r2) ? b.r2 : null;
    if (r2a !== null && r2b !== null && r2a !== r2b) return r2b - r2a;
    if (r2a !== null && r2b === null) return -1;
    if (r2a === null && r2b !== null) return 1;
    return (a.rmse || 0) - (b.rmse || 0);
}

function summariseHeldOutSet(ev) {
    // Caveats about the size/composition of the evaluation set
    const notes = [];
    // testCount is the held-out set size; totalSamples (from evaluate()) is the same number
    const n = ev.testCount || ev.totalSamples || 0;
    if (n > 0 && n < 30) {
        notes.push(`Only ${n} test cases: performance estimates are imprecise${Array.isArray(ev.accuracyCI) && ev.accuracyCI.length === 2 ? ` (accuracy 95% CI ${(ev.accuracyCI[0] * 100).toFixed(0)}–${(ev.accuracyCI[1] * 100).toFixed(0)}%)` : ''}. Treat differences between models with caution.`);
    }
    if (ev.classesMissingFromTest && ev.classesMissingFromTest.length > 0) {
        notes.push(`Class(es) absent from the test set: ${ev.classesMissingFromTest.map(escapeHtml).join(', ')}. Their recall/F1 cannot be evaluated and are counted as 0 in macro averages.`);
    }
    return notes;
}

export { RESEARCH_USE_NOTICE, AppState, MODEL_INFO, ALL_MODEL_IDS, escapeHtml, actionAttrs, trainingMedian, renderExploratoryCutoffTable, PROBABILITY_LABELS, probabilityLabel, displayModelName, parseSeed, fmtValue, fmtR2, describeNNTraining, describeNNTrainingCore, cssEscape, APP_VERSION, RANKING_METRICS, rankingMetricKey, rankingMetricLabel, compareEvalResults, summariseHeldOutSet };
