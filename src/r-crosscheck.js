// r-crosscheck.js: "R cross-check script" for the analysis on screen. Collects the analysis description (no
// patient data) and the app's results, and downloads the R script built by r-script.js, which rebuilds the analysis
// independently from the user's CSV file and prints both sets of results side by side.
import { APP_VERSION, AppState } from './core.js';
import { readMultipleImputation } from './multiple-imputation.js';
import { SURV } from './survival.js';
import { survivalSpec } from './outcome-type.js';
import { buildRScript } from './r-script.js';

const fin = v => (typeof v === 'number' && isFinite(v) ? v : null);

// Predictors with their coding and missing-data method, as configured in Steps 2 and 3
function predictorSpecs(prep) {
    return prep.featureCols.map(col => {
        const cfg = AppState.columnConfigs[col] || {};
        const m = prep.featureMetadata[col] || {};
        return { name: col, type: m.type || cfg.type, impute: AppState.completeCase ? 'drop' : (cfg.imputationStrategy || 'median'), constant: cfg.customImputeVal ?? null,
            levels: m.type === 'ordinal' ? (m.ranks || []).map(String) : null, derivedFrom: cfg.derivedFrom || null };
    });
}

function missingSpec() {
    const m = AppState.completeCase ? 0 : readMultipleImputation();
    return { missing: AppState.completeCase ? 'complete' : m ? 'mice' : 'single', miceM: m || null };
}

function validationOf(mode) { return mode === 'option-c' || mode === 'bootstrap' ? 'C' : mode === 'option-b' || mode === 'separate' ? 'B' : 'A'; }

function standardSpec() {
    const prep = AppState.prepMeta;
    const results = AppState.allTestedResults || [];
    const glm = results.find(r => r.modelId === 'glm');
    if (!prep || !glm) throw new Error('Train the logistic / linear regression model (GLM) in Step 4 first: the R cross-check reproduces that model, which R can fit identically.');
    if (prep.isClassification && (prep.targetClasses || []).length !== 2) throw new Error('The R cross-check covers yes/no and numerical outcomes (and time-to-event outcomes); this outcome has more than two categories.');
    const ev = glm.evalRes, validation = validationOf(ev.protocol);
    const oc = ev.optimismCorrection || null;
    const binary = prep.isClassification;
    const y = Array.isArray(ev.actuals) ? ev.actuals : [];
    const app = binary ? {
        n: y.length || null, events: y.length ? y.filter(v => v === 1).length : null,
        auc: fin(oc ? oc.apparent && oc.apparent.rocAuc : ev.rocAuc), auc_ci: validation === 'C' ? null : (ev.ci && ev.ci.rocAuc) || null,
        brier: fin(oc ? oc.apparent && oc.apparent.brierScore : ev.brierScore),
        slope: fin(ev.calibration && ev.calibration.slope), intercept: fin(ev.calibration && ev.calibration.intercept),
        auc_corrected: fin(oc && oc.harrell ? oc.harrell.rocAuc : null), slope_corrected: fin(oc ? oc.calibrationSlope : null)
    } : { n: y.length || null, r2: fin(ev.r2), rmse: fin(ev.rmse), mae: fin(ev.mae) };
    return {
        outcomeType: binary ? 'binary' : 'continuous', outcome: prep.targetCol,
        positive: binary ? String(prep.targetIdxToClass ? prep.targetIdxToClass[1] : prep.targetClasses[1]) : null,
        predictors: predictorSpecs(prep), ...missingSpec(), validation,
        testRows: validation === 'A' ? AppState.lastSplitTestRowNumbers || [] : null,
        testCheck: validation === 'A' && y.length ? { n: y.length, events: binary ? y.filter(v => v === 1).length : null } : null,
        B: oc ? oc.B : null, seed: glm.hyperparams && glm.hyperparams.seed, lambda: glm.model && typeof glm.model.lambdaL2 === 'number' ? glm.model.lambdaL2 : 0.01,
        externalFile: validation === 'B' ? AppState.separateTestFileName : null, modelName: 'Logistic / ridge regression (GLM)', app
    };
}

function survivalSpecForR() {
    const r = SURV.result, spec = survivalSpec();
    if (!r || !spec) throw new Error('Fit the time-to-event model in Step 4 first.');
    const validation = validationOf(r.validation);
    return {
        outcomeType: 'survival', timeCol: r.tc, eventCol: r.ec, eventValue: r.ev, competingValue: r.competing ? r.cv : null, horizon: r.horizon,
        predictors: predictorSpecs(r.prep), ...missingSpec(), validation,
        testRows: validation === 'A' ? r.testRowNumbers || [] : null,
        testCheck: validation === 'A' ? { n: r.nTest, events: r.eventsTest } : null,
        B: r.optimism ? r.optimism.B : null, seed: r.seed, externalFile: validation === 'B' ? AppState.separateTestFileName : null,
        modelName: r.competing ? 'Cause-specific Cox models (competing risks)' : 'Cox proportional-hazards model',
        app: { n: r.nTest, events: r.eventsTest, c_index: fin(r.cIndex), td_auc: fin(r.tdAuc), brier: fin(r.brier), observed: fin(r.obsAll), predicted: fin(r.meanPred),
            c_corrected: fin(r.optimism && r.optimism.cCorrected), slope_corrected: r.competing ? null : fin(r.optimism && r.optimism.slope) }
    };
}

function downloadRCrossCheck(kind) {
    try {
        const spec = kind === 'survival' ? survivalSpecForR() : standardSpec();
        const date = new Date().toISOString().slice(0, 10);
        const base = String(AppState.sourceFile || AppState.fileName || 'analysis').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_').slice(0, 40);
        const scriptName = `crosscheck_${base}_${date}.R`;
        const text = buildRScript({ ...spec, appVersion: APP_VERSION, generated: date, dataFile: AppState.sourceFile || AppState.fileName, nRows: AppState.parsedData.rows.length, scriptName });
        const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
        const a = document.createElement('a');
        a.href = url; a.download = scriptName;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (e) {
        alert(e.message || String(e));
    }
}

// The survival results panel calls this (its button is created there)
SURV.onRCheck = () => downloadRCrossCheck('survival');

export { downloadRCrossCheck };
