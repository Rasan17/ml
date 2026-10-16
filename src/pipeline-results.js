// pipeline-results.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { AppState, MODEL_INFO, compareEvalResults, escapeHtml, fmtR2 } from './core.js';
import { executeImputation } from './imputation.js';
import { updateModelExportButton } from './model-io.js';
import { goToStep } from './navigation.js';
import { goToPipelineSubstep, renderPipelineLeaderboardOnly } from './pipeline.js';
import { renderPipelineVsBaselineComparison } from './reports.js';
import { DOM } from './shell.js';
import { renderSinglePredictionForm } from './single-prediction.js';
import { deployTestedModel, showSeedToast, startModelTraining } from './standard-training.js';


// ---------------------------------------------------------------------------------------------
// "Too good to be true?" check: near-perfect test results are explained in a balanced way
// (small test set, easily separable data, constrained models) and checked for common causes of
// spurious performance (a single predictor that almost restates the outcome; repeated patients).
// Uses the Training set only for the single-predictor check.
// ---------------------------------------------------------------------------------------------
const TOO_GOOD_THRESHOLD = 0.98;        // best model's accuracy / AUC / R² at or above this triggers the panel
const SINGLE_PREDICTOR_THRESHOLD = 0.95; // a predictor alone with AUC (or r²) at or above this is named

function singlePredictorSeparation(trainRows, featureCols, targetCol, configs, isClassification, positiveClass) {
    const found = [];
    if (!trainRows || trainRows.length < 4) return found;
    const toNum = v => (typeof v === 'number' ? v : Number(String(v).trim()));
    let y;
    if (isClassification) {
        const labels = trainRows.map(r => String(r[targetCol]));
        const distinct = [...new Set(labels)];
        if (distinct.length !== 2) return found; // binary outcomes only
        const pos = positiveClass !== null && positiveClass !== undefined && distinct.includes(String(positiveClass)) ? String(positiveClass) : distinct.sort()[1];
        y = labels.map(l => (l === pos ? 1 : 0));
    } else {
        y = trainRows.map(r => toNum(r[targetCol]));
    }
    featureCols.forEach(col => {
        const raw = trainRows.map(r => r[col]);
        const numeric = raw.every(v => v !== null && v !== undefined && String(v).trim() !== '' && !isNaN(toNum(v)));
        let score;
        if (numeric && (configs[col]?.type !== 'nominal')) {
            score = raw.map(toNum);
        } else {
            // Categories: score each level by its outcome rate (classification) or mean (regression) in the Training set
            const sums = {}, counts = {};
            raw.forEach((v, i) => { const k = String(v); sums[k] = (sums[k] || 0) + y[i]; counts[k] = (counts[k] || 0) + 1; });
            score = raw.map(v => sums[String(v)] / counts[String(v)]);
        }
        if (isClassification) {
            const auc = ModelEngine._calculateBinaryAUC(y, score);
            if (auc === null) return;
            const a = Math.max(auc, 1 - auc);
            if (a >= SINGLE_PREDICTOR_THRESHOLD) found.push({ col, label: `AUC ${a.toFixed(2)}`, value: a });
        } else {
            const n = y.length;
            const mx = score.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
            let sxy = 0, sxx = 0, syy = 0;
            for (let i = 0; i < n; i++) { sxy += (score[i] - mx) * (y[i] - my); sxx += (score[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
            if (sxx === 0 || syy === 0) return;
            const r2 = (sxy * sxy) / (sxx * syy);
            if (r2 >= SINGLE_PREDICTOR_THRESHOLD) found.push({ col, label: `r² ${r2.toFixed(2)}`, value: r2 });
        }
    });
    return found.sort((a, b) => b.value - a.value);
}

function countRepeatedPatients(trainRows, testRows, featureCols) {
    if (!trainRows || !testRows) return 0;
    const key = r => featureCols.map(c => String(r[c])).join('\u241F');
    const seen = new Set(trainRows.map(key));
    return testRows.filter(r => seen.has(key(r))).length;
}

function renderTooGoodToBeTrueCheck(sorted, isClassification) {
    const best = sorted && sorted[0];
    if (!best) return '';
    const ev = best.eval;
    const p = AppState.pipeline;
    const metrics = [];
    let tooGood = false;
    if (isClassification) {
        if (typeof ev.accuracy === 'number' && ev.accuracy >= TOO_GOOD_THRESHOLD) tooGood = true;
        if (typeof ev.rocAuc === 'number' && ev.rocAuc >= TOO_GOOD_THRESHOLD) tooGood = true;
        if (typeof ev.accuracy === 'number') metrics.push(`test accuracy ${(ev.accuracy * 100).toFixed(1)}%`);
        if (typeof ev.rocAuc === 'number') metrics.push(`AUC ${ev.rocAuc.toFixed(3)}`);
        if (typeof ev.brierScore === 'number') metrics.push(`Brier ${ev.brierScore.toFixed(3)}`);
    } else if (typeof ev.r2 === 'number') {
        if (ev.r2 >= TOO_GOOD_THRESHOLD) tooGood = true;
        metrics.push(`R² ${ev.r2.toFixed(3)}`);
    }
    if (!tooGood) return '';

    const activeCols = (p.featureCols || []).filter(c => !p.droppedFeatures.has(c));
    const nTest = ev.testCount || ev.totalSamples || (p.testRows ? p.testRows.length : 0);
    const nTrain = p.trainRows ? p.trainRows.length : 0;
    const separators = singlePredictorSeparation(p.trainRows, activeCols, p.targetCol, p.columnConfigs, isClassification, ev.positiveClass);
    const repeated = countRepeatedPatients(p.trainRows, p.testRows, activeCols);
    const ci = Array.isArray(ev.accuracyCI) && ev.accuracyCI.length === 2
        ? ` The 95% confidence interval for accuracy is ${(ev.accuracyCI[0] * 100).toFixed(0)}–${(ev.accuracyCI[1] * 100).toFixed(0)}%, so the true accuracy could be lower.` : '';
    const nPerfect = sorted.filter(it => isClassification ? it.eval.accuracy >= TOO_GOOD_THRESHOLD : it.eval.r2 >= TOO_GOOD_THRESHOLD).length;

    const reasons = [];
    if (nTest > 0 && nTest < 100) {
        reasons.push(`<strong>Small samples:</strong> ${nTrain} training and ${nTest} test patients. Getting every one of a few test patients right is far easier than with hundreds.${ci}`);
    }
    if (separators.length > 0) {
        reasons.push(`<strong>Easily separable data:</strong> in the Training set, ${separators.length === 1 ? 'one predictor alone almost perfectly separates' : `${separators.length} predictors each alone almost perfectly separate`} the outcome${isClassification ? ' groups' : ''}: ${separators.slice(0, 5).map(s => `<strong>${escapeHtml(s.col)}</strong> (${s.label})`).join(', ')}${separators.length > 5 ? ` and ${separators.length - 5} more` : ''}. When one measurement clearly divides the groups, any sensible model will score near-perfectly.`);
    }
    if (nPerfect > 1) {
        reasons.push(`<strong>Agreement across models:</strong> ${nPerfect} of ${sorted.length} quite different models score near-perfectly. That points to a strong signal in the data rather than one model memorising its training patients.`);
    }
    reasons.push(`<strong>Constrained models, unseen test patients:</strong> the models were limited (feature pruning, limited tree depth, ridge/L2 penalties) and scored only on real patients they never saw. Classic overfitting shows up as <em>poor</em> test results, so a perfect test score is not in itself a sign of overfitting.`);

    const checks = [];
    checks.push(`<strong>Data leakage:</strong> make sure no predictor was recorded after the outcome, or restates it (e.g. a post-operative score used to predict shunt response)${separators.length ? `; look especially at ${separators.slice(0, 3).map(s => `<strong>${escapeHtml(s.col)}</strong>`).join(', ')}` : ''}. If in doubt, exclude the variable (Step 4) and re-run: a large drop in performance suggests it was leaking the answer.${separators.length > 2 ? ' Here several predictors each separate the groups on their own, so removing one will change little; that pattern is more typical of very clean or simulated data (such as the sample cohorts) than of leakage through a single variable.' : ''}`);
    checks.push(repeated > 0
        ? `<strong>Repeated patients:</strong> ⚠️ ${repeated} test patient(s) have exactly the same predictor values as a training patient. If these are the same people (e.g. repeat visits), the test is not independent; keep each patient in one set only.`
        : `<strong>Repeated patients:</strong> ✓ no test patient has exactly the same predictor values as a training patient.`);
    checks.push(`<strong>Independent confirmation:</strong> test the model on a separate cohort (Option B: separate test file), ideally from another time period or hospital, before relying on it clinically.`);

    return `
        <div class="too-good-box" role="note">
            <div class="too-good-title">🔎 Too good to be true? Near-perfect test performance (best model, ${escapeHtml(best.name)}: ${metrics.join(', ')})</div>
            <p>Near-perfect results are uncommon with real clinical data, but they are <strong>not necessarily wrong</strong>. In this analysis they are plausible because:</p>
            <ul>${reasons.map(r => `<li>${r}</li>`).join('')}</ul>
            <p>Before relying on the result, check:</p>
            <ul>${checks.map(c => `<li>${c}</li>`).join('')}</ul>
        </div>`;
}

function renderPipelineBenchmarkResults(benchmarkList, targetClasses, isClassification) {
    // Sort best-first (same criterion as the leaderboard)
    const sorted = [...benchmarkList].sort((a, b) => compareEvalResults(a.eval, b.eval, isClassification));

    // 1. Leaderboard Table
    renderPipelineLeaderboardOnly();

    // 2. 2x2 Confusion Matrices Grid
    let cmHtml = '';
    sorted.forEach(item => {
        const ev = item.eval;
        const cm = ev.confMatrix;

        if (!isClassification) {
            // Regression: report error metrics (a median-split 2×2 table is shown in Step 4 only as a descriptive aid)
            cmHtml += `
                <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <strong style="font-size: 13.5px; color: var(--text-primary);">${escapeHtml(item.name)}</strong>
                        <span class="pill pill-continuous" style="font-size: 10.5px;">R²: ${fmtR2(ev.r2)}</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 10px;">
                        ${escapeHtml(ev.featuresUsed)} &bull; ${escapeHtml(ev.constraintNote)}
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; font-size: 11px; text-align: center;">
                        <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">RMSE: <strong>${ev.rmse}</strong></div>
                        <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">MAE: <strong>${ev.mae}</strong></div>
                        <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">MSE: <strong>${ev.mse}</strong></div>
                    </div>
                </div>
            `;
        } else if (cm && cm.length === 2) {
            // Positive class from the evaluation result (set in Step 2); index lookup keeps TP/FN/FP/TN
            // correct even if the class order ever differs from [negative, positive]
            const posLabel = ev.positiveClass !== null && ev.positiveClass !== undefined ? ev.positiveClass : targetClasses[1];
            const posIdx = targetClasses.indexOf(posLabel) === 0 ? 0 : 1;
            const negIdx = 1 - posIdx;
            const negLabel = targetClasses[negIdx];
            const tn = cm[negIdx][negIdx];
            const fp = cm[negIdx][posIdx];
            const fn = cm[posIdx][negIdx];
            const tp = cm[posIdx][posIdx];
            // Undefined ratios (zero denominator) are shown as N/A rather than 0%
            const sens = (tp + fn) > 0 ? ((tp / (tp + fn)) * 100).toFixed(1) + '%' : 'N/A';
            const spec = (tn + fp) > 0 ? ((tn / (tn + fp)) * 100).toFixed(1) + '%' : 'N/A';
            const ppv = (tp + fp) > 0 ? ((tp / (tp + fp)) * 100).toFixed(1) + '%' : 'N/A';
            const npv = (tn + fn) > 0 ? ((tn / (tn + fn)) * 100).toFixed(1) + '%' : 'N/A';

            cmHtml += `
                <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <strong style="font-size: 13.5px; color: var(--text-primary);">${escapeHtml(item.name)}</strong>
                        <span class="pill pill-continuous" style="font-size: 10.5px;">Acc: ${(ev.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 10px;">
                        ${escapeHtml(ev.featuresUsed)} &bull; ${escapeHtml(ev.constraintNote)}
                    </div>
                    <div style="font-size: 10.5px; color: var(--text-muted); margin-bottom: 6px;">
                        "Positive" = <strong>${escapeHtml(posLabel)}</strong> (the positive/event class chosen in Step 2); check this matches your clinical event of interest.
                    </div>
                    <div class="cm-table-wrapper" style="margin-bottom: 10px;">
                        <table class="cm-table">
                            <thead>
                                <tr>
                                    <th>Actual \\ Pred</th>
                                    <th>Negative (${escapeHtml(negLabel)})</th>
                                    <th>Positive (${escapeHtml(posLabel)})</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Negative (${escapeHtml(negLabel)})</strong></td>
                                    <td class="cm-cell-tn">TN: <strong>${tn}</strong></td>
                                    <td class="cm-cell-fp">FP: <strong>${fp}</strong></td>
                                </tr>
                                <tr>
                                    <td><strong>Positive (${escapeHtml(posLabel)})</strong></td>
                                    <td class="cm-cell-fn">FN: <strong>${fn}</strong></td>
                                    <td class="cm-cell-tp">TP: <strong>${tp}</strong></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; font-size: 11px; text-align: center;">
                        <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">Sens: <strong>${sens}</strong></div>
                        <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">Spec: <strong>${spec}</strong></div>
                        <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">PPV: <strong>${ppv}</strong></div>
                        <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">NPV: <strong>${npv}</strong></div>
                    </div>
                </div>
            `;
        } else if (cm && cm.length > 2) {
            // Multiclass outcome: full K×K matrix (rows = actual, columns = predicted)
            cmHtml += `
                <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <strong style="font-size: 13.5px; color: var(--text-primary);">${escapeHtml(item.name)}</strong>
                        <span class="pill pill-continuous" style="font-size: 10.5px;">Acc: ${(ev.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 10px;">
                        ${escapeHtml(ev.featuresUsed)} &bull; ${escapeHtml(ev.constraintNote)}
                    </div>
                    <div class="cm-table-wrapper" style="margin-bottom: 10px; overflow-x: auto;">
                        <table class="cm-table" style="font-size: 11.5px;">
                            <thead>
                                <tr>
                                    <th>Actual \\ Pred</th>
                                    ${targetClasses.map(c => `<th>${escapeHtml(c)}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${targetClasses.map((actual, r) => `
                                    <tr>
                                        <th style="text-align: left;">${escapeHtml(actual)}</th>
                                        ${targetClasses.map((_, c) => `<td class="${r === c ? 'cm-cell-hit' : 'cm-cell-miss'}">${cm[r] && cm[r][c] !== undefined ? cm[r][c] : 0}</td>`).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                    <div style="font-size: 11px; color: var(--text-secondary);">
                        Macro F1: <strong>${(ev.macroF1 * 100).toFixed(1)}%</strong> &bull;
                        Balanced accuracy: <strong>${typeof ev.balancedAccuracy === 'number' ? (ev.balancedAccuracy * 100).toFixed(1) + '%' : 'N/A'}</strong>
                        (computed over all ${targetClasses.length} classes; sensitivity/PPV apply per class, not to the whole table)
                    </div>
                </div>
            `;
        }
    });

    if (DOM.pipelineCmGrid) DOM.pipelineCmGrid.innerHTML = cmHtml;

    // Auto-deploy top-performing model if nothing deployed yet or previous deployment was from pipeline
    if (!AppState.trainedModel || AppState.deployedSource === '7step') {
        deployPipelineModel(sorted[0].id, false, true); // skipRerender = true to prevent recursion
    } else {
        updatePipelineActiveDeploymentStatus();
        updatePredictionProvenanceBar();
    }
}

function deployPipelineModel(modelId, showNotice = true, skipRerender = false) {
    if (!AppState.pipeline.benchmarkResults) return;
    const targetItem = AppState.pipeline.benchmarkResults.find(b => b.id === modelId);
    if (!targetItem) return;

    AppState.trainedModel = targetItem.model;
    AppState.prepMeta = targetItem.prep;
    AppState.evalResults = targetItem.eval;
    AppState.deployedModelId = targetItem.id;
    AppState.deployedSource = '7step';
    AppState.deployedPipelineModel = targetItem;

    // Enable Step 5 & Step 6 buttons
    if (DOM.toStep5Btn) DOM.toStep5Btn.disabled = false;
    if (DOM.toStep6Btn) DOM.toStep6Btn.disabled = false;

    updatePipelineActiveDeploymentStatus();
    updatePredictionProvenanceBar();
    renderSinglePredictionForm();

    // Update leaderboard row styling without recursive mutual autoDeploy
    if (!skipRerender) {
        renderPipelineLeaderboardOnly();
    }

    if (showNotice) {
        showSeedToast(`✓ Deployed "${targetItem.name}" as the active predictor for Steps 5 & 6`);
    }
}
window.deployPipelineModel = deployPipelineModel;

function autoDeployBestPipelineModel() {
    if (!AppState.pipeline.benchmarkResults || AppState.pipeline.benchmarkResults.length === 0) return;
    const isClassification = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous';
    const sorted = [...AppState.pipeline.benchmarkResults].sort((a, b) => compareEvalResults(a.eval, b.eval, isClassification));
    deployPipelineModel(sorted[0].id, false);
}

function updatePipelineActiveDeploymentStatus() {
    if (!DOM.pipelineActiveDeploymentStatus) return;
    if (AppState.deployedPipelineModel) {
        DOM.pipelineActiveDeploymentStatus.innerHTML = `
            Currently Active Deployed Model: <strong style="color: var(--accent-emerald);">${escapeHtml(AppState.deployedPipelineModel.name)}</strong> 
            (${escapeHtml(AppState.deployedPipelineModel.eval.featuresUsed)}). 
            ${pipelineProvenanceSummary()} &bull; Ready for single-case and batch predictions (research use).
        `;
    } else {
        DOM.pipelineActiveDeploymentStatus.innerHTML = `Select any model from the leaderboard to deploy for real-time patient predictions.`;
    }
}

// Factual one-line summary of what the 7-step pipeline actually did
function pipelineProvenanceSummary() {
    const p = AppState.pipeline;
    const parts = [p.completeCase
        ? `Complete-case analysis (${p.excludedIncomplete} incomplete row(s) removed); scaling fitted on training split only`
        : 'Imputation & scaling fitted on training split only'];
    const vifDropped = (p.vifResults || []).filter(v => v.highVIF && p.droppedFeatures.has(v.feature)).length;
    parts.push(p.droppedFeatures.size > 0 ? `${p.droppedFeatures.size} feature(s) excluded (correlation/VIF/univariate review)` : 'no features excluded');
    if (vifDropped > 0) parts.push(`${vifDropped} with VIF > 5`);
    if (p.smotencResult) {
        parts.push(p.smotencResult.applied ? `SMOTENC applied (+${p.smotencResult.syntheticCountGenerated} synthetic training rows)` : 'SMOTENC not applied');
    }
    return parts.join(' &bull; ');
}

function updatePredictionProvenanceBar() {
    updateModelExportButton();
    const singleBar = DOM.singlePredictProvenanceBar;
    const batchBar = DOM.batchPredictProvenanceBar;
    if (!singleBar && !batchBar) return;

    const hasPipeline = AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0;
    const hasBaseline = AppState.allTestedResults && AppState.allTestedResults.length > 0;

    let contentHtml = '';
    if (AppState.deployedSource === 'imported' && AppState.importedModelInfo) {
        const m = AppState.importedModelInfo;
        contentHtml = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 22px;">📥</span>
                <div>
                    <div style="font-size: 13.5px; font-weight: 700; color: var(--accent-cyan);">Active Predictor: ${escapeHtml(m.name)} &bull; Imported model file</div>
                    <div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 2px;">Trained on ${escapeHtml(m.dataset || 'an unnamed dataset')}; exported ${escapeHtml((m.exportedAt || '').slice(0, 10))} with app version ${escapeHtml(m.version || '?')}. Uses the preprocessing fitted during training.</div>
                </div>
            </div>`;
        [singleBar, batchBar].forEach(bar => {
            if (!bar) return;
            bar.className = 'pipeline-alert pipeline-alert-info';
            bar.style.display = 'flex';
            bar.innerHTML = contentHtml;
        });
    } else if (AppState.deployedSource === '7step' && AppState.deployedPipelineModel) {
        const m = AppState.deployedPipelineModel;
        const featCount = AppState.prepMeta ? AppState.prepMeta.featureCols.length : (AppState.pipeline.top70Features?.length || 'Optimal');
        contentHtml = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 22px;">🔬</span>
                <div>
                    <div style="font-size: 13.5px; font-weight: 700; color: var(--accent-emerald);">
                        Active Predictor: ${escapeHtml(m.name)} &bull; Trained via 7-Step Clinical Pipeline
                    </div>
                    <div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 2px;">
                        ${pipelineProvenanceSummary()} &bull; ${featCount} active features
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                ${hasBaseline ? `<button class="btn btn-secondary btn-sm" onclick="window.switchToBaselineModel()">🔄 Switch to Baseline Model</button>` : ''}
                <button class="btn btn-secondary btn-sm" onclick="window.goToStep && window.goToStep('pipeline')">🔬 View 7-Step Pipeline</button>
            </div>
        `;
        if (singleBar) {
            singleBar.className = 'pipeline-alert pipeline-alert-success';
            singleBar.style.display = 'flex';
            singleBar.innerHTML = contentHtml;
        }
        if (batchBar) {
            batchBar.className = 'pipeline-alert pipeline-alert-success';
            batchBar.style.display = 'flex';
            batchBar.innerHTML = contentHtml;
        }
    } else if (AppState.trainedModel) {
        const mInfo = MODEL_INFO[AppState.deployedModelId] || { name: (AppState.deployedModelId || 'ML').toUpperCase() };
        const featCount = AppState.prepMeta ? AppState.prepMeta.featureCols.length : 'All';
        contentHtml = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 22px;">📋</span>
                <div>
                    <div style="font-size: 13.5px; font-weight: 700; color: var(--accent-cyan);">
                        Active Predictor: ${mInfo.name} &bull; Trained via Pre-existing Standard Workflow
                    </div>
                    <div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 2px;">
                        Standard Workflow &bull; ${AppState.evalResults && AppState.evalResults.protocol === 'option-b' ? 'Preprocessing fitted on Dataset (i)' : (AppState.evalResults && AppState.evalResults.retrainOn100 ? 'Final model refitted on the full cohort after split-first evaluation' : 'Preprocessing fitted on the training split only')} &bull; ${featCount} features
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                ${hasPipeline ? `<button class="btn btn-emerald btn-sm" onclick="window.switchToPipelineModel()">🔬 Switch to 7-Step Pipeline Model</button>` : `<button class="btn btn-emerald btn-sm" onclick="window.goToStep && window.goToStep('pipeline')">🔬 Test 7-Step Pipeline to Compare</button>`}
            </div>
        `;
        if (singleBar) {
            singleBar.className = 'pipeline-alert pipeline-alert-info';
            singleBar.style.display = 'flex';
            singleBar.innerHTML = contentHtml;
        }
        if (batchBar) {
            batchBar.className = 'pipeline-alert pipeline-alert-info';
            batchBar.style.display = 'flex';
            batchBar.innerHTML = contentHtml;
        }
    } else {
        if (singleBar) singleBar.style.display = 'none';
        if (batchBar) batchBar.style.display = 'none';
    }
}

window.switchToBaselineModel = function () {
    if (!AppState.allTestedResults || AppState.allTestedResults.length === 0) {
        alert("No baseline model found. Train a model in Step 4 first.");
        return;
    }
    const bestBaseline = AppState.allTestedResults[0];
    deployTestedModel(bestBaseline.modelId);
    updatePredictionProvenanceBar();
};

window.switchToPipelineModel = function () {
    if (!AppState.pipeline.benchmarkResults || AppState.pipeline.benchmarkResults.length === 0) {
        alert("7-Step Pipeline has not been run yet. Please run the 7-Step Pipeline first.");
        if (window.goToStep) window.goToStep('pipeline');
        return;
    }
    const bestPipeline = AppState.pipeline.benchmarkResults[0];
    deployPipelineModel(bestPipeline.id, true);
};

function togglePipelineBaselineComparison() {
    if (!DOM.pipelineVsBaselinePanel) return;
    const isHidden = DOM.pipelineVsBaselinePanel.style.display === 'none';
    if (isHidden) {
        DOM.pipelineVsBaselinePanel.style.display = 'block';
        renderPipelineVsBaselineComparison();
        DOM.pipelineVsBaselinePanel.scrollIntoView({ behavior: 'smooth' });
    } else {
        DOM.pipelineVsBaselinePanel.style.display = 'none';
    }
}

// Full-screen "Comparison is underway" message that blocks other buttons until the comparison finishes.
// It mirrors the training progress and offers a Stop button (the Step 4 Force Stop is hidden behind it).
function showComparisonOverlay() {
    let overlay = document.getElementById('comparison-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'comparison-overlay';
        overlay.className = 'comparison-overlay';
        overlay.setAttribute('role', 'alertdialog');
        overlay.setAttribute('aria-modal', 'true');
        overlay.setAttribute('aria-labelledby', 'comparison-overlay-title');
        overlay.innerHTML = `
            <div class="comparison-overlay-card">
                <div class="comparison-overlay-spinner" aria-hidden="true"></div>
                <div id="comparison-overlay-title" class="comparison-overlay-title">⚖️ Comparison is underway</div>
                <p class="comparison-overlay-text">The standard baseline is being trained so it can be compared with the 7-step pipeline. Please do not click other buttons until it finishes; you will be taken to the comparison automatically.</p>
                <div id="comparison-overlay-progress" class="comparison-overlay-progress" aria-live="polite"></div>
                <button type="button" id="comparison-overlay-stop" class="btn btn-secondary btn-sm" style="color: var(--accent-rose); border-color: rgba(244,63,94,0.4);">🛑 Stop comparison</button>
            </div>`;
        document.body.appendChild(overlay);
        overlay.querySelector('#comparison-overlay-stop').addEventListener('click', () => {
            const wd = AppState.watchdogs ? AppState.watchdogs.step4 : null;
            if (wd) wd.forceAbort('Comparison stopped by user.');
        });
        // Keep keyboard focus inside the dialog
        overlay.addEventListener('keydown', e => { if (e.key === 'Tab') { e.preventDefault(); overlay.querySelector('#comparison-overlay-stop').focus(); } });
    }
    overlay.style.display = 'flex';
    document.body.classList.add('comparison-running');
    overlay.querySelector('#comparison-overlay-stop').focus();
    const progress = overlay.querySelector('#comparison-overlay-progress');
    clearInterval(overlay._timer);
    overlay._timer = setInterval(() => {
        const txt = DOM.trainStatus ? DOM.trainStatus.innerText.trim() : '';
        if (progress.textContent !== txt) progress.textContent = txt;
    }, 400);
}

function hideComparisonOverlay() {
    const overlay = document.getElementById('comparison-overlay');
    if (!overlay) return;
    clearInterval(overlay._timer);
    overlay.style.display = 'none';
    document.body.classList.remove('comparison-running');
}

// "Train baseline & compare": runs the Step 4 standard workflow with the pipeline's split (so both are
// tested on the same patients where possible), then returns to Step 7 and shows the comparison
window.trainStandardBaselineFromPipeline = async function () {
    if (!AppState.parsedData || !AppState.parsedData.rows || AppState.parsedData.rows.length === 0) {
        alert("No dataset loaded yet. Please select a sample cohort or upload a CSV first.");
        return;
    }

    // Match the pipeline's train/test split: Option A, same ratio and seed
    const valA = document.getElementById('val-mode-a');
    if (valA && !valA.checked) { valA.checked = true; valA.dispatchEvent(new Event('change', { bubbles: true })); }
    const pct = Math.round((AppState.pipeline.trainRatio || 0.8) * 100);
    const pctInput = document.getElementById('train-pct-input');
    if (pctInput) { pctInput.value = pct; pctInput.dispatchEvent(new Event('input', { bubbles: true })); pctInput.dispatchEvent(new Event('change', { bubbles: true })); }
    const seed = AppState.pipeline.seed;
    const seedMode = document.getElementById(seed === 42 ? 'split-seed-mode-default' : 'split-seed-mode-custom');
    if (seedMode) { seedMode.checked = true; seedMode.dispatchEvent(new Event('change', { bubbles: true })); }
    const seedInput = document.getElementById('split-seed-input');
    if (seedInput) seedInput.value = seed;

    // Step 3 (data imputation) with the current choices, if it has not been run yet
    if (!AppState.imputedData) {
        try { executeImputation(); } catch (err) { console.warn("Auto-imputation error:", err); }
        if (!AppState.imputedData) { goToStep(3); return; }
    }

    // Remember the deployed pipeline model so it stays active after the baseline run
    const keepPipelineDeployment = AppState.deployedSource === '7step' ? AppState.deployedModelId : null;
    goToStep(4);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">⚖️ Training the standard baseline for the comparison (same ${pct}/${100 - pct} split and seed ${seed} as the pipeline). You will be returned to the comparison when it finishes...</span>`;

    showComparisonOverlay();
    const shownAt = Date.now();
    let ok = false;
    try {
        ok = await startModelTraining();
        // Keep the message up briefly even when training is very fast, so the user sees what happened
        const remaining = 1500 - (Date.now() - shownAt);
        if (ok && remaining > 0) {
            const progress = document.getElementById('comparison-overlay-progress');
            if (progress) progress.textContent = '✓ Baseline trained. Opening the comparison...';
            await new Promise(r => setTimeout(r, remaining));
        }
    } finally {
        hideComparisonOverlay();
    }
    if (!ok) return; // training failed or was stopped: stay on Step 4 so the message is visible

    if (keepPipelineDeployment && AppState.pipeline.benchmarkResults.some(b => b.id === keepPipelineDeployment)) {
        deployPipelineModel(keepPipelineDeployment, false);
    }
    goToStep('pipeline');
    goToPipelineSubstep(7);
    if (DOM.pipelineVsBaselinePanel) DOM.pipelineVsBaselinePanel.style.display = 'block';
    AppState.baselineJustTrained = true;
    renderPipelineVsBaselineComparison();
    AppState.baselineJustTrained = false;
    setTimeout(() => { if (DOM.pipelineVsBaselinePanel) DOM.pipelineVsBaselinePanel.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 200);
    showSeedToast(`✓ Standard baseline trained; comparison updated.${keepPipelineDeployment ? ' The pipeline model remains the active predictor.' : ''}`);
};

export { TOO_GOOD_THRESHOLD, SINGLE_PREDICTOR_THRESHOLD, singlePredictorSeparation, countRepeatedPatients, renderTooGoodToBeTrueCheck, renderPipelineBenchmarkResults, deployPipelineModel, autoDeployBestPipelineModel, updatePipelineActiveDeploymentStatus, pipelineProvenanceSummary, updatePredictionProvenanceBar, togglePipelineBaselineComparison, showComparisonOverlay, hideComparisonOverlay };
