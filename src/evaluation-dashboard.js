// evaluation-dashboard.js: part of the Universal Medical ML/NeuralNet Predictor application controller (split from standard-training.js)
import { AppState, MODEL_INFO, actionAttrs, displayModelName, escapeHtml, fmtR2, rankingMetricLabel, renderExploratoryCutoffTable, summariseHeldOutSet } from './core.js';
import { ciScopeNote, ciSpan, describeSplit, formatCvScore } from './cv-tuning.js';
import { renderCalibrationPanel, renderExplainPanel, renderOddsRatioPanel, renderOptimismPanel, renderMiceDiagnostics } from './panels.js';
import { DOM } from './shell.js';
import { renderSubgroupPanel } from './subgroups.js';
import { syncRankingSelectors } from './training-actions.js';
import { renderInterpretation } from './interpretation.js';

// Accuracy interval: the bootstrap interval when one was computed (it resamples patients, not rows, when a
// patient-ID column is set, so it respects repeated records); otherwise Wilson's interval on the rows
function accuracyInterval(ev) {
    if (ev && ev.ci && Array.isArray(ev.ci.accuracy) && ev.ci.accuracy.length === 2) return { ci: ev.ci.accuracy, label: ev.ci.clustered ? 'bootstrap over patients' : 'bootstrap' };
    if (ev && Array.isArray(ev.accuracyCI) && ev.accuracyCI.length === 2) return { ci: ev.accuracyCI, label: 'Wilson' };
    return null;
}

function renderEvaluationDashboard() {
    const evalRes = AppState.evalResults;
    const prep = AppState.prepMeta;
    const allResults = AppState.allTestedResults || [];
    DOM.resultsDashboard.style.display = 'flex';

    const isOptionB = evalRes.protocol === 'option-b';
    const isOptionC = evalRes.protocol === 'option-c';
    syncRankingSelectors(evalRes.isClassification);
    const trainPct = evalRes.trainPct || 80;
    const testPct = evalRes.testPct || 20;
    const trainCount = evalRes.trainCount || Math.round(evalRes.totalCount * 0.8);
    const testCount = evalRes.testCount || (evalRes.totalCount - trainCount);
    const totalCount = evalRes.totalCount || evalRes.totalSamples;

    // Protocol Status Banner
    if (DOM.retrainStatusBanner) {
        if (isOptionC) {
            const oc = evalRes.optimismCorrection || {};
            DOM.retrainStatusBanner.innerHTML = `
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px;">
                    <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-emerald);">Option C: Internal Validation by Bootstrap Optimism Correction</div>
                    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                        Every model was trained on all <strong>${totalCount}</strong> patients. Performance figures are <strong>optimism-corrected</strong> using ${oc.B || '?'} bootstrap resamples (fewer for neural networks / ensembles), each repeating missing-value filling, encoding, scaling and training. There is no separate test set; confirm on an independent cohort (Option B) before clinical use.
                    </div>
                </div>`;
        } else if (isOptionB) {
            DOM.retrainStatusBanner.innerHTML = `
                <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-purple); display: flex; align-items: center; gap: 6px;">
                            Option B: Independent External Validation Cohort
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                            Tested on separate external <strong>Dataset (ii) (${testCount} test cases)</strong> with zero data leakage. All models trained on 100% of <strong>Dataset (i) (${trainCount} patients)</strong>.
                        </div>
                    </div>
                    <span class="pill" style="background: rgba(168, 85, 247, 0.18); color: var(--accent-purple); border: 1px solid rgba(168, 85, 247, 0.4); font-size: 12px; font-weight: 700;">
                        External Cohort Validated
                    </span>
                </div>
            `;
        } else if (evalRes.retrainOn100) {
            DOM.retrainStatusBanner.innerHTML = `
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.28); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-emerald); display: flex; align-items: center; gap: 6px;">
                            <span>✓</span> Option A: Two-Phase Retraining Protocol &bull; 100% Retrained Models Active
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                            Accuracy evaluated on held-out <strong>${testPct}% test split (${testCount} cases)</strong> after fitting on <strong>${trainPct}% (${trainCount} cases)</strong>. The models were then retrained on <strong>100% of cohort data (${totalCount} cases)</strong> for use in Steps 5 and 6. <strong>The results shown are those of the models fitted on ${trainPct}% and scored on the held-out test cases.</strong> The retrained models have not themselves been tested (no unseen patients remain): their expected performance is taken from this evaluation, and they should be confirmed on new patients.
                        </div>
                    </div>
                    <span class="pill pill-target" style="background: rgba(16, 185, 129, 0.18); color: var(--tag-emerald); border-color: rgba(16, 185, 129, 0.4); font-size: 12px; font-weight: 700;">
                        100% Retrained Models Active
                    </span>
                </div>
            `;
        } else {
            DOM.retrainStatusBanner.innerHTML = `
                <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.28); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-cyan); display: flex; align-items: center; gap: 6px;">
                            <span>ℹ️</span> Option A: Single-Phase Protocol &bull; Split-Trained Models Active
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                            Evaluated on held-out <strong>${testPct}% test split (${testCount} cases)</strong>. Models maintained strictly on the <strong>${trainPct}% training split (${trainCount} cases)</strong> without 100% retraining.
                        </div>
                    </div>
                    <span class="pill" style="background: rgba(6, 182, 212, 0.18); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.4); font-size: 12px; font-weight: 700;">
                        ${trainPct}% Split Models Active
                    </span>
                </div>
            `;
        }
    }

    renderOddsRatioPanel('odds-ratio-container', AppState.prepMeta);

    renderOptimismPanel('optimism-container', isOptionC ? allResults : []);

    // Explainability panel (permutation importance on the Testing set)
    renderExplainPanel('explain-panel-container',
        (allResults.length ? allResults : []).map(r => ({ name: (MODEL_INFO[r.modelId] || { name: r.modelId }).name, model: r.model, prep: r.prep, inSample: Boolean(r.retrained) })),
        AppState.lastTestRows, Boolean(evalRes && evalRes.isClassification));

    // Subgroup (fairness) panel on the Testing set; the original values are used for grouping when available
    renderSubgroupPanel('subgroup-panel-container', isOptionC ? [] :
        (allResults.length ? allResults : []).map(r => ({ name: (MODEL_INFO[r.modelId] || { name: r.modelId }).name, model: r.model, prep: r.prep, threshold: r.evalRes && r.evalRes.decisionThreshold, inSample: Boolean(r.retrained) && !isOptionB })),
        AppState.lastTestRows, Boolean(evalRes && evalRes.isClassification),
        !isOptionB && AppState.baselineTestRawRows && AppState.lastTestRows && AppState.baselineTestRawRows.length === AppState.lastTestRows.length ? AppState.baselineTestRawRows : null);

    // Plain-language reading of the active model's results
    renderInterpretation('interpretation-container', {
        ev: evalRes, prep, name: displayModelName(AppState.deployedModelId || AppState.selectedModelId, AppState.trainedModel),
        design: isOptionB ? 'B' : isOptionC ? 'C' : 'A', dataset: AppState.fileName
    });

    // Convergence of the multiple imputation behind the active model (when MICE was used)
    renderMiceDiagnostics('mice-diagnostics-container', prep && prep.multipleImputation);

    // Calibration and decision-threshold panel (binary outcomes)
    renderCalibrationPanel('calibration-panel-container', isOptionC ? [] :
        (allResults.length ? allResults : [{ modelId: AppState.selectedModelId, evalRes }]).map(r => ({
            name: (MODEL_INFO[r.modelId] || { name: r.modelId || 'Model' }).name, ev: r.evalRes
        })), Boolean(evalRes && evalRes.isClassification));

    // Methodological notes that affect how the metrics should be read
    if (DOM.retrainStatusBanner) {
        const notes = [];
        const resultsForNotes = allResults.length ? allResults.map(r => r.evalRes) : [evalRes];
        if (evalRes && evalRes.splitInfo && evalRes.splitInfo.method !== 'random') {
            notes.push(`Training/Testing ${escapeHtml(describeSplit(evalRes.splitInfo))}.${evalRes.splitInfo.method === 'time' ? ' Cross-validation inside the Training period uses folds that are not time-ordered.' : ''}`);
        }
        if (evalRes && evalRes.isClassification && prep && Array.isArray(prep.y) && prep.y.length && prep.X && prep.X[0]) {
            const counts = {};
            prep.y.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
            const minority = Math.min(...Object.values(counts));
            // Parameters: 1 per numerical or ordered predictor, (categories − 1) per unordered categorical one
            const nPred = Math.max(1, prep.X[0].length - Object.values(prep.featureMetadata || {}).filter(m => m.type === 'nominal' && m.length > 1).length);
            const epv = minority / nPred;
            if (epv < 10) {
                notes.push(`⚠︎ <strong>Small sample for the number of predictors:</strong> the smaller outcome group has ${minority} patient(s) for ${nPred} predictor parameter(s) (${epv.toFixed(1)} per parameter; at least 10–20 is a common rule of thumb, and see the Riley criteria in Step 4). Models may overfit; consider fewer predictors or more patients, and treat the results as exploratory.`);
            }
        }
        if (evalRes && evalRes.multipleImputation) {
            const mi = evalRes.multipleImputation;
            notes.push(`<strong>Multiple imputation:</strong> missing predictor values${mi.imputedColumns && mi.imputedColumns.length ? ` (${mi.imputedColumns.map(escapeHtml).join(', ')})` : ''} were imputed ${mi.m} times by chained equations with predictive mean matching, fitted on the Training patients only and without the outcome. One model was trained on each completed dataset and the predictions are averaged; Testing and new patients with missing values are imputed in the same way.${evalRes.cv && evalRes.cv.imputation ? ` Cross-validation folds (model selection, tuning, recalibration): ${escapeHtml(evalRes.cv.imputation)}.` : ''}${mi.insideResamples ? ` Bootstrap resamples: ${escapeHtml(mi.insideResamples)}.` : ''}`);
        }
        if (evalRes && evalRes.multipleImputationSkipped) notes.push(`${escapeHtml(evalRes.multipleImputationSkipped)}`);
        if (isOptionC && evalRes && evalRes.recalibration) {
            notes.push(`<strong>Recalibrated</strong> with ${evalRes.recalibration.method === 'platt' ? 'Platt scaling' : 'isotonic regression'}, fitted on ${escapeHtml(evalRes.recalibration.fittedOn)}. The recalibration was refitted inside ${typeof evalRes.recalibrationInResamples === 'number' ? evalRes.recalibrationInResamples : 'every'} bootstrap resample(s) on their own cross-validated predictions, so the optimism-corrected results include its effect.`);
        }
        // Option A: one random split is a single, noisy draw
        if (!isOptionB && !isOptionC && evalRes) notes.push('<strong>One split:</strong> these results come from a single division of the patients into Training and Testing sets. Another split could give noticeably different figures, so a good result here is not evidence of reproducibility. Bootstrap validation (Option C) uses every patient and is more stable; an independent cohort (Option B) gives the strongest evidence.');
        if (isOptionC && evalRes && evalRes.recalibrationSkipped) notes.push(`${escapeHtml(evalRes.recalibrationSkipped)}`);
        if (evalRes && evalRes.cv) {
            notes.push(`Best model chosen by ${evalRes.cv.folds}-fold cross-validation inside the Training set${evalRes.cv.grouped ? ' (each patient\'s records kept in one fold)' : ''}; missing-value filling (${escapeHtml(evalRes.cv.imputation || 'Step 3 methods')}), encoding and scaling were refitted inside every fold.`);
        } else if (allResults.length > 1 && !isOptionC) {
            notes.push(`⚠︎ <strong>Exploratory model choice:</strong> the leading model was chosen by its score on the same Testing patients it is reported on, so its test results are optimistic (the best of ${allResults.length} scores is partly luck). For a reportable result, choose "Cross-validation within the Training set" for model selection, so the Testing set is used once, only to evaluate the chosen model.`);
        }
        // Who the test metrics rest on: patients, events and non-events, with a warning when they are few
        if (!isOptionC && evalRes && evalRes.isClassification && Array.isArray(evalRes.actuals) && evalRes.actuals.length) {
            const n = evalRes.actuals.length, counts = {};
            evalRes.actuals.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
            const groups = Object.keys(counts).map(k => `${counts[k]} "${escapeHtml(String(prep && prep.targetIdxToClass ? prep.targetIdxToClass[k] : k))}"`);
            const fewest = Math.min(...Object.values(counts)), nGroups = Object.keys(counts).length;
            const expected = prep && prep.targetClasses ? prep.targetClasses.length : nGroups;
            let warn = '';
            if (nGroups < expected) warn = ' <strong>Not every outcome group is present, so AUC and per-group results cannot be estimated.</strong>';
            else if (fewest < 10) warn = ' <strong>Fewer than 10 patients in the smallest group: AUC, sensitivity, calibration and their intervals are unreliable; treat them as illustrative only.</strong>';
            else if (fewest < 50) warn = ' With fewer than about 50 patients in the smallest group these estimates are imprecise (see the 95% intervals); calibration needs about 100 in each group.';
            notes.push(`${fewest < 50 || nGroups < expected ? '⚠︎ ' : ''}<strong>Testing set:</strong> ${n} ${isOptionB ? 'patients in Dataset (ii)' : 'test records'} (${groups.join(', ')}).${warn}`);
        }
        if (prep && prep.isClassification && prep.positiveClass !== null && prep.positiveClass !== undefined) {
            notes.push(`Positive (event) class for ROC-AUC, Brier score, sensitivity and TP/FN: <strong>${escapeHtml(prep.positiveClass)}</strong>. Change it in Step 2 (target card) if this is not the outcome of interest.`);
        }
        if (AppState.completeCase) {
            notes.push(isOptionB
                ? `Complete-case analysis: training rows with an empty cell were removed${evalRes.completeCaseDropped ? `; ${evalRes.completeCaseDropped} Dataset (ii) row(s) with an empty predictor cell were also removed` : ''}.`
                : 'Complete-case analysis: rows with an empty cell were removed from both the training and test splits (no imputation).');
        }
        if (isOptionB && evalRes.imputedFeatureCells > 0) {
            notes.push(`${evalRes.imputedFeatureCells} missing/unrecognised predictor cell(s) in Dataset (ii) were filled from training-cohort statistics.`);
        }
        if (isOptionB && (evalRes.missingTargetCount > 0 || evalRes.unknownClassCount > 0)) {
            notes.push(`Dataset (ii) rows excluded: ${evalRes.missingTargetCount || 0} with a missing outcome, ${evalRes.unknownClassCount || 0} with an outcome class not seen in training.`);
        }
        if (isOptionC) {
            notes.push('Option C: headline metrics are optimism-corrected (apparent performance on all patients minus the average bootstrap optimism). Testing-set panels (curves, calibration, permutation importance) are not shown because there is no separate test set; the bootstrap table below gives apparent, optimism and corrected values.');
        } else if (!isOptionB) {
            notes.push(`Option A: records were split first; imputation, category encoding and scaling were fitted on the training split only and applied unchanged to the test split.${(!evalRes.splitInfo || evalRes.splitInfo.method === 'random')
                ? ' The split is random at row level: if patients contribute several rows, choose "keep each patient\'s records together"; for temporal data choose "by date"; for multicentre data use an external cohort (Option B).'
                : ''}`);
            if (evalRes.testImputedCells > 0) {
                notes.push(`${evalRes.testImputedCells} missing predictor value(s) in the test split were filled with training-derived values.`);
            }
            if (evalRes.unknownClassCount > 0) {
                notes.push(`${evalRes.unknownClassCount} test row(s) had an outcome class absent from the training split and were excluded.`);
            }
        }
        if (prep && prep.isClassification) {
            notes.push(`Class "probabilities" are uncalibrated: Random Forest and hard voting give vote shares, k-NN gives neighbour shares. The Brier score (lower is better) summarises probability accuracy; a calibration plot is recommended before clinical use.`);
        }
        summariseHeldOutSet(evalRes).forEach(n => notes.push(n));
        resultsForNotes.forEach(r => {
            if (r && r.fallbackNote) notes.push(`⚠︎ ${escapeHtml(r.fallbackNote)}`);
            if (r && r.weightingNote) notes.push(`Ensemble: ${escapeHtml(r.weightingNote)}`);
            if (r && r.nnNote) notes.push(`Neural network: ${escapeHtml(r.nnNote)}`);
        });
        if (notes.length > 0) {
            DOM.retrainStatusBanner.innerHTML += `
                <div style="font-size: 11.5px; color: var(--text-secondary); margin: -6px 0 14px 0; padding: 8px 12px; border-left: 3px solid var(--accent-amber); background: var(--bg-surface-subtle); border-radius: var(--radius-sm);">
                    ${Array.from(new Set(notes)).map(n => `<div>• ${n}</div>`).join('')}
                </div>
            `;
        }
    }

    const cohortDesc = isOptionC ? `bootstrap internal validation of all ${totalCount} patients (optimism-corrected)` : isOptionB ? `External Testing Cohort (Dataset ii: ${testCount} Patients)` : `Held-out ${testPct}% Test Cohort (${testCount} Patients)`;

    // RENDER BASED ON NUMBER OF TESTED MODELS
    if (allResults.length > 1) {
        // MULTI-MODEL TOURNAMENT / COMPARISON MODE
        if (DOM.dashboardSingleHeading) DOM.dashboardSingleHeading.style.display = 'none';
        if (DOM.metricsSummaryRow) DOM.metricsSummaryRow.style.display = 'none';
        if (DOM.evaluationDetailContainer) DOM.evaluationDetailContainer.style.display = 'none';

        // 1. Benchmark Leaderboard Table
        if (DOM.multiModelLeaderboardContainer) {
            DOM.multiModelLeaderboardContainer.style.display = 'block';
            let lbRows = '';
            const anyCV = allResults.some(r => r.evalRes && r.evalRes.cv);
            allResults.forEach((resItem, idx) => {
                const mInfo = resItem.info;
                const rEval = resItem.evalRes;
                const isChampion = idx === 0;
                const isDeployed = AppState.deployedModelId === resItem.modelId;

                const rankBadge = isChampion ? '#1' : (idx === 1 ? '#2' : (idx === 2 ? '#3' : `#${idx + 1}`));
                const scoreDisplay = rEval.isClassification 
                    ? `<strong style="color: var(--accent-cyan); font-size: 14.5px;">${(rEval.accuracy * 100).toFixed(1)}%</strong>${accuracyInterval(rEval) ? `<div style="font-size: 10px; color: var(--text-muted);">95% CI ${(accuracyInterval(rEval).ci[0] * 100).toFixed(0)}–${(accuracyInterval(rEval).ci[1] * 100).toFixed(0)}%</div>` : ''}`
                    : `<strong style="color: var(--accent-cyan); font-size: 14.5px;">R²: ${fmtR2(rEval.r2)}</strong>${ciSpan(rEval, 'r2')}`;

                const rocAucDisplay = rEval.isClassification 
                    ? (rEval.rocAuc !== null ? `${rEval.rocAuc.toFixed(3)}${ciSpan(rEval, 'rocAuc')}` : '<span style="color:var(--text-muted)">N/A</span>')
                    : `RMSE: ${rEval.rmse.toFixed(3)}${ciSpan(rEval, 'rmse')}`;

                const f1Display = rEval.isClassification 
                    ? `${(rEval.macroF1 * 100).toFixed(1)}%${ciSpan(rEval, 'macroF1', true)}`
                    : `MAE: ${rEval.mae.toFixed(3)}${ciSpan(rEval, 'mae')}`;

                const precDisplay = rEval.isClassification 
                    ? `${(rEval.macroPrecision * 100).toFixed(1)}%`
                    : `MSE: ${rEval.mse.toFixed(3)}`;

                const recDisplay = rEval.isClassification 
                    ? `${(rEval.macroRecall * 100).toFixed(1)}%`
                    : '-';
                const balDisplay = rEval.isClassification && typeof rEval.balancedAccuracy === 'number'
                    ? `${(rEval.balancedAccuracy * 100).toFixed(1)}%${ciSpan(rEval, 'balancedAccuracy', true)}` : '-';
                const brierDisplay = rEval.isClassification && typeof rEval.brierScore === 'number'
                    ? `${rEval.brierScore.toFixed(3)}${ciSpan(rEval, 'brierScore')}` : '-';

                const seedDisplay = (rEval.seed !== null && rEval.seed !== undefined)
                    ? `<span class="seed-pill" title="Click to copy reproducibility seed: ${rEval.seed}" ${actionAttrs('copySeedToClipboard', rEval.seed)}>seed ${rEval.seed} <span class="copy-icon"><span class="ico ico-clipboard" aria-hidden="true"></span></span></span>`
                    : `<span style="font-size: 11px; color: var(--text-muted); font-family: monospace;">N/A</span>`;

                const deployBtn = isDeployed
                    ? `<span class="pill pill-target" style="background: rgba(16, 185, 129, 0.2); color: var(--tag-emerald); font-weight: 700; border-color: rgba(16, 185, 129, 0.4);">✓Active research model</span>`
                    : `<button class="btn-preset-action deploy-model-btn" ${actionAttrs('deployTestedModel', resItem.modelId)} style="padding: 4px 10px; font-size: 11px; cursor: pointer;" title="Make this the active research model for Steps 5 and 6">Use this model</button>`;

                lbRows += `
                    <tr style="${isDeployed ? 'background: rgba(6, 182, 212, 0.05);' : ''}">
                        <td><span class="leaderboard-rank ${isChampion ? 'rank-1' : ''}">${rankBadge}</span></td>
                        ${anyCV ? `<td>${formatCvScore(rEval, prep.isClassification)}</td>` : ''}
                        <td>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 16px;">${mInfo.icon}</span>
                                <div>
                                    <div style="font-weight: 700; color: var(--text-primary);">${escapeHtml(displayModelName(resItem.modelId, resItem.model))}</div>
                                    <div style="font-size: 11px; color: var(--text-muted);">${isOptionC ? 'Bootstrap-validated (all patients)' : resItem.retrained ? '100% Cohort Retrained' : (isOptionB ? 'External Validated' : 'Split-Trained')}</div>
                                </div>
                            </div>
                        </td>
                        <td><span class="pill ${mInfo.pillClass}" style="font-size: 11px;">${mInfo.tag}</span></td>
                        <td>${scoreDisplay}</td>
                        <td>${rocAucDisplay}</td>
                        <td>${f1Display}</td>
                        <td>${precDisplay}</td>
                        <td>${recDisplay}</td>
                        <td>${balDisplay}</td>
                        <td>${brierDisplay}</td>
                        <td>${seedDisplay}</td>
                        <td>${deployBtn}</td>
                    </tr>
                `;
            });

            DOM.multiModelLeaderboardContainer.innerHTML = `
                <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-color); border-radius: var(--radius-lg); padding: 18px 20px; margin-bottom: 24px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px;">
                        <div>
                            <h3 style="font-size: 16px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
                                Benchmark Leaderboard &mdash; ${allResults.length} Tested Models Ranked by ${escapeHtml(rankingMetricLabel(prep.isClassification))}
                            </h3>
                            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
                                ${anyCV
                                    ? `Models are ranked by <strong>cross-validated ${escapeHtml(rankingMetricLabel(prep.isClassification))} inside the Training set</strong> (CV column); the Testing set (${cohortDesc}) was not used to choose. The top-ranked model (${allResults[0].info.name}) is set as the active predictor, and its Testing-set score is an honest estimate. The other models' test scores are shown for information; picking a different winner from them would reintroduce optimism. Report all metrics, not only the winner. Click any seed pill to copy it.`
                                    : `Comparative performance on ${cohortDesc}, ranked by <strong>${escapeHtml(rankingMetricLabel(prep.isClassification))}</strong> (change the primary metric above). The top-ranked model (${allResults[0].info.name}) is set as the active predictor. A single train/test split does not show that one model is generally superior, and because the leader was selected on these same test cases its score is optimistically biased. Choosing by cross-validation (option above) avoids this. Report all metrics, not only the winner. Click any seed pill to copy it.`}
                            </div>
                        </div>
                        <span class="pill pill-target" style="font-size: 12px; font-weight: 700;">
                            ${allResults.length} Models Tested
                        </span>
                    </div>
                    <div class="table-container">
                        <table class="leaderboard-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    ${anyCV ? `<th title="Score inside the Training set by cross-validation (mean ± SD across folds); used to choose the best model">CV ${escapeHtml(rankingMetricLabel(prep.isClassification))}</th>` : ''}
                                    <th>Model Algorithm</th>
                                    <th>Paradigm</th>
                                    <th>${isOptionC ? (prep.isClassification ? 'Corrected Accuracy' : 'Corrected R²') : (prep.isClassification ? 'Test Accuracy' : 'Test R²')}</th>
                                    <th>${prep.isClassification ? 'ROC-AUC' : 'RMSE'}</th>
                                    <th>${prep.isClassification ? 'Macro F1' : 'MAE'}</th>
                                    <th>${prep.isClassification ? 'Precision' : 'MSE'}</th>
                                    <th>${prep.isClassification ? 'Recall' : '-'}</th>
                                    <th>${prep.isClassification ? 'Balanced Acc' : '-'}</th>
                                    <th>${prep.isClassification ? 'Brier ↓' : '-'}</th>
                                    <th>Seed (Ref)</th>
                                    <th>Status / Deploy</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${lbRows}
                            </tbody>
                        </table>
                    </div>
                    ${!isOptionC && allResults.some(r => r.evalRes && r.evalRes.ci) ? ciScopeNote() : ''}
                </div>
            `;
        }

        // 2. Individual Confusion Matrices & Accuracy Cards for ALL Chosen Models
        if (DOM.multiModelCmContainer) {
            DOM.multiModelCmContainer.style.display = 'block';
            let cmCardsHtml = '';
            allResults.forEach((resItem, idx) => {
                const mInfo = resItem.info;
                const rEval = resItem.evalRes;
                const isChampion = idx === 0;
                const isDeployed = AppState.deployedModelId === resItem.modelId;

                let cmContent = '';
                if (rEval.isClassification) {
                    const cm = rEval.confMatrix;
                    const classes = prep.targetClasses;
                    cmContent = `
                        <div class="table-container" style="margin-top: 10px;">
                            <table class="cm-table" style="font-size: 11.5px;">
                                <thead>
                                    <tr>
                                        <th style="padding: 6px 8px;">Actual \\ Pred</th>
                                        ${classes.map(c => `<th style="padding: 6px 8px;">${escapeHtml(c)}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${classes.map((actualClass, r) => `
                                        <tr>
                                            <th style="text-align: left; padding: 6px 8px;">${escapeHtml(actualClass)}</th>
                                            ${classes.map((_, c) => {
                                                const val = cm[r][c];
                                                const isHit = r === c;
                                                return `<td class="${isHit ? 'cm-cell-hit' : 'cm-cell-miss'}" style="padding: 6px 8px; font-weight: 700;">${val}</td>`;
                                            }).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    `;
                } else {
                    cmContent = `
                        ${renderExploratoryCutoffTable(rEval, true)}
                        <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: var(--text-secondary); padding: 0 4px;">
                            <span>RMSE: <strong>${rEval.rmse.toFixed(3)}</strong></span>
                            <span>MAE: <strong>${rEval.mae.toFixed(3)}</strong></span>
                        </div>
                    `;
                }

                const deployBtn = isDeployed
                    ? `<button class="btn-preset-action" disabled style="background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.4); color: var(--tag-emerald); font-weight: 700; width: 100%;">✓Active research model (Steps 5 &amp; 6)</button>`
                    : `<button class="btn-preset-action deploy-model-btn" ${actionAttrs('deployTestedModel', resItem.modelId)} style="width: 100%; font-weight: 600; cursor: pointer;">Use as active research model</button>`;

                cmCardsHtml += `
                    <div class="multi-cm-card ${isDeployed ? 'active-deployed' : ''}">
                        <div class="multi-cm-card-header">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 18px;">${mInfo.icon}</span>
                                <div>
                                    <div class="cm-card-title">${escapeHtml(displayModelName(resItem.modelId, resItem.model))}</div>
                                    <span class="pill ${mInfo.pillClass}" style="font-size: 10px; padding: 1px 6px;">${mInfo.tag}</span>
                                </div>
                            </div>
                            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
                                ${isChampion ? `<span class="champion-badge">Top-ranked (${escapeHtml(rankingMetricLabel(prep.isClassification))})</span>` : `<span class="pill" style="font-size: 10.5px;">Rank #${idx + 1}</span>`}
                                ${isDeployed ? `<span class="pill pill-target" style="font-size: 10px; background: rgba(16, 185, 129, 0.15); color: var(--tag-emerald);">Active</span>` : ''}
                            </div>
                        </div>

                        <div class="cm-card-metrics">
                            <div class="cm-metric-item">
                                <span class="cm-m-label">${rEval.isClassification ? 'Test Accuracy' : 'R² Score'}</span>
                                <span class="cm-m-val">${rEval.isClassification ? `${(rEval.accuracy * 100).toFixed(1)}%` : fmtR2(rEval.r2)}</span>
                            </div>
                            <div class="cm-metric-item">
                                <span class="cm-m-label">${rEval.isClassification ? 'ROC-AUC' : 'RMSE'}</span>
                                <span class="cm-m-val" style="color: var(--accent-purple);">${rEval.isClassification ? (rEval.rocAuc !== null ? rEval.rocAuc.toFixed(3) : 'N/A') : rEval.rmse.toFixed(3)}</span>
                            </div>
                            <div class="cm-metric-item">
                                <span class="cm-m-label">${rEval.isClassification ? 'Macro F1' : 'MAE'}</span>
                                <span class="cm-m-val" style="color: var(--accent-amber);">${rEval.isClassification ? `${(rEval.macroF1 * 100).toFixed(1)}%` : rEval.mae.toFixed(3)}</span>
                            </div>
                        </div>

                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; padding-top: 6px; border-top: 1px dashed var(--border-color);">
                            <div class="cm-seed-card-badge" title="Click to copy reproducibility seed: ${rEval.seed}" ${actionAttrs('copySeedToClipboard', rEval.seed)}>
                                <span>Seed:</span> <strong>${rEval.seed !== null && rEval.seed !== undefined ? rEval.seed : 'N/A'}</strong> <span style="font-size: 10px; opacity: 0.8;"><span class="ico ico-clipboard" aria-hidden="true"></span></span>
                            </div>
                            ${rEval.splitSeed ? `<span style="font-size: 10.5px; color: var(--text-muted); font-family: monospace;">Split: ${rEval.splitSeed}</span>` : ''}
                        </div>

                        <div style="font-size: 12px; font-weight: 700; color: var(--text-primary); margin-top: 10px;">
                            ${rEval.isClassification ? 'Confusion Matrix:' : 'Error metrics &amp; exploratory 2×2 split:'}
                        </div>
                        ${cmContent}

                        <div style="margin-top: 12px;">
                            ${deployBtn}
                        </div>
                    </div>
                `;
            });

            DOM.multiModelCmContainer.innerHTML = `
                <div style="margin-bottom: 20px;">
                    <h3 style="font-size: 16px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
                        Confusion Matrix &amp; Diagnostic Accuracy Breakdown &mdash; All Chosen Models
                    </h3>
                    <p style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
                        Individual confusion matrices and test metrics for every model chosen in this benchmark. Each card includes its exact reproducibility seed (click to copy).${isOptionC ? ' <strong>Option C: these matrices are apparent (measured on the patients each model was trained on), not optimism-corrected.</strong>' : ''}
                    </p>
                </div>
                <div class="multi-cm-grid">
                    ${cmCardsHtml}
                </div>
            `;
        }
    } else {
        // SINGLE MODEL TESTING MODE
        if (DOM.multiModelLeaderboardContainer) DOM.multiModelLeaderboardContainer.style.display = 'none';
        if (DOM.multiModelCmContainer) DOM.multiModelCmContainer.style.display = 'none';
        if (DOM.dashboardSingleHeading) DOM.dashboardSingleHeading.style.display = 'block';
        if (DOM.metricsSummaryRow) DOM.metricsSummaryRow.style.display = 'grid';
        if (DOM.evaluationDetailContainer) DOM.evaluationDetailContainer.style.display = 'block';

        const activeInfo = MODEL_INFO[AppState.deployedModelId || AppState.selectedModelId] || { name: 'Model', icon: '<span class="ico ico-network" aria-hidden="true"></span>' };
        if (DOM.dashboardSingleHeading) {
            DOM.dashboardSingleHeading.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span>${activeInfo.icon}</span> Single Model Diagnostic Performance: <strong>${activeInfo.name}</strong>
                </div>
            `;
        }

        const testAccLabel = isOptionC ? 'Optimism-corrected Accuracy (bootstrap)' : isOptionB ? 'External Test Accuracy (Dataset ii)' : `Test Accuracy (${testPct}% Split)`;
        const testR2Label = isOptionC ? 'Optimism-corrected R² (bootstrap)' : isOptionB ? 'External Test R² (Dataset ii)' : `Test R² Score (${testPct}% Split)`;

        if (evalRes.isClassification) {
            DOM.metricsSummaryRow.innerHTML = `
                <div class="metric-pill-card">
                    <span class="m-title">${testAccLabel}</span>
                    <span class="m-val">${(evalRes.accuracy * 100).toFixed(1)}%</span>
                    ${accuracyInterval(evalRes) ? `<span style="font-size: 11px; color: var(--text-muted);">95% CI ${(accuracyInterval(evalRes).ci[0] * 100).toFixed(1)}–${(accuracyInterval(evalRes).ci[1] * 100).toFixed(1)}% (${accuracyInterval(evalRes).label})</span>` : ''}
                </div>
                ${typeof evalRes.balancedAccuracy === 'number' ? `
                <div class="metric-pill-card">
                    <span class="m-title">Balanced Accuracy</span>
                    <span class="m-val">${(evalRes.balancedAccuracy * 100).toFixed(1)}%</span>
                </div>` : ''}
                ${typeof evalRes.brierScore === 'number' ? `
                <div class="metric-pill-card">
                    <span class="m-title">Brier Score (lower is better)</span>
                    <span class="m-val">${evalRes.brierScore.toFixed(3)}</span>
                </div>` : ''}
                <div class="metric-pill-card">
                    <span class="m-title">Macro F1-Score</span>
                    <span class="m-val">${(evalRes.macroF1 * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">Macro Precision</span>
                    <span class="m-val">${(evalRes.macroPrecision * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">Macro Recall</span>
                    <span class="m-val">${(evalRes.macroRecall * 100).toFixed(1)}%</span>
                </div>
                ${evalRes.rocAuc !== null ? `
                    <div class="metric-pill-card">
                        <span class="m-title">ROC AUC</span>
                        <span class="m-val">${(evalRes.rocAuc).toFixed(3)}</span>
                    </div>
                ` : ''}
                <div class="metric-pill-card" style="cursor: pointer;" ${actionAttrs('copySeedToClipboard', evalRes.seed)} title="Click to copy reproducibility seed: ${evalRes.seed}">
                    <span class="m-title">Reproducibility Seed</span>
                    <span class="m-val" style="color: var(--accent-purple); font-family: monospace;">seed ${evalRes.seed !== null && evalRes.seed !== undefined ? evalRes.seed : 'Auto'} </span>
                </div>
            `;

            // Confusion Matrix Table
            const cm = evalRes.confMatrix;
            const classes = prep.targetClasses;
            let cmHtml = `
                <div style="display: flex; flex-direction: column; gap: 16px;">
                    <h3 style="font-size: 15px; color: var(--text-primary);">Confusion Matrix (${isOptionC ? `apparent: the final model on the ${totalCount} patients it was trained on, <em>not</em> optimism-corrected` : cohortDesc})</h3>
                    ${isOptionC ? '<p class="tune-warn" style="margin: 0;">⚠︎ With bootstrap validation no patients are held out, so this matrix and the per-class sensitivity and precision below are measured on the training patients and are optimistic (flexible models can look almost perfect). Only the optimism-corrected measures in the summary and leaderboard (accuracy, balanced accuracy, F1, AUC, Brier, calibration slope) estimate performance on new patients.</p>' : ''}
                    <div class="table-container" style="max-width: 500px;">
                        <table class="cm-table">
                            <thead>
                                <tr>
                                    <th>Actual \\ Predicted</th>
                                    ${classes.map(c => `<th>${escapeHtml(c)}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${classes.map((actualClass, r) => `
                                    <tr>
                                        <th style="text-align: left;">${escapeHtml(actualClass)}</th>
                                        ${classes.map((_, c) => {
                                            const val = cm[r][c];
                                            const isHit = r === c;
                                            return `<td class="${isHit ? 'cm-cell-hit' : 'cm-cell-miss'}">${val}</td>`;
                                        }).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
            DOM.evaluationDetailContainer.innerHTML = cmHtml;
        } else {
            // Regression Metrics
            DOM.metricsSummaryRow.innerHTML = `
                <div class="metric-pill-card">
                    <span class="m-title">${testR2Label}</span>
                    <span class="m-val">${fmtR2(evalRes.r2)}</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">RMSE</span>
                    <span class="m-val">${evalRes.rmse.toFixed(3)}</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">MAE</span>
                    <span class="m-val">${evalRes.mae.toFixed(3)}</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">MSE</span>
                    <span class="m-val">${evalRes.mse.toFixed(3)}</span>
                </div>
                <div class="metric-pill-card" style="cursor: pointer;" ${actionAttrs('copySeedToClipboard', evalRes.seed)} title="Click to copy reproducibility seed: ${evalRes.seed}">
                    <span class="m-title">Reproducibility Seed</span>
                    <span class="m-val" style="color: var(--accent-purple); font-family: monospace;">seed ${evalRes.seed !== null && evalRes.seed !== undefined ? evalRes.seed : 'Auto'} </span>
                </div>
            `;

            const regCmHtml = renderExploratoryCutoffTable(evalRes, false);

            DOM.evaluationDetailContainer.innerHTML = `
                ${regCmHtml}
                <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
                    Model validated on ${cohortDesc} with R² = ${fmtR2(evalRes.r2)} and Mean Absolute Error = ${evalRes.mae}. ${isOptionB ? `Fitted strictly on 100% of Training Dataset (i) (${trainCount} patients).` : (evalRes.retrainOn100 ? `Active model was then retrained on 100% of data (${totalCount} patients); the results shown are from the split-trained model on the test patients, and the retrained model itself has not been tested.` : `Active model maintained strictly on ${trainPct}% training split (${trainCount} patients; 100% retraining declined).`)}
                </p>
                <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 6px;">
                    <span class="ico ico-bulb" aria-hidden="true"></span> <em>R², RMSE and MAE are the primary performance measures for a continuous outcome. If the outcome has a prespecified clinical decision threshold, analyse it as a binary outcome (Nominal) instead of relying on the exploratory split above.</em>
                </div>
            `;
        }
    }
}

export { renderEvaluationDashboard };
