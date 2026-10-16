// reports.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { APP_VERSION, AppState, MODEL_INFO, RESEARCH_USE_NOTICE, escapeHtml, rankingMetricLabel, actionAttrs } from './core.js';
import { describeSplit } from './cv-tuning.js';
import { executePipelineStep7 } from './pipeline-train.js';
import { SETTINGS_APP } from './settings.js';
import { DOM } from './shell.js';


function renderPipelineVsBaselineComparison() {
    if (!DOM.pipelineVsBaselineContent) return;

    const hasPipeline = AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0;
    const hasBaseline = AppState.allTestedResults && AppState.allTestedResults.length > 0;

    if (!hasPipeline) {
        DOM.pipelineVsBaselineContent.innerHTML = `
            <p style="padding: 16px; color: var(--text-muted);">Please complete Step 7 of the pipeline first to generate metrics.</p>
        `;
        return;
    }

    const pipeBest = AppState.pipeline.benchmarkResults[0]; // stored best-first in executePipelineStep7
    const pEv = pipeBest.eval;

    if (!hasBaseline) {
        const f1Text = (pEv && typeof pEv.macroF1 === 'number' && !isNaN(pEv.macroF1))
            ? `F1: ${(pEv.macroF1 * 100).toFixed(1)}%`
            : (pEv && typeof pEv.r2 === 'number' && !isNaN(pEv.r2) ? `R²: ${pEv.r2.toFixed(3)}` : '');
        const accText = (pEv && typeof pEv.accuracy === 'number' && !isNaN(pEv.accuracy))
            ? `Acc: ${(pEv.accuracy * 100).toFixed(1)}%`
            : (pEv && typeof pEv.rmse === 'number' && !isNaN(pEv.rmse) ? `RMSE: ${pEv.rmse.toFixed(3)}` : '');
        const metricSummary = [f1Text, accText].filter(Boolean).join(', ') || 'Evaluated';

        DOM.pipelineVsBaselineContent.innerHTML = `
            <div class="pipeline-alert pipeline-alert-info">
                <div style="font-size: 20px;">💡</div>
                <div>
                    <strong>Baseline Benchmark Not Run Yet:</strong><br>
                    You have successfully completed the 7-Step Pipeline with <strong>${escapeHtml(pipeBest.name)}</strong> (${metricSummary}).<br>
                    To see whether the 7-Step Pipeline predicts better than the standard workflow on this cohort, click below:
                    <div style="margin-top: 10px;">
                        <button class="btn btn-primary btn-sm" ${actionAttrs('trainStandardBaselineFromPipeline')}>
                            ⚖️ Train Standard Baseline &amp; Compare
                        </button>
                        <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 6px;">Uses your Step 4 model choices and the same train/test split as the pipeline, then brings you straight back here with the comparison.</div>
                    </div>
                </div>
            </div>
        `;
        return;
    }

    // Both baseline and pipeline exist: side-by-side (unpaired) comparison
    const baseBest = AppState.allTestedResults[0];
    const bEv = baseBest.evalRes;
    const isClassification = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous';

    if (Boolean(bEv.isClassification) !== Boolean(isClassification)) {
        DOM.pipelineVsBaselineContent.innerHTML = `
            <div class="pipeline-alert pipeline-alert-info">
                <div style="font-size: 20px;">ℹ️</div>
                <div>The Step 4 baseline and the 7-Step Pipeline were run with different outcome types (classification vs regression), so they cannot be compared. Re-run one of them with the same target configuration.</div>
            </div>`;
        return;
    }

    // Plain-language difference: says which way the change goes and whether that is better or worse
    const fmtChange = (base, pipe, higherIsBetter, digits, unit = '') => {
        if (base === null || pipe === null || isNaN(base) || isNaN(pipe)) return '—';
        const d = pipe - base;
        if (Math.abs(d) < Math.pow(10, -digits) / 2) return '<span style="color: var(--text-secondary);">No difference</span>';
        const better = higherIsBetter ? d > 0 : d < 0;
        return `<strong style="color: ${better ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${Math.abs(d).toFixed(digits)}${unit} ${d > 0 ? 'higher' : 'lower'}</strong> <span style="color: var(--text-secondary);">(pipeline ${better ? 'better' : 'worse'})</span>`;
    };
    const row = (label, meaning, bVal, pVal, change) => `
        <tr>
            <td><strong>${label}</strong>${meaning ? `<div class="cmp-meaning">${meaning}</div>` : ''}</td>
            <td>${bVal}</td>
            <td><strong>${pVal}</strong></td>
            <td>${change}</td>
        </tr>`;
    const num = v => (typeof v === 'number' && isFinite(v) ? v : null);
    const pct = v => (num(v) !== null ? v * 100 : null);
    const showPct = v => (v !== null ? `${v.toFixed(1)}%` : 'N/A');
    const show3 = v => (v !== null ? v.toFixed(3) : 'N/A');
    const targetName = escapeHtml(AppState.pipeline.targetCol || 'the outcome');

    let metricRows = '';
    let guide = [];
    let headline;
    if (isClassification) {
        const bBr = num(bEv.brierScore), pBr = num(pEv.brierScore);
        const bAcc = pct(bEv.accuracy), pAcc = pct(pEv.accuracy);
        const bF1 = pct(bEv.macroF1), pF1 = pct(pEv.macroF1);
        const bAuc = num(bEv.rocAuc), pAuc = num(pEv.rocAuc);
        const bBal = pct(bEv.balancedAccuracy), pBal = pct(pEv.balancedAccuracy);
        metricRows += row('Brier score ↓', 'How close the predicted probabilities were to what happened. 0 = perfect; lower is better.', show3(bBr), show3(pBr), fmtChange(bBr, pBr, false, 3));
        metricRows += row('Test accuracy', 'Percentage of test patients whose outcome was predicted correctly. Higher is better.', showPct(bAcc), showPct(pAcc), fmtChange(bAcc, pAcc, true, 1, ' percentage points'));
        metricRows += row('Macro F1-score', 'Balance of precision and recall, averaged over the outcome groups. Higher is better.', showPct(bF1), showPct(pF1), fmtChange(bF1, pF1, true, 1, ' percentage points'));
        if (bAuc !== null && pAuc !== null) {
            metricRows += row('ROC-AUC', 'How well the model ranks higher-risk above lower-risk patients. 0.5 = chance, 1 = perfect.', show3(bAuc), show3(pAuc), fmtChange(bAuc, pAuc, true, 3));
        }
        metricRows += row('Balanced accuracy', 'Average of the accuracy within each outcome group; fairer than accuracy when one outcome is rare.', showPct(bBal), showPct(pBal), fmtChange(bBal, pBal, true, 1, ' percentage points'));
        guide = [
            '<strong>Test accuracy</strong> is the simplest measure: the share of test patients predicted correctly. If one outcome is uncommon, look at <strong>balanced accuracy</strong> as well.',
            '<strong>Brier score</strong> matters if you will quote the predicted percentage to a patient (e.g. "70% chance of improving"): the lower it is, the more trustworthy those percentages are.',
            '<strong>ROC-AUC</strong> shows how well the model separates patients who will and will not have the outcome, regardless of the cut-off used.',
            '<strong>Macro F1</strong> combines how many predicted cases were right (precision) and how many real cases were found (recall).'
        ];
        headline = (bAcc !== null && pAcc !== null)
            ? `On the test patients, the pipeline's best model (${escapeHtml(pipeBest.name)}) predicted ${pAcc.toFixed(1)}% correctly and the standard workflow's best model (${escapeHtml(MODEL_INFO[baseBest.modelId]?.name || baseBest.modelId)}) ${bAcc.toFixed(1)}%${bBr !== null && pBr !== null ? `; their Brier scores were ${pBr.toFixed(3)} and ${bBr.toFixed(3)} (lower is better)` : ''}.`
            : 'The accuracy of the two workflows could not be compared.';
    } else {
        const bR2 = num(bEv.r2), pR2 = num(pEv.r2);
        const bRmse = num(bEv.rmse), pRmse = num(pEv.rmse);
        const bMae = num(bEv.mae), pMae = num(pEv.mae);
        metricRows += row('R²', `Share of the differences in ${targetName} between patients that the model explains. 0 = no better than predicting the average, 1 = perfect. Higher is better.`, show3(bR2), show3(pR2), fmtChange(bR2, pR2, true, 3));
        metricRows += row('MAE ↓', `Average size of a prediction error, in ${targetName} units. Lower is better.`, show3(bMae), show3(pMae), fmtChange(bMae, pMae, false, 3));
        metricRows += row('RMSE ↓', `Typical size of a prediction error, in ${targetName} units, giving extra weight to large misses. Lower is better.`, show3(bRmse), show3(pRmse), fmtChange(bRmse, pRmse, false, 3));
        guide = [
            `For a numerical outcome there is no "accuracy" in the yes/no sense. <strong>R²</strong> is the closest equivalent: it says how much of the variation in ${targetName} the model explains (e.g. 0.60 = 60%).`,
            `<strong>MAE</strong> is the easiest to explain to colleagues: "on average, the prediction was off by this many ${targetName} units". Judge it against what counts as a clinically important difference on that scale.`,
            '<strong>RMSE</strong> is similar to MAE but punishes large misses more. If RMSE is much larger than MAE, a few patients were predicted badly.'
        ];
        headline = (bR2 !== null && pR2 !== null)
            ? `On the test patients, the pipeline's best model (${escapeHtml(pipeBest.name)}) explained ${(pR2 * 100).toFixed(0)}% of the variation in ${targetName} (R² ${pR2.toFixed(3)}) and the standard workflow's best model (${escapeHtml(MODEL_INFO[baseBest.modelId]?.name || baseBest.modelId)}) ${(bR2 * 100).toFixed(0)}% (R² ${bR2.toFixed(3)})${bMae !== null && pMae !== null ? `; predictions were off by ${pMae.toFixed(2)} and ${bMae.toFixed(2)} units on average` : ''}.`
            : 'R² could not be compared.';
    }

    const sm = AppState.pipeline.smotencResult;
    const baseProtocol = bEv.protocol === 'option-c'
        ? 'bootstrap internal validation of all patients (no separate test patients)'
        : bEv.protocol === 'option-b'
        ? `a separate test file (Dataset ii)`
        : `a ${bEv.testPct || 20}% random test split (split seed ${bEv.splitSeed})`;
    const baseTestN = bEv.testCount || bEv.totalSamples || 0;
    const pipeTestN = AppState.pipeline.testRows ? AppState.pipeline.testRows.length : 0;
    const pipeRatio = Math.round((AppState.pipeline.trainRatio || 0.8) * 100);
    const ciText = ev => (Array.isArray(ev.accuracyCI) && ev.accuracyCI.length === 2 ? ` (accuracy 95% CI ${(ev.accuracyCI[0] * 100).toFixed(0)}–${(ev.accuracyCI[1] * 100).toFixed(0)}%)` : '');

    // Cautions, worded for the actual analysis
    const cautions = [];
    // Same test patients? (same raw records, e.g. same split ratio and seed, no rows dropped)
    const baseRaw = bEv.protocol === 'option-b' ? null : AppState.baselineTestRawRows;
    const pipeRaw = AppState.pipeline.testRawRows;
    const sameTestPatients = Boolean(baseRaw && pipeRaw && baseRaw.length === pipeRaw.length && pipeTestN === pipeRaw.length &&
        (() => { const set = new Set(baseRaw); return pipeRaw.every(r => set.has(r)); })());
    const smallest = Math.min(baseTestN || Infinity, pipeTestN || Infinity);
    if (smallest < 30) {
        cautions.push(`<strong>Very small test sets:</strong> ${baseTestN} test patients for the standard workflow${isClassification ? ciText(bEv) : ''} and ${pipeTestN} for the pipeline${isClassification ? ciText(pEv) : ''}. With so few patients, one or two well- or badly-predicted cases can change the results a lot, so these numbers are rough indications only.`);
    } else if (smallest < 100) {
        cautions.push(`<strong>Modest test sets</strong> (${baseTestN} and ${pipeTestN} patients): differences of a few points may be due to chance.`);
    }
    cautions.push(sameTestPatients
        ? `<strong>Same test patients:</strong> both workflows were tested on the same ${pipeTestN} patients, which makes the comparison fairer; the cautions about sample size still apply.`
        : `<strong>Different test patients:</strong> the standard workflow was tested on ${baseProtocol}; the pipeline on its own ${pipeRatio}/${100 - pipeRatio} split (seed ${AppState.pipeline.seed}). Part of any difference may simply be that one set of test patients was easier to predict than the other. Using the same split ratio and seed in both workflows (and no row removal) tests them on the same patients.`);
    // A single trained model involves no selection, so it needs no selection caution
    const bCV = Boolean(bEv.cv) || (AppState.allTestedResults || []).length <= 1, pCV = Boolean(pEv.cv);
    if (!bCV || !pCV) {
        const which = !bCV && !pCV ? 'each workflow\'s "top" model was' : (!bCV ? 'the standard workflow\'s "top" model was' : 'the pipeline\'s "top" model was');
        cautions.push(`<strong>Best model picked on its own test set:</strong> ${which} chosen because it did best on the very test patients it is scored on, so that score is somewhat optimistic. Choosing by cross-validation avoids this.`);
    }
    cautions.push('<strong>No statistical test</strong> has been applied, so the difference is not shown to be statistically significant.');
    const nearPerfect = isClassification ? (num(pEv.accuracy) !== null && pEv.accuracy >= 0.98) || (num(bEv.accuracy) !== null && bEv.accuracy >= 0.98)
        : (num(pEv.r2) !== null && pEv.r2 >= 0.98) || (num(bEv.r2) !== null && bEv.r2 >= 0.98);
    if (nearPerfect) {
        cautions.push('<strong>Near-perfect results</strong> are uncommon with real clinical data; see the "Too good to be true?" check under the pipeline results before relying on them.');
    }
    if ((AppState.pipeline.droppedFeatures && AppState.pipeline.droppedFeatures.size > 0) || (sm && sm.applied) || AppState.pipeline.tuningReport || AppState.tuningReport) {
        cautions.push('<strong>Different preparation:</strong> the workflows differ in which predictors were used, balancing and/or settings tuning (see the rows above), so a difference reflects the whole approach, not one step.');
    }

    const tuned = r => (r ? 'Automatically tuned (cross-validation in the Training set)' : '<span style="color: var(--text-muted);">Default / user settings</span>');

    const trainedAt = AppState.baselineTrainedAt instanceof Date
        ? AppState.baselineTrainedAt.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }) : null;
    const compHtml = `
        <div class="cmp-baseline-note">
            <span>${AppState.baselineJustTrained
                ? `✓ Standard baseline trained just now${trainedAt ? ` (${trainedAt})` : ''} with the same train/test split as the pipeline.`
                : `ℹ️ This comparison uses the standard-workflow (Step 4) results already trained${trainedAt ? ` at ${trainedAt}` : ''}, so no new training was needed. If you have since changed settings, re-run it to compare like with like.`}</span>
            <button type="button" class="btn btn-outline btn-sm" ${actionAttrs('trainStandardBaselineFromPipeline')}>↻ Re-run baseline &amp; compare</button>
        </div>
        <div style="margin-bottom: 14px; font-size: 13px; color: var(--text-secondary);">
            Results of each workflow's best model for <strong>${escapeHtml(AppState.fileName || 'Cohort')}</strong>, ${sameTestPatients ? 'both scored on the <strong>same</strong> test patients' : 'each scored on its <strong>own</strong> test patients'}.
        </div>

        <div class="table-container"><table class="data-table cmp-table" style="margin-bottom: 16px;">
            <thead>
                <tr>
                    <th>Measure &amp; what it means</th>
                    <th style="color: var(--accent-cyan);">Step 4 Standard Workflow</th>
                    <th style="color: var(--accent-emerald);">🔬 7-Step Pipeline</th>
                    <th>Difference (pipeline vs standard)</th>
                </tr>
            </thead>
            <tbody>
                ${row('Best model', '', escapeHtml(MODEL_INFO[baseBest.modelId]?.name || baseBest.modelId.toUpperCase()), escapeHtml(pipeBest.name), '')}
                ${row('Test patients', 'Number of patients each result is based on. More patients = more reliable.', String(baseTestN || '?'), String(pipeTestN || '?'), '')}
                ${metricRows}
                ${row('Missing data / scaling', '', bEv.protocol === 'option-b' ? 'Fitted on Dataset (i) only' : 'Fitted on the training split only (split first)', AppState.pipeline.completeCase ? `Complete-case (${AppState.pipeline.excludedIncomplete} incomplete row(s) removed); scaling fitted on the training split only` : 'Fitted on the training split only', '')}
                ${row('Overlapping predictors', 'Predictors that measure nearly the same thing (collinearity).', '<span style="color: var(--text-muted);">None removed (all kept)</span>', `Correlation &amp; VIF review; ${AppState.pipeline.droppedFeatures.size} predictor(s) excluded overall`, '')}
                ${row('Class imbalance', 'Whether a rare outcome was topped up with synthetic patients in training.', '<span style="color: var(--text-muted);">Original distribution</span>',
                    isClassification ? (sm && sm.applied ? `SMOTENC on training split (+${sm.syntheticCountGenerated} synthetic patients)` : 'SMOTENC not needed (not imbalanced)') : 'Not applicable (numerical outcome)', '')}
                ${row('Predictor pruning', '', '<span style="color: var(--text-muted);">All predictors</span>', 'Weakest 30% (by Random Forest importance) removed for k-NN &amp; Naive Bayes only', '')}
                ${row('Model settings', '', tuned(AppState.tuningReport), tuned(AppState.pipeline.tuningReport), '')}
            </tbody>
        </table></div>

        <div class="cmp-guide">
            <div class="cmp-guide-title">📖 How to read this comparison</div>
            <ul>${guide.map(g => `<li>${g}</li>`).join('')}
                <li>In the <strong>Difference</strong> column, green means the pipeline did better on that measure and red means it did worse.</li>
            </ul>
        </div>

        <div class="cmp-cautions">
            <div class="cmp-cautions-title">⚠️ Cautions before drawing conclusions</div>
            <ul>${cautions.map(c => `<li>${c}</li>`).join('')}</ul>
        </div>

        <div class="pipeline-alert pipeline-alert-info" style="margin-top: 10px;">
            <div style="font-size: 20px;">🔬</div>
            <div>
                <strong>Summary:</strong> ${headline} Because of the cautions above, this is <strong>not proof</strong> that one workflow is better. To establish a genuine difference, test both on the <strong>same, larger group of patients</strong>, ideally an independent validation cohort.
            </div>
        </div>
    `;

    DOM.pipelineVsBaselineContent.innerHTML = compHtml;
}

// Standard workflow (Step 4) analysis report: validation design, settings and results of every model
function downloadStandardAnalysisReport() {
    const results = AppState.allTestedResults || [];
    if (!results.length) { alert("No Step 4 results available to download yet."); return; }
    const top = results[0].evalRes;
    const cfgs = AppState.columnConfigs || {};
    const target = Object.keys(cfgs).find(c => cfgs[c].role === 'target');
    const report = {
        title: 'Standard Workflow (Step 4) Analysis Report',
        notice: RESEARCH_USE_NOTICE,
        software: { name: SETTINGS_APP, version: APP_VERSION, url: location.href.split('?')[0] },
        generated_at: new Date().toISOString(),
        dataset: { name: AppState.fileName, records: AppState.parsedData ? AppState.parsedData.rows.length : null, columns: AppState.parsedData ? AppState.parsedData.headers.length : null },
        outcome: target, positive_class: top.positiveClass ?? null,
        variables: Object.fromEntries(Object.entries(cfgs).map(([k, v]) => [k, { role: v.role, type: v.type, missing_data: v.role === 'feature' ? v.imputationStrategy : undefined }])),
        exclusions: { complete_case_analysis: Boolean(AppState.completeCase), test_rows_dropped: top.testDroppedRows ?? null },
        validation_design: {
            protocol: top.protocol === 'option-c' ? `internal validation by bootstrap optimism correction (${(top.optimismCorrection || {}).B || '?'} resamples)` : top.protocol === 'option-b' ? 'separate testing dataset (Dataset ii)' : 'internal train/test split',
            optimism_correction: top.protocol === 'option-c' ? results.map(r => ({ model: (MODEL_INFO[r.modelId] || {}).name || r.modelId, ...(r.evalRes.optimismCorrection || {}) })) : undefined,
            split: top.protocol === 'option-b' ? null : describeSplit(top.splitInfo),
            train_percent: top.trainPct ?? null, split_seed: top.splitSeed ?? null,
            train_cases: top.trainCount, test_cases: top.testCount,
            model_selection: top.cv ? `${top.cv.folds}-fold cross-validation inside the training set${top.cv.grouped ? ' (patient-grouped folds)' : ''}; preprocessing refitted in each fold` : 'best test-set score (optimistic)',
            settings_tuning: AppState.tuningReport ? 'automatic (cross-validation inside the training set)' : 'user settings',
            ranking_metric: rankingMetricLabel(Boolean(top.isClassification)),
            decision_threshold: top.decisionThreshold ?? null,
            deployed_model_retrained_on_all_data: Boolean(top.retrainOn100)
        },
        deployed_model: AppState.deployedSource === 'standard' ? (MODEL_INFO[AppState.deployedModelId] || {}).name || AppState.deployedModelId : null,
        models: results.map((r, i) => ({
            rank: i + 1, model: (MODEL_INFO[r.modelId] || {}).name || r.modelId, seed: r.seed,
            settings: r.hyperparams || null,
            cross_validated_training_scores: r.evalRes.cv || null,
            tuning: r.evalRes.tuning || null,
            test_metrics: r.evalRes.isClassification ? {
                accuracy: r.evalRes.accuracy, accuracy_95ci_wilson: r.evalRes.accuracyCI, balanced_accuracy: r.evalRes.balancedAccuracy,
                f1_macro: r.evalRes.macroF1, precision_macro: r.evalRes.macroPrecision, recall_macro: r.evalRes.macroRecall,
                auc_roc: r.evalRes.rocAuc, brier_score: r.evalRes.brierScore, probability_type: r.evalRes.probabilityType
            } : { r2: r.evalRes.r2, rmse: r.evalRes.rmse, mae: r.evalRes.mae, mse: r.evalRes.mse },
            calibration: r.evalRes.calibration || null,
            confusion_matrix: r.evalRes.confMatrix || null
        })),
        limitations: [
            'For research and education only; not validated for clinical decision-making.',
            'Probabilities may be poorly calibrated; see the calibration results before interpreting them as risks.',
            'A single train/test split gives an imprecise estimate; confirm on an independent (external) cohort.'
        ]
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Clinical_ML_Standard_Analysis_Report_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

export { renderPipelineVsBaselineComparison, downloadStandardAnalysisReport };
