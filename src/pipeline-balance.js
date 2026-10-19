// pipeline-balance.js: part of the Universal Medical ML/NeuralNet Predictor application controller (split from pipeline.js)
import { fitModel } from './compute.js';
import { AppState, actionAttrs, escapeHtml } from './core.js';
import { DOM } from './shell.js';

// =========================================================================
// STEP 5: Imbalance Handling (SMOTENC)
// =========================================================================
function renderPipelineStep5() {
    if (!AppState.pipeline.trainRows) return;

    const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
    const smotencRes = DataEngine.applySMOTENC(
        AppState.pipeline.trainRows,
        activeCols,
        AppState.pipeline.targetCol,
        AppState.pipeline.columnConfigs,
        AppState.pipeline.seed
    );

    AppState.pipeline.smotencResult = smotencRes;
    AppState.pipeline.balancedTrainRows = smotencRes.balancedRows;

    let contentHtml = '';
    if (smotencRes.applied) {
        contentHtml = `
            <div class="pipeline-alert pipeline-alert-warning" style="margin-bottom: 20px;">
                <div style="font-size: 20px;"><span class="ico ico-warn" aria-hidden="true"></span></div>
                <div>
                    <strong>Class Imbalance Detected:</strong><br>
                    ${escapeHtml(smotencRes.reason)}<br>
                    Smallest class <code>${escapeHtml(smotencRes.minorityClass)}</code>: ${smotencRes.minorityPctBefore}% of training rows before &rarr; ${smotencRes.minorityPctAfter}% after resampling.
                    Synthetic rows are used for training only; all test-set metrics are computed on real patients at their true prevalence.
                    <strong>Balancing distorts predicted risks</strong> (the models learn a 50:50 world): for two-outcome models the app shifts the probabilities back to the real event rate, or recalibrates them if you choose recalibration in Step 7. Balancing rarely improves discrimination, and is often unnecessary for well-specified models (van den Goorbergh et al., JAMIA 2022).
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 16px;">
                <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                    <h5 style="font-size: 13px; font-weight: 700; margin-bottom: 10px; color: var(--text-secondary);">Original Training Class Distribution</h5>
                    ${Object.keys(smotencRes.originalCounts).map(c => `
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
                            <span>Class <strong>${escapeHtml(c)}</strong>:</span>
                            <strong>${smotencRes.originalCounts[c]} cases</strong>
                        </div>
                    `).join('')}
                </div>
                <div class="card" style="padding: 16px; background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.3);">
                    <h5 style="font-size: 13px; font-weight: 700; margin-bottom: 10px; color: var(--accent-emerald);">Post-SMOTENC Training Class Distribution</h5>
                    ${Object.keys(smotencRes.balancedCounts).map(c => `
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
                            <span>Class <strong>${escapeHtml(c)}</strong>:</span>
                            <strong style="color: var(--accent-emerald);">${smotencRes.balancedCounts[c]} cases</strong>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    } else {
        contentHtml = `
            <div class="pipeline-alert pipeline-alert-success" style="margin-bottom: 20px;">
                <div style="font-size: 20px;"><span class="ico ico-check" aria-hidden="true"></span></div>
                <div>
                    <strong>No Synthetic Resampling Applied:</strong><br>
                    ${escapeHtml(smotencRes.reason || 'No synthetic resampling needed.')}
                </div>
            </div>
            <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                <h5 style="font-size: 13px; font-weight: 700; margin-bottom: 10px; color: var(--text-secondary);">Training Class Distribution</h5>
                ${Object.keys(smotencRes.classCounts || {}).map(c => `
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
                        <span>Class <strong>${escapeHtml(c)}</strong>:</span>
                        <strong>${smotencRes.classCounts[c]} cases</strong>
                    </div>
                `).join('')}
            </div>
        `;
    }

    if (DOM.pipelineSmotencContainer) {
        DOM.pipelineSmotencContainer.innerHTML = contentHtml;
    }
}

// =========================================================================
// STEP 6: Model-Driven Feature Ranking (Random Forest Scout)
// =========================================================================
function renderPipelineStep6UI() {
    if (!AppState.pipeline.rfImportance || AppState.pipeline.rfImportance.length === 0) {
        if (DOM.pipelineRfImportanceContainer) {
            DOM.pipelineRfImportanceContainer.innerHTML = `
                <p style="padding: 24px; color: var(--text-muted); text-align: center;">
                    Click the purple 'Run Random Forest Scout & Feature Ranking' button above to train a Random Forest classifier on the balanced training set and extract feature importances.
                </p>
            `;
        }
    }
}

async function executePipelineStep6() {
    const btn = DOM.pipelineRunRfScoutBtn;
    const label = btn ? btn.innerHTML : '';
    try {
        if (btn) { btn.disabled = true; btn.textContent = 'Training Random Forest scout…'; }
        await runPipelineStep6();
    } catch (err) {
        alert(`Random Forest scout could not run: ${err.message}`);
    } finally {
        if (btn) { btn.disabled = false; btn.innerHTML = label; }
    }
}

async function runPipelineStep6() {
    // Always recompute Step 5 from the current split & feature set so the scout never uses stale rows
    renderPipelineStep5();

    const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
    if (activeCols.length === 0) {
        alert("No active predictor features remain. Restore at least one feature in Step 4.");
        return;
    }
    const subConfigs = {};
    activeCols.forEach(c => subConfigs[c] = AppState.pipeline.columnConfigs[c]);
    subConfigs[AppState.pipeline.targetCol] = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol];

    const prep = DataEngine.prepareDataset(
        AppState.pipeline.balancedTrainRows,
        subConfigs,
        activeCols,
        AppState.pipeline.targetCol
    );

    // Train Random Forest Scout
    // Scout trained in the background worker when available
    const rf = await fitModel('rf', prep.isClassification, prep.targetClasses.length || 2, { numTrees: 40, maxDepth: 6, minSamplesSplit: 2, featureRatio: 'sqrt', seed: AppState.pipeline.seed }, prep.X, prep.y);

    // Map feature importances back to original clinical features
    const featureImpMap = {};
    activeCols.forEach(col => {
        const meta = prep.featureMetadata[col];
        let sumImp = 0;
        if (meta) {
            for (let k = 0; k < meta.length; k++) {
                sumImp += rf.featureImportances[meta.startIndex + k] || 0;
            }
        }
        featureImpMap[col] = sumImp;
    });

    const totalImp = Object.values(featureImpMap).reduce((a, b) => a + b, 0) || 1;
    const ranked = activeCols.map(col => ({
        name: col,
        importance: Number((featureImpMap[col] / totalImp).toFixed(4)),
        pct: Number(((featureImpMap[col] / totalImp) * 100).toFixed(1))
    }));

    ranked.sort((a, b) => b.importance - a.importance);

    // Bottom 30% cutoff
    const k = Math.max(1, Math.round(activeCols.length * 0.30));
    const topCount = activeCols.length - k;

    AppState.pipeline.rfImportance = ranked;
    AppState.pipeline.rfTopCount = topCount;
    // Keep earlier reinstatements that still fall below the cutoff
    const bottomNames = ranked.slice(topCount).map(f => f.name);
    const kept = Array.from(AppState.pipeline.reinstatedFeatures || []).filter(f => bottomNames.includes(f));
    AppState.pipeline.reinstatedFeatures = new Set(kept);
    applyRfFeatureSelection();
    renderRfImportanceChart();
    if (DOM.pipelineToStep7Btn) DOM.pipelineToStep7Btn.disabled = false;
}

// Top 70% plus any bottom-30% features the user reinstated (in importance order) go to k-NN & Naive Bayes
function applyRfFeatureSelection() {
    const ranked = AppState.pipeline.rfImportance || [];
    const topCount = AppState.pipeline.rfTopCount;
    const reinstated = AppState.pipeline.reinstatedFeatures || new Set();
    AppState.pipeline.top70Features = ranked.filter((f, idx) => idx < topCount || reinstated.has(f.name)).map(f => f.name);
    AppState.pipeline.bottom30Features = ranked.filter((f, idx) => idx >= topCount && !reinstated.has(f.name)).map(f => f.name);
}

window.pipelineToggleReinstateFeature = function (name) {
    const set = AppState.pipeline.reinstatedFeatures || (AppState.pipeline.reinstatedFeatures = new Set());
    if (set.has(name)) set.delete(name); else set.add(name);
    applyRfFeatureSelection();
    renderRfImportanceChart();
};

function renderRfImportanceChart() {
    const ranked = AppState.pipeline.rfImportance || [];
    const topCount = AppState.pipeline.rfTopCount;
    const reinstated = AppState.pipeline.reinstatedFeatures || new Set();
    const hasResults = (AppState.pipeline.benchmarkResults || []).length > 0;
    const maxPct = Math.max(...ranked.map(r => r.pct), 1);
    let chartHtml = `
        <div class="rf-importance-bar-container">
    `;

    ranked.forEach((r, idx) => {
        const isBottomTier = idx >= topCount;
        const isReinstated = isBottomTier && reinstated.has(r.name);
        const isPruned = isBottomTier && !isReinstated;
        const barWidth = Math.max(4, Math.round((r.pct / maxPct) * 100));

        if (idx === topCount) {
            chartHtml += `
                <div class="cutoff-line-notice">
                    <span><strong>Bottom 30% Cutoff:</strong> Features below this line are removed for k-NN and Naive Bayes only in Step 7 (other models use all surviving features). Click <strong>↩️ Reinstate</strong> to keep a feature for k-NN and Naive Bayes anyway (for example, if it is clinically important). MDI importance is computed on the SMOTENC-balanced training set and can favour features with many distinct values.</span>
                </div>
            `;
        }

        const pill = isReinstated
            ? `<span class="pill" style="font-size: 10px; background: rgba(56,189,248,0.15); color: var(--accent-cyan); border-color: rgba(56,189,248,0.4);">Reinstated</span>`
            : `<span class="pill" style="font-size: 10px; ${isPruned ? 'background: rgba(245,158,11,0.15); color: var(--accent-amber); border-color: rgba(245,158,11,0.4);' : 'background: rgba(16,185,129,0.15); color: var(--accent-emerald); border-color: rgba(16,185,129,0.4);'}">
                        ${isPruned ? 'Bottom 30%' : 'Top 70%'}
                    </span>`;
        const action = !isBottomTier ? '' : (isReinstated
            ? `<button type="button" class="btn btn-outline btn-sm" ${actionAttrs('pipelineToggleReinstateFeature', r.name)} title="Remove this feature again for k-NN and Naive Bayes">Remove again</button>`
            : `<button type="button" class="btn btn-outline btn-sm" ${actionAttrs('pipelineToggleReinstateFeature', r.name)} title="Keep this feature for k-NN and Naive Bayes in Step 7">↩️ Reinstate</button>`);

        chartHtml += `
            <div class="rf-importance-row">
                <div>
                    <strong style="color: ${isPruned ? 'var(--accent-amber)' : 'var(--text-primary)'};">#${idx + 1} ${escapeHtml(r.name)}</strong>
                </div>
                <div class="rf-meter-track">
                    <div class="rf-meter-fill ${isPruned ? 'bottom-tier' : 'top-tier'}" style="width: ${barWidth}%;"></div>
                </div>
                <div style="font-weight: 700; color: ${isPruned ? 'var(--accent-amber)' : 'var(--accent-emerald)'}; text-align: right;">
                    ${r.pct}%
                </div>
                <div>${pill}</div>
                <div class="rf-importance-action">${action}</div>
            </div>
        `;
    });

    const nReinstated = ranked.filter((r, idx) => idx >= topCount && reinstated.has(r.name)).length;
    chartHtml += `
            <div class="rf-importance-summary">
                k-NN and Naive Bayes will use <strong>${AppState.pipeline.top70Features.length}</strong> of ${ranked.length} features
                (${topCount} in the top 70%${nReinstated ? ` + ${nReinstated} reinstated` : ''}).
                ${hasResults ? '<span style="color: var(--accent-amber);">Re-run Step 7 to apply any change to the results.</span>' : ''}
            </div>
        </div>`;
    if (DOM.pipelineRfImportanceContainer) DOM.pipelineRfImportanceContainer.innerHTML = chartHtml;
}

export { renderPipelineStep5, renderPipelineStep6UI, executePipelineStep6, runPipelineStep6, applyRfFeatureSelection, renderRfImportanceChart };
