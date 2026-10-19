// pipeline-screening.js: part of the Universal Medical ML/NeuralNet Predictor application controller (split from pipeline.js)
import { AppState, actionAttrs, escapeHtml } from './core.js';
import { DOM } from './shell.js';
import { renderPredictorBudget } from './predictor-budget.js';
import { pipelineGroupFunction } from './cv-tuning.js';

// =========================================================================
// STEP 2: Continuous Variable Correlation Check (Pearson vs. Spearman)
// =========================================================================
function renderPipelineStep2() {
    if (!AppState.pipeline.trainRows) return;

    const contCols = AppState.pipeline.featureCols.filter(col =>
        !AppState.pipeline.droppedFeatures.has(col) &&
        (AppState.pipeline.columnConfigs[col]?.type === 'continuous' || AppState.pipeline.columnConfigs[col]?.type === 'ordinal')
    );

    if (contCols.length < 2) {
        if (DOM.pipelineNormalityTableContainer) {
            DOM.pipelineNormalityTableContainer.innerHTML = `<p style="padding: 16px; color: var(--text-muted);">Less than two continuous features available for pairwise correlation analysis.</p>`;
        }
        if (DOM.pipelineCorrPairsContainer) {
            DOM.pipelineCorrPairsContainer.innerHTML = `<p style="padding: 16px; color: var(--text-muted);">No continuous feature pairs to evaluate.</p>`;
        }
        if (DOM.pipelineCorrPairCounter) DOM.pipelineCorrPairCounter.innerText = `0 High Pairs`;
        return;
    }

    const corrRes = DataEngine.checkPairwiseCorrelations(AppState.pipeline.trainRows, contCols, 0.8, AppState.pipeline.columnConfigs);
    AppState.pipeline.normalityMap = corrRes.normalityMap;
    AppState.pipeline.corrPairs = corrRes.allPairs;

    // 1. Render Normality Table
    let normHtml = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Continuous Feature</th>
                    <th>Sample Mean</th>
                    <th>Std Deviation</th>
                    <th>Skewness (adjusted Fisher–Pearson G₁)</th>
                    <th>Skewness Screen (rule of thumb)</th>
                    <th>Correlation Method</th>
                </tr>
            </thead>
            <tbody>
    `;

    contCols.forEach(col => {
        const norm = corrRes.normalityMap[col] || { mean: 0, std: 1, skewness: 0, isNormal: true };
        normHtml += `
            <tr>
                <td><strong>${escapeHtml(col)}</strong></td>
                <td>${norm.mean}</td>
                <td>${norm.std}</td>
                <td><strong style="color: ${norm.isNormal ? 'var(--accent-emerald)' : 'var(--accent-amber)'};">${norm.skewness}</strong></td>
                <td>
                    <span class="pill" style="${norm.isNormal ? 'background: rgba(16, 185, 129, 0.15); color: var(--tag-emerald); border-color: rgba(16, 185, 129, 0.4);' : 'background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border-color: rgba(245, 158, 11, 0.4);'}">
                        ${norm.isNormal ? '✓Approx. symmetric (|G₁| ≤ 1.0)' : '⚠︎ Skewed (|G₁| > 1.0)'}
                    </span>
                </td>
                <td style="font-size: 12px; color: var(--text-secondary);">
                    ${norm.isNormal ? 'Pearson if its partner is also approx. symmetric' : 'Spearman rank correlation for its pairs'}
                </td>
            </tr>
        `;
    });
    normHtml += `</tbody></table>`;
    if (DOM.pipelineNormalityTableContainer) DOM.pipelineNormalityTableContainer.innerHTML = normHtml;

    // 2. Render High Correlation Pairs (|r| or |rho| >= 0.8)
    const highPairs = corrRes.highPairs;
    if (DOM.pipelineCorrPairCounter) {
        DOM.pipelineCorrPairCounter.innerText = `${highPairs.length} High Pairs`;
    }

    if (highPairs.length === 0) {
        DOM.pipelineCorrPairsContainer.innerHTML = `
            <div class="pipeline-alert pipeline-alert-success">
                <div style="font-size: 18px;"><span class="ico ico-check" aria-hidden="true"></span></div>
                <div>
                    <strong>No Excessive Pairwise Collinearity Detected:</strong> All continuous variable pairs have |correlation| &lt; 0.80. No automatic dropping is strictly required; you may proceed to Step 3.
                </div>
            </div>
        `;
    } else {
        let pairsHtml = '';
        highPairs.forEach((p, idx) => {
            const var1Dropped = AppState.pipeline.droppedFeatures.has(p.var1);
            const var2Dropped = AppState.pipeline.droppedFeatures.has(p.var2);

            pairsHtml += `
                <div class="corr-pair-card high-corr ${var1Dropped || var2Dropped ? 'dropped' : ''}">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                            <strong style="font-size: 14px; color: var(--text-primary);">${escapeHtml(p.var1)}</strong>
                            <span style="color: var(--text-muted);">&harr;</span>
                            <strong style="font-size: 14px; color: var(--text-primary);">${escapeHtml(p.var2)}</strong>
                            <span class="vif-badge-danger">${p.method} = ${Number(p.coeff).toFixed(3)}</span>
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary);">
                            <strong>Method:</strong> ${p.bothNormal ? 'Both variables approximately symmetric (|G₁| ≤ 1) &rarr; Pearson product-moment correlation.' : `Skewness detected (${escapeHtml([p.norm1.isNormal ? '' : p.var1, p.norm2.isNormal ? '' : p.var2].filter(Boolean).join(', '))}) &rarr; Spearman rank correlation.`}
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <button class="btn btn-rose btn-sm" ${actionAttrs('pipelineDropFeature', p.var1)} ${var1Dropped ? 'disabled' : ''}>
                            ${var1Dropped ? `✓Dropped ${escapeHtml(p.var1)}` : `Drop ${escapeHtml(p.var1)}`}
                        </button>
                        <button class="btn btn-rose btn-sm" ${actionAttrs('pipelineDropFeature', p.var2)} ${var2Dropped ? 'disabled' : ''}>
                            ${var2Dropped ? `✓Dropped ${escapeHtml(p.var2)}` : `Drop ${escapeHtml(p.var2)}`}
                        </button>
                        <button class="btn btn-secondary btn-sm" ${actionAttrs('pipelineKeepBothPairs', p.var1, p.var2)}>
                            Keep Both
                        </button>
                    </div>
                </div>
            `;
        });
        DOM.pipelineCorrPairsContainer.innerHTML = pairsHtml;
    }
}

window.pipelineDropFeature = function (featureName) {
    AppState.pipeline.droppedFeatures.add(featureName);
    outcomeFreeDrops().add(featureName); // Step 2 (correlation) does not use the outcome
    renderPipelineStep2();
    if (AppState.pipeline.substep === 3) renderPipelineStep3();
};

window.pipelineKeepBothPairs = function (var1, var2) {
    alert(`Pair [${var1}] and [${var2}] retained. Record the clinical justification for keeping both in your study protocol.`);
};

// =========================================================================
// STEP 3: Variance Inflation Factor (VIF) Check
// =========================================================================
function renderPipelineStep3() {
    if (!AppState.pipeline.trainRows) return;

    const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
    let vifResults;
    try {
        vifResults = DataEngine.calculateVIF(AppState.pipeline.trainRows, activeCols, AppState.pipeline.columnConfigs);
    } catch (err) {
        if (DOM.pipelineVifTableContainer) {
            DOM.pipelineVifTableContainer.innerHTML = `<div class="pipeline-alert pipeline-alert-warning"><div><span class="ico ico-warn" aria-hidden="true"></span></div><div>${escapeHtml(err.message)}</div></div>`;
        }
        if (DOM.pipelineDropVifBtn) DOM.pipelineDropVifBtn.disabled = true;
        return;
    }
    AppState.pipeline.vifResults = vifResults;

    const hasHighVif = vifResults.some(v => v.highVIF);
    if (DOM.pipelineDropVifBtn) DOM.pipelineDropVifBtn.disabled = !hasHighVif;

    let tableHtml = `
        <table class="data-table">
            <thead>
                <tr>
                    <th style="width: 40px; text-align: center;">Select</th>
                    <th>Feature Name</th>
                    <th>Measurement Scale</th>
                    <th>df (Encoded Columns)</th>
                    <th>Equivalent R²</th>
                    <th>VIF (GVIF<sup>1/df</sup> for df &gt; 1)</th>
                    <th>Collinearity Status</th>
                </tr>
            </thead>
            <tbody>
    `;

    vifResults.forEach(item => {
        const isHigh = item.highVIF;
        const isDropped = AppState.pipeline.droppedFeatures.has(item.feature);

        tableHtml += `
            <tr style="${isHigh ? 'background: rgba(244, 63, 94, 0.04);' : ''}">
                <td style="text-align: center;">
                    <input type="checkbox" class="pipeline-vif-checkbox" data-feature="${escapeHtml(item.feature)}" ${isHigh && !isDropped ? 'checked' : ''} ${isDropped ? 'disabled' : ''}>
                </td>
                <td><strong>${escapeHtml(item.feature)}</strong> ${isDropped ? '<span class="pill pill-meta" style="font-size: 10px;">Dropped</span>' : ''}</td>
                <td><span class="pill pill-${item.type}">${item.type}</span></td>
                <td style="font-size: 12px; color: var(--text-secondary);">${item.dummyCount}</td>
                <td>${item.rSquared}</td>
                <td><strong style="color: ${isHigh ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; font-size: 13.5px;">${item.vif >= 999 ? '≥ 999 (≈ perfect collinearity)' : item.vif}</strong>${item.dummyCount > 1 && item.vif < 999 ? `<div style="font-size: 10.5px; color: var(--text-muted);">GVIF ${item.gvif}; GVIF<sup>1/(2·df)</sup> ${item.adjustedGVIF}</div>` : ''}</td>
                <td>
                    <span class="${isHigh ? 'vif-badge-danger' : 'vif-badge-ok'}">
                        ${isHigh ? '⚠︎ VIF > 5.0 (High Multicollinearity)' : '✓VIF ≤ 5.0 (Acceptable)'}
                    </span>
                </td>
            </tr>
        `;
    });

    tableHtml += `</tbody></table>`;
    if (DOM.pipelineVifTableContainer) DOM.pipelineVifTableContainer.innerHTML = tableHtml;
}

function pipelineDropSelectedVIF() {
    const checkboxes = document.querySelectorAll('.pipeline-vif-checkbox:checked');
    if (checkboxes.length === 0) {
        alert("No variables selected to eliminate.");
        return;
    }

    const droppedNames = [];
    checkboxes.forEach(cb => {
        const f = cb.dataset.feature;
        if (f) {
            AppState.pipeline.droppedFeatures.add(f);
            outcomeFreeDrops().add(f); // Step 3 (VIF) does not use the outcome
            droppedNames.push(f);
        }
    });

    alert(`Excluded ${droppedNames.length} feature(s) from the pipeline (Train and Test): ${droppedNames.join(', ')}. They can be restored in Step 4.`);
    renderPipelineStep3();
}

// =========================================================================
// STEP 4: Univariate Statistical Testing Check
// =========================================================================
// Predictors removed in Steps 2–3 (correlation, VIF), which never look at the outcome
function outcomeFreeDrops() {
    if (!AppState.pipeline.outcomeFreeDrops) AppState.pipeline.outcomeFreeDrops = new Set();
    return AppState.pipeline.outcomeFreeDrops;
}

// Step 4 screening uses the outcome, so cross-validation repeats it inside every fold (cv-tuning.js). Each Step 4
// decision becomes a rule: a predictor kept because it was significant, or dropped because it was not, is kept in a
// fold only if it is significant on that fold's training patients; the user's overrides (keeping a non-significant
// predictor, dropping a significant one) are kept as they are.
function recordUnivariateScreen(results) {
    const P = AppState.pipeline;
    const sig = new Map(results.map(r => [r.feature, Boolean(r.significant)]));
    const candidates = P.featureCols.filter(c => !outcomeFreeDrops().has(c) && sig.has(c));
    const dropped = c => P.droppedFeatures.has(c);
    P.univScreen = {
        active: candidates.some(dropped),
        rule: candidates.filter(c => dropped(c) !== sig.get(c)),
        alwaysKeep: candidates.filter(c => !dropped(c) && !sig.get(c)),
        alwaysDrop: candidates.filter(c => dropped(c) && sig.get(c))
    };
}

function renderPipelineStep4() {
    if (!AppState.pipeline.trainRows) return;

    // Run univariate test on all pipeline features so user can see rankings and restore any previously dropped variables
    const allFeatures = AppState.pipeline.featureCols;
    const results = DataEngine.runUnivariateTesting(
        AppState.pipeline.trainRows,
        allFeatures,
        AppState.pipeline.targetCol,
        AppState.pipeline.columnConfigs
    );

    AppState.pipeline.univariateResults = results;
    recordUnivariateScreen(results);

    const activeCount = allFeatures.filter(c => !AppState.pipeline.droppedFeatures.has(c)).length;
    const sigCount = results.filter(r => r.significant && !AppState.pipeline.droppedFeatures.has(r.feature)).length;
    const nonSigCount = activeCount - sigCount;

    if (DOM.pipelineUnivSelectedSummary) {
        DOM.pipelineUnivSelectedSummary.innerHTML = `
            Active in Pipeline: <strong>${activeCount} / ${allFeatures.length}</strong> features 
            (${sigCount} significant, ${nonSigCount} non-significant). 
            ${AppState.pipeline.droppedFeatures.size > 0 ? `<span style="color: var(--accent-rose);">(${AppState.pipeline.droppedFeatures.size} currently excluded)</span>` : ''}
        `;
    }

    let tableHtml = `
        <table class="data-table">
            <thead>
                <tr>
                    <th style="width: 40px; text-align: center;">
                        <input type="checkbox" id="pipeline-univ-master-check" title="Toggle all active" checked>
                    </th>
                    <th style="width: 50px;">Rank</th>
                    <th>Feature</th>
                    <th>p-value</th>
                    <th style="width: 110px;">Action</th>
                    <th>Significance</th>
                    <th>Measurement Scale</th>
                    <th>Statistical Test</th>
                    <th>Test Statistic</th>
                    <th>Degrees of Freedom</th>
                </tr>
            </thead>
            <tbody>
    `;

    results.forEach((r, idx) => {
        const isSig = r.significant;
        const isDropped = AppState.pipeline.droppedFeatures.has(r.feature);

        tableHtml += `
            <tr style="${isDropped ? 'opacity: 0.55; background: rgba(244, 63, 94, 0.04);' : (isSig ? 'background: rgba(16, 185, 129, 0.02);' : '')}">
                <td style="text-align: center;">
                    <input type="checkbox" class="pipeline-univ-checkbox" data-feature="${escapeHtml(r.feature)}" data-sig="${isSig}" ${!isDropped ? 'checked' : ''}>
                </td>
                <td style="font-weight: 700; color: var(--accent-cyan);">#${idx + 1}</td>
                <td>
                    <strong style="${isDropped ? 'text-decoration: line-through;' : ''}">${escapeHtml(r.feature)}</strong>
                    ${isDropped ? '<span class="pill pill-meta" style="font-size: 10px; margin-left: 6px;">Excluded</span>' : ''}
                </td>
                <td><strong style="color: ${isSig ? 'var(--accent-emerald)' : 'var(--text-muted)'}; font-size: 13.5px;">${r.pValue < 0.0001 ? '< 0.0001' : r.pValue}</strong></td>
                <td>
                    ${isDropped ? `
                        <button class="btn btn-secondary btn-sm" ${actionAttrs('pipelineRestoreFeatureStep4', r.feature)} title="Restore variable into modeling pipeline">
                            ↩️ Include
                        </button>
                    ` : `
                        <button class="btn btn-rose btn-sm" ${actionAttrs('pipelineDropFeatureStep4', r.feature)} title="Exclude this variable from modeling">
                            Exclude
                        </button>
                    `}
                </td>
                <td>
                    <span class="pill" style="${isSig ? 'background: rgba(16, 185, 129, 0.15); color: var(--tag-emerald); border-color: rgba(16, 185, 129, 0.4);' : 'background: rgba(100, 116, 139, 0.15); color: var(--text-muted);'}">
                        ${isSig ? '✓p < 0.05 (Sig)' : 'p ≥ 0.05 (Non-Sig)'}
                    </span>
                </td>
                <td><span class="pill pill-${r.scale.toLowerCase().includes('nominal') ? 'nominal' : 'continuous'}">${r.scale}</span></td>
                <td style="font-weight: 600;">${escapeHtml(r.testName)}</td>
                <td>${escapeHtml(r.statisticLabel)}: <strong>${r.statisticValue === Infinity ? '∞' : r.statisticValue}</strong></td>
                <td style="font-size: 12px; color: var(--text-secondary);">${r.df}</td>
            </tr>
        `;
    });

    tableHtml += `</tbody></table>`;
    if (DOM.pipelineUnivariateTableContainer) {
        DOM.pipelineUnivariateTableContainer.innerHTML = tableHtml;

        // Master checkbox event
        const masterCheck = document.getElementById('pipeline-univ-master-check');
        if (masterCheck) {
            masterCheck.addEventListener('change', (e) => {
                const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
                cbs.forEach(cb => cb.checked = e.target.checked);
                updateUnivSelectionSummary();
            });
        }

        // Individual checkbox events
        const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
        cbs.forEach(cb => {
            cb.addEventListener('change', updateUnivSelectionSummary);
        });
    }
    renderPipelineBudget();
}

// How many predictors the pipeline Training set supports, for the ticked predictors (all predictors that
// entered the pipeline count as candidates considered)
function renderPipelineBudget() {
    const P = AppState.pipeline;
    if (!P || !P.trainRows) return;
    const cbs = Array.from(document.querySelectorAll('.pipeline-univ-checkbox'));
    const ticked = cbs.length ? cbs.filter(cb => cb.checked).map(cb => cb.dataset.feature).filter(Boolean) : P.featureCols.filter(c => !P.droppedFeatures.has(c));
    renderPredictorBudget('pipeline-budget-container', {
        rows: P.trainRows, featureCols: ticked, targetCol: P.targetCol, columnConfigs: P.columnConfigs,
        trainLabel: 'the pipeline Training set', candidateCols: P.featureCols, groupOf: pipelineGroupFunction()
    });
}

function updateUnivSelectionSummary() {
    renderPipelineBudget();
    const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
    const checkedCount = Array.from(cbs).filter(cb => cb.checked).length;
    if (DOM.pipelineUnivSelectedSummary) {
        DOM.pipelineUnivSelectedSummary.innerHTML = `
            Checked for Inclusion: <strong>${checkedCount} / ${cbs.length}</strong> features. 
            Click <strong>"Exclude Unchecked Features"</strong> to apply changes to subsequent pipeline steps.
        `;
    }
}

function filterSignificantUnivariateFeatures() {
    const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
    let uncheckCount = 0;
    cbs.forEach(cb => {
        const isSig = cb.dataset.sig === 'true';
        if (!isSig) {
            cb.checked = false;
            uncheckCount++;
        } else {
            cb.checked = true;
        }
    });
    updateUnivSelectionSummary();
    alert(`Unchecked ${uncheckCount} non-significant feature(s) (p ≥ 0.05). Click "Exclude Unchecked Features" to finalize.`);
}

function selectAllUnivariateFeatures() {
    const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
    cbs.forEach(cb => cb.checked = true);
    const masterCheck = document.getElementById('pipeline-univ-master-check');
    if (masterCheck) masterCheck.checked = true;
    updateUnivSelectionSummary();
}

function dropUnselectedUnivariateFeatures() {
    const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
    let newlyDropped = [];
    let newlyRestored = [];

    cbs.forEach(cb => {
        const f = cb.dataset.feature;
        if (!f) return;
        if (!cb.checked) {
            if (!AppState.pipeline.droppedFeatures.has(f)) {
                AppState.pipeline.droppedFeatures.add(f);
                newlyDropped.push(f);
            }
        } else {
            if (AppState.pipeline.droppedFeatures.has(f)) {
                AppState.pipeline.droppedFeatures.delete(f);
                newlyRestored.push(f);
            }
        }
    });

    const activeCount = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c)).length;
    if (activeCount === 0) {
        alert("Warning: At least 1 predictor variable must remain active in the pipeline!");
        return;
    }

    let msg = `Pipeline features updated successfully! (${activeCount} active)`;
    if (newlyDropped.length > 0) msg += `\nExcluded: ${newlyDropped.join(', ')}`;
    if (newlyRestored.length > 0) msg += `\nRestored: ${newlyRestored.join(', ')}`;
    alert(msg);

    renderPipelineStep4();
}

window.pipelineDropFeatureStep4 = function (featureName) {
    AppState.pipeline.droppedFeatures.add(featureName);
    renderPipelineStep4();
};

window.pipelineRestoreFeatureStep4 = function (featureName) {
    AppState.pipeline.droppedFeatures.delete(featureName);
    renderPipelineStep4();
};

export { renderPipelineStep2, renderPipelineStep3, pipelineDropSelectedVIF, renderPipelineStep4, updateUnivSelectionSummary, filterSignificantUnivariateFeatures, selectAllUnivariateFeatures, dropUnselectedUnivariateFeatures };
