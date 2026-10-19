// imputation.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { AppState, cssEscape, escapeHtml, fmtValue } from './core.js';
import { DOM } from './shell.js';
import { renderCompleteCaseBox } from './variables.js';
import { readMultipleImputation } from './multiple-imputation.js';


// =========================================================================
// STEP 3: MISSING DATA IMPUTATION
// =========================================================================
function renderImputationTable() {
    const { headers, rows } = AppState.parsedData;
    const analysis = AppState.columnAnalysis;

    // Complete-case option: while on, every used predictor is set to "drop rows with missing"
    // (previous choices are remembered and restored when it is switched off)
    const activeCols = headers.filter(h => AppState.columnConfigs[h].role !== 'meta');
    activeCols.forEach(h => {
        const c = AppState.columnConfigs[h];
        if (c.role !== 'feature') return;
        if (AppState.completeCase) {
            if (c.imputationStrategy !== 'drop' && c.prevImputationStrategy === undefined) c.prevImputationStrategy = c.imputationStrategy;
            c.imputationStrategy = 'drop';
        } else if (c.prevImputationStrategy !== undefined) {
            c.imputationStrategy = c.prevImputationStrategy;
            delete c.prevImputationStrategy;
        }
    });
    renderCompleteCaseBox(rows, activeCols);
    // One choice governs the analysis: complete-case analysis leaves nothing to impute, so multiple imputation is
    // not offered with it; with multiple imputation, MICE fills every incomplete predictor (no per-variable methods)
    const miEl = document.getElementById('mi-select');
    if (miEl) {
        if (AppState.completeCase && miEl.value !== 'single') { miEl.value = 'single'; miEl.dispatchEvent(new Event('change', { bubbles: true })); }
        miEl.disabled = Boolean(AppState.completeCase);
        miEl.title = AppState.completeCase ? 'Complete-case analysis is on: there is nothing to impute' : '';
    }
    const mice = !AppState.completeCase && readMultipleImputation() > 0;
    const methodsHint = document.getElementById('impute-methods-hint');
    if (methodsHint) methodsHint.textContent = mice ? 'Multiple imputation: MICE fills every incomplete predictor, in the final models, every cross-validation fold and every bootstrap resample.' : 'Continuous: Median, Mean, KNN, Constant • Ordinal: Mode, Median • Nominal: Mode, "Missing" category';

    let rowsHtml = '';
    let missingColCount = 0;

    headers.forEach(col => {
        const a = analysis[col];
        const cfg = AppState.columnConfigs[col];

        // Only show columns with missing values and non-meta roles
        if (cfg.role === 'meta') return;
        if (a.missingCount === 0) return;

        missingColCount++;

        if (cfg.role === 'target') {
            // Outcomes are never imputed: rows with a missing outcome are always excluded
            rowsHtml += `
                <tr>
                    <td style="font-weight: 600;">${escapeHtml(col)}</td>
                    <td><span class="pill pill-target">target</span></td>
                    <td><span class="pill pill-${cfg.type}">${cfg.type}</span></td>
                    <td><span class="pill pill-missing">${a.missingCount} / ${a.total} (${a.missingPercent}%)</span></td>
                    <td colspan="2" style="font-size: 12px; color: var(--text-secondary);">
                        Rows with a missing outcome are <strong>excluded</strong> (an outcome is never imputed).
                    </td>
                </tr>
            `;
            return;
        }

        let strategyOptionsHtml = '';
        if (cfg.type === 'continuous') {
            strategyOptionsHtml = `
                <option value="median" ${cfg.imputationStrategy === 'median' ? 'selected' : ''}>Median (${a.stats.median})</option>
                <option value="mean" ${cfg.imputationStrategy === 'mean' ? 'selected' : ''}>Mean (${a.stats.mean})</option>
                <option value="knn" ${cfg.imputationStrategy === 'knn' ? 'selected' : ''}>KNN (K-Nearest Neighbors)</option>
                <option value="mode" ${cfg.imputationStrategy === 'mode' ? 'selected' : ''}>Mode (${a.stats.mode})</option>
                <option value="constant" ${cfg.imputationStrategy === 'constant' ? 'selected' : ''}>Constant Number...</option>
                <option value="drop" ${cfg.imputationStrategy === 'drop' ? 'selected' : ''}>Drop Rows with Missing</option>
            `;
        } else if (cfg.type === 'ordinal') {
            strategyOptionsHtml = `
                <option value="mode" ${cfg.imputationStrategy === 'mode' ? 'selected' : ''}>Mode / Most Frequent (${escapeHtml(a.stats.mode)})</option>
                <option value="median" ${cfg.imputationStrategy === 'median' ? 'selected' : ''}>${a.isNumeric ? `Median Rank (${a.stats.median})` : 'Median Rank (text levels: falls back to mode)'}</option>
                <option value="constant" ${cfg.imputationStrategy === 'constant' ? 'selected' : ''}>Constant Rank...</option>
                <option value="drop" ${cfg.imputationStrategy === 'drop' ? 'selected' : ''}>Drop Rows with Missing</option>
            `;
        } else {
            strategyOptionsHtml = `
                <option value="mode" ${cfg.imputationStrategy === 'mode' ? 'selected' : ''}>Mode / Most Frequent ("${escapeHtml(a.stats.mode)}")</option>
                <option value="constant" ${cfg.imputationStrategy === 'constant' ? 'selected' : ''}>New Category ("Missing" / "Unknown")</option>
                <option value="drop" ${cfg.imputationStrategy === 'drop' ? 'selected' : ''}>Drop Rows with Missing</option>
            `;
        }

        if (mice) {
            rowsHtml += `
            <tr>
                <td style="font-weight: 600;">${escapeHtml(col)}</td>
                <td><span class="pill pill-${cfg.role}">${cfg.role}</span></td>
                <td><span class="pill pill-${cfg.type}">${cfg.type}</span></td>
                <td><span class="pill pill-missing">${a.missingCount} / ${a.total} (${a.missingPercent}%)</span></td>
                <td colspan="2"><span class="pill pill-feature">MICE (multiple imputation)</span></td>
            </tr>`;
            return;
        }

        rowsHtml += `
            <tr>
                <td style="font-weight: 600;">${escapeHtml(col)}</td>
                <td><span class="pill pill-${cfg.role}">${cfg.role}</span></td>
                <td><span class="pill pill-${cfg.type}">${cfg.type}</span></td>
                <td><span class="pill pill-missing">${a.missingCount} / ${a.total} (${a.missingPercent}%)</span></td>
                <td>
                    <select class="control-select impute-strat-select" data-col="${escapeHtml(col)}" aria-label="Missing-data method for ${escapeHtml(col)}" style="min-width: 200px;" ${AppState.completeCase ? 'disabled title="Complete-case option is on: rows with an empty cell are removed"' : ''}>
                        ${strategyOptionsHtml}
                    </select>
                </td>
                <td>
                    <input type="text" class="control-input impute-custom-val" data-col="${escapeHtml(col)}" aria-label="Constant fill value for ${escapeHtml(col)}" placeholder="${cfg.type === 'continuous' ? '0' : 'Missing'}" value="${escapeHtml(cfg.customImputeVal || '')}" style="${cfg.imputationStrategy === 'constant' ? 'display:inline-block; width: 110px;' : 'display:none; width: 110px;'}">
                </td>
            </tr>
        `;
    });

    // Button reflects the data: nothing to impute -> "Full data | Move Next" (validates and opens Step 4)
    if (DOM.applyImputationBtn) {
        const complete = missingColCount === 0;
        DOM.applyImputationBtn.dataset.mode = complete ? 'complete' : 'impute';
        DOM.applyImputationBtn.textContent = complete ? '✓Full data | Move Next'
            : (AppState.completeCase ? 'Remove Incomplete Rows & Validate Dataset' : mice ? 'Continue with Multiple Imputation (MICE)' : 'Apply Imputation & Validate Dataset');
    }

    if (missingColCount === 0) {
        DOM.imputeTableBody.innerHTML = `
            <tr>
                <td colspan="6" style="text-align: center; padding: 28px; color: var(--accent-emerald);">
                    ✓No missing data detected in active variables. The dataset is complete and ready for machine learning!
                </td>
            </tr>
        `;
    } else {
        DOM.imputeTableBody.innerHTML = rowsHtml;
    }

    // Attach listeners
    document.querySelectorAll('.impute-strat-select').forEach(sel => {
        sel.addEventListener('change', (e) => {
            const col = e.target.dataset.col;
            const strat = e.target.value;
            AppState.columnConfigs[col].imputationStrategy = strat;
            const customInput = document.querySelector(`.impute-custom-val[data-col="${cssEscape(col)}"]`);
            if (customInput) {
                customInput.style.display = strat === 'constant' ? 'inline-block' : 'none';
            }
        });
    });

    document.querySelectorAll('.impute-custom-val').forEach(input => {
        input.addEventListener('input', (e) => {
            const col = e.target.dataset.col;
            AppState.columnConfigs[col].customImputeVal = e.target.value;
        });
    });
}

function executeImputation() {
    try {
        const { rows } = AppState.parsedData;
        const m = AppState.completeCase ? 0 : readMultipleImputation();
        // With multiple imputation nothing is filled now: MICE runs when models are trained (and inside every
        // cross-validation fold and bootstrap resample). A plain fill is kept only for the Step 4 counts and
        // sample-size panel, never for training; it keeps every patient ("drop rows" does not apply to MICE)
        const cfgs = m ? Object.fromEntries(Object.entries(AppState.columnConfigs).map(([c, v]) => [c, v.imputationStrategy === 'drop' || v.imputationStrategy === 'knn' ? { ...v, imputationStrategy: v.type === 'continuous' ? 'median' : 'mode' } : v])) : AppState.columnConfigs;
        const res = DataEngine.imputeDataset(rows, cfgs, AppState.columnAnalysis);
        AppState.imputedData = res;
        if (m) {
            const cells = res.audit.filter(x => x.type !== 'drop_rows').reduce((a, x) => a + (x.imputedCount || 0), 0);
            DOM.imputeAuditBox.innerHTML = `
            <div style="background: var(--bg-surface-elevated); border: 1px solid var(--accent-emerald); border-radius: var(--radius-md); padding: 16px;">
                <h4 style="color: var(--accent-emerald);">✓Multiple imputation selected: ${rows.length} records kept</h4>
                <p style="margin: 8px 0 0; font-size: 13px; color: var(--text-secondary);">${cells} missing value(s) in ${res.audit.filter(x => x.type !== 'drop_rows').length} predictor(s) will be filled by MICE when the models are trained (${m} completed datasets, each built over 10 rounds), from the training patients only and without the outcome. The same procedure is repeated inside every cross-validation fold and bootstrap resample; no single-imputation method is used.</p>
            </div>`;
            DOM.toStep4Btn.disabled = false;
            return;
        }

        let auditHtml = `
            <div style="background: var(--bg-surface-elevated); border: 1px solid var(--accent-emerald); border-radius: var(--radius-md); padding: 16px;">
                <h4 style="color: var(--accent-emerald); display: flex; align-items: center; gap: 8px;">
                    ${AppState.completeCase
                        ? `✓Complete-case dataset ready: ${res.totalRowsAfter} complete records (${rows.length - res.totalRowsAfter} row(s) with an empty cell removed; no imputation)`
                        : `✓Imputation Complete: ${res.totalRowsAfter} clean records ready`}
                </h4>
                <ul style="margin: 10px 0 0 20px; font-size: 13px; color: var(--text-secondary);">
                    ${res.audit.length === 0 ? '<li>No missing cells required imputation.</li>' : res.audit.map(a => `
                        <li>${a.type === 'drop_rows'
                            ? `<strong>Rows</strong>: ${escapeHtml(a.description)}`
                            : `<strong>${escapeHtml(a.column)}</strong>: ${escapeHtml(a.strategy)} &rarr; ${a.imputedCount} value(s) imputed (fill: ${escapeHtml(fmtValue(a.filledValue))})`}</li>
                    `).join('')}
                </ul>
            </div>
        `;
        DOM.imputeAuditBox.innerHTML = auditHtml;
        DOM.toStep4Btn.disabled = false;
    } catch (e) {
        alert(`Imputation error: ${e.message}`);
    }
}

export { renderImputationTable, executeImputation };
