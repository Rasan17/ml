// variables.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { AppState, escapeHtml } from './core.js';
import { renderImputationTable } from './imputation.js';
import { DOM } from './shell.js';
import { renderPredictorBudget } from './predictor-budget.js';
import { splitGroupFunction } from './cv-tuning.js';


// =========================================================================
// STEP 2: VARIABLE ROLES & TYPES CONFIGURATION
// =========================================================================
function renderVariableConfigGrid() {
    const { headers } = AppState.parsedData;
    const analysis = AppState.columnAnalysis;

    let gridHtml = '';
    headers.forEach((col, colIdx) => {
        const a = analysis[col];
        const cfg = AppState.columnConfigs[col];

        gridHtml += `
            <div class="var-card ${cfg.role === 'target' ? 'is-target' : (cfg.role === 'meta' ? 'is-meta' : '')}" id="card-col-${escapeHtml(col)}">
                <div class="var-header">
                    <label class="var-include" title="${cfg.role === 'target' ? 'The outcome is always included; change its role to exclude it' : 'Tick to include this variable in the analysis; untick to exclude it'}">
                        <input type="checkbox" class="var-include-check" data-col="${escapeHtml(col)}" ${cfg.role !== 'meta' ? 'checked' : ''} ${cfg.role === 'target' ? 'disabled' : ''} aria-label="Include ${escapeHtml(col)} in the analysis">
                        <span class="var-name" title="${escapeHtml(col)}">${escapeHtml(col)}</span>
                    </label>
                    ${a.missingCount > 0 ? `<span class="pill pill-missing">${a.missingCount} missing (${a.missingPercent}%)</span>` : `<span class="pill" style="background: rgba(16, 185, 129, 0.1); color: var(--tag-emerald);">Complete</span>`}
                </div>

                <div style="font-size: 11.5px; color: var(--text-muted);">
                    ${a.isNumeric ? `Numeric (range: ${a.stats.min} to ${a.stats.max})` : `Distinct values: ${a.distinctCount} (e.g. ${escapeHtml(a.distinctValues.slice(0, 3).join(', '))})`}
                </div>

                <div class="var-row-control">
                    <label for="var-role-${colIdx}">Variable Role</label>
                    <select id="var-role-${colIdx}" class="control-select var-role-select" data-col="${escapeHtml(col)}">
                        <option value="feature" ${cfg.role === 'feature' ? 'selected' : ''}>Predictor Feature</option>
                        <option value="target" ${cfg.role === 'target' ? 'selected' : ''}>Target (Outcome to Predict)</option>
                        <option value="meta" ${cfg.role === 'meta' ? 'selected' : ''}>Meta / ID (Ignore from Model)</option>
                    </select>
                </div>

                <div class="var-row-control">
                    <label for="var-type-${colIdx}">Variable Type</label>
                    <select id="var-type-${colIdx}" class="control-select var-type-select" data-col="${escapeHtml(col)}">
                        <option value="continuous" ${cfg.type === 'continuous' ? 'selected' : ''}>Continuous (Numerical / Scale)</option>
                        <option value="ordinal" ${cfg.type === 'ordinal' ? 'selected' : ''}>Ordinal (Ranked Categorical)</option>
                        <option value="nominal" ${cfg.type === 'nominal' ? 'selected' : ''}>Nominal (Unordered Categorical)</option>
                    </select>
                </div>
                ${a.autoMetaReason && cfg.role === 'meta' ? `<div style="font-size: 11px; color: var(--text-muted);">ℹ️ ${escapeHtml(a.autoMetaReason)}</div>` : ''}
                ${renderTargetAndOrdinalControls(col, colIdx, a, cfg)}
                ${cfg.role === 'feature' && cfg.type === 'nominal' && (a.distinctCount > 20 || (a.validCount > 0 && a.distinctCount / a.validCount > 0.5))
                    ? `<div style="font-size: 11px; color: var(--accent-amber);">⚠️ High cardinality (${a.distinctCount} levels): one-hot encoding adds a column per level, which slows training and invites overfitting. If this is an identifier, set it to Meta / ID; otherwise consider grouping rare levels.</div>`
                    : ''}
            </div>
        `;
    });

    // Up-front size check (mirrors the engine's limit): warn before training would be refused
    const encodedWidth = headers.reduce((sum, h) => {
        const c = AppState.columnConfigs[h];
        if (c.role !== 'feature') return sum;
        return sum + (c.type === 'nominal' ? analysis[h].distinctCount : 1);
    }, 0);
    if (encodedWidth > 2000) {
        const worst = headers.filter(h => AppState.columnConfigs[h].role === 'feature' && AppState.columnConfigs[h].type === 'nominal' && analysis[h].distinctCount > 50)
            .sort((x, y) => analysis[y].distinctCount - analysis[x].distinctCount)
            .map(h => `${h} (${analysis[h].distinctCount.toLocaleString('en-GB')})`);
        gridHtml = `<div style="grid-column: 1 / -1; padding: 12px 14px; border-radius: var(--radius-sm); background: rgba(244, 63, 94, 0.12); border: 1px solid var(--accent-rose); color: var(--text-primary); font-size: 13px;">
            ⚠️ <strong>Too many encoded predictor columns (${encodedWidth.toLocaleString('en-GB')}; limit 2,000).</strong>
            Categorical predictors are one-hot encoded, one column per category, and this would exceed browser memory.
            ${worst.length ? `Set these to <strong>Meta / ID</strong>: ${escapeHtml(worst.join(', '))}.` : ''}
        </div>` + gridHtml;
    }

    DOM.varConfigGrid.innerHTML = gridHtml;
    renderVariableIncludeSummary();

    // Attach listeners for role & type updates
    document.querySelectorAll('.var-role-select').forEach(sel => {
        sel.addEventListener('change', (e) => {
            const col = e.target.dataset.col;
            const newRole = e.target.value;

            if (newRole === 'target') {
                // Reset any previously selected target
                Object.keys(AppState.columnConfigs).forEach(otherCol => {
                    if (otherCol !== col && AppState.columnConfigs[otherCol].role === 'target') {
                        AppState.columnConfigs[otherCol].role = 'feature';
                        const otherCard = document.getElementById(`card-col-${otherCol}`);
                        if (otherCard) {
                            otherCard.classList.remove('is-target');
                            otherCard.querySelector('.var-role-select').value = 'feature';
                        }
                    }
                });
            }

            AppState.columnConfigs[col].role = newRole;
            // Re-render so target-only controls (positive class) appear/disappear
            renderVariableConfigGrid();
        });
    });

    document.querySelectorAll('.var-include-check').forEach(box => {
        box.addEventListener('change', (e) => {
            setVariableIncluded(e.target.dataset.col, e.target.checked);
            renderVariableConfigGrid();
        });
    });

    document.querySelectorAll('.var-type-select').forEach(sel => {
        sel.addEventListener('change', (e) => {
            const col = e.target.dataset.col;
            AppState.columnConfigs[col].type = e.target.value;
            // update default suggested imputation strategy in step 3
            if (e.target.value === 'continuous') {
                AppState.columnConfigs[col].imputationStrategy = 'median';
            } else {
                AppState.columnConfigs[col].imputationStrategy = 'mode';
            }
            // Re-render so ordinal / positive-class controls reflect the new type
            renderVariableConfigGrid();
        });
    });

    document.querySelectorAll('.var-positive-select').forEach(sel => {
        sel.addEventListener('change', (e) => {
            AppState.columnConfigs[e.target.dataset.col].positiveClass = e.target.value;
        });
    });

    document.querySelectorAll('.var-ordinal-order').forEach(input => {
        input.addEventListener('change', (e) => {
            const col = e.target.dataset.col;
            const levels = e.target.value.split(',').map(v => v.trim()).filter(v => v.length > 0);
            const known = (AppState.columnAnalysis[col]?.distinctValues || []).map(v => String(v).trim().toLowerCase());
            const unknown = levels.filter(l => !known.includes(l.toLowerCase()));
            AppState.columnConfigs[col].ordinalOrder = levels.length > 0 ? levels : undefined;
            const hint = document.getElementById(e.target.id + '-hint');
            if (hint) {
                hint.textContent = unknown.length > 0
                    ? `⚠️ Not found in the data (ignored): ${unknown.join(', ')}`
                    : (levels.length > 0 ? '✓ Custom order saved' : 'Using automatic order');
            }
        });
    });
}

// Include (ticked) / exclude (unticked) a variable. Excluded variables use the existing Meta / ID role, which
// every step already ignores; the role held before exclusion is remembered so re-ticking restores it.
function setVariableIncluded(col, include) {
    const cfg = AppState.columnConfigs[col];
    if (!cfg || cfg.role === 'target') return;
    if (!include) {
        if (cfg.role !== 'meta') { cfg.prevRole = cfg.role; cfg.role = 'meta'; }
        return;
    }
    if (cfg.role !== 'meta') return;
    const otherTarget = Object.keys(AppState.columnConfigs).some(c => c !== col && AppState.columnConfigs[c].role === 'target');
    cfg.role = cfg.prevRole === 'target' && !otherTarget ? 'target' : 'feature';
    delete cfg.prevRole;
}

// Columns the app itself flagged as identifiers or dates: "Include all" leaves these unticked
function isAutoExcludedColumn(col) {
    const a = AppState.columnAnalysis[col];
    return Boolean(a && a.role === 'meta');
}

function renderVariableIncludeSummary() {
    const el = document.getElementById('var-include-summary');
    if (!el) return;
    const headers = AppState.parsedData.headers;
    const cfgs = AppState.columnConfigs;
    const target = headers.find(h => cfgs[h].role === 'target');
    const included = headers.filter(h => cfgs[h].role === 'feature').length;
    const excluded = headers.filter(h => cfgs[h].role === 'meta').length;
    el.textContent = `${included} predictor(s) included, ${excluded} excluded` + (target ? `; outcome: ${target}` : '; no outcome selected yet');
    // How many predictors the sample supports, for the share of patients that will train the model
    const mode = document.querySelector('input[name="validation-mode"]:checked')?.value || 'split';
    const pct = Math.min(95, Math.max(50, parseFloat(document.getElementById('train-pct-input')?.value) || 80));
    renderPredictorBudget('predictor-budget-container', {
        rows: AppState.parsedData.rows, featureCols: headers.filter(h => cfgs[h].role === 'feature'), targetCol: target, columnConfigs: cfgs,
        trainFraction: mode === 'split' ? pct / 100 : 1,
        trainLabel: mode === 'split' ? `the ${pct}% Training set of Option A; Options B and C train on all loaded patients` : (mode === 'bootstrap' ? 'all patients, Option C' : 'all loaded patients, Option B'),
        groupOf: splitGroupFunction()
    });
}

function setAllVariablesIncluded(include) {
    const headers = AppState.parsedData.headers;
    let skipped = 0;
    headers.forEach(col => {
        if (include && isAutoExcludedColumn(col)) {
            if (AppState.columnConfigs[col].role === 'meta') skipped++;
            return;
        }
        setVariableIncluded(col, include);
    });
    renderVariableConfigGrid();
    if (skipped > 0) {
        const el = document.getElementById('var-include-summary');
        if (el) el.textContent += ` (${skipped} identifier/date column(s) left unticked; tick them individually if needed)`;
    }
}

// Extra Step 2 controls: positive (event) class for a binary target, level order for ordinals
function renderTargetAndOrdinalControls(col, colIdx, a, cfg) {
    let html = '';
    const levels = (a.distinctValues || []).map(v => String(v));
    if (cfg.role === 'target' && cfg.type !== 'continuous' && levels.length === 2) {
        const sorted = [...levels].sort();
        const positive = cfg.positiveClass !== undefined && levels.includes(String(cfg.positiveClass)) ? String(cfg.positiveClass) : sorted[1];
        html += `
            <div class="var-row-control">
                <label for="var-positive-${colIdx}">Positive (event) class</label>
                <select id="var-positive-${colIdx}" class="control-select var-positive-select" data-col="${escapeHtml(col)}">
                    ${sorted.map(v => `<option value="${escapeHtml(v)}" ${v === positive ? 'selected' : ''}>${escapeHtml(v)}</option>`).join('')}
                </select>
            </div>
            <div style="font-size: 11px; color: var(--text-muted);">Used as "positive" for ROC-AUC, Brier score, sensitivity, PPV and TP/FN.</div>
        `;
    }
    if (cfg.type === 'ordinal' && cfg.role !== 'meta' && levels.length > 1) {
        const autoOrder = DataEngine.sortOrdinalLevels(levels);
        const current = Array.isArray(cfg.ordinalOrder) && cfg.ordinalOrder.length ? cfg.ordinalOrder : autoOrder;
        html += `
            <div class="var-row-control" style="flex-direction: column; align-items: stretch; gap: 4px;">
                <label for="var-order-${colIdx}">Level order (lowest &rarr; highest, comma-separated)</label>
                <input type="text" id="var-order-${colIdx}" class="control-input var-ordinal-order" data-col="${escapeHtml(col)}" value="${escapeHtml(current.join(', '))}">
                <span id="var-order-${colIdx}-hint" style="font-size: 11px; color: var(--text-muted);">${Array.isArray(cfg.ordinalOrder) && cfg.ordinalOrder.length ? '✓ Custom order saved' : 'Automatic order: please check it matches the clinical order'}</span>
            </div>
        `;
    }
    return html;
}

// Rows with at least one empty cell among the variables used for modelling (predictors + outcome)
function countIncompleteRows(rows, activeCols) {
    let n = 0;
    for (let i = 0; i < rows.length; i++) {
        for (let j = 0; j < activeCols.length; j++) {
            if (DataEngine.isMissing(rows[i][activeCols[j]])) { n++; break; }
        }
    }
    return n;
}

function renderCompleteCaseBox(rows, activeCols) {
    const box = document.getElementById('complete-case-box');
    if (!box) return;
    const incomplete = countIncompleteRows(rows, activeCols);
    if (incomplete === 0) {
        AppState.completeCase = false;
        box.innerHTML = '';
        return;
    }
    const total = rows.length;
    const remaining = total - incomplete;
    const pct = (100 * incomplete / total).toFixed(1);
    const heavy = incomplete / total > 0.2 || remaining < 30;
    box.innerHTML = `
        <div style="margin-bottom: 16px; padding: 12px 14px; border-radius: var(--radius-sm); border: 1px solid ${AppState.completeCase ? 'var(--accent-cyan)' : 'var(--border-color)'}; background: var(--bg-surface-subtle);">
            <label for="complete-case-toggle" style="display: flex; gap: 10px; align-items: flex-start; cursor: pointer; font-size: 13px; color: var(--text-primary);">
                <input type="checkbox" id="complete-case-toggle" ${AppState.completeCase ? 'checked' : ''} style="margin-top: 3px; accent-color: var(--accent-cyan);">
                <span>
                    <strong>Remove all rows with one or more empty cells</strong> (complete-case analysis) instead of imputing.
                    <span style="display: block; font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                        ${incomplete.toLocaleString('en-GB')} of ${total.toLocaleString('en-GB')} rows (${pct}%) have an empty cell in the variables used for modelling
                        (predictors and outcome; columns set to Meta / ID are ignored). ${remaining.toLocaleString('en-GB')} complete rows would remain.
                        The same rule is applied to the test data.
                    </span>
                    ${heavy ? `<span style="display: block; font-size: 12px; color: var(--accent-amber); margin-top: 4px;">
                        ⚠️ This removes a large share of the data${remaining < 30 ? ' and leaves very few rows' : ''}. Complete-case analysis is unbiased only if
                        values are missing completely at random; otherwise imputation is usually preferable.</span>` : ''}
                </span>
            </label>
            <label for="missing-indicator-toggle" style="display: flex; gap: 10px; align-items: flex-start; cursor: pointer; font-size: 13px; color: var(--text-primary); margin-top: 10px; ${AppState.completeCase ? 'opacity: 0.5;' : ''}">
                <input type="checkbox" id="missing-indicator-toggle" ${AppState.missingIndicators ? 'checked' : ''} ${AppState.completeCase ? 'disabled' : ''} style="margin-top: 3px; accent-color: var(--accent-cyan);">
                <span>
                    <strong>Add a "was missing" indicator for each predictor with empty cells</strong>
                    <span style="display: block; font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                        Adds a Yes/No predictor recording whether the value was missing, alongside the imputed value, so the models can learn from the fact that a measurement was not taken (e.g. a test not ordered). It is derived automatically for new patients. It can introduce bias if missingness reflects data-collection practice that differs elsewhere.
                    </span>
                </span>
            </label>
        </div>`;
    document.getElementById('missing-indicator-toggle')?.addEventListener('change', (e) => {
        setMissingIndicators(e.target.checked);
        AppState.imputedData = null;
        if (DOM.toStep4Btn) DOM.toStep4Btn.disabled = true;
        if (DOM.imputeAuditBox) DOM.imputeAuditBox.innerHTML = '';
        renderImputationTable();
    });
    document.getElementById('complete-case-toggle').addEventListener('change', (e) => {
        AppState.completeCase = e.target.checked;
        AppState.imputedData = null; // previous result no longer matches the chosen strategy
        if (DOM.toStep4Btn) DOM.toStep4Btn.disabled = true;
        if (DOM.imputeAuditBox) DOM.imputeAuditBox.innerHTML = '';
        renderImputationTable();
    });
}

// Adds (or removes) derived "was missing" indicator columns for every predictor that has empty cells in the
// training data; the same columns are derived for a separate testing file and for new patients
function setMissingIndicators(on) {
    const pd = AppState.parsedData;
    if (!pd) return;
    const cfgs = AppState.columnConfigs;
    // remove existing indicators
    Object.keys(cfgs).filter(c => cfgs[c].derivedFrom).forEach(ind => {
        delete cfgs[ind];
        if (AppState.columnAnalysis) delete AppState.columnAnalysis[ind];
        pd.headers = pd.headers.filter(h => h !== ind);
        pd.rows.forEach(r => { delete r[ind]; });
        (AppState.separateTestRawRows || []).forEach(r => { delete r[ind]; });
    });
    AppState.missingIndicators = Boolean(on);
    if (!on) return;
    const sources = pd.headers.filter(c => cfgs[c] && cfgs[c].role === 'feature' && pd.rows.some(r => DataEngine.isMissing(r[c])));
    sources.forEach(src => {
        const ind = DataEngine.missingIndicatorName(src);
        const derive = r => { r[ind] = DataEngine.isMissing(r[src]) ? 'Yes' : 'No'; };
        pd.rows.forEach(derive);
        (AppState.separateTestRawRows || []).forEach(derive);
        pd.headers.push(ind);
        cfgs[ind] = { role: 'feature', type: 'nominal', imputationStrategy: 'mode', derivedFrom: src };
        if (AppState.columnAnalysis) AppState.columnAnalysis[ind] = DataEngine.analyzeColumns([ind], pd.rows)[ind];
    });
}

export { setMissingIndicators, renderVariableConfigGrid, setVariableIncluded, isAutoExcludedColumn, renderVariableIncludeSummary, setAllVariablesIncluded, renderTargetAndOrdinalControls, countIncompleteRows, renderCompleteCaseBox };
