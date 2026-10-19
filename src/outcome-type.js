// outcome-type.js: Step 2 choice of the outcome type. A standard outcome (yes/no, categories or a number) is the
// column marked Target in the variable list; a time-to-event (survival) outcome is a follow-up time column plus an
// event column and the value meaning "event occurred". The same predictors, missing-data handling, validation
// design and saving apply to both; Step 4 then fits a Cox model for time-to-event outcomes.
/* global DataEngine */
import { AppState, escapeHtml } from './core.js';

function numericCols(rows, headers) {
    return headers.filter(h => { const v = rows.map(r => r[h]).filter(x => !DataEngine.isMissing(x)); return v.length > 0 && v.every(x => isFinite(Number(x))); });
}
function twoValueCols(rows, headers) {
    return headers.filter(h => new Set(rows.map(r => r[h]).filter(x => !DataEngine.isMissing(x)).map(x => String(x).trim())).size === 2);
}

// The survival specification in force, or null for a standard outcome
function survivalSpec() {
    return AppState.outcomeType === 'survival' && AppState.survival && AppState.survival.timeCol && AppState.survival.eventCol ? AppState.survival : null;
}

// Applies the specification to the variable roles: the event column is the Target (two values), the time column is
// kept out of the predictors (it is part of the outcome)
function applySurvivalRoles() {
    const s = survivalSpec(), cfgs = AppState.columnConfigs;
    Object.keys(cfgs).forEach(c => { if (cfgs[c].survivalTime) { delete cfgs[c].survivalTime; if (cfgs[c].role === 'meta') cfgs[c].role = cfgs[c].prevRole && cfgs[c].prevRole !== 'meta' ? cfgs[c].prevRole : 'feature'; } });
    if (!s) return;
    Object.keys(cfgs).forEach(c => { if (cfgs[c].role === 'target' && c !== s.eventCol) cfgs[c].role = 'feature'; });
    if (cfgs[s.eventCol]) { cfgs[s.eventCol].role = 'target'; cfgs[s.eventCol].type = 'nominal'; cfgs[s.eventCol].positiveClass = s.eventValue; }
    if (cfgs[s.timeCol]) { cfgs[s.timeCol].prevRole = cfgs[s.timeCol].role; cfgs[s.timeCol].role = 'meta'; cfgs[s.timeCol].survivalTime = true; }
}

function renderOutcomeTypePanel(onChange) {
    const box = document.getElementById('outcome-type-container');
    if (!box || !AppState.parsedData) return;
    const { headers, rows } = AppState.parsedData;
    const surv = AppState.outcomeType === 'survival';
    const timeCols = numericCols(rows, headers), eventCols = twoValueCols(rows, headers);
    const s = AppState.survival || {};
    const tc = s.timeCol && timeCols.includes(s.timeCol) ? s.timeCol : (timeCols.find(h => /time|month|day|year|follow|surv|duration/i.test(h)) || timeCols[0] || '');
    const ec = s.eventCol && eventCols.includes(s.eventCol) ? s.eventCol : (eventCols.find(h => /event|death|died|revision|status|outcome|fail|recur/i.test(h)) || eventCols[0] || '');
    const vals = ec ? [...new Set(rows.map(r => r[ec]).filter(v => !DataEngine.isMissing(v)).map(v => String(v).trim()))].sort() : [];
    const ev = s.eventValue && vals.includes(s.eventValue) ? s.eventValue : (vals.find(v => /^(yes|1|true|dead|died|event|revised|revision)$/i.test(v)) || vals[vals.length - 1] || '');
    const opts = (list, sel) => list.map(h => `<option value="${escapeHtml(h)}"${h === sel ? ' selected' : ''}>${escapeHtml(h)}</option>`).join('');
    box.innerHTML = `
        <div class="cal-panel outcome-type-panel">
            <div class="tune-title">🎯 What kind of outcome do you want to predict?</div>
            <div class="outcome-type-options" role="radiogroup" aria-label="Outcome type">
                <label><input type="radio" name="outcome-type" value="standard"${surv ? '' : ' checked'}> <strong>Yes/no, categories or a number</strong>: mark that column as <em>Target</em> in the list below</label>
                <label><input type="radio" name="outcome-type" value="survival"${surv ? ' checked' : ''}> <strong>Time to an event</strong> (survival, e.g. time to shunt revision), with patients followed for different lengths of time</label>
            </div>
            ${surv ? `<div class="surv-grid">
                <label class="form-group"><span class="form-label">Follow-up time column</span><select id="ot-time-col" class="form-select">${opts(timeCols, tc)}</select></label>
                <label class="form-group"><span class="form-label">Event column (did the event happen?)</span><select id="ot-event-col" class="form-select">${opts(eventCols, ec)}</select></label>
                <label class="form-group"><span class="form-label">Value meaning "event occurred"</span><select id="ot-event-value" class="form-select">${opts(vals, ev)}</select></label>
            </div>
            <p class="tune-muted">The time column is the time from the start of follow-up to the event, or to the last follow-up for patients without the event (censored). Both columns form the outcome: they are never used as predictors or to fill in missing values. Step 4 fits a Cox proportional-hazards model.</p>` : ''}
        </div>`;
    const commit = () => {
        const type = box.querySelector('input[name="outcome-type"]:checked')?.value || 'standard';
        AppState.outcomeType = type;
        if (type === 'survival') {
            const t = document.getElementById('ot-time-col'), e = document.getElementById('ot-event-col'), v = document.getElementById('ot-event-value');
            AppState.survival = { timeCol: t ? t.value : tc, eventCol: e ? e.value : ec, eventValue: v ? v.value : ev };
        }
        applySurvivalRoles();
        if (onChange) onChange();
    };
    box.querySelectorAll('input[name="outcome-type"]').forEach(r => r.addEventListener('change', () => { commit(); renderOutcomeTypePanel(onChange); }));
    ['ot-time-col', 'ot-event-col', 'ot-event-value'].forEach(id => document.getElementById(id)?.addEventListener('change', e => {
        if (id === 'ot-event-col' && AppState.survival) AppState.survival.eventValue = null;
        commit(); renderOutcomeTypePanel(onChange);
    }));
    if (surv && (!AppState.survival || AppState.survival.timeCol !== tc || AppState.survival.eventCol !== ec || AppState.survival.eventValue !== ev)) { AppState.survival = { timeCol: tc, eventCol: ec, eventValue: ev }; applySurvivalRoles(); }
}

export { applySurvivalRoles, renderOutcomeTypePanel, survivalSpec };
