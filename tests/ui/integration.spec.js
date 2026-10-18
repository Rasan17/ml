// Integration tests of combinations that cross several parts of the app: settings saved and restored with the
// newer controls; a multiply imputed and recalibrated model exported and re-imported (identical predictions);
// batch prediction with blanks and an out-of-range value; Option B with multiple imputation; the 7-step pipeline
// with recalibration and its printable report.
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', '..');
function watch(page) {
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('dialog', d => d.accept());
    return errors;
}
async function toStep4(page) {
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
}

test('settings with the newer controls are saved and restored', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await page.selectOption('#mi-select', 'mice10');
    await page.selectOption('#recalibration-select', 'isotonic');
    await page.selectOption('#rf-split-select', 'exact', { force: true }); // in the Random Forest settings panel (hidden until RF is configured)
    await page.check('input[name="validation-mode"][value="bootstrap"]', { force: true });
    await page.selectOption('#bootstrap-reps', '200');
    await page.selectOption('#bootstrap-group-col', 'Patient_ID');
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#settings-save-btn')]);
    const file = await dl.path();
    await page.goto('/index.html');
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.setInputFiles('#settings-file-input', file);
    await page.click('button.step-tab[data-step="4"]');
    await expect(page.locator('#mi-select')).toHaveValue('mice10');
    await expect(page.locator('#recalibration-select')).toHaveValue('isotonic');
    await expect(page.locator('#rf-split-select')).toHaveValue('exact');
    await expect(page.locator('input[name="validation-mode"][value="bootstrap"]')).toBeChecked();
    await expect(page.locator('#bootstrap-reps')).toHaveValue('200');
    await expect(page.locator('#bootstrap-group-col')).toHaveValue('Patient_ID');
    expect(errors).toEqual([]);
});

test('a multiply imputed, recalibrated model gives identical predictions after export and re-import', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await page.selectOption('#mi-select', 'mice5');
    await page.selectOption('#recalibration-select', 'platt');
    await page.locator('.model-card[data-model-id="glm"]').click();
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    await page.click('button.step-tab[data-step="5"]');
    await page.fill('#predict-field-11', '');
    await page.fill('#predict-field-12', '');
    await page.click('#run-single-predict-btn');
    const probs = async () => (await page.locator('#single-result-banner').innerText()).match(/(Responder|Non-Responder):\s*[\d.]+%/g).join(' ');
    const before = await probs();
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#model-export-btn')]);
    const file = await dl.path();
    const values = [];
    for (let k = 0; k < 14; k++) values.push(await page.locator(`#predict-field-${k}`).inputValue());
    await page.goto('/index.html');
    await page.setInputFiles('#model-file-input', file);
    await expect(page.locator('#step-5-content')).toBeVisible();
    for (let k = 0; k < 14; k++) {
        const el = page.locator(`#predict-field-${k}`);
        if (await el.evaluate(e => e.tagName) === 'SELECT') await el.selectOption(values[k]); else await el.fill(values[k]);
    }
    await page.click('#run-single-predict-btn');
    expect(await probs()).toBe(before);
    expect(errors).toEqual([]);
});

test('batch prediction with blanks and an out-of-range value', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await page.selectOption('#mi-select', 'mice5');
    await page.locator('.model-card[data-model-id="rf"]').click();
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    const lines = fs.readFileSync(path.join(ROOT, 'medical_shunt_validation_cohort.csv'), 'utf8').trim().split('\n').slice(0, 21);
    const head = lines[0].split(','), age = head.indexOf('Age');
    const rows = lines.slice(1).map((l, i) => { const v = l.split(','); if (i === 0) v[age] = '650'; return v.join(','); });
    await page.click('button.step-tab[data-step="6"]');
    await page.setInputFiles('#batch-file-input', { name: 'batch.csv', mimeType: 'text/csv', buffer: Buffer.from([lines[0], ...rows].join('\n')) });
    await page.click('#run-batch-predict-btn');
    await expect(page.locator('#batch-results-card')).toBeVisible();
    await expect(page.locator('#batch-results-card')).toContainText('Outside_Training_Range');
    await expect(page.locator('#batch-results-card')).toContainText('Age=650');
    expect(errors).toEqual([]);
});

test('Option B with multiple imputation on the external validation cohort', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await page.selectOption('#mi-select', 'mice5');
    await page.check('input[name="validation-mode"][value="separate"]', { force: true });
    await page.getByRole('button', { name: 'Load Built-In Validation Cohort' }).click();
    await expect(page.locator('#option-b-container')).toContainText('Hospital D');
    await page.locator('.model-card[data-model-id="glm"]').click();
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    await expect(page.locator('body')).toContainText('Multiple imputation:');
    expect(errors).toEqual([]);
});

test('7-step pipeline with recalibration and its printable report', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-shunt-large-btn');
    await page.click('#pipeline-direct-launch-btn');
    await page.click('#pipeline-step1-run-btn');
    for (const n of [2, 3, 4, 5, 6]) await page.click(`#subpill-${n}`);
    await page.click('#pipeline-run-rf-scout-btn');
    await expect(page.locator('.rf-importance-summary')).toBeVisible({ timeout: 60000 });
    await page.click('#subpill-7');
    await page.selectOption('#pipeline-recalibration-select', 'platt');
    await page.click('#pipeline-train-models-btn');
    await expect(page.locator('#pipeline-train-status-text')).toContainText('Evaluated Successfully', { timeout: 360000 });
    await expect(page.locator('#pipeline-leaderboard-container tbody tr')).toHaveCount(7);
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#pipeline-html-report-btn')]);
    const report = fs.readFileSync(await dl.path(), 'utf8');
    expect(report).toContain('7-step pipeline');
    expect(report).toContain('Calibration');
    expect(errors).toEqual([]);
});

test('external validation: an exported model is imported and validated on a separate labelled cohort', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await page.locator('.model-card[data-model-id="glm"]').click();
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#model-export-btn')]);
    const modelFile = await dl.path();
    // A colleague's fresh copy of the app: import the model, then a cohort that includes the outcome
    await page.goto('/index.html');
    await page.setInputFiles('#model-file-input', modelFile);
    await expect(page.locator('#step-5-content')).toBeVisible();
    await expect(page.locator('#analysis-summary')).toContainText('(imported file)');
    await page.click('button.step-tab[data-step="6"]');
    await page.setInputFiles('#batch-file-input', path.join(ROOT, 'medical_shunt_validation_cohort.csv'));
    await page.click('#run-batch-predict-btn');
    const panel = page.locator('#external-validation-container');
    await expect(panel).toContainText('ROC-AUC', { timeout: 60000 });
    await expect(panel).toContainText('87 of 150');
    await expect(page.locator('#external-validation-calibration')).toContainText('Calibration');
    const [rep] = await Promise.all([page.waitForEvent('download'), page.click('#external-validation-report-btn')]);
    const html = fs.readFileSync(await rep.path(), 'utf8');
    expect(html).toContain('Validation report');
    expect(html).toContain('<svg');
    expect(errors).toEqual([]);
});

test('multiple imputation is done once: reused for a second model, and saved and loaded for a later session', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await page.selectOption('#mi-select', 'mice5');
    // record every progress message of the imputation rounds
    await page.evaluate(() => {
        window.__miRounds = 0;
        new MutationObserver(() => { if (/Multiple imputation of .*(round \d+\/\d+|% of rounds done)/.test(document.getElementById('train-status').textContent)) window.__miRounds++; })
            .observe(document.getElementById('train-status'), { childList: true, subtree: true, characterData: true });
    });
    await page.locator('.model-card[data-model-id="glm"]').click();
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    expect(await page.evaluate(() => window.__miRounds)).toBeGreaterThan(0);
    await page.evaluate(() => { window.__miRounds = 0; });
    await page.locator('.model-card[data-model-id="nb"]').click();
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    expect(await page.evaluate(() => window.__miRounds)).toBe(0);
    await expect(page.locator('body')).toContainText('Multiple imputation:');
    // Saved imputed data lets a later session skip the imputation
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#mi-save-btn')]);
    expect(dl.suggestedFilename()).toContain('PATIENT-DATA');
    const saved = await dl.path();
    await page.goto('/index.html');
    await toStep4(page);
    await page.selectOption('#mi-select', 'mice5');
    await page.setInputFiles('#mi-file-input', saved);
    await expect(page.locator('#mi-file-status')).toContainText('Loaded');
    await page.evaluate(() => {
        window.__miRounds = 0;
        new MutationObserver(() => { if (/Multiple imputation of .*(round \d+\/\d+|% of rounds done)/.test(document.getElementById('train-status').textContent)) window.__miRounds++; })
            .observe(document.getElementById('train-status'), { childList: true, subtree: true, characterData: true });
    });
    await page.locator('.model-card[data-model-id="glm"]').click();
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    expect(await page.evaluate(() => window.__miRounds)).toBe(0);
    expect(errors).toEqual([]);
});

test('multiple imputation split across parallel workers gives exactly the single-run result', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    await page.goto('/index.html');
    const r = await page.evaluate(async () => {
        const { miceFitRun } = await import('./src/compute.js');
        const rng = window.DataEngine.createPRNG(9);
        const rows = Array.from({ length: 2500 }, () => {
            const o = { Y: rng() < 0.4 ? 'Yes' : 'No', Cat: rng() < 0.15 ? '' : ['A', 'B', 'C'][Math.floor(rng() * 3)] };
            for (let j = 0; j < 10; j++) o[`X${j}`] = rng() < 0.15 ? '' : (rng() * 10).toFixed(2);
            return o;
        });
        const cfg = { Y: { role: 'target', type: 'nominal' }, Cat: { role: 'feature', type: 'nominal' } };
        for (let j = 0; j < 10; j++) cfg[`X${j}`] = { role: 'feature', type: 'continuous' };
        let parallel = 0;
        const par = await miceFitRun(rows, cfg, { m: 5, seed: 3 }, { onProgress: (l, m, it, iters, k) => { parallel = Math.max(parallel, k || 1); } });
        const single = window.DataEngine.miceFit(rows, cfg, { m: 5, seed: 3 });
        return { same: JSON.stringify(par) === JSON.stringify(single), parallel, cores: navigator.hardwareConcurrency };
    });
    expect(r.same).toBe(true);
    if (r.cores > 2) expect(r.parallel).toBeGreaterThan(1);
    expect(errors).toEqual([]);
});

test('"How many predictors?" (Step 2, pipeline Step 4) and per-model data needs (Step 4) follow the selection', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.click('button.step-tab[data-step="2"]');
    const guide = page.locator('#predictor-budget-container');
    await expect(guide).toContainText('336 training records');
    await expect(guide).toContainText('104 are in the smaller outcome group');
    await expect(guide).toContainText('14 predictor(s) = 15 parameter(s)');
    await expect(guide).toContainText('Riley');
    await page.click('#var-exclude-all-btn');
    await expect(guide).toContainText('0 predictor(s) = 0 parameter(s)');
    await expect(guide).toContainText('meets these planning criteria');
    await page.click('#var-include-all-btn');
    // Step 4: data needed by each selected model
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    const models = page.locator('#model-sample-guide');
    await page.locator('.model-card[data-model-id="glm"]').click({ position: { x: 12, y: 12 } }); // corner: the seed box in the middle ignores clicks
    await expect(models).toContainText('Enough data for each model');
    await expect(models).toContainText('GLM');
    await expect(models).toContainText('150–300 events');
    await page.locator('.model-card[data-model-id="rf"]').click({ position: { x: 12, y: 12 } }); // corner: the seed box in the middle ignores clicks
    await expect(models).toContainText('Random Forest');
    await expect(models).toContainText('3,000 events');
    await page.click('button.step-tab[data-step="1"]'); // the pipeline is launched from Step 1
    await page.click('#pipeline-direct-launch-btn');
    await page.click('#pipeline-step1-run-btn');
    for (const n of [2, 3, 4]) await page.click(`#subpill-${n}`);
    const pg = page.locator('#pipeline-budget-container');
    await expect(pg).toContainText('15 parameter(s)');
    await page.locator('.pipeline-univ-checkbox').first().uncheck();
    await expect(pg).toContainText('candidate parameters considered');
    expect(errors).toEqual([]);
});

test('a tampered model file is refused before anything is installed; Clear session empties the page', async ({ page }) => {
    test.setTimeout(400000);
    const errors = [], dialogs = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('dialog', d => { dialogs.push(d.message()); d.accept(); });
    await page.goto('/index.html');
    await toStep4(page);
    await page.locator('.model-card[data-model-id="glm"]').click({ position: { x: 12, y: 12 } });
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#model-export-btn')]);
    const bundle = JSON.parse(fs.readFileSync(await dl.path(), 'utf8'));
    const tampered = JSON.parse(JSON.stringify(bundle));
    (tampered.parameters.state || tampered.parameters.base?.state || {}).predictSingle = 1;
    if (tampered.parameters.type === 'calibrated') tampered.parameters.base.state.predictSingle = 1;
    await page.goto('/index.html');
    await page.setInputFiles('#model-file-input', { name: 'tampered.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(tampered)) });
    await expect.poll(() => dialogs.join(' ')).toContain('could not be imported');
    await expect(page.locator('#step-5-content')).not.toBeVisible();
    // the genuine file still imports
    await page.setInputFiles('#model-file-input', { name: 'genuine.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(bundle)) });
    await expect(page.locator('#step-5-content')).toBeVisible();
    // Clear session: everything held by the page is discarded
    await page.click('button.step-tab[data-step="1"]');
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await Promise.all([page.waitForEvent('load'), page.click('#clear-session-btn')]);
    await expect(page.locator('#data-stats-grid')).not.toContainText('420');
    await expect(page.locator('#model-export-btn')).toBeDisabled();
    expect(errors).toEqual([]);
});
