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
// The imputation choice is in Step 3 (one pane for single and multiple imputation)
async function chooseMI(page, value) {
    await page.click('button.step-tab[data-step="3"]');
    await page.selectOption('#mi-select', value);
    await page.click('button.step-tab[data-step="4"]');
}
async function toStep4(page) {
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
}

test('the imputation step gives the share of incomplete patients and suggests the number of imputations; 20 imputations can be chosen', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.click('button.step-tab[data-step="3"]');
    const advice = page.locator('#mi-advice');
    await expect(advice).toContainText('101 of 420 patients (24%)');
    await expect(advice).toContainText('MICE 20');
    await expect(advice).toContainText('about 24 would be ideal');
    await page.selectOption('#mi-select', 'mice10');
    await expect(advice).toContainText('The current choice (10) is fewer');
    await page.selectOption('#mi-select', 'mice20');
    await expect(advice).not.toContainText('is fewer');
    await expect(page.locator('#analysis-summary')).toContainText('multiple imputation (20)');
    expect(errors).toEqual([]);
});

test('Step 4 estimates the training time; the leaderboard shows the Brier score fifth; the results explain the top-ranked model', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await page.check('#scope-multi', { force: true });
    const selected = () => page.evaluate(() => [...document.querySelectorAll('.model-card')].filter(e => /selected|active/.test(e.className)).map(e => e.dataset.modelId));
    for (const id of ['glm', 'nb']) if (!(await selected()).includes(id)) await page.locator(`.model-card[data-model-id="${id}"]`).click({ position: { x: 12, y: 12 } });
    for (const id of await selected()) if (!['glm', 'nb'].includes(id)) await page.locator(`.model-card[data-model-id="${id}"]`).click({ position: { x: 12, y: 12 } });
    const est = page.locator('#train-estimate');
    await expect(est).toContainText('Estimated training time on this computer');
    await expect(est).toContainText('2 models');
    await page.check('input[name="validation-mode"][value="bootstrap"]', { force: true });
    await expect(est).toContainText('100 bootstrap resamples');
    await page.check('input[name="validation-mode"][value="split"]', { force: true });
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    // The confusion-matrix section explains the ranking and every card shows the Brier score
    const why = page.locator('.explain-box', { hasText: 'is top-ranked' });
    await expect(why).toContainText('ranked first because it had the highest Macro F1 in cross-validation');
    await expect(why).toContainText(/effectively tied|clear lead/);
    await expect(page.locator('.multi-cm-card .cm-m-label', { hasText: 'Brier' })).toHaveCount(2);
    expect(await page.evaluate(() => Boolean(document.getElementById('sample-size-container').compareDocumentPosition(document.getElementById('model-sample-guide')) & Node.DOCUMENT_POSITION_FOLLOWING))).toBe(true);
    const heads = await page.locator('.leaderboard-table thead th').allInnerTexts();
    expect(heads[4]).toMatch(/brier/i);
    expect(heads.filter(h => /brier/i.test(h)).length).toBe(1);
    expect(errors).toEqual([]);
});

test('the chosen task is highlighted until another is chosen; both workflow buttons show at every level', async ({ page }) => {
    const errors = watch(page);
    await page.addInitScript(() => { try { localStorage.clear(); } catch (e) { /* ignore */ } });
    await page.goto('/index.html');
    await page.selectOption('#ui-level-select', 'beginner');
    await page.click('#task-demo');
    await expect(page.locator('#task-demo')).toHaveClass(/chosen/);
    await expect(page.locator('#task-demo')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    page.on('filechooser', fc => fc.setFiles([]).catch(() => {}));
    await page.click('#task-validate');
    await expect(page.locator('#task-validate')).toHaveClass(/chosen/);
    await expect(page.locator('#task-demo')).not.toHaveClass(/chosen/);
    await expect(page.locator('#to-step-2-btn')).toBeVisible();
    await expect(page.locator('#pipeline-direct-launch-btn')).toBeVisible();
    expect(errors).toEqual([]);
});

test('settings with the newer controls are saved and restored', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await chooseMI(page, 'mice10');
    await page.selectOption('#recalibration-select', 'isotonic');
    await page.selectOption('#rf-split-select', 'exact', { force: true }); // in the Random Forest settings panel (hidden until RF is configured)
    await page.check('input[name="validation-mode"][value="bootstrap"]', { force: true });
    await page.selectOption('#bootstrap-reps', '200');
    await page.selectOption('#bootstrap-group-col', 'Patient_ID');
    // the analysis summary follows the choices
    const summary = page.locator('#analysis-summary');
    await expect(summary).toContainText('positive class "Responder"');
    await expect(summary).toContainText('multiple imputation (10)');
    await expect(summary).toContainText('Option C: bootstrap, 200 resamples, by patient (Patient_ID)');
    await expect(summary).toContainText('isotonic regression');
    await expect(summary).toContainText('none yet');
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
    await chooseMI(page, 'mice5');
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
    await chooseMI(page, 'mice5');
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
    await chooseMI(page, 'mice5');
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
    await chooseMI(page, 'mice5');
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
    await page.click('button.step-tab[data-step="3"]'); // the imputed-data buttons are in the Step 3 imputation pane
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#mi-save-btn')]);
    expect(dl.suggestedFilename()).toContain('PATIENT-DATA');
    const saved = await dl.path();
    await page.goto('/index.html');
    await toStep4(page);
    await chooseMI(page, 'mice5');
    await page.click('button.step-tab[data-step="3"]');
    await page.setInputFiles('#mi-file-input', saved);
    await expect(page.locator('#mi-file-status')).toContainText('Loaded');
    await page.click('button.step-tab[data-step="4"]');
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
    // With Option A, the larger budget when all patients train (Options B and C) is shown alongside
    await expect(guide).toContainText('With Option B or C (all 420 records)');
    await expect(guide).toContainText('allow up to 5 parameter(s) instead of 4 (you have 15)');
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

test('network audit: during a full analysis every request is a download of the app\'s own files (nothing is sent out)', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    const requests = [];
    page.on('request', r => requests.push({ url: r.url(), method: r.method(), body: r.postData() }));
    await page.goto('/index.html');
    const origin = new URL(page.url()).origin;
    await toStep4(page);
    await page.locator('.model-card[data-model-id="glm"]').click({ position: { x: 12, y: 12 } });
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    await expect(page.locator('#interpretation-container')).toContainText('In plain language');
    await page.click('button.step-tab[data-step="5"]');
    await page.click('#run-single-predict-btn');
    await expect(page.locator('#single-result-banner')).toBeVisible();
    await page.click('button.step-tab[data-step="6"]');
    await page.setInputFiles('#batch-file-input', path.join(ROOT, 'medical_shunt_validation_cohort.csv'));
    await page.click('#run-batch-predict-btn');
    await expect(page.locator('#external-validation-container')).toContainText('ROC-AUC', { timeout: 60000 });
    const outside = requests.filter(r => !r.url.startsWith(origin + '/') && !r.url.startsWith('blob:') && !r.url.startsWith('data:'));
    const sending = requests.filter(r => r.method !== 'GET' || r.body);
    expect(outside, `requests to other addresses: ${outside.map(r => r.url).join(', ')}`).toEqual([]);
    expect(sending, `requests sending data: ${sending.map(r => `${r.method} ${r.url}`).join(', ')}`).toEqual([]);
    expect(requests.length).toBeGreaterThan(5);
    expect(errors).toEqual([]);
});

test('one imputation choice: with MICE the Step 3 table shows MICE only, folds and bootstrap resamples repeat MICE', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.click('button.step-tab[data-step="3"]');
    await expect(page.locator('.impute-strat-select').first()).toBeVisible();
    await page.selectOption('#mi-select', 'mice5');
    await expect(page.locator('.impute-strat-select')).toHaveCount(0); // no per-variable methods with MICE
    await expect(page.locator('#impute-table-body')).toContainText('MICE (multiple imputation)');
    await expect(page.locator('#apply-imputation-btn')).toContainText('Multiple Imputation');
    await page.click('#apply-imputation-btn');
    await expect(page.locator('#impute-audit-box')).toContainText('no single-imputation method is used');
    await page.click('button.step-tab[data-step="4"]');
    // cross-validated selection of two models: MICE inside the folds
    await page.evaluate(async () => { const { AppState } = await import('./src/core.js'); AppState.applyModelSelection('multi', ['glm', 'nb']); });
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Complete/, { timeout: 170000 });
    await expect(page.locator('#retrain-status-banner')).toContainText('MICE (5 completed datasets, pooled models) refitted in each fold');
    await expect(page.locator('#mice-diagnostics-container')).toContainText('Did the multiple imputation converge');
    await expect(page.locator('#mice-diagnostics-container')).toContainText('Gelman');
    // bootstrap validation: MICE inside every resample, pooled MICE final model
    await page.check('#val-mode-c', { force: true });
    await page.selectOption('#bootstrap-reps', { index: 0 });
    await page.evaluate(async () => { const { AppState } = await import('./src/core.js'); AppState.applyModelSelection('single', ['glm']); });
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Complete/, { timeout: 300000 });
    await expect(page.locator('#retrain-status-banner')).toContainText('Bootstrap resamples: MICE (one completed dataset per resample');
    // back to single imputation: the per-variable methods return
    await page.click('button.step-tab[data-step="3"]');
    await page.selectOption('#mi-select', 'single');
    await expect(page.locator('.impute-strat-select').first()).toBeVisible();
    expect(errors).toEqual([]);
});

test('Option C with recalibration: the final model is recalibrated and the resamples repeat it', async ({ page }) => {
    test.setTimeout(400000);
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await page.selectOption('#recalibration-select', 'platt');
    await page.check('#val-mode-c', { force: true });
    await page.selectOption('#bootstrap-reps', { index: 0 });
    await page.locator('.model-card[data-model-id="glm"]').click({ position: { x: 12, y: 12 } });
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Complete/, { timeout: 300000 });
    await expect(page.locator('#retrain-status-banner')).toContainText('Recalibrated with Platt scaling');
    await expect(page.locator('#retrain-status-banner')).toContainText('refitted inside');
    await page.click('button.step-tab[data-step="5"]');
    await page.click('#run-single-predict-btn');
    await expect(page.locator('#single-result-banner')).toBeVisible();
    expect(errors).toEqual([]);
});

test('7-step pipeline with multiple imputation: pooled models for all seven, then prediction', async ({ page }) => {
    test.setTimeout(500000);
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-shunt-large-btn');
    await page.click('#pipeline-direct-launch-btn');
    await page.selectOption('#pipeline-impute-strategy', 'mice5');
    await page.click('#pipeline-step1-run-btn');
    await expect(page.locator('#pipeline-step1-results')).toContainText('MICE (5 completed datasets)');
    for (const n of [2, 3, 4, 5, 6]) await page.click(`#subpill-${n}`);
    await page.click('#pipeline-run-rf-scout-btn');
    await expect(page.locator('.rf-importance-summary')).toBeVisible({ timeout: 60000 });
    await page.click('#subpill-7');
    await page.click('#pipeline-train-models-btn');
    await expect(page.locator('#pipeline-train-status-text')).toContainText('Evaluated Successfully', { timeout: 450000 });
    await expect(page.locator('#pipeline-leaderboard-container tbody tr')).toHaveCount(7);
    const pooled = await page.evaluate(async () => { const { AppState } = await import('./src/core.js'); return AppState.pipeline.benchmarkResults.every(b => b.prep.imputation && b.prep.imputation.mice); });
    expect(pooled).toBe(true);
    await page.click('button.step-tab[data-step="5"]');
    await page.click('#run-single-predict-btn');
    await expect(page.locator('#single-result-banner')).toBeVisible();
    expect(errors).toEqual([]);
});

test('interface levels: Beginner and Intermediate hide expert settings and reset them to defaults', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await toStep4(page);
    await page.check('#autotune-check');
    await page.selectOption('#recalibration-select', 'isotonic');
    await page.click('button.step-tab[data-step="1"]');
    await page.selectOption('#ui-level-select', 'intermediate');
    await page.click('button.step-tab[data-step="4"]');
    await expect(page.locator('#autotune-check')).toBeVisible(); // automatic tuning is offered at every level
    await expect(page.locator('#recalibration-select')).toBeVisible();
    await expect(page.locator('#autotune-check')).toBeChecked(); // visible at every level, so kept
    await expect(page.locator('#recalibration-select')).toHaveValue('isotonic'); // still visible, so kept
    await page.click('button.step-tab[data-step="1"]');
    await page.selectOption('#ui-level-select', 'beginner');
    await page.click('button.step-tab[data-step="4"]');
    await expect(page.locator('#recalibration-select')).toBeHidden();
    await expect(page.locator('#recalibration-select')).toHaveValue('none');
    await expect(page.locator('#autotune-check')).toBeVisible();
    await expect(page.locator('#autotune-check')).toBeChecked();
    await expect(page.locator('#decision-threshold')).toBeHidden();
    await expect(page.locator('input[name="validation-mode"][value="split"]')).toBeAttached(); // validation choices stay
    await page.reload(); // the level is remembered in this browser
    await expect(page.locator('#ui-level-select')).toHaveValue('beginner');
    await page.selectOption('#ui-level-select', 'advanced');
    expect(errors).toEqual([]);
});

test('7-step pipeline: Step 4 significance screening is repeated inside every cross-validation fold', async ({ page }) => {
    test.setTimeout(500000);
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-shunt-large-btn');
    await page.click('#pipeline-direct-launch-btn');
    await page.click('#pipeline-step1-run-btn');
    for (const n of [2, 3, 4]) await page.click(`#subpill-${n}`);
    await page.click('#pipeline-filter-significant-btn');
    await page.click('#pipeline-drop-unselected-univ-btn');
    const scr = await page.evaluate(async () => { const { AppState } = await import('./src/core.js'); return AppState.pipeline.univScreen; });
    expect(scr.active).toBe(true);
    expect(scr.rule.length).toBeGreaterThan(0);
    for (const n of [5, 6]) await page.click(`#subpill-${n}`);
    await page.click('#pipeline-run-rf-scout-btn');
    await expect(page.locator('.rf-importance-summary')).toBeVisible({ timeout: 60000 });
    await page.click('#subpill-7');
    await page.click('#pipeline-train-models-btn');
    await expect(page.locator('#pipeline-train-status-text')).toContainText('Evaluated Successfully', { timeout: 450000 });
    await expect(page.locator('body')).toContainText('Step 4 screening repeated inside cross-validation');
    expect(errors).toEqual([]);
});

test.describe('first visit', () => {
    test.use({ storageState: { cookies: [], origins: [] } });
    test('a first-time visitor starts at the Beginner level', async ({ page }) => {
        const errors = watch(page);
        await page.goto('/index.html');
        await expect(page.locator('#ui-level-select')).toHaveValue('beginner');
        await expect(page.locator('.ui-level-note')).toBeVisible();
        // A simpler first screen: settings files appear from Intermediate; both workflows can be started at every level
        await expect(page.locator('#settings-save-btn')).toBeHidden();
        await expect(page.locator('#pipeline-direct-launch-btn')).toBeVisible();
        await expect(page.locator('#clear-session-btn')).toBeVisible();
        expect(errors).toEqual([]);
    });
});

test('model card: a one-page summary of the active model, and the research-use confirmation before Step 5', async ({ page }) => {
    test.setTimeout(300000);
    const errors = [], dialogs = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('dialog', d => { dialogs.push(d.message()); d.accept(); });
    await page.goto('/index.html');
    await toStep4(page);
    await page.locator('.model-card[data-model-id="glm"]').click({ position: { x: 12, y: 12 } });
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Complete/, { timeout: 170000 });
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#model-card-btn')]);
    const html = fs.readFileSync(await dl.path(), 'utf8');
    for (const t of ['Model card:', 'Intended use', 'Known limitations', 'ROC-AUC', 'Content-Security-Policy', 'Shunt_Response']) expect(html).toContain(t);
    await page.click('button.step-tab[data-step="5"]');
    await expect.poll(() => dialogs.join(' ')).toContain('Research and education use only');
    await expect(page.locator('#step-5-content')).toBeVisible();
    expect(errors).toEqual([]);
});
