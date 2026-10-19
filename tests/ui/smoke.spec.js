// Browser smoke tests: the main user journeys through the real interface (Chromium, headless).
// Run: npx playwright test   (starts a local web server; see playwright.config.js)
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

// Fail a test on any uncaught page error; answer alert/confirm dialogs automatically
function watch(page) {
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('dialog', d => d.accept());
    return errors;
}

async function loadShuntAndImpute(page) {
    await page.goto('/index.html');
    await page.click('#sample-cohort-btn');
    await expect(page.locator('#data-preview-container table')).toBeVisible();
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    await expect(page.locator('#step-4-content')).toBeVisible();
}

async function selectModels(page, ids) {
    await page.click('#scope-multi', { force: true });
    for (const box of await page.locator('.model-select-checkbox').all()) {
        const id = await box.getAttribute('data-model-id');
        if ((await box.isChecked()) !== ids.includes(id)) await box.click();
    }
}

test('app loads without errors and shows the sample data', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await expect(page).toHaveTitle(/Universal Medical ML/);
    await page.click('#sample-cohort-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('40');
    await page.click('button.step-tab[data-step="2"]');
    await expect(page.locator('.var-include-check').first()).toBeVisible();
    expect(errors).toEqual([]);
});

test('the main iNPH demo (420 simulated patients) and its external validation cohort load', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    await page.check('input[name="validation-mode"][value="separate"]', { force: true });
    await page.getByRole('button', { name: 'Load Built-In Validation Cohort' }).click();
    await expect(page.locator('#option-b-container')).toContainText('Hospital D');
    expect(errors).toEqual([]);
});

test('standard workflow: train, compare, predict, explain', async ({ page }) => {
    const errors = watch(page);
    await loadShuntAndImpute(page);
    await selectModels(page, ['glm', 'rf', 'gbm']);
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Tournament Complete', { timeout: 150000 });
    await expect(page.locator('#multi-model-leaderboard-container tbody tr')).toHaveCount(3);
    await expect(page.locator('#calibration-panel-container')).toContainText('Discrimination');
    await page.click('button.step-tab[data-step="5"]');
    await page.click('#run-single-predict-btn');
    await expect(page.locator('#single-result-banner .pred-outcome-val')).toBeVisible();
    await expect(page.locator('#single-result-banner')).toContainText('Why this prediction');
    expect(errors).toEqual([]);
});

test('settings and trained model can be saved and loaded again', async ({ page }) => {
    const errors = watch(page);
    await loadShuntAndImpute(page);
    await selectModels(page, ['glm', 'rf']);
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Tournament Complete', { timeout: 150000 });
    // settings file
    const [settingsDl] = await Promise.all([page.waitForEvent('download'), page.click('#settings-save-btn')]);
    const settings = JSON.parse(fs.readFileSync(await settingsDl.path(), 'utf8'));
    expect(settings.schema).toBe(1);
    // printable HTML report with methods, results and plots
    const [reportDl] = await Promise.all([page.waitForEvent('download'), page.click('#standard-html-report-btn')]);
    const report = fs.readFileSync(await reportDl.path(), 'utf8');
    expect(report).toContain('Prediction model report');
    expect(report).toContain('Not a certified medical device');
    expect(report).toContain('<svg');
    // What the figures are: how the model was chosen, what the estimate rests on, baseline, external validation
    expect(report).toContain('What these performance figures are');
    expect(report).toContain('cross-validation inside the development data');
    expect(report).toContain('not evidence of reproducibility');
    expect(report).toContain('Always predicting the most common outcome');
    await expect(page.locator('#results-dashboard')).toContainText('One split');
    // trained model file, re-imported into a fresh page
    const [modelDl] = await Promise.all([page.waitForEvent('download'), page.click('#model-export-btn')]);
    const modelPath = await modelDl.path();
    const bundle = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
    expect(bundle.format).toBe('ummp-trained-model');
    await page.goto('/index.html');
    await page.setInputFiles('#model-file-input', modelPath);
    await expect(page.locator('#step-5-content')).toBeVisible();
    await page.click('#run-single-predict-btn');
    await expect(page.locator('#single-result-banner .pred-outcome-val')).toBeVisible();
    // Provenance and the reproducibility self-check recorded in the file and confirmed on import
    expect(bundle.environment && bundle.environment.app).toBeTruthy();
    expect(bundle.selfCheck.inputs.length).toBe(3);
    const [cardDl] = await Promise.all([page.waitForEvent('download'), page.click('#model-card-btn')]);
    expect(fs.readFileSync(await cardDl.path(), 'utf8')).toContain('reproducibility check on import: reproduced exactly');
    // A file whose recorded predictions no longer match (altered parameters) is questioned before use
    const tampered = JSON.parse(JSON.stringify(bundle));
    tampered.selfCheck.outputs[0] = tampered.selfCheck.outputs[0].map((v, k) => (k ? Math.min(1, v + 0.2) : v));
    const tamperedPath = modelPath + '.tampered.json';
    fs.writeFileSync(tamperedPath, JSON.stringify(tampered));
    const messages = [];
    page.on('dialog', d => messages.push(d.message()));
    await page.goto('/index.html');
    await page.setInputFiles('#model-file-input', tamperedPath);
    await expect.poll(() => messages.some(m => /Reproducibility check/.test(m))).toBe(true);
    expect(errors).toEqual([]);
});

test('Content Security Policy: the page cannot send data to another server', async ({ page }) => {
    const errors = watch(page);
    await page.route('https://example.org/**', route => route.fulfill({ status: 200, body: 'reached' })); // never the real network
    await page.goto('/index.html');
    const result = await page.evaluate(async () => {
        const violations = [];
        document.addEventListener('securitypolicyviolation', e => violations.push(e.violatedDirective));
        let sent = false;
        try { await fetch('https://example.org/collect', { method: 'POST', body: 'patient data' }); sent = true; } catch (e) { sent = false; }
        await new Promise(r => setTimeout(r, 200));
        return { sent, violations };
    });
    expect(result.sent).toBe(false);
    expect(result.violations.some(v => /connect-src/.test(v))).toBe(true);
    expect(errors).toEqual([]);
});

test('Option C bootstrap validation reports Harrell and .632+ estimates', async ({ page }) => {
    const errors = watch(page);
    await loadShuntAndImpute(page);
    await page.check('input[name="validation-mode"][value="bootstrap"]', { force: true });
    await page.selectOption('#bootstrap-reps', '50');
    await selectModels(page, ['glm', 'rf']);
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Tournament Complete', { timeout: 170000 });
    const panel = page.locator('#optimism-container');
    await expect(panel).toContainText('Bootstrap internal validation (Option C)');
    await expect(panel).toContainText('.632+ estimate');
    await expect(panel).toContainText('Choosing the best of 2 models');
    await expect(page.locator('#multi-model-leaderboard-container')).toContainText('Corrected Accuracy');
    expect(errors).toEqual([]);
});

test('recalibration, subgroup analysis and Shapley explanation in the standard workflow', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-breast-btn');
    await expect(page.locator('#data-preview-container table')).toBeVisible();
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    await page.selectOption('#recalibration-select', 'platt');
    await selectModels(page, ['glm', 'rf']);
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Tournament Complete', { timeout: 150000 });
    await expect(page.locator('#calibration-panel-container')).toContainText('Recalibrated');
    await page.click('#subgroup-panel-container-btn');
    await expect(page.locator('#subgroup-panel-container-out table')).toContainText('All Testing patients');
    await expect(page.locator('#subgroup-panel-container-out table')).toContainText('AUC vs other patients (95% CI)');
    await expect(page.locator('#subgroup-panel-container-out')).toContainText('percentile bootstrap 95% intervals');
    await page.click('button.step-tab[data-step="5"]');
    await page.click('#run-single-predict-btn');
    await expect(page.locator('#single-result-banner')).toContainText('Shapley');
    expect(errors).toEqual([]);
});

test('multiple imputation: pooled models on the iNPH cohort, with blanks at prediction time', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    await page.click('button.step-tab[data-step="3"]'); // the imputation choice is in Step 3
    await page.selectOption('#mi-select', 'mice5');
    await page.click('button.step-tab[data-step="4"]');
    await selectModels(page, ['glm', 'rf']);
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Tournament Complete', { timeout: 150000 });
    await expect(page.locator('body')).toContainText('Multiple imputation:');
    await page.click('button.step-tab[data-step="5"]');
    await page.fill('#predict-field-11', '');
    await page.click('#run-single-predict-btn');
    await expect(page.locator('#single-result-banner .pred-outcome-val')).toBeVisible();
    expect(errors).toEqual([]);
});

test('7-step pipeline runs to the benchmark', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-cohort-btn');
    await page.click('#pipeline-direct-launch-btn');
    await page.click('#pipeline-step1-run-btn');
    for (const n of [2, 3, 4, 5, 6]) await page.click(`#subpill-${n}`);
    await page.click('#pipeline-run-rf-scout-btn');
    await expect(page.locator('.rf-importance-summary')).toBeVisible({ timeout: 60000 });
    await page.click('#subpill-7');
    await page.click('#pipeline-train-models-btn');
    await expect(page.locator('#pipeline-train-status-text')).toContainText('Evaluated Successfully', { timeout: 170000 });
    await expect(page.locator('#pipeline-leaderboard-container tbody tr')).toHaveCount(7);
    expect(errors).toEqual([]);
});

test('time-to-event sample opens Step 2 with the outcome chosen; Option C bootstrap corrects the Cox model', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-survival-btn');
    await expect(page.locator('#step-2-content')).toBeVisible();
    await expect(page.locator('input[name="outcome-type"][value="survival"]')).toBeChecked();
    await expect(page.locator('#ot-time-col')).toHaveValue('Months_To_Revision_Or_Censoring');
    await page.fill('#ot-horizon', '24');
    await page.dispatchEvent('#ot-horizon', 'change');
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    await page.check('input[name="validation-mode"][value="bootstrap"]', { force: true });
    await page.selectOption('#bootstrap-reps', '50');
    await page.click('#train-model-btn');
    const box = page.locator('#survival-step4-results');
    await expect(box).toContainText('Bootstrap internal validation (optimism-corrected)', { timeout: 120000 });
    await expect(box).toContainText('Calibration slope');
    const optTable = box.locator('table', { hasText: 'Optimism' });
    expect(await optTable.locator('tbody tr').count()).toBe(3);
    console.log('Option C Cox:', (await optTable.innerText()).replace(/\s+/g, ' '));
    // Model card and settings for the Cox model
    const [cardDl] = await Promise.all([page.waitForEvent('download'), page.click('#model-card-btn')]);
    const card = fs.readFileSync(await cardDl.path(), 'utf8');
    expect(card).toContain('Cox proportional-hazards model');
    expect(card).toContain('optimism-corrected');
    const [settingsDl] = await Promise.all([page.waitForEvent('download'), page.click('#settings-save-btn')]);
    const settingsPath = await settingsDl.path();
    const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
    expect(settings.outcome).toMatchObject({ type: 'survival', timeCol: 'Months_To_Revision_Or_Censoring', horizon: 24 });
    await page.goto('/index.html');
    await page.setInputFiles('#csv-file-input', path.join(__dirname, '..', '..', 'medical_shunt_revision_survival.csv'));
    await page.click('button.step-tab[data-step="2"]');
    await expect(page.locator('input[name="outcome-type"][value="standard"]')).toBeChecked();
    await page.setInputFiles('#settings-file-input', settingsPath);
    await page.click('button.step-tab[data-step="2"]');
    await expect(page.locator('input[name="outcome-type"][value="survival"]')).toBeChecked();
    await expect(page.locator('#ot-horizon')).toHaveValue('24');
    expect(errors).toEqual([]);
});

test('time-to-event outcome chosen in Step 2 is fitted by Step 4 (Cox model, single imputation and MICE)', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-survival-btn');
    await page.click('button.step-tab[data-step="2"]');
    await page.check('input[name="outcome-type"][value="survival"]');
    await page.fill('#ot-horizon', '24');
    await page.dispatchEvent('#ot-horizon', 'change');
    for (const mode of ['single', 'mice5']) {
        await page.click('button.step-tab[data-step="3"]');
        await page.selectOption('#mi-select', mode);
        await page.click('#apply-imputation-btn');
        await page.click('button.step-tab[data-step="4"]');
        await expect(page.locator('.survival-step4-note')).toBeVisible();
        await expect(page.locator('.model-cards-grid')).toBeHidden();
        await page.evaluate(() => { document.getElementById('survival-step4-results').innerHTML = ''; });
        await page.click('#train-model-btn');
        await expect(page.locator('#survival-step4-results')).toContainText(mode === 'single' ? 'single imputation' : 'MICE, 5 datasets', { timeout: 120000 });
        await expect(page.locator('#survival-step4-results')).toContainText("Harrell's C-index");
    }
    // Option B: fit on all patients, validate on a separate cohort (here the same file uploaded as Dataset ii)
    await page.check('input[name="validation-mode"][value="separate"]', { force: true });
    await page.setInputFiles('#test-csv-file-input', path.join(__dirname, '..', '..', 'medical_shunt_revision_survival.csv'));
    await page.click('#train-model-btn');
    await expect(page.locator('#survival-step4-results')).toContainText('validation cohort (Dataset ii)', { timeout: 120000 });
    // Step 5: one patient; Step 6: a file of patients, validated because it has the time and event columns
    await expect(page.locator('#train-model-btn')).toContainText('Cox');
    await expect(page.locator('#step-4-content .chart-container-box')).toBeHidden();
    await expect(page.locator('#to-step-6-btn')).toBeEnabled();
    await page.click('#to-step-5-btn');
    await expect(page.locator('#predict-form-grid')).toBeHidden();
    await page.click('#surv-predict-btn');
    await expect(page.locator('#surv-predict-out')).toContainText('Predicted probability of no event');
    await page.click('button.step-tab[data-step="6"]');
    await page.setInputFiles('#batch-file-input', path.join(__dirname, '..', '..', 'medical_shunt_revision_survival.csv'));
    await page.click('#run-batch-predict-btn');
    await expect(page.locator('#batch-table-container')).toContainText('Predicted_event_free_at_24');
    await expect(page.locator('#external-validation-container')).toContainText("Validation of the Cox model");
    expect(errors).toEqual([]);
});

test('time-to-event outcome columns are protected: follow-up time never a predictor, an earlier outcome excluded; report hazard ratios per unit', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    // The last column (an outcome) would be the target; choosing a time-to-event outcome must exclude it, not make it a predictor
    const csv = fs.readFileSync(path.join(__dirname, '..', '..', 'medical_shunt_revision_survival.csv'), 'utf8').trim().split(/\r?\n/)
        .map((l, i) => l + ',' + (i === 0 ? 'Outcome_Status' : (l.endsWith('Yes') ? 'Revised' : 'Not revised'))).join('\n');
    await page.setInputFiles('#csv-file-input', { name: 'survival_extra.csv', mimeType: 'text/csv', buffer: Buffer.from(csv) });
    await page.click('button.step-tab[data-step="2"]');
    await page.check('input[name="outcome-type"][value="survival"]');
    await page.selectOption('#ot-event-col', 'Shunt_Revision');
    await page.selectOption('#ot-event-value', 'Yes');
    await page.fill('#ot-horizon', '24');
    await page.dispatchEvent('#ot-horizon', 'change');
    await expect(page.locator('#outcome-type-container')).toContainText('"Outcome_Status" was the outcome column');
    await page.click('#var-include-all-btn');
    await expect(page.locator('.var-include-check[data-col="Months_To_Revision_Or_Censoring"]')).not.toBeChecked();
    await expect(page.locator('.var-include-check[data-col="Months_To_Revision_Or_Censoring"]')).toBeDisabled();
    // Include-all ticks the earlier outcome (user's choice); untick it again for the model
    await page.locator('.var-include-check[data-col="Outcome_Status"]').uncheck();
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    await page.check('input[name="validation-mode"][value="split"]', { force: true });
    await page.click('#train-model-btn');
    const box = page.locator('#survival-step4-results');
    await expect(box).toContainText("Harrell's C-index", { timeout: 120000 });
    const hrTable = box.locator('table').first();
    await expect(hrTable).not.toContainText('Months_To_Revision');
    await expect(hrTable).not.toContainText('Outcome_Status');
    const screenAge = await hrTable.locator('tr', { hasText: 'Age' }).locator('td').nth(1).innerText();
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#surv-report-btn')]);
    const rep = JSON.parse(fs.readFileSync(await dl.path(), 'utf8'));
    const age = rep.hazard_ratios.find(h => h.term === 'Age');
    expect(age.hr.toFixed(2)).toBe(screenAge.split(' ')[0]);
    expect(rep.design.validation).toContain('Option A');
    expect(errors).toEqual([]);
});

test('leakage safeguard: the leakage teaching example triggers a warning, and declining stops the analysis', async ({ page }) => {
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    const dialogs = [];
    page.on('dialog', d => { dialogs.push(d.message()); if (/Possible leakage/.test(d.message())) d.dismiss(); else d.accept(); });
    await page.goto('/index.html');
    await page.click('#sample-leakage-btn'); // the built-in teaching example: a post-operative column that largely defines response
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Stopped: check the predictors flagged as possible leakage');
    expect(dialogs.some(m => /Possible leakage/.test(m) && /Gait_Improvement_6mo_Points/.test(m))).toBe(true);
    expect(errors).toEqual([]);
});

test('competing risks: death before revision as a competing event, Options A and C, Steps 5 and 6', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-competing-btn');
    await expect(page.locator('#step-2-content')).toBeVisible();
    await expect(page.locator('#ot-event-value')).toHaveValue('Revision');
    await expect(page.locator('#ot-competing-value')).toHaveValue('Death');
    await page.fill('#ot-horizon', '36');
    await page.dispatchEvent('#ot-horizon', 'change');
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    const box = page.locator('#survival-step4-results');
    await page.check('input[name="validation-mode"][value="split"]', { force: true });
    await page.click('#train-model-btn');
    await expect(box).toContainText('Competing-risk model (cause-specific Cox)', { timeout: 120000 });
    await expect(box).toContainText("Wolbers' C-index");
    await expect(box).toContainText('Cause-specific hazard ratios for the competing event "Death"');
    await expect(box).toContainText('Why competing risks matter here');
    await page.check('input[name="validation-mode"][value="bootstrap"]', { force: true });
    await page.selectOption('#bootstrap-reps', '50');
    await page.evaluate(() => { document.getElementById('survival-step4-results').innerHTML = ''; });
    await page.click('#train-model-btn');
    await expect(box).toContainText('Not defined for the cumulative incidence with competing risks', { timeout: 180000 });
    console.log('Competing risks, Option C:', (await box.locator('table', { hasText: 'Optimism' }).innerText()).replace(/\s+/g, ' '));
    await page.click('#to-step-5-btn');
    await page.click('#surv-predict-btn');
    await expect(page.locator('#surv-predict-out')).toContainText('Neither yet');
    await page.click('button.step-tab[data-step="6"]');
    await page.setInputFiles('#batch-file-input', path.join(__dirname, '..', '..', 'medical_shunt_revision_competing.csv'));
    await page.click('#run-batch-predict-btn');
    await expect(page.locator('#batch-table-container')).toContainText('Predicted_competing_risk_by_36');
    await expect(page.locator('#external-validation-container')).toContainText('Validation of the competing-risk model');
    const [cardDl] = await Promise.all([page.waitForEvent('download'), page.click('#model-card-btn')]);
    expect(fs.readFileSync(await cardDl.path(), 'utf8')).toContain('competing-risk model (cause-specific Cox)');
    expect(errors).toEqual([]);
});

test('several records per patient: the patient split is used, and models are compared with the clustered DeLong test', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    // Each of 200 simulated patients appears twice (two admissions with slightly different measurements)
    const lines = fs.readFileSync(path.join(__dirname, '..', '..', 'medical_shunt_cohort_large.csv'), 'utf8').trim().split(/\r?\n/);
    const head = lines[0], rows = lines.slice(1, 201);
    const twice = rows.flatMap(l => [l, l.replace(/,(\d+),(Male|Female),/, (m, a, sx) => `,${Number(a) + 1},${sx},`)]);
    await page.setInputFiles('#csv-file-input', { name: 'repeated.csv', mimeType: 'text/csv', buffer: Buffer.from([head, ...twice].join('\n')) });
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    await page.selectOption('#split-method', 'group');
    await page.selectOption('#split-group-col', 'Patient_ID');
    await selectModels(page, ['glm', 'rf']);
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Tournament Complete', { timeout: 150000 });
    const [reportDl] = await Promise.all([page.waitForEvent('download'), page.click('#standard-html-report-btn')]);
    const report = fs.readFileSync(await reportDl.path(), 'utf8');
    expect(report).toContain('clustered DeLong (Obuchowski)');
    expect(errors).toEqual([]);
});

test('opened directly from disk (file://), the fallback bundle runs', async ({ page }) => {
    const errors = watch(page);
    const root = path.join(__dirname, '..', '..');
    await page.goto('file://' + path.join(root, 'index.html'));
    await page.setInputFiles('#csv-file-input', path.join(root, 'medical_shunt_cohort.csv'));
    await expect(page.locator('#data-preview-container table')).toBeVisible();
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    await selectModels(page, ['glm', 'rf']);
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Tournament Complete', { timeout: 150000 });
    // module-loading errors on file:// are expected; only application errors fail the test
    expect(errors.filter(e => !/module|Failed to fetch|CORS|importmap/i.test(e))).toEqual([]);
});
