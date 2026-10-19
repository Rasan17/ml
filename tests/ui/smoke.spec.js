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
    await page.click('button.step-tab[data-step="5"]');
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
