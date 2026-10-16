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
    await expect(page.locator('#pipeline-leaderboard-container tbody tr')).toHaveCount(6);
    expect(errors).toEqual([]);
});

test('survival analysis tab fits a Cox model', async ({ page }) => {
    const errors = watch(page);
    await page.goto('/index.html');
    await page.click('#sample-survival-btn');
    await expect(page.locator('#step-survival-content')).toBeVisible();
    await page.fill('#surv-horizon', '24');
    await page.click('#surv-run-btn');
    await expect(page.locator('#survival-results')).toContainText("Harrell's C-index", { timeout: 60000 });
    expect(await page.locator('#survival-results table').first().locator('tbody tr').count()).toBeGreaterThan(3);
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
