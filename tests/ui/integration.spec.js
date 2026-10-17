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
    await page.selectOption('#rf-split-select', 'exact');
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
