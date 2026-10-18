// Browser tests with numerical and multi-category outcomes (the smoke tests mostly use yes/no outcomes):
// training, multiple imputation, Option C, subgroups, explanations and the printable report.
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

function watch(page) {
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('dialog', d => d.accept());
    return errors;
}

// The iNPH demo cohort with its outcome replaced: a numerical score, or three categories
function variantCsv(kind) {
    const lines = fs.readFileSync(path.join(__dirname, '..', '..', 'medical_shunt_cohort_large.csv'), 'utf8').trim().split('\n');
    const head = lines[0].split(',');
    const yi = head.indexOf('Shunt_Response'), tap = head.indexOf('Tap_Test_Response'), age = head.indexOf('Age');
    let s = 11;
    const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    const out = [head.map((h, i) => (i === yi ? (kind === 'numeric' ? 'Improvement_Points' : 'Outcome_Grade') : h)).join(',')];
    lines.slice(1).forEach(l => {
        const v = l.split(',');
        const score = 2 + (v[yi] === 'Responder' ? 2.5 : 0) + (v[tap] === 'Positive' ? 0.8 : 0) - 0.03 * (Number(v[age]) - 75) + (rnd() - 0.5) * 2;
        v[yi] = kind === 'numeric' ? score.toFixed(2) : (score > 4.5 ? 'Good' : score > 3 ? 'Partial' : 'Poor');
        out.push(v.join(','));
    });
    return Buffer.from(out.join('\n'));
}

async function loadVariant(page, kind) {
    await page.goto('/index.html');
    await page.setInputFiles('#csv-file-input', { name: `inph_${kind}.csv`, mimeType: 'text/csv', buffer: variantCsv(kind) });
    await expect(page.locator('#data-stats-grid')).toContainText('420');
    await page.click('button.step-tab[data-step="3"]');
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
}

async function selectModels(page, ids) {
    await page.click('#scope-multi', { force: true });
    for (const box of await page.locator('.model-select-checkbox').all()) {
        const id = await box.getAttribute('data-model-id');
        if ((await box.isChecked()) !== ids.includes(id)) await box.click();
    }
}

test('numerical outcome: multiple imputation, subgroups, explanation and report', async ({ page }) => {
    const errors = watch(page);
    await loadVariant(page, 'numeric');
    await page.click('button.step-tab[data-step="3"]'); // the imputation choice is in Step 3
    await page.selectOption('#mi-select', 'mice5');
    await page.click('button.step-tab[data-step="4"]');
    await selectModels(page, ['glm', 'rf']);
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Tournament Complete', { timeout: 170000 });
    await expect(page.locator('#calibration-panel-container')).toContainText('Prediction errors');
    await page.selectOption('#subgroup-panel-container-col', 'Site');
    await page.click('#subgroup-panel-container-btn');
    await expect(page.locator('#subgroup-panel-container-out')).toContainText('Mean observed');
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#standard-html-report-btn')]);
    expect(fs.readFileSync(await dl.path(), 'utf8')).toContain('Prediction errors');
    await page.click('button.step-tab[data-step="5"]');
    await page.fill('#predict-field-11', '');
    await page.click('#run-single-predict-btn');
    const banner = page.locator('#single-result-banner');
    await expect(banner).toContainText('Why this prediction');
    // The explanation ends at the prediction shown (multiple imputation of the blank value included)
    const text = await banner.innerText();
    const shown = Number((text.match(/IMPROVEMENT_POINTS\s*\n\s*([-\d.]+)/i) || [])[1]);
    const moved = Number((text.match(/moved this patient to ([-\d.]+)/) || [])[1]);
    expect(Math.abs(shown - moved)).toBeLessThan(0.006);
    expect(errors).toEqual([]);
});

test('numerical outcome: Option C reports Harrell and .632+ estimates', async ({ page }) => {
    const errors = watch(page);
    await loadVariant(page, 'numeric');
    await page.check('input[name="validation-mode"][value="bootstrap"]', { force: true });
    await page.selectOption('#bootstrap-reps', '50');
    await page.locator('.model-card[data-model-id="rf"]').click();
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 170000 });
    await expect(page.locator('#optimism-container')).toContainText('.632+ estimate');
    await expect(page.locator('#optimism-container')).toContainText('R²');
    expect(errors).toEqual([]);
});

test('three-category outcome: training, subgroups, explanation and report', async ({ page }) => {
    const errors = watch(page);
    await loadVariant(page, 'multi');
    await selectModels(page, ['glm', 'gbm']);
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText('Tournament Complete', { timeout: 170000 });
    await page.click('#subgroup-panel-container-btn');
    await expect(page.locator('#subgroup-panel-container-out')).toContainText('Accuracy');
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#standard-html-report-btn')]);
    expect(fs.readFileSync(await dl.path(), 'utf8')).toContain('Confusion matrix');
    await page.click('button.step-tab[data-step="5"]');
    await page.click('#run-single-predict-btn');
    await expect(page.locator('#single-result-banner')).toContainText('Why this prediction');
    expect(errors).toEqual([]);
});
