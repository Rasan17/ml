// The R cross-check script (Step 4): generated from five real analyses (logistic regression with Options A and C,
// linear regression, a Cox model and a competing-risk model). The scripts' content is always checked; where R and
// its packages are installed (the "R cross-check" workflow, or locally), each script is also run and the app's
// and R's results must agree within the stated tolerances.
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.join(__dirname, '..', '..');
const hasR = (() => {
    try { execFileSync('Rscript', ['-e', 'for (p in c("survival","pROC","rms","riskRegression","prodlim")) if (!requireNamespace(p, quietly = TRUE)) quit(status = 3)'], { stdio: 'ignore' }); return true; } catch (e) { return false; }
})();

async function selectModels(page, ids) {
    await page.click('#scope-multi', { force: true });
    for (const box of await page.locator('.model-select-checkbox').all()) {
        const id = await box.getAttribute('data-model-id');
        if ((await box.isChecked()) !== ids.includes(id)) await box.click();
    }
}

// Rows of the printed table: { measure, app, r, how }
function parseTable(out) {
    const lines = out.split('\n');
    const start = lines.findIndex(l => l.startsWith('Measure'));
    const rows = [];
    for (let i = start + 1; i < lines.length && lines[i].trim(); i++) {
        const parts = lines[i].split(/\s{2,}/);
        if (parts.length >= 4) rows.push({ measure: parts[0], app: parts[1], r: parts[2], how: parts.slice(3).join(' ') });
    }
    return rows;
}

const SCENARIOS = [
    { name: 'logistic, Option A', data: 'medical_shunt_cohort_large.csv', kind: 'std', sample: '#sample-shunt-large-btn', mode: 'split',
      tol: { 'ROC-AUC': 0.005, 'Brier score': 0.005, 'Calibration slope': 0.03, 'Calibration intercept': 0.03 } },
    { name: 'logistic, Option C', data: 'medical_shunt_cohort_large.csv', kind: 'std', sample: '#sample-shunt-large-btn', mode: 'bootstrap',
      tol: { 'ROC-AUC': 0.005, 'Brier score': 0.005, 'Calibration slope': 0.03, 'Optimism-corrected C (AUC)': 0.02, 'Optimism-corrected calibration slope': 0.12 } },
    { name: 'linear regression, Option A', data: 'medical_recovery_score.csv', kind: 'std', sample: '#sample-regression-btn', mode: 'split',
      tol: { 'R-squared': 0.005, RMSE: 0.01, MAE: 0.01 } },
    { name: 'Cox model, Option A', data: 'medical_shunt_revision_survival.csv', kind: 'surv', sample: '#sample-survival-btn', horizon: '24',
      tol: { "Harrell's C-index": 0.002, 'Time-dependent AUC at 24': 0.03, 'Brier score at 24': 0.03, 'Observed risk by 24 (Kaplan-Meier)': 0.002, 'Mean predicted risk by 24': 0.005 } },
    { name: 'competing risks, Option A', data: 'medical_shunt_revision_competing.csv', kind: 'surv', sample: '#sample-competing-btn', horizon: '36',
      tol: { "Wolbers' C-index": 0.002, 'Competing-risk AUC at 36': 0.03, 'Competing-risk Brier score at 36': 0.03, 'Observed risk by 36 (Aalen-Johansen)': 0.002, 'Mean predicted risk by 36': 0.01 } }
];

for (const sc of SCENARIOS) {
    test(`R cross-check script: ${sc.name}${hasR ? ' (run in R)' : ' (content only: R not installed)'}`, async ({ page, browserName }) => {
        test.skip(browserName !== 'chromium', 'generated once, in Chromium');
        test.setTimeout(600000);
        page.on('dialog', d => d.accept());
        await page.goto('/index.html');
        await page.click(sc.sample);
        if (sc.kind === 'surv') { await page.fill('#ot-horizon', sc.horizon); await page.dispatchEvent('#ot-horizon', 'change'); }
        await page.click('button.step-tab[data-step="3"]');
        await page.click('#apply-imputation-btn');
        await page.click('button.step-tab[data-step="4"]');
        await page.check(`input[name="validation-mode"][value="${sc.mode || 'split'}"]`, { force: true });
        if (sc.mode === 'bootstrap') await page.selectOption('#bootstrap-reps', '100');
        let btn = '#surv-r-check-btn';
        if (sc.kind === 'std') {
            await selectModels(page, ['glm', 'rf']);
            await page.click('#train-model-btn');
            await expect(page.locator('#train-status')).toContainText('Complete', { timeout: 300000 });
            btn = '#standard-r-check-btn';
        } else {
            await page.click('#train-model-btn');
            await expect(page.locator(btn)).toBeVisible({ timeout: 300000 });
        }
        const [dl] = await Promise.all([page.waitForEvent('download'), page.click(btn)]);
        const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rcheck-'));
        const script = path.join(dir, 'crosscheck.R');
        await dl.saveAs(script);
        const text = fs.readFileSync(script, 'utf8');
        expect(text).toContain(`data_file = "${sc.data}"`);
        expect(text).toContain('APP <- list(');
        expect(text).toContain('contains no patient data');
        if (sc.mode !== 'bootstrap') expect(text).toMatch(/test_rows = c\(\d+/);
        if (!hasR) { if (process.env.CI_REQUIRE_R) throw new Error('R and its packages were expected (CI_REQUIRE_R) but are not available'); return; }
        fs.copyFileSync(path.join(ROOT, sc.data), path.join(dir, sc.data));
        const out = execFileSync('Rscript', [script], { cwd: dir, encoding: 'utf8', timeout: 540000 });
        const rows = parseTable(out);
        for (const [measure, tol] of Object.entries(sc.tol)) {
            const row = rows.find(x => x.measure === measure && x.app !== '-');
            expect(row, `${measure} row in the R output`).toBeTruthy();
            const diff = Math.abs(Number(row.app) - Number(row.r));
            expect(diff, `${measure}: app ${row.app}, R ${row.r}`).toBeLessThanOrEqual(tol);
        }
        const counts = rows.filter(x => /^(Patients|Events) evaluated/.test(x.measure));
        counts.forEach(x => expect(x.r, x.measure).toBe(x.app));
        console.log(`R cross-check (${sc.name}) ran in R and agreed: ${Object.keys(sc.tol).map(m => { const r = rows.find(x => x.measure === m && x.app !== '-'); return `${m} app ${r.app} / R ${r.r}`; }).join('; ')}`);
    });
}
