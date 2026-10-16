// Automated accessibility scan (axe-core, WCAG 2.1 A and AA rules) of each main screen, including the panels
// that appear after training. Fails on any violation, with the rule and the affected elements in the message.
const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

const RULES = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];

async function scan(page, label) {
    const problems = [];
    for (const theme of ['dark', 'light']) {
        // switch theme with transitions off, so colours are final when measured
        await page.evaluate(t => {
            if (!document.getElementById('no-transitions')) { const st = document.createElement('style'); st.id = 'no-transitions'; st.textContent = '*,*::before,*::after{transition:none!important;animation:none!important}'; document.head.appendChild(st); }
            if ((document.documentElement.getAttribute('data-theme') || 'dark') !== t) document.getElementById('theme-toggle').click();
        }, theme);
        const res = await new AxeBuilder({ page }).withTags(RULES).analyze();
        problems.push(...res.violations.map(v => `${label} (${theme}): [${v.impact}] ${v.id} (${v.help}) at ${v.nodes.slice(0, 4).map(n => n.target.join(' ')).join(' | ')}${v.nodes.length > 4 ? ` and ${v.nodes.length - 4} more` : ''}`));
    }
    return problems;
}

test('accessibility of every main screen (axe-core, WCAG 2.1 AA), dark and light themes', async ({ page }) => {
    test.setTimeout(400000);
    page.on('dialog', d => d.accept());
    const problems = [];
    await page.goto('/index.html');
    problems.push(...await scan(page, 'start page'));
    await page.click('#sample-shunt-large-btn');
    await expect(page.locator('#data-preview-container table')).toBeVisible();
    problems.push(...await scan(page, 'step 1 (data loaded)'));
    await page.click('button.step-tab[data-step="2"]');
    problems.push(...await scan(page, 'step 2 (variables)'));
    await page.click('button.step-tab[data-step="3"]');
    problems.push(...await scan(page, 'step 3 (missing data)'));
    await page.click('#apply-imputation-btn');
    await page.click('button.step-tab[data-step="4"]');
    problems.push(...await scan(page, 'step 4 (training set-up)'));
    await page.locator('.model-card[data-model-id="glm"]').click();
    await page.click('#train-model-btn');
    await expect(page.locator('#train-status')).toContainText(/Deployed|Complete/, { timeout: 150000 });
    await page.click('#subgroup-panel-container-btn');
    problems.push(...await scan(page, 'step 4 (results)'));
    await page.click('button.step-tab[data-step="5"]');
    await page.click('#run-single-predict-btn');
    await expect(page.locator('#single-result-banner')).toContainText('Why this prediction');
    problems.push(...await scan(page, 'step 5 (prediction)'));
    await page.click('button.step-tab[data-step="1"]');
    await page.click('#sample-survival-btn');
    await page.fill('#surv-horizon', '24');
    await page.click('#surv-run-btn');
    await expect(page.locator('#survival-results')).toContainText("Harrell's C-index", { timeout: 60000 });
    problems.push(...await scan(page, 'survival tab (results)'));
    expect(problems, problems.join('\n')).toEqual([]);
});
