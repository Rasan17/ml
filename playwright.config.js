// Browser tests of the interface in Chromium (Chrome, Edge), Firefox and WebKit (Safari's engine).
// Run: npm ci && npx playwright install --with-deps && npm run test:ui   (one browser: npx playwright test --project=webkit)
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
    testDir: 'tests/ui',
    timeout: 180000,
    expect: { timeout: 30000 },
    retries: 1,
    // Tests start at the Advanced interface level (first-time visitors get Beginner), so every setting is reachable
    use: { baseURL: 'http://127.0.0.1:8777', headless: true, acceptDownloads: true, storageState: 'tests/ui/advanced-level.json' },
    webServer: { command: 'python3 -m http.server 8777 --bind 127.0.0.1', url: 'http://127.0.0.1:8777/index.html', reuseExistingServer: true, timeout: 30000 },
    projects: [
        { name: 'chromium', use: { browserName: 'chromium' } },
        { name: 'firefox', use: { browserName: 'firefox' } },
        { name: 'webkit', use: { browserName: 'webkit' } }
    ]
});
