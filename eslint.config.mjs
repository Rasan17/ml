// Lint check run in CI: catches names used but never defined or imported (e.g. a helper deleted while still in use),
// which the module check (cross-module imports only) cannot see. Only no-undef is enforced.
const browser = ['document', 'window', 'alert', 'confirm', 'prompt', 'console', 'URL', 'Blob', 'FileReader', 'fetch',
    'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'localStorage', 'sessionStorage', 'Event', 'navigator',
    'location', 'Worker', 'requestAnimationFrame', 'getComputedStyle', 'performance', 'self'];
const app = ['DataEngine', 'ModelEngine', 'Chart', 'tf'];
export default [{
    files: ['src/**/*.js'],
    languageOptions: { ecmaVersion: 2023, sourceType: 'module', globals: Object.fromEntries([...browser, ...app].map(g => [g, 'readonly'])) },
    rules: { 'no-undef': 'error' }
}];
