// Regression tests for the defects reported by the external (Genspark) audit of October 2026: each check
// reproduces the reported failure and confirms it is fixed. No browser needed.
// Usage (from the project folder):  node tests/audit_regressions.js
const fs = require('fs');
const path = require('path');
const dir = path.join(__dirname, '..');
global.window = {};
eval(fs.readFileSync(path.join(dir, 'data-engine.js'), 'utf8'));
eval(fs.readFileSync(path.join(dir, 'models.js'), 'utf8'));
const DE = window.DataEngine;
const ME = window.ModelEngine;
console.warn = () => {};

let passed = 0, failed = 0;
const results = [];
async function check(name, fn) {
    try {
        const msg = await fn();
        passed++;
        results.push(`PASS  ${name}${msg ? `  (${msg})` : ''}`);
    } catch (e) {
        failed++;
        results.push(`FAIL  ${name}: ${e && e.message ? e.message : e}`);
    }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }
const rejects = async (fn, what) => { try { await fn(); } catch (e) { return e.message; } throw new Error(`${what} was accepted`); };

(async () => {
    await check('Model files cannot install code (typed-array allowlist, no method or prototype fields)', async () => {
        const glm = ME.createModel('glm', true, 2, { seed: 1 });
        glm.fit([[0, 1], [1, 0], [0.2, 0.9], [0.9, 0.1]], [0, 1, 0, 1]);
        const ser = await ME.serializeModel(glm);
        const ok = await ME.deserializeModel(JSON.parse(JSON.stringify(ser)));
        assert(typeof ok.predictSingle === 'function' && isFinite(ok.predictSingle([0, 1]).pred), 'a genuine model no longer loads');
        const evil = (state) => ({ type: 'glm', state: { ...ser.state, ...state } });
        const m1 = await rejects(() => ME.deserializeModel(evil({ weights: { __typed: 'Function', data: ['return 1'] } })), 'a Function "typed array"');
        const m2 = await rejects(() => ME.deserializeModel(evil({ predictSingle: 1 })), 'a field replacing a method');
        const m3 = await rejects(() => ME.deserializeModel(JSON.parse(`{"type":"glm","state":{"__proto__":{"x":1}}}`)), 'a __proto__ field');
        const m4 = await rejects(() => ME.deserializeModel(evil({ weights: { __typed: 'Float64Array', data: ['1'] } })), 'a non-numeric typed array');
        return [m1, m2, m3, m4].map(m => m.replace('Invalid model file: ', '')).join('; ');
    });

    await check('Isotonic recalibration does not depend on the order of tied predictions', async () => {
        const p = new Array(20).fill(0.5);
        const a = ME.fitIsotonic([...new Array(10).fill(0), ...new Array(10).fill(1)], p);
        const b = ME.fitIsotonic([...new Array(10).fill(1), ...new Array(10).fill(0)], p);
        assert(JSON.stringify(a) === JSON.stringify(b), `fits differ: ${JSON.stringify(a.y)} vs ${JSON.stringify(b.y)}`);
        assert(Math.abs(a.y[0] - 0.5) < 1e-12, `tied block should calibrate to 0.5, got ${a.y[0]}`);
        // random permutations of data with many ties give the same fit
        const rng = DE.createPRNG(3);
        const pp = Array.from({ length: 200 }, () => Math.round(rng() * 10) / 10), yy = pp.map(v => (rng() < v ? 1 : 0));
        const ref = JSON.stringify(ME.fitIsotonic(yy, pp));
        for (let r = 0; r < 20; r++) {
            const order = pp.map((_, i) => i).sort(() => rng() - 0.5);
            assert(JSON.stringify(ME.fitIsotonic(order.map(i => yy[i]), order.map(i => pp[i]))) === ref, 'a permutation changed the fit');
        }
        return 'tied block → 0.5 in either order; 20 random permutations identical';
    });

    await check('CSV: renamed duplicate headers never collide; unclosed quotes rejected; infinite numbers treated as missing', async () => {
        const p = DE.parseCSV('a,a,a_2\nfirst,second,third\n');
        assert(new Set(p.headers).size === 3, `headers not unique: ${p.headers}`);
        assert(p.rows[0].a === 'first' && p.rows[0].a_2 === 'third' && Object.values(p.rows[0]).includes('second'), `a value was lost: ${JSON.stringify(p.rows[0])}`);
        const q = DE.parseCSV('__proto__,b\n1,2\n');
        assert(Object.getPrototypeOf(q.rows[0]) === Object.prototype && q.headers[0] !== '__proto__', 'a "__proto__" header altered the row object');
        const msg = await rejects(() => DE.parseCSV('x,y\n"open,1\n2,3\n'), 'an unclosed quotation mark');
        const r = DE.parseCSV('x,y\n"ab"cd,1\n');
        assert(r.malformedQuoteCount === 1 && DE.parseIssues(r).length === 1, 'text after a closing quote not reported');
        ['Infinity', '-inf', '1e999', 'INF'].forEach(v => assert(DE.isMissing(v), `${v} not treated as missing`));
        ['0', '-3.5', '1e10', 'Inferior'].forEach(v => assert(!DE.isMissing(v), `${v} wrongly treated as missing`));
        return `headers ${p.headers.join(', ')}; ${msg.slice(0, 60)}…`;
    });

    await check('Cross-validation with one patient in an outcome group is reported as not possible (no silent 2 folds)', async () => {
        const code = fs.readFileSync(path.join(dir, 'src', 'validation-core.js'), 'utf8').replace(/^export\s*\{[^}]*\};?\s*$/gm, '').replace(/^import .*$/gm, '');
        const VC = new Function('DataEngine', 'ModelEngine', `${code}\nreturn { effectiveFolds };`)(DE, ME);
        const rows = [...new Array(30).fill(0).map(() => ({ Y: 'No' })), { Y: 'Yes' }];
        const m = await rejects(() => VC.effectiveFolds(rows, 5, true, 'Y'), 'one patient with the outcome');
        assert(VC.effectiveFolds([...rows, { Y: 'Yes' }, { Y: 'Yes' }], 5, true, 'Y') === 3, 'three events should give 3 folds');
        const grouped = [...new Array(10).fill(0).map((_, i) => ({ Y: 'No', id: i })), { Y: 'Yes', id: 'p1' }, { Y: 'Yes', id: 'p1' }];
        await rejects(() => VC.effectiveFolds(grouped, 5, true, 'Y', r => r.id), 'two records of the same patient as two patients');
        return m.slice(0, 90) + '…';
    });

    await check('Survival Brier score is "not estimable" (null) without censoring support, never a number', async () => {
        const time = [1, 2, 3, 4, 5, 6], event = [1, 0, 1, 0, 0, 0], surv = [0.5, 0.6, 0.4, 0.7, 0.8, 0.9];
        const censKM = ME.censoringKM ? ME.censoringKM(time, event) : null;
        assert(censKM, 'censoringKM unavailable');
        const ok = ME.ipcwBrier(time, event, surv, 2.5, censKM);
        assert(typeof ok === 'number' && isFinite(ok), 'an estimable horizon should give a number');
        const late = ME.ipcwBrier(time, event, surv, 10, censKM);
        assert(late === null || (typeof late === 'number' && late > 0), `beyond follow-up gave ${late}`);
        const none = ME.ipcwBrier([1, 2], [0, 0], [0.5, 0.5], 5, ME.censoringKM([1, 2], [0, 0]));
        assert(none === null, `no support gave ${none}`);
        return `horizon 2.5 → ${ok.toFixed(3)}; all censored before horizon → not estimable`;
    });

    await check('Patient-data warning on export finds k-NN inside recalibrated, pooled, ensemble and fallback wrappers', async () => {
        const src = fs.readFileSync(path.join(dir, 'src', 'model-io.js'), 'utf8');
        const fnText = src.slice(src.indexOf('function modelContainsKnn'), src.indexOf('\n}\n', src.indexOf('function modelContainsKnn')) + 2);
        const modelContainsKnn = new Function(`${fnText}; return modelContainsKnn;`)();
        const knn = ME.createModel('knn', true, 2, { seed: 1 });
        knn.fit([[0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 0, 1]);
        const s = await ME.serializeModel(knn);
        const cases = {
            calibrated: await ME.serializeModel(ME.wrapCalibrated(knn, { type: 'platt', a: 0, b: 1 })),
            pooled: await ME.serializeModel(ME.createMIPooled([knn, knn])),
            calibratedPooled: await ME.serializeModel(ME.wrapCalibrated(ME.createMIPooled([knn]), { type: 'platt', a: 0, b: 1 })),
            ensemble: { type: 'ensemble', models: [{ model: s }] }
        };
        Object.entries(cases).forEach(([k, v]) => assert(modelContainsKnn(v), `k-NN inside a ${k} model was missed`));
        const glm = ME.createModel('glm', true, 2, { seed: 1 });
        glm.fit([[0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 0, 1]);
        assert(!modelContainsKnn(await ME.serializeModel(ME.wrapCalibrated(glm, { type: 'platt', a: 0, b: 1 }))), 'a model without k-NN was flagged');
        return Object.keys(cases).join(', ') + ' all flagged; logistic not flagged';
    });

    console.log(results.join('\n'));
    console.log(`\n${passed}/${passed + failed} audit regression checks passed`);
    if (failed) process.exit(1);
})();
