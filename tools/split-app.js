// One-off migration: splits the single-closure app.js into ES modules under src/.
// Cuts only at top-level boundaries (4-space indent inside the original IIFE), exports every top-level
// declaration and adds the imports each module needs. Usage: node tools/split-app.js
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const lines = fs.readFileSync(path.join(root, 'app.js'), 'utf8').split('\n');

// Module boundaries: [module name, first line of the section (1-based, a top-level declaration or banner)]
const CUTS = [
    ['core', 11],
    ['watchdog', 263],
    ['shell', 476],
    ['navigation', 1129],
    ['ingest', 1390],
    ['variables', 1616],
    ['imputation', 1889],
    ['training-setup', 2054],
    ['panels', 2331],
    ['cv-tuning', 2747],
    ['standard-training', 3194],
    ['single-prediction', 4386],
    ['batch-prediction', 4544],
    ['pipeline', 4726],
    ['pipeline-results', 6069],
    ['reports', 6624],
    ['model-io', 6904],
    ['settings', 7142]
];

const body = lines.slice(0); // 1-based indexing helpers below
const lastLine = lines.length;
// The IIFE closes with "})();" on the final non-empty line
let end = lastLine;
while (end > 0 && lines[end - 1].trim() === '') end--;
if (lines[end - 1].trim() !== '})();') throw new Error('unexpected file end');

// Snap a cut upwards over the comment/blank block that belongs to the declaration below it
function snap(lineNo) {
    let l = lineNo;
    while (l > 12) {
        const prev = lines[l - 2];
        if (/^\s*$/.test(prev) || /^    \/\//.test(prev) || /^    \/\*\*/.test(prev) || /^     \*/.test(prev)) l--;
        else break;
    }
    return l;
}

const starts = CUTS.map(([, l], i) => (i === 0 ? l : snap(l)));
const chunks = CUTS.map(([name], i) => ({ name, from: starts[i], to: (i + 1 < CUTS.length ? starts[i + 1] : end) - 1 }));

const DECL = /^    (?:async\s+)?(?:function\s*\*?\s*([A-Za-z_$][\w$]*)|const\s+([A-Za-z_$][\w$]*)|class\s+([A-Za-z_$][\w$]*))/;
chunks.forEach(c => {
    c.src = lines.slice(c.from - 1, c.to);
    c.decls = [];
    c.src.forEach(l => { const m = l.match(DECL); if (m) c.decls.push(m[1] || m[2] || m[3]); });
});

const owner = new Map();
chunks.forEach(c => c.decls.forEach(d => {
    if (owner.has(d)) throw new Error(`duplicate top-level name ${d}`);
    owner.set(d, c.name);
}));

// Identifiers used in a chunk (strings and comments are not stripped: this can only over-import, which is harmless)
const IDENT = /[A-Za-z_$][\w$]*/g;
chunks.forEach(c => {
    const used = new Set(c.src.join('\n').match(IDENT));
    c.imports = {};
    used.forEach(id => {
        const o = owner.get(id);
        if (o && o !== c.name) (c.imports[o] = c.imports[o] || []).push(id);
    });
});

const outDir = path.join(root, 'src');
fs.mkdirSync(outDir, { recursive: true });
const header = lines.slice(0, 8).join('\n'); // original file comment
chunks.forEach(c => {
    // The pipeline state used to be initialised at load time; main.js now does it before start-up
    const code = c.src
        .filter(l => !/^    AppState\.pipeline = createEmptyPipelineState\(42\);/.test(l))
        .map(l => (l.startsWith('    ') ? l.slice(4) : l))
        .join('\n')
        .replace(/^'use strict';\n/m, '');
    const imp = Object.keys(c.imports).sort().map(m => `import { ${c.imports[m].sort().join(', ')} } from './${m}.js';`).join('\n');
    const exp = c.decls.length ? `\nexport { ${c.decls.join(', ')} };\n` : '\n';
    const text = `// ${c.name}.js: part of the Universal Medical ML/NeuralNet Predictor application controller\n${imp}${imp ? '\n' : ''}\n${code.replace(/\s+$/, '')}\n${exp}`;
    fs.writeFileSync(path.join(outDir, `${c.name}.js`), text);
    console.log(`${c.name}.js  lines ${c.from}-${c.to}  ${c.decls.length} exports, imports from ${Object.keys(c.imports).length} module(s)`);
});
void header; void body;
