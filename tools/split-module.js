// One-off refactoring helper: moves whole top-level sections of one ES module in src/ into new modules,
// without changing any code. Each new module gets the imports it needs (from the original module's imports
// or from its sibling parts) and exports every top-level declaration; other modules' imports are redirected
// to wherever each name now lives, and main.js imports the new modules so the file:// bundle includes them.
// Usage: node tools/split-module.js <module> <newModule>:<firstLine> [<newModule>:<firstLine> ...]
//   e.g. node tools/split-module.js pipeline pipeline-step1:103 pipeline-screening:323
// Lines are 1-based in the current file; each cut should be at a top-level declaration (or its comment).
'use strict';
const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, '..', 'src');
const [base, ...cutArgs] = process.argv.slice(2);
if (!base || !cutArgs.length) { console.error('usage: node tools/split-module.js <module> <new>:<line> ...'); process.exit(1); }
const file = name => path.join(srcDir, `${name}.js`);
const lines = fs.readFileSync(file(base), 'utf8').split('\n');

// Header (comment + imports) and the trailing export list
const importRe = /^import\s*\{([^}]*)\}\s*from\s*'\.\/([\w-]+)\.js';\s*$/;
const exportIdx = lines.findIndex(l => /^export\s*\{/.test(l));
if (exportIdx < 0) throw new Error('no export list');
const imports = []; // { names, from }
lines.forEach(l => { const m = l.match(importRe); if (m) imports.push({ names: m[1].split(',').map(s => s.trim()).filter(Boolean), from: m[2] }); });

const cuts = cutArgs.map(a => { const [name, line] = a.split(':'); return { name, line: Number(line) }; }).sort((a, b) => a.line - b.line);
const firstImport = lines.findIndex(l => l.startsWith('import '));
let lastImport = firstImport;
lines.forEach((l, i) => { if (l.startsWith('import ')) lastImport = i; });
const firstBody = lastImport + 1; // the body keeps its section banners and comments
const parts = [{ name: base, from: firstBody, to: cuts[0].line - 1 }];
cuts.forEach((c, i) => parts.push({ name: c.name, from: c.line - 1, to: i + 1 < cuts.length ? cuts[i + 1].line - 1 : exportIdx }));
parts.forEach(p => {
    p.body = lines.slice(p.from, p.to).join('\n').replace(/\s+$/, '') + '\n';
    p.decls = [...p.body.matchAll(/^(?:async\s+)?(?:function\s*\*?\s*|const\s+|let\s+|class\s+)([A-Za-z_$][\w$]*)/gm)].map(m => m[1]);
});
const owner = new Map();
parts.forEach(p => p.decls.forEach(d => owner.set(d, p.name)));
const uses = (body, name) => new RegExp(`(^|\\.\\.\\.|[^\\w$.'"])${name.replace(/\$/g, '\\$')}(?![\\w$])`).test(body);

const banner = lines.slice(0, firstImport).filter(l => l.startsWith('//') || /^\/\*/.test(l));
parts.forEach((p, i) => {
    const need = {};
    imports.forEach(im => im.names.forEach(n => { if (uses(p.body, n)) (need[im.from] = need[im.from] || []).push(n); }));
    parts.forEach(q => { if (q !== p) q.decls.forEach(d => { if (uses(p.body, d)) (need[q.name] = need[q.name] || []).push(d); }); });
    const head = i === 0 ? banner : [`// ${p.name}.js: part of the Universal Medical ML/NeuralNet Predictor application controller (split from ${base}.js)`,
        ...banner.filter(l => /^\/\* global/.test(l))];
    const importLines = Object.keys(need).sort().map(f => `import { ${[...new Set(need[f])].sort().join(', ')} } from './${f}.js';`);
    const out = [...head, ...importLines, '', p.body.replace(/^\n+/, ''), `export { ${p.decls.join(', ')} };`, ''].join('\n');
    fs.writeFileSync(file(p.name), out);
    console.log(`${p.name}.js: ${p.body.split('\n').length} lines, ${p.decls.length} declarations`);
});

// Redirect other modules' imports of moved names
fs.readdirSync(srcDir).filter(f => f.endsWith('.js') && !parts.some(p => `${p.name}.js` === f)).forEach(f => {
    const txt = fs.readFileSync(path.join(srcDir, f), 'utf8');
    const re = new RegExp(`^import\\s*\\{([^}]*)\\}\\s*from\\s*'\\./${base}\\.js';\\s*$`, 'm');
    const m = txt.match(re);
    if (!m) return;
    const byFile = {};
    m[1].split(',').map(s => s.trim()).filter(Boolean).forEach(n => { const o = owner.get(n) || base; (byFile[o] = byFile[o] || []).push(n); });
    const repl = Object.keys(byFile).sort().map(o => `import { ${byFile[o].join(', ')} } from './${o}.js';`).join('\n');
    fs.writeFileSync(path.join(srcDir, f), txt.replace(re, repl));
    console.log(`  imports redirected in ${f}`);
});

// main.js: import the new modules right after the original one (bundle order)
const mainPath = file('main');
let main = fs.readFileSync(mainPath, 'utf8');
const lineRe = new RegExp(`^import[^\\n]*'\\./${base}\\.js';$`, 'm');
const mm = main.match(lineRe);
if (mm && !mm[0].includes('from')) main = main.replace(lineRe, `${mm[0]}\n${cuts.map(c => `import './${c.name}.js';`).join('\n')}`);
else if (mm) main = main.replace(lineRe, `${mm[0]}\n${cuts.map(c => `import './${c.name}.js';`).join('\n')}`);
fs.writeFileSync(mainPath, main);
console.log('main.js updated');
