// Static check of the ES modules in src/: every imported name is exported by the module it is imported
// from, and no module calls a function exported by another module without importing it (which would only
// fail at run time, when that code path is reached). Usage: node tests/module_check.js
'use strict';
const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, '..', 'src');
const files = fs.readdirSync(srcDir).filter(f => f.endsWith('.js'));
const names = s => s.split(',').map(x => x.trim().split(/\s+as\s+/).pop()).filter(Boolean);
const mod = {};
files.forEach(f => {
    const text = fs.readFileSync(path.join(srcDir, f), 'utf8');
    const code = text.replace(/^import[^\n]*$/gm, '').replace(/\/\/[^\n]*/g, '');
    const declared = new Set([
        ...[...code.matchAll(/^\s*(?:async\s+)?(?:function\s*\*?\s*|const\s+|let\s+|var\s+|class\s+)([A-Za-z_$][\w$]*)/gm)].map(m => m[1]),
        ...[...code.matchAll(/(?:const|let|var)\s*\{([^}]*)\}\s*=/g)].flatMap(m => names(m[1]).map(n => n.split(':').pop().split('=')[0].trim()))
    ]);
    mod[f] = {
        code, declared,
        exports: new Set([...text.matchAll(/^export\s*\{([^}]*)\}/gm)].flatMap(m => names(m[1]))),
        imports: [...text.matchAll(/^import\s*\{([^}]*)\}\s*from\s*'\.\/([\w-]+\.js)';/gm)].map(m => ({ names: names(m[1]), from: m[2] }))
    };
});

const problems = [];
files.forEach(f => {
    const m = mod[f];
    const imported = new Set();
    m.imports.forEach(im => im.names.forEach(n => {
        imported.add(n);
        if (!mod[im.from]) problems.push(`${f}: imports from missing module ${im.from}`);
        else if (!mod[im.from].exports.has(n)) problems.push(`${f}: imports ${n}, which ${im.from} does not export`);
    }));
    files.filter(g => g !== f).forEach(g => mod[g].exports.forEach(n => {
        if (imported.has(n) || m.declared.has(n)) return;
        const call = new RegExp(`(^|\\.\\.\\.|[^\\w$.'"-])${n.replace(/\$/g, '\\$')}\\s*\\(`, 'm');
        if (call.test(m.code)) problems.push(`${f}: calls ${n}() (exported by ${g}) without importing it`);
    }));
});

// The file:// fallback joins every module into one script, so top-level names must be unique across modules
const vm = require('vm');
const bundlePath = path.join(__dirname, '..', 'dist', 'app.bundle.js');
try { new vm.Script(fs.readFileSync(bundlePath, 'utf8'), { filename: 'dist/app.bundle.js' }); } catch (e) { problems.push(`dist/app.bundle.js does not compile (${e.message}): rename the clashing top-level name and run node tools/build.js`); }

// Every data-action used in the modules or the page must be in main.js's list of allowed actions
{
    const allowed = new Set((fs.readFileSync(path.join(__dirname, '..', 'src', 'main.js'), 'utf8').match(/const ACTIONS = new Set\(\[([\s\S]*?)\]\)/) || [, ''])[1].match(/'[^']+'/g)?.map(x => x.slice(1, -1)) || []);
    const used = new Set();
    files.forEach(f => { for (const mm of mod[f].code.matchAll(/actionAttrs\('([^']+)'/g)) used.add(mm[1]); });
    for (const mm of fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8').matchAll(/data-action="([^"]+)"/g)) used.add(mm[1]);
    used.forEach(a => { if (!allowed.has(a)) problems.push(`data-action "${a}" is not in the ACTIONS list of src/main.js, so it would not run`); });
}

if (problems.length) {
    console.log(problems.join('\n'));
    console.log(`\n${problems.length} module problem(s)`);
    process.exit(1);
}
console.log(`${files.length} modules: every import resolves, every cross-module call is imported, and the file:// bundle compiles`);
