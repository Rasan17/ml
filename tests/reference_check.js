// Computes app statistics on deterministic synthetic data and writes JSON for comparison with Python references
// Usage (from the project folder):  node tests/reference_check.js && python3 tests/reference_check.py
const fs=require('fs');const path=require('path');const dir=process.argv[2]||path.join(__dirname,'..');
global.window={};eval(fs.readFileSync(dir+'/data-engine.js','utf8'));eval(fs.readFileSync(dir+'/models.js','utf8'));
const DE=window.DataEngine, ME=window.ModelEngine;
let s=12345; const rnd=()=>{s=(s*1103515245+12345)%2147483648; return s/2147483648;};
const gauss=()=>{let u=rnd()||1e-9,v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
const N=120; const rows=[];
for(let i=0;i<N;i++){const x1=gauss(), x2=0.6*x1+0.8*gauss(), x3=gauss()*2+1; const g=['A','B','C'][i%3];
 const lin=0.8*x1-0.5*x3+(g==='B'?0.7:0); const y=lin+gauss()*0.7; const cls=(lin+gauss()*0.8)>0?'Yes':'No';
 const grp=['G1','G2'][Math.floor(rnd()*2)];
 rows.push({x1,x2,x3,g,grp,y,cls});}
const out={rows};
const cfg={x1:{type:'continuous'},x2:{type:'continuous'},x3:{type:'continuous'},g:{type:'nominal'},grp:{type:'nominal'},y:{type:'continuous'},cls:{type:'nominal'}};
out.univ_reg=DE.runUnivariateTesting(rows,['x1','g'],'y',cfg);
out.univ_cls=DE.runUnivariateTesting(rows,['x3','grp','g'],'cls',cfg);
out.pearson=DE.calculatePearson(rows.map(r=>r.x1),rows.map(r=>r.x2));
out.spearman=DE.calculateSpearman(rows.map(r=>r.x1),rows.map(r=>r.x3));
out.skew=DE.calculateSkewness(rows.map(r=>r.x3)).skewness;
out.vif=DE.calculateVIF(rows,['x1','x2','x3'],cfg);
out.gvif=DE.calculateVIF(rows,['x1','x2','x3','g'],cfg);
// classification metrics
const yt=[],sc=[];for(let i=0;i<200;i++){const y=rnd()<0.4?1:0; yt.push(y); sc.push(Math.round((0.35*y+rnd()*0.65)*10)/10);} // rounded -> many ties
out.auc_y=yt; out.auc_s=sc; out.auc=ME._calculateBinaryAUC(yt,sc);
const fake={predictSingle:x=>{const p=x[0];return {pred:p>=0.5?1:0,probabilities:{0:1-p,1:p}}}, predict(X){return X.map(x=>this.predictSingle(x).pred)}};
out.cls_eval=ME.evaluate(fake,sc.map(v=>[v]),yt,true,['No','Yes']);
// multiclass metrics with a missing predicted class
const yt3=[],yp3=[];for(let i=0;i<90;i++){const y=i%3; yt3.push(y); yp3.push(rnd()<0.6?y:(y+1)%3);}
const fake3={predictSingle:x=>({pred:x[0],probabilities:{0:x[0]===0?0.7:0.15,1:x[0]===1?0.7:0.15,2:x[0]===2?0.7:0.15}}),predict(X){return X.map(x=>x[0])}};
out.yt3=yt3; out.yp3=yp3; out.mc_eval=ME.evaluate(fake3,yp3.map(v=>[v]),yt3,true,['a','b','c']);
// regression metrics
const regY=rows.map(r=>r.y), regP=rows.map(r=>r.x1*0.8-0.5*r.x3);
out.reg_eval=ME.evaluate({predict:X=>X.map(x=>x[0])},regP.map(v=>[v]),regY,false,[],{});
out.regY=regY; out.regP=regP;
// ridge regression via GD vs closed form
const X=rows.map(r=>[r.x1,r.x2,r.x3]);
const glm=new window.GeneralizedLinearModel(false,1,0.05,5000,0.1,'momentum',1); glm.fit(X,regY);
out.ridge={w:glm.weights,b:glm.bias,lambda:0.1};
// logistic (softmax, 2 classes) via GD
const yc=rows.map(r=>r.cls==='Yes'?1:0);
const lg=new window.GeneralizedLinearModel(true,2,0.1,6000,0.01,'momentum',1); lg.fit(X,yc);
out.logit={probs:X.map(x=>lg.predictSingle(x).probabilities[1]),lambda:0.01, yc};
// Naive Bayes: continuous only (Gaussian) and binary only (Bernoulli)
const nbG=new window.GaussianNaiveBayes(1e-12,'empirical'); nbG.fit(X,yc);
out.nbG=X.map(x=>nbG.predictSingle(x).probabilities[1]);
const XB=rows.map(r=>[r.g==='A'?1:0,r.g==='B'?1:0,r.grp==='G1'?1:0]);
const nbB=new window.GaussianNaiveBayes(1e-4,'empirical'); nbB.fit(XB,yc);
out.XB=XB; out.nbB=XB.map(x=>nbB.predictSingle(x).probabilities[1]);
// Calibration slope and intercept of moderately miscalibrated probabilities
const calY=[],calP=[];for(let i=0;i<600;i++){const lp=-1+3*rnd(); const ptrue=1/(1+Math.exp(-lp)); calY.push(rnd()<ptrue?1:0); calP.push(1/(1+Math.exp(-(0.3+1.6*lp))));}
out.calY=calY; out.calP=calP; out.cal=ME.calibrationSummary(calY,calP);
// DeLong paired AUC test, average precision, decision curve, and logistic regression by IRLS (odds ratios)
const dlY=[],dlA=[],dlB=[];for(let i=0;i<300;i++){const y=rnd()<0.35?1:0; dlY.push(y); const z=rnd(); dlA.push(Math.round((0.6*y+z)*20)/20); dlB.push(0.3*y+rnd());}
out.dlY=dlY; out.dlA=dlA; out.dlB=dlB; out.delong=ME.delongTest(dlY,dlA,dlB); out.pr=ME.prCurve(dlY,dlA).averagePrecision;
out.dca=ME.decisionCurve(dlY,dlA,[0.1,0.2,0.3]);
const irX=rows.map(r=>[r.x1,r.x2,r.g==='B'?1:0,r.g==='C'?1:0]); out.irX=irX; out.irls=ME.logisticIRLS(irX,yc);
// Recalibration: Platt scaling and isotonic regression fitted on the miscalibrated probabilities above
out.platt=ME.fitPlatt(calY,calP); const iso=ME.fitIsotonic(calY,calP); out.isoFitted=calP.map(p=>ME.applyCalibrator(iso,p));
// Rubin's rules pooling and the t distribution
out.rubinIn=[{coef:[0.52,1.10],se:[0.11,0.30]},{coef:[0.47,1.31],se:[0.12,0.28]},{coef:[0.58,0.95],se:[0.10,0.33]},{coef:[0.50,1.22],se:[0.11,0.29]},{coef:[0.55,1.05],se:[0.12,0.31]}];
out.rubin=ME.poolRubin(out.rubinIn,120); out.tq=[ME.tQuantile(0.975,4.3),ME.tQuantile(0.9,17)]; out.tc=[ME.tCdf(1.7,6),ME.tCdf(-2.4,11.5)];
// Cox proportional hazards (Efron ties), Breslow baseline, Harrell's C-index, Kaplan-Meier
const cxX=[],cxT=[],cxE=[];for(let i=0;i<250;i++){const a=rnd()*2-1,b=rnd()<0.5?1:0,c=rnd()*2-1; const h=Math.exp(0.8*a-0.5*b+0.3*c); const t=-Math.log(rnd()||1e-9)/h, cens=rnd()*2.5; cxX.push([a,b,c]); cxT.push(Math.round(Math.min(t,cens)*20)/20+0.05); cxE.push(t<=cens?1:0);}
const cox=new window.CoxModel().fit(cxX,cxT,cxE); out.cxX=cxX; out.cxT=cxT; out.cxE=cxE; out.cox={beta:cox.beta,se:cox.se,loglik:cox.loglik,converged:cox.converged,H:[0.3,0.8,1.5].map(t=>cox.cumHazAt(t))};
out.cindex=ME.concordanceIndex(cxT,cxE,cxX.map(x=>cox.linearPredictor(x))); const km=ME.kaplanMeier(cxT,cxE); out.km=[0.5,1,2].map(t=>ME.kmAt(km,t)); out.kmSE=[0.5,1,2].map(t=>ME.kmStepAt(km,t).se); out.kmCI=[0.5,1,2].map(t=>{const s=ME.kmStepAt(km,t); return [s.lower,s.upper];});
// Time-dependent AUC and censoring-weighted Brier score (checked against scikit-survival); a train/test pair with
// ties and censoring; censoring weights from the training patients, as in the app's survival tab
{
    const r2 = DE.createPRNG(77);
    const mk = n => Array.from({ length: n }, () => { const x = r2() * 2 - 1; const t = Math.round((-Math.log(r2() || 1e-9) / Math.exp(x)) * 20) / 10 + 0.1; const c = Math.round((-Math.log(r2() || 1e-9) / 0.5) * 20) / 10 + 0.1; return { x, t: Math.min(t, c), e: t <= c ? 1 : 0 }; });
    const tr = mk(300), te = mk(200), taus = [0.5, 1, 1.5];
    const cens = ME.censoringKM(tr.map(d => d.t), tr.map(d => d.e));
    out.survTD = { trT: tr.map(d => d.t), trE: tr.map(d => d.e), teT: te.map(d => d.t), teE: te.map(d => d.e), risk: te.map(d => d.x), taus,
        auc: taus.map(tau => ME.timeDependentAUC(te.map(d => d.t), te.map(d => d.e), te.map(d => d.x), tau, cens)),
        surv: taus.map(tau => te.map(d => Math.exp(-Math.exp(d.x) * tau))),
        brier: taus.map(tau => ME.ipcwBrier(te.map(d => d.t), te.map(d => d.e), te.map(d => Math.exp(-Math.exp(d.x) * tau)), tau, cens)) };
}
// Riley et al. minimum sample size, binary outcome (checked against the pmsampsize package)
out.riley = [[24, 0.174, 0.288], [10, 0.3, 0.1], [30, 0.05, 0.04], [15, 0.5, 0.25]].map(([p, prev, r2]) => {
    const r = ME.rileySampleSize({ type: 'binary', p, prevalence: prev, r2 });
    return { p, prev, r2, n: r.n, crit: r.criteria.map(c => c.n) };
});
// Multiple imputation (MICE with predictive mean matching) on MAR data: pooled estimates, compared in Python with
// scikit-learn's IterativeImputer (posterior draws). The methods differ (PMM vs Bayesian ridge draws), so the check
// is agreement of the pooled estimates within a tolerance, not identical imputations.
{
    const r3 = DE.createPRNG(77), g3 = () => Math.sqrt(-2 * Math.log(r3() || 1e-12)) * Math.cos(2 * Math.PI * r3());
    const miRows = Array.from({ length: 2000 }, () => {
        const x1 = g3(), x3 = 0.5 * x1 + g3() * 0.8, x2 = 1 + 0.7 * x1 + 0.3 * x3 + 0.6 * g3();
        const miss = r3() < 1 / (1 + Math.exp(-(-1 + 1.5 * x1)));
        return { x1: Number(x1.toFixed(4)), x2: miss ? null : Number(x2.toFixed(4)), x3: Number(x3.toFixed(4)) };
    });
    const cfg = { x1: { role: 'feature', type: 'continuous' }, x2: { role: 'feature', type: 'continuous' }, x3: { role: 'feature', type: 'continuous' }, Y: { role: 'target', type: 'nominal' } };
    const asRows = miRows.map(r => ({ x1: String(r.x1), x2: r.x2 === null ? '' : String(r.x2), x3: String(r.x3), Y: 'a' }));
    const mi = DE.miceFit(asRows, cfg, { m: 10, seed: 5 });
    const stats = d => {
        const a = d.map(r => Number(r.x1)), b = d.map(r => Number(r.x2)), n = a.length;
        const ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n;
        let sab = 0, saa = 0, sbb = 0; a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; });
        return [mb, Math.sqrt(sbb / (n - 1)), sab / Math.sqrt(saa * sbb), sab / saa];
    };
    const per = mi.datasets.map(stats);
    out.mice = { rows: miRows, pooled: [0, 1, 2, 3].map(j => per.reduce((x, p) => x + p[j], 0) / per.length) };
}
// Survival measures with heavy ties: integer follow-up times, several events and censorings at the same time
{
    const r3 = DE.createPRNG(31);
    const mk = n => Array.from({ length: n }, () => { const x = Math.round((r3() * 2 - 1) * 4) / 4; const t = Math.min(6, Math.ceil(-Math.log(r3() || 1e-9) / Math.exp(x) * 2)); const c = Math.ceil(r3() * 7); return { x, t: Math.min(t, c), e: t <= c ? 1 : 0 }; });
    const tr = mk(150), te = mk(120), taus = [2, 3, 4];
    const cw = ME.censoringKM(tr.map(d => d.t), tr.map(d => d.e));
    const tT = te.map(d => d.t), tE = te.map(d => d.e), risk = te.map(d => d.x);
    const surv = taus.map(tau => te.map(d => Math.exp(-0.2 * tau * Math.exp(d.x))));
    out.survTies = { trT: tr.map(d => d.t), trE: tr.map(d => d.e), teT: tT, teE: tE, risk, taus, surv,
        cindex: ME.concordanceIndex(tT, tE, risk), auc: taus.map(tau => ME.timeDependentAUC(tT, tE, risk, tau, cw)), brier: taus.map((tau, k) => ME.ipcwBrier(tT, tE, surv[k], tau, cw)) };
}
// Competing risks: Aalen-Johansen, cause-specific Cox models and their cumulative incidence, Wolbers' C, and the
// time-dependent AUC and Brier score with a competing event (ties at integer-ish times, censoring)
{
    const r4 = DE.createPRNG(55);
    const mk = n => Array.from({ length: n }, () => {
        const x1 = r4() * 2 - 1, x2 = r4() < 0.5 ? 1 : 0;
        const t1 = -Math.log(r4() || 1e-9) / (0.25 * Math.exp(0.8 * x1 + 0.4 * x2)), t2 = -Math.log(r4() || 1e-9) / (0.15 * Math.exp(-0.5 * x1 + 0.6 * x2)), c = r4() * 8;
        const t = Math.round(Math.min(t1, t2, c) * 4) / 4 + 0.25;
        return { x: [x1, x2], t, s: c < Math.min(t1, t2) ? 0 : (t1 <= t2 ? 1 : 2) };
    });
    const tr = mk(220), te = mk(150), taus = [1.5, 2.5, 4];
    const tT = tr.map(d => d.t), sT = tr.map(d => d.s), X = tr.map(d => d.x);
    const m1 = new window.CoxModel().fit(X, tT, sT.map(v => (v === 1 ? 1 : 0)));
    const m2 = new window.CoxModel().fit(X, tT, sT.map(v => (v === 2 ? 1 : 0)));
    const aj = ME.aalenJohansen(tT, sT);
    const eT = te.map(d => d.t), eS = te.map(d => d.s);
    const cw = ME.censoringKM(tT, sT.map(v => (v > 0 ? 1 : 0)));
    const risk = te.map(d => ME.competingCIF(m1, m2, d.x, d.x, 2.5));
    const cif = taus.map(tau => te.map(d => ME.competingCIF(m1, m2, d.x, d.x, tau)));
    out.cr = { trT: tT, trS: sT, trX: X, teT: eT, teS: eS, teX: te.map(d => d.x), taus,
        beta1: m1.beta, beta2: m2.beta, aj: taus.map(t => ME.cifAt(aj, t)),
        cif: cif.map(c => c.slice(0, 5)), risk,
        cindex: ME.concordanceIndexCR(eT, eS, risk),
        auc: taus.map(tau => ME.timeDependentAUCcr(eT, eS, cif[taus.indexOf(tau)], tau, cw)),
        brier: taus.map((tau, k) => ME.ipcwBrierCR(eT, eS, cif[k], tau, cw)) };
    // Without competing events every competing-risk measure must equal its standard counterpart
    const st = out.survTies, cw2 = ME.censoringKM(st.trT, st.trE);
    out.crReduces = {
        c: [ME.concordanceIndexCR(st.teT, st.teE, st.risk), ME.concordanceIndex(st.teT, st.teE, st.risk)],
        auc: st.taus.map(tau => [ME.timeDependentAUCcr(st.teT, st.teE, st.risk, tau, cw2), ME.timeDependentAUC(st.teT, st.teE, st.risk, tau, cw2)]),
        brier: st.taus.map((tau, k) => [ME.ipcwBrierCR(st.teT, st.teE, st.surv[k].map(v => 1 - v), tau, cw2), ME.ipcwBrier(st.teT, st.teE, st.surv[k], tau, cw2)]),
        aj: [1, 2, 3, 4].map(t => [ME.cifAt(ME.aalenJohansen(st.teT, st.teE), t), 1 - ME.kmAt(ME.kaplanMeier(st.teT, st.teE), t)])
    };
}
// Clustered comparison of two AUCs (Obuchowski 1997): several records per patient
{
    const r5 = DE.createPRNG(91);
    const g = () => { const u = r5() || 1e-9, v = r5(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const y = [], a = [], b = [], cl = [];
    for (let p = 0; p < 50; p++) { const u = g(), k = 1 + Math.floor(r5() * 4); for (let j = 0; j < k; j++) { const yy = r5() < 1 / (1 + Math.exp(-u)) ? 1 : 0; y.push(yy); a.push(Math.round((yy + g()) * 4) / 4); b.push(Math.round((0.7 * yy + g() + 0.3 * u) * 4) / 4); cl.push('P' + p); } }
    const d = ME.delongTest(y, a, b, cl);
    out.obu = { y, a, b, cl, aucA: d.aucA, aucB: d.aucB, se: d.se, p: d.p, seA: d.seA };
}
// Harrell's bootstrap optimism for a Cox model (Option C for time-to-event outcomes): each resample refits the Cox
// model and is scored by the app's own ModelEngine.coxResampleOptimism (C-index, time-dependent AUC, calibration slope)
{
    const n = cxT.length, rng = DE.createPRNG(2024), horizon = 1.0, censKM = ME.censoringKM(cxT, cxE);
    const idxs = Array.from({ length: 25 }, () => Array.from({ length: n }, () => Math.floor(rng() * n)));
    const per = idxs.map(idx => {
        const Xb = idx.map(i => cxX[i]), tb = idx.map(i => cxT[i]), eb = idx.map(i => cxE[i]);
        const m = new window.CoxModel().fit(Xb, tb, eb);
        return ME.coxResampleOptimism({ t: tb, e: eb, lp: Xb.map(x => m.linearPredictor(x)) }, { t: cxT, e: cxE, lp: cxX.map(x => m.linearPredictor(x)) }, horizon, censKM);
    });
    out.coxOpt = { idxs, horizon, c: per.map(p => p.c), td: per.map(p => p.td), slope: per.map(p => p.slope) };
}
fs.writeFileSync(path.join(__dirname,'reference_values.json'),JSON.stringify(out));
console.log('App values written to tests/reference_values.json; now run: python3 tests/reference_check.py');
