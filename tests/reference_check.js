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
out.cindex=ME.concordanceIndex(cxT,cxE,cxX.map(x=>cox.linearPredictor(x))); const km=ME.kaplanMeier(cxT,cxE); out.km=[0.5,1,2].map(t=>ME.kmAt(km,t));
fs.writeFileSync(path.join(__dirname,'reference_values.json'),JSON.stringify(out));
console.log('App values written to tests/reference_values.json; now run: python3 tests/reference_check.py');
