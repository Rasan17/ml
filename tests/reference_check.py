import json, numpy as np, warnings
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
warnings.filterwarnings('ignore')
# Compares the app's statistics with SciPy / scikit-learn / statsmodels (requires those packages).
import os
o=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'reference_values.json'))); R=o['rows']
res=[]
def chk(name, app, ref, tol):
    app=np.atleast_1d(np.array(app,dtype=float)); ref=np.atleast_1d(np.array(ref,dtype=float))
    d=float(np.max(np.abs(app-ref)))
    res.append((('PASS' if d<=tol else 'FAIL'), name, d, tol))
col=lambda k:np.array([r[k] for r in R])
# Univariate: regression target
u={r['feature']:r for r in o['univ_reg']}
r_,p_=stats.pearsonr(col('x1'),col('y')); chk('Pearson r (x1 vs y)',u['x1']['statisticValue'],r_,6e-4); chk('Pearson t-test p',u['x1']['pValue'],p_,max(1e-6,abs(p_)*1e-3))
F,p=stats.f_oneway(*[col('y')[col('g')==k] for k in 'ABC']); chk('ANOVA F (nominal vs continuous target)',u['g']['statisticValue'],F,6e-4); chk('ANOVA p',u['g']['pValue'],p,max(1e-6,p*1e-3))
u={r['feature']:r for r in o['univ_cls']}
F,p=stats.f_oneway(*[col('x3')[col('cls')==k] for k in ['No','Yes']]); chk('t-test/ANOVA F (continuous vs binary)',u['x3']['statisticValue'],F,6e-4); chk('  p',u['x3']['pValue'],p,max(1e-6,p*1e-3))
for f in ['grp','g']:
    tab=np.array([[np.sum((col(f)==a)&(col('cls')==b)) for b in ['No','Yes']] for a in sorted(set(col(f)))])
    c2,p,dof,_=stats.chi2_contingency(tab,correction=False); chk(f'Chi-square stat ({f})',u[f]['statisticValue'],c2,6e-4); chk(f'Chi-square p ({f})',u[f]['pValue'],p,max(1e-6,p*1e-3))
chk('Pearson r (x1,x2)',o['pearson'],stats.pearsonr(col('x1'),col('x2'))[0],6e-5)
chk('Spearman rho (x1,x3)',o['spearman'],stats.spearmanr(col('x1'),col('x3'))[0],6e-5)
chk('Skewness g1 (bias-corrected)',o['skew'],stats.skew(col('x3'),bias=False),6e-4)
Xv=sm.add_constant(np.column_stack([col('x1'),col('x2'),col('x3')]))
vif={v['feature']:v['vif'] for v in o['vif']}
for i,f in enumerate(['x1','x2','x3']): chk(f'VIF {f}',vif[f],variance_inflation_factor(Xv,i+1),6e-3)
# Generalised VIF (Fox & Monette): x1..x3 plus 3-level nominal g (dummies B, C; reference A)
D=np.column_stack([col('x1'),col('x2'),col('x3'),(col('g')=='B').astype(float),(col('g')=='C').astype(float)])
Rm=np.corrcoef(D,rowvar=False)
gv={v['feature']:v for v in o['gvif']}
for f,S in [('x1',[0]),('x2',[1]),('x3',[2]),('g',[3,4])]:
    O=[j for j in range(5) if j not in S]
    G=np.linalg.det(Rm[np.ix_(S,S)])*np.linalg.det(Rm[np.ix_(O,O)])/np.linalg.det(Rm)
    chk(f'GVIF^(1/df) {f} (df={len(S)})',gv[f]['vif'],G**(1/len(S)),6e-3)
Xg=sm.add_constant(D)
for i,f in enumerate(['x1','x2','x3']): chk(f'GVIF(df=1) == statsmodels VIF {f}',gv[f]['vif'],variance_inflation_factor(Xg,i+1),6e-3)
# classification metrics
y=np.array(o['auc_y']); s=np.array(o['auc_s'])
chk('ROC-AUC with ties',o['auc'],metrics.roc_auc_score(y,s),6e-5)
e=o['cls_eval']; pred=(s>=0.5).astype(int)
chk('Brier score (binary)',e['brierScore'],metrics.brier_score_loss(y,s),6e-5)
chk('Accuracy',e['accuracy'],metrics.accuracy_score(y,pred),6e-5)
chk('Balanced accuracy',e['balancedAccuracy'],metrics.balanced_accuracy_score(y,pred),6e-5)
chk('Macro F1',e['macroF1'],metrics.f1_score(y,pred,average='macro',zero_division=0),2e-4)
lo,hi=proportion_confint((pred==y).sum(),len(y),method='wilson'); chk('Wilson 95% CI',e['accuracyCI'],[lo,hi],6e-5)
yt3=np.array(o['yt3']); yp3=np.array(o['yp3']); m=o['mc_eval']
chk('Multiclass macro precision',m['macroPrecision'],metrics.precision_score(yt3,yp3,average='macro',zero_division=0),2e-4)
chk('Multiclass macro recall',m['macroRecall'],metrics.recall_score(yt3,yp3,average='macro',zero_division=0),2e-4)
chk('Multiclass macro F1',m['macroF1'],metrics.f1_score(yt3,yp3,average='macro',zero_division=0),2e-4)
P=np.array([[0.7 if c==p else 0.15 for c in range(3)] for p in yp3]); Y=np.eye(3)[yt3]
chk('Multiclass Brier',m['brierScore'],np.mean(np.sum((P-Y)**2,axis=1)),6e-5)
yR=np.array(o['regY']); pR=np.array(o['regP']); r=o['reg_eval']
chk('R²',r['r2'],metrics.r2_score(yR,pR),6e-5); chk('RMSE',r['rmse'],np.sqrt(metrics.mean_squared_error(yR,pR)),6e-5); chk('MAE',r['mae'],metrics.mean_absolute_error(yR,pR),6e-5)
X=np.column_stack([col('x1'),col('x2'),col('x3')]); n=len(X)
rd=Ridge(alpha=n*o['ridge']['lambda']).fit(X,yR)
chk('Ridge coefficients (GD vs closed form)',o['ridge']['w'],rd.coef_,1e-3); chk('Ridge intercept',o['ridge']['b'],rd.intercept_,1e-3)
yc=np.array(o['logit']['yc'])
# 2-class softmax with L2 on BOTH class vectors (lambda/2 each) is equivalent to binary logistic
# regression with penalty (lambda/4)||w||^2, i.e. scikit-learn C = 2/(n*lambda)
lr=LogisticRegression(C=2/(n*o['logit']['lambda']),tol=1e-12,max_iter=20000).fit(X,yc)
chk('Logistic probabilities (GD vs L-BFGS)',o['logit']['probs'],lr.predict_proba(X)[:,1],2e-3)
g=GaussianNB(var_smoothing=1e-12).fit(X,yc); chk('Gaussian NB probabilities',o['nbG'],g.predict_proba(X)[:,1],2e-4)
XB=np.array(o['XB']); b=BernoulliNB(alpha=1.0,binarize=None).fit(XB,yc); chk('Bernoulli NB probabilities',o['nbB'],b.predict_proba(XB)[:,1],2e-4)
# Calibration slope (logistic regression of y on logit p) and intercept (logit p as offset) vs statsmodels
cy=np.array(o['calY']); cp=np.clip(np.array(o['calP']),1e-4,1-1e-4); lp=np.log(cp/(1-cp))
fs=sm.GLM(cy,sm.add_constant(lp),family=sm.families.Binomial()).fit()
fi=sm.GLM(cy,np.ones((len(cy),1)),family=sm.families.Binomial(),offset=lp).fit()
chk('Calibration slope',o['cal']['slope'],fs.params[1],1e-3); chk('Calibration intercept (calibration-in-the-large)',o['cal']['intercept'],fi.params[0],1e-3)
# DeLong test: independent NumPy implementation of DeLong et al. (1988) structural components
def delong(y,a,b):
    y=np.array(y); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; m,n=len(pos),len(neg)
    def comps(s):
        s=np.array(s); psi=(s[pos][:,None]>s[neg][None,:]).astype(float)+0.5*(s[pos][:,None]==s[neg][None,:])
        return psi.mean(1),psi.mean(0),psi.mean()
    a10,a01,A=comps(a); b10,b01,Bv=comps(b)
    S10=np.cov(np.vstack([a10,b10])); S01=np.cov(np.vstack([a01,b01])); L=np.array([1,-1])
    var=L@S10@L/m+L@S01@L/n; z=(A-Bv)/np.sqrt(var); return A,Bv,z,2*stats.norm.sf(abs(z))
A,Bv,z,pv=delong(o['dlY'],o['dlA'],o['dlB']); d=o['delong']
chk('DeLong AUC A / AUC B',[d['aucA'],d['aucB']],[metrics.roc_auc_score(o['dlY'],o['dlA']),metrics.roc_auc_score(o['dlY'],o['dlB'])],1e-9)
chk('DeLong z and p (paired AUC difference)',[d['z'],d['p']],[z,pv],1e-6)
chk('Average precision (PR curve)',o['pr'],metrics.average_precision_score(o['dlY'],o['dlA']),1e-9)
yy=np.array(o['dlY']); pa=np.array(o['dlA']); nb=[]
for t in [0.1,0.2,0.3]:
    tp=np.sum((pa>=t)&(yy==1)); fp=np.sum((pa>=t)&(yy==0)); nb.append(tp/len(yy)-fp/len(yy)*t/(1-t))
chk('Decision curve net benefit',[x['model'] for x in o['dca']],nb,1e-12)
irX=np.array(o['irX']); lg=sm.Logit(yc,sm.add_constant(irX)).fit(disp=0); ir=o['irls']['rows']
chk('Logistic IRLS coefficients (vs statsmodels Logit)',[r['coef'] for r in ir],lg.params,1e-6)
chk('Logistic IRLS standard errors',[r['se'] for r in ir],lg.bse,1e-6)
chk('Logistic IRLS p-values',[r['p'] for r in ir],lg.pvalues,1e-6)
from sklearn.isotonic import IsotonicRegression
pl=LogisticRegression(penalty=None,tol=1e-12,max_iter=10000).fit(lp.reshape(-1,1),cy)
chk('Platt scaling a, b (vs scikit-learn unpenalised logistic)',[o['platt']['a'],o['platt']['b']],[pl.intercept_[0],pl.coef_[0][0]],1e-5)
ir=IsotonicRegression(out_of_bounds='clip').fit(np.array(o['calP']),cy)
chk('Isotonic regression fitted values (vs scikit-learn)',o['isoFitted'],ir.predict(np.array(o['calP'])),1e-9)
# Rubin's rules (Barnard-Rubin df with complete-data df 120), independent NumPy implementation
E=o['rubinIn']; m=len(E); Q=np.array([e['coef'] for e in E]); U=np.array([e['se'] for e in E])**2
qb=Q.mean(0); W=U.mean(0); Bv=Q.var(0,ddof=1); T=W+(1+1/m)*Bv; lam=(1+1/m)*Bv/T; dfo=(m-1)/lam**2; dc=120; dfobs=(dc+1)/(dc+3)*dc*(1-lam); df=dfo*dfobs/(dfo+dfobs)
lo=qb-stats.t.ppf(.975,df)*np.sqrt(T); pv=2*stats.t.sf(np.abs(qb/np.sqrt(T)),df); R_=o['rubin']
chk("Rubin's rules: pooled estimate, SE, df",[[r['coef'],r['se'],r['df']] for r in R_],np.column_stack([qb,np.sqrt(T),df]),1e-8)
chk("Rubin's rules: 95% CI lower bound and p-value",[[r['low'],r['p']] for r in R_],np.column_stack([lo,pv]),1e-6)
chk('t distribution quantiles and CDF',o['tq']+o['tc'],[stats.t.ppf(.975,4.3),stats.t.ppf(.9,17),stats.t.cdf(1.7,6),stats.t.cdf(-2.4,11.5)],1e-6)
# Cox model vs statsmodels PHReg (Efron ties); C-index and Kaplan-Meier vs independent NumPy implementations
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight
cT=np.array(o['cxT']); cE=np.array(o['cxE']); cX=np.array(o['cxX'])
ph=PHReg(cT,cX,status=cE,ties='efron').fit()
chk('Cox coefficients (Efron ties, vs statsmodels PHReg)',o['cox']['beta'],ph.params,1e-6)
chk('Cox standard errors',o['cox']['se'],ph.bse,1e-6)
chk('Cox log partial likelihood',o['cox']['loglik'],ph.llf,1e-6)
lp=cX@np.array(o['cox']['beta']); w=np.exp(lp); Hb=[]
for t in [0.3,0.8,1.5]:
    Hb.append(sum(np.sum((cT==u)&(cE==1))/w[cT>=u].sum() for u in np.unique(cT[(cE==1)&(cT<=t)])))
chk('Breslow baseline cumulative hazard',o['cox']['H'],Hb,1e-9)
conc=comp=0.0
for i in range(len(cT)):
    if cE[i]!=1: continue
    m=(cT>cT[i])|((cT==cT[i])&(cE==0)); comp+=m.sum(); conc+=(lp[i]>lp[m]).sum()+0.5*(lp[i]==lp[m]).sum()
chk("Harrell's C-index",o['cindex'],conc/comp,1e-12)
sf=SurvfuncRight(cT,cE)
chk('Kaplan-Meier survival (vs statsmodels SurvfuncRight)',o['km'],[sf.surv_prob[sf.surv_times<=t][-1] for t in [0.5,1,2]],1e-12)
chk('Kaplan-Meier Greenwood standard error (vs statsmodels SurvfuncRight)',o['kmSE'],[sf.surv_prob_se[sf.surv_times<=t][-1] for t in [0.5,1,2]],1e-10)
def _loglog(S,se):
    w=1.959964*se/(S*abs(np.log(S))); ll=np.log(-np.log(S)); return [np.exp(-np.exp(ll+w)),np.exp(-np.exp(ll-w))]
chk('Kaplan-Meier 95% CI on the log(-log) scale (from statsmodels S and SE)',[v for p in o['kmCI'] for v in p],[v for t in [0.5,1,2] for v in _loglog(sf.surv_prob[sf.surv_times<=t][-1],sf.surv_prob_se[sf.surv_times<=t][-1])],1e-9)
# Time-dependent AUC (Uno, cumulative/dynamic) and Brier score with censoring weights, vs scikit-survival
try:
    from sksurv.metrics import cumulative_dynamic_auc, brier_score
    from sksurv.util import Surv
    sd=o['survTD']
    ytr=Surv.from_arrays(np.array(sd['trE'],bool),np.array(sd['trT'])); yte=Surv.from_arrays(np.array(sd['teE'],bool),np.array(sd['teT']))
    auc_ref=[cumulative_dynamic_auc(ytr,yte,np.array(sd['risk']),[t])[0][0] for t in sd['taus']]
    chk('Time-dependent AUC with censoring weights (vs scikit-survival cumulative_dynamic_auc)',sd['auc'],auc_ref,1e-10)
    bs_ref=[brier_score(ytr,yte,np.array(sd['surv'][k]).reshape(-1,1),[t])[1][0] for k,t in enumerate(sd['taus'])]
    chk('Brier score with censoring weights (vs scikit-survival brier_score)',sd['brier'],bs_ref,1e-10)
    # Heavy ties (integer times, events and censorings at the same times, tied risk scores) vs scikit-survival
    from sksurv.metrics import concordance_index_censored
    st = o['survTies']
    ytr2 = Surv.from_arrays(np.array(st['trE'], bool), np.array(st['trT'], float)); yte2 = Surv.from_arrays(np.array(st['teE'], bool), np.array(st['teT'], float))
    chk("Harrell's C-index with heavy ties (vs scikit-survival concordance_index_censored)", st['cindex'], concordance_index_censored(np.array(st['teE'], bool), np.array(st['teT'], float), np.array(st['risk']))[0], 1e-12)
    chk('Time-dependent AUC with heavy ties (vs scikit-survival)', st['auc'], [cumulative_dynamic_auc(ytr2, yte2, np.array(st['risk']), [t])[0][0] for t in st['taus']], 1e-10)
    chk('Censoring-weighted Brier score with heavy ties (vs scikit-survival)', st['brier'], [brier_score(ytr2, yte2, np.array(st['surv'][k]).reshape(-1, 1), [t])[1][0] for k, t in enumerate(st['taus'])], 1e-10)
    # Competing risks, against scikit-survival (Aalen-Johansen), statsmodels (cause-specific Cox) and independent
    # NumPy implementations of the cause-specific cumulative incidence, Wolbers' C and the weighted AUC and Brier score
    from sksurv.nonparametric import cumulative_incidence_competing_risks
    cr = o['cr']
    tT, sT, X = np.array(cr['trT']), np.array(cr['trS']), np.array(cr['trX'])
    xs, ci = cumulative_incidence_competing_risks(sT, tT)
    chk('Aalen-Johansen cumulative incidence (vs scikit-survival)', cr['aj'], [ci[1][xs <= t][-1] for t in cr['taus']], 1e-10)
    b1 = PHReg(tT, X, status=(sT == 1).astype(int), ties='efron').fit().params
    b2 = PHReg(tT, X, status=(sT == 2).astype(int), ties='efron').fit().params
    chk('Cause-specific Cox coefficients, event and competing event (vs statsmodels)', list(cr['beta1']) + list(cr['beta2']), list(b1) + list(b2), 1e-6)
    def breslow(beta, ev):
        w = np.exp(X @ beta); us = np.unique(tT[ev == 1])
        return us, np.array([np.sum((tT == u) & (ev == 1)) / w[tT >= u].sum() for u in us])
    u1, d1 = breslow(b1, (sT == 1).astype(int)); u2, d2 = breslow(b2, (sT == 2).astype(int))
    def cif_np(x, tau):
        e1, e2 = np.exp(np.dot(x, b1)), np.exp(np.dot(x, b2)); F = 0.0
        for sv in sorted(set(u1[u1 <= tau]) | set(u2[u2 <= tau])):
            H1 = d1[u1 < sv].sum() * e1; H2 = d2[u2 < sv].sum() * e2
            F += np.exp(-H1 - H2) * d1[u1 == sv].sum() * e1
        return F
    chk('Cause-specific cumulative incidence F1(t|x) (independent NumPy)', [v for row in cr['cif'] for v in row], [cif_np(x, tau) for tau in cr['taus'] for x in cr['teX'][:5]], 1e-8)
    eT, eS, rk = np.array(cr['teT']), np.array(cr['teS']), np.array(cr['risk'])
    conc = comp = 0.0
    for i in range(len(eT)):
        if eS[i] != 1: continue
        m = ((eT > eT[i]) | ((eT == eT[i]) & (eS == 0)) | ((eS == 2) & (eT <= eT[i]))); m[i] = False
        comp += m.sum(); conc += (rk[i] > rk[m]).sum() + 0.5 * (rk[i] == rk[m]).sum()
    chk("Wolbers' concordance for competing risks (independent NumPy)", cr['cindex'], conc / comp, 1e-12)
    def G_at(t):  # censoring Kaplan-Meier from the training patients, events before censoring at tied times
        g = 1.0
        for u in np.unique(tT[tT <= t]):
            n = (tT >= u).sum(); d = ((tT == u) & (sT > 0)).sum(); c = ((tT == u) & (sT == 0)).sum()
            if n - d > 0: g *= 1 - c / (n - d)
        return g
    cifs = [np.array([cif_np(x, tau) for x in cr['teX']]) for tau in cr['taus']]
    aucs, briers = [], []
    for k, tau in enumerate(cr['taus']):
        gt = G_at(tau); cases = [(i, 1 / G_at(eT[i])) for i in range(len(eT)) if eS[i] == 1 and eT[i] <= tau]
        ctrls = [(j, 1 / gt) for j in range(len(eT)) if eT[j] > tau] + [(j, 1 / G_at(eT[j])) for j in range(len(eT)) if eS[j] == 2 and eT[j] <= tau]
        num = den = 0.0
        for i, wi in cases:
            for j, wj in ctrls:
                w = wi * wj; den += w; num += w * (1.0 if cifs[k][i] > cifs[k][j] else 0.5 if cifs[k][i] == cifs[k][j] else 0.0)
        aucs.append(num / den)
        tot = 0.0
        for i in range(len(eT)):
            if eT[i] <= tau and eS[i] > 0: tot += ((1.0 if eS[i] == 1 else 0.0) - cifs[k][i]) ** 2 / G_at(eT[i])
            elif eT[i] > tau: tot += cifs[k][i] ** 2 / gt
        briers.append(tot / len(eT))
    chk('Time-dependent AUC with a competing event (independent NumPy)', cr['auc'], aucs, 1e-8)
    chk('Brier score with a competing event (independent NumPy)', cr['brier'], briers, 1e-8)
    rd = o['crReduces']
    chk('Without competing events, Wolbers C, AUC, Brier and Aalen-Johansen equal Harrell C, AUC, Brier and 1 - Kaplan-Meier',
        [rd['c'][0]] + [p[0] for p in rd['auc']] + [p[0] for p in rd['brier']] + [p[0] for p in rd['aj']],
        [rd['c'][1]] + [p[1] for p in rd['auc']] + [p[1] for p in rd['brier']] + [p[1] for p in rd['aj']], 1e-12)
    # Harrell's bootstrap optimism for a Cox model: per resample, refit with statsmodels PHReg (Efron ties), then
    # Harrell's C (NumPy), Uno's time-dependent AUC (scikit-survival) and the calibration slope (PHReg on the
    # original patients' linear predictor), each on the resample and on the original patients
    def _harrell(t, e, lp):
        conc = comp = 0.0
        for i in range(len(t)):
            if e[i] != 1: continue
            m = (t > t[i]) | ((t == t[i]) & (e == 0)); comp += m.sum(); conc += (lp[i] > lp[m]).sum() + 0.5 * (lp[i] == lp[m]).sum()
        return conc / comp
    co = o['coxOpt']; H = co['horizon']; yO = Surv.from_arrays(cE.astype(bool), cT)
    rc, rtd, rsl = [], [], []
    for idx in co['idxs']:
        idx = np.array(idx); Xb, tb, eb = cX[idx], cT[idx], cE[idx]
        beta = PHReg(tb, Xb, status=eb, ties='efron').fit().params
        lpB, lpO = Xb @ beta, cX @ beta
        rc.append(_harrell(tb, eb, lpB) - _harrell(cT, cE, lpO))
        yB = Surv.from_arrays(eb.astype(bool), tb)
        rtd.append(cumulative_dynamic_auc(yB, yB, lpB, [H])[0][0] - cumulative_dynamic_auc(yO, yO, lpO, [H])[0][0])
        rsl.append(PHReg(cT, lpO.reshape(-1, 1), status=cE, ties='efron').fit().params[0])
    chk("Cox bootstrap optimism: C-index (resample minus original), per resample", co['c'], rc, 1e-6)
    chk("Cox bootstrap optimism: time-dependent AUC at the horizon, per resample", co['td'], rtd, 1e-6)
    chk("Cox bootstrap calibration slope on the original patients, per resample", co['slope'], rsl, 1e-6)
except ImportError:
    print('scikit-survival not installed: time-dependent AUC and Brier checks skipped (pip install scikit-survival)')
# Riley minimum sample size (binary outcome) vs the pmsampsize package, every criterion and the final size
try:
    import io, contextlib
    import pmsampsize.pmsampsize as _pms
    app_n, ref_n = [], []
    for c in o['riley']:
        with contextlib.redirect_stdout(io.StringIO()):
            r = _pms.pmsampsize(type='b', parameters=c['p'], csrsquared=c['r2'], prevalence=c['prev'], noprint=True)
        rows = [x for x in r['results_table'] if isinstance(x, list)]
        app_n += c['crit'] + [c['n']]
        ref_n += [rows[0][1], rows[1][1], rows[2][1], r['sample_size']]
    chk('Riley minimum sample size, binary outcome: 3 criteria and final size (vs pmsampsize)', app_n, ref_n, 0.5)
except ImportError:
    print('pmsampsize not installed: Riley sample-size check skipped (pip install pmsampsize)')
# Multiple imputation under MAR: the app's MICE (PMM) vs scikit-learn's IterativeImputer with posterior draws,
# pooled over 10 imputations: mean and SD of the imputed variable, its correlation and slope with the predictor.
# Different algorithms, so agreement within a tolerance is checked (both should recover the same quantities).
if 'mice' in o:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    M = np.array([[r['x1'], np.nan if r['x2'] is None else r['x2'], r['x3']] for r in o['mice']['rows']], dtype=float)
    per = []
    for k in range(10):
        Z = IterativeImputer(sample_posterior=True, max_iter=10, random_state=k).fit_transform(M)
        a, b = Z[:, 0], Z[:, 1]
        per.append([b.mean(), b.std(ddof=1), np.corrcoef(a, b)[0, 1], np.cov(a, b, ddof=1)[0, 1] / a.var(ddof=1)])
    ref = np.mean(per, axis=0)
    # Tolerance 0.05: PMM (5 donors, as in R's mice) slightly attenuates when missingness is concentrated where
    # observed donors are scarce (here about 1% in the mean and 3% in the slope, see README "Known limitations")
    chk('MICE (PMM) pooled mean, SD, correlation, slope under MAR vs scikit-learn IterativeImputer', o['mice']['pooled'], ref, 0.05)
w=max(len(x[1]) for x in res)
for st,nm,d,t in res: print(f'{st}  {nm:<{w}}  max|diff|={d:.2e}  (tol {t:.0e})')
print(f"\n{sum(1 for x in res if x[0]=='PASS')}/{len(res)} checks passed")
import sys; sys.exit(0 if all(x[0]=='PASS' for x in res) else 1)
