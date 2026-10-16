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
w=max(len(x[1]) for x in res)
for st,nm,d,t in res: print(f'{st}  {nm:<{w}}  max|diff|={d:.2e}  (tol {t:.0e})')
print(f"\n{sum(1 for x in res if x[0]=='PASS')}/{len(res)} checks passed")
import sys; sys.exit(0 if all(x[0]=='PASS' for x in res) else 1)
