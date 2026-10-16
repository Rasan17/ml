"""Independent check of the app's subgroup (fairness) panel with Fairlearn.

The app's own engine (tests/audit_app_runner.js) trains three models on the simulated iNPH cohort and
writes their test predictions and the subgroup figures computed by src/subgroups.js. Fairlearn's MetricFrame
recomputes every figure by sex, hospital and age band; they must agree. Fairlearn's standard disparity
measures (demographic parity and equalised odds differences) are reported as well.

Usage (Fairlearn needs its own environment):
    python3 -m venv .venv-fairlearn && .venv-fairlearn/bin/pip install fairlearn pandas
    node tests/audit_app_runner.js /tmp/ummp-audit
    .venv-fairlearn/bin/python tests/audit_fairlearn.py /tmp/ummp-audit
Writes tests/AUDIT_FAIRLEARN.md; exits with 1 if any figure disagrees.
"""
import json
import os
import sys

import numpy as np
from fairlearn.metrics import (MetricFrame, demographic_parity_difference, equalized_odds_difference,
                               false_negative_rate, true_negative_rate, true_positive_rate)
from sklearn.metrics import brier_score_loss, roc_auc_score
import fairlearn

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ummp-audit'
res = json.load(open(os.path.join(src, 'app_results.json')))
TOL = 1e-9
TOL_AUC = 5e-5 + 1e-12  # the app rounds AUC to 4 decimal places


def metrics(threshold):
    hard = lambda p: (np.asarray(p) >= threshold).astype(int)
    def auc(y, p):
        return roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    def net_benefit(y, p):
        yp = hard(p); n = len(y)
        tp = np.sum((yp == 1) & (np.asarray(y) == 1)); fp = np.sum((yp == 1) & (np.asarray(y) == 0))
        return tp / n - fp / n * threshold / (1 - threshold)
    return {
        'n': lambda y, p: len(y),
        'events': lambda y, p: int(np.sum(y)),
        'observed': lambda y, p: float(np.mean(y)),
        'meanPred': lambda y, p: float(np.mean(p)),
        'oe': lambda y, p: float(np.mean(y) / np.mean(p)),
        'auc': auc,
        'brier': lambda y, p: brier_score_loss(y, p),
        'sens': lambda y, p: true_positive_rate(y, hard(p)) if np.sum(y) else np.nan,
        'spec': lambda y, p: true_negative_rate(y, hard(p)) if np.sum(1 - np.asarray(y)) else np.nan,
        'netBenefit': net_benefit,
    }


lines = [f'# Fairlearn cross-check of the subgroup panel', '',
         f'Fairlearn {fairlearn.__version__}. Cohort: {res["cohort"]} (simulated), Option A split (seed 42): '
         f'{res["nTrain"]} training and {res["nTest"]} test patients; positive class "{res["positiveClass"]}"; threshold 0.5.', '',
         'Every subgroup figure of the app (src/subgroups.js) was recomputed with Fairlearn MetricFrame and scikit-learn.', '',
         '| Model | Subgroup column | Groups | Figures compared | Largest difference (AUC) | Largest difference (all other figures) | Result |', '|---|---|---|---|---|---|---|']
disparity = ['', '## Fairlearn disparity measures (test patients, threshold 0.5)', '',
             'Demographic parity difference: the largest difference between groups in the share predicted as responders. '
             'Equalised odds difference: the larger of the largest between-group differences in sensitivity and in false-positive rate. '
             '0 means no difference; the groups are small, so these figures are imprecise.', '',
             '| Model | Column | Demographic parity difference | Equalised odds difference |', '|---|---|---|---|']
failures = 0
for mid, m in res['models'].items():
    y = np.array(m['yTrue']); p = np.array(m['prob'])
    for col, labels in m['groups'].items():
        mf = MetricFrame(metrics=metrics(m['threshold']), y_true=y, y_pred=p, sensitive_features=np.array(labels))
        app = m['appSubgroupMetrics'][col]
        worst, worst_auc, compared, bad = 0.0, 0.0, 0, []
        for g, row in mf.by_group.iterrows():
            a = app[str(g)]
            for k in ['n', 'events', 'observed', 'meanPred', 'oe', 'auc', 'brier', 'sens', 'spec', 'netBenefit']:
                ref, val = row[k], a.get(k)
                if (ref is None or (isinstance(ref, float) and np.isnan(ref))) and val is None:
                    continue
                d = abs(float(ref) - float(val)); compared += 1
                if d > (TOL_AUC if k == 'auc' else TOL): bad.append(f'{g} {k}: app {val}, Fairlearn {ref}')
                if k != 'auc': worst = max(worst, d)
                else: worst_auc = max(worst_auc, d)
        ok = not bad
        if bad: print(f'{mid} {col}: ' + '; '.join(bad))
        failures += 0 if ok else 1
        lines.append(f'| {mid} | {col} | {len(mf.by_group)} | {compared} | {worst_auc:.1e} | {worst:.1e} | {"PASS" if ok else "FAIL"} |')
        hard = (p >= m['threshold']).astype(int)
        disparity.append(f'| {mid} | {col} | {demographic_parity_difference(y, hard, sensitive_features=labels):.3f} | '
                         f'{equalized_odds_difference(y, hard, sensitive_features=labels):.3f} |')

lines.append('')
lines.append(f'**{"All figures agree" if not failures else f"{failures} comparison(s) disagree"}** (tolerance {TOL:g}; AUC 5e-5, because the app rounds it to 4 decimal places).')
report = '\n'.join(lines + disparity) + '\n'
open(os.path.join(HERE, 'AUDIT_FAIRLEARN.md'), 'w').write(report)
print(report)
sys.exit(1 if failures else 0)
