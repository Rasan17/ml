"""Generates the simulated idiopathic normal-pressure hydrocephalus (iNPH) shunt-response cohorts.

medical_shunt_cohort_large.csv        420 patients from three hospitals (development cohort)
medical_shunt_validation_cohort.csv   150 patients from a fourth hospital (external validation, Option B)
medical_shunt_cohort.csv              40-patient mini demo (one hospital): the same model, to show how unstable
                                      and imprecise results are with a small cohort
medical_batch_unseen_cases.csv        10 further patients from the same model, for batch prediction (Step 6)
medical_shunt_cohort_leakage.csv      teaching example of leakage: the 420-patient cohort plus one column measured
                                      after shunting (gait improvement at 6 months), which largely defines response

The patients are simulated, not real. Distributions and effect sizes are chosen to be clinically plausible
(about two thirds respond to shunting; a positive CSF tap test, DESH, a narrow callosal angle, high
outflow resistance and shorter symptom duration favour response), so that models reach realistic
discrimination (AUC around 0.75), never the near-perfect separation that real clinical predictors do not give.
Missing values follow common practice: infusion-test results (ICP, Rout, PVI) are missing together when
no infusion study was done, and a few other values are unrecorded.

Usage: python3 tools/make_shunt_cohort.py   (reproducible: fixed seeds)
"""
import csv
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
COLS = ['Patient_ID', 'Site', 'Age', 'Sex', 'Symptom_Duration_Months', 'Gait_Score_Pre', 'Cognitive_MMSE_Pre',
        'Incontinence_Score_Pre', 'Evans_Index', 'Callosal_Angle_Deg', 'DESH_Sign', 'Baseline_ICP_mmHg',
        'Rout_mmHg_min_ml', 'PVI_ml', 'Tap_Test_Response', 'Shunt_Response']


def simulate(n, seed, sites, prefix, shift=None):
    shift = shift or {}
    rng = np.random.default_rng(seed)
    site = rng.choice(sites, size=n, p=[1 / len(sites)] * len(sites))
    age = np.clip(rng.normal(75 + shift.get('age', 0), 6, n), 60, 92)
    male = rng.random(n) < 0.56
    duration = np.clip(np.exp(rng.normal(np.log(18) + shift.get('log_duration', 0), 0.6, n)), 3, 120)
    # DESH and the callosal angle are correlated (both reflect the iNPH pattern of CSF distribution)
    desh = rng.random(n) < (0.55 + shift.get('desh', 0))
    callosal = np.clip(rng.normal(np.where(desh, 78, 98), 14), 45, 140)
    evans = np.clip(rng.normal(np.where(desh, 0.37, 0.34), 0.025), 0.30, 0.46)
    gait = np.clip(np.round(rng.normal(2.6, 0.9, n)), 0, 4)
    mmse = np.clip(np.round(rng.normal(23.5 - 0.08 * (age - 75) - 0.04 * (duration - 18), 3.5)), 8, 30)
    incont = np.clip(np.round(rng.normal(1.5 + 0.01 * (duration - 18), 0.9)), 0, 3)
    rout = np.clip(rng.normal(np.where(desh, 15.5, 12.0), 4.5), 4, 35)
    icp = np.clip(rng.normal(11 + 0.15 * (rout - 13), 3.0), 3, 22)
    pvi = np.clip(rng.normal(19 - 0.25 * (rout - 13), 5.5), 6, 40)

    # True response model (log-odds); about two thirds respond
    lp = (0.55
          + 0.75 * desh
          - 0.022 * (callosal - 88)
          + 0.07 * (rout - 13)
          - 0.55 * np.log(duration / 18)
          + 0.30 * (gait - 2.6)
          + 0.05 * (mmse - 23)
          - 0.025 * (age - 75)
          + shift.get('intercept', 0))
    respond = rng.random(n) < 1 / (1 + np.exp(-lp))
    # The CSF tap test is informative but imperfect (positive in ~65% of responders, ~25% of non-responders)
    tap = rng.random(n) < np.where(respond, 0.65, 0.25)

    # Missing values: no infusion study in ~12% (ICP, Rout and PVI missing together); a few others
    no_infusion = rng.random(n) < 0.12
    miss = {'Symptom_Duration_Months': rng.random(n) < 0.05, 'Cognitive_MMSE_Pre': rng.random(n) < 0.04,
            'Callosal_Angle_Deg': rng.random(n) < 0.03, 'PVI_ml': no_infusion | (rng.random(n) < 0.04),
            'Baseline_ICP_mmHg': no_infusion, 'Rout_mmHg_min_ml': no_infusion}

    rows = []
    for i in range(n):
        r = {
            'Patient_ID': f'{prefix}-{i + 1:03d}', 'Site': site[i], 'Age': int(round(age[i])), 'Sex': 'Male' if male[i] else 'Female',
            'Symptom_Duration_Months': int(round(duration[i])), 'Gait_Score_Pre': int(gait[i]), 'Cognitive_MMSE_Pre': int(mmse[i]),
            'Incontinence_Score_Pre': int(incont[i]), 'Evans_Index': f'{evans[i]:.2f}', 'Callosal_Angle_Deg': int(round(callosal[i])),
            'DESH_Sign': 'Present' if desh[i] else 'Absent', 'Baseline_ICP_mmHg': f'{icp[i]:.1f}', 'Rout_mmHg_min_ml': f'{rout[i]:.1f}',
            'PVI_ml': f'{pvi[i]:.1f}', 'Tap_Test_Response': 'Positive' if tap[i] else 'Negative',
            'Shunt_Response': 'Responder' if respond[i] else 'Non-Responder'
        }
        for c, m in miss.items():
            if m[i]:
                r[c] = ''
        rows.append(r)
    return rows, lp, respond


def write(name, rows, cols=COLS):
    with open(os.path.join(ROOT, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.sum()
    return (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


if __name__ == '__main__':
    dev, lp_d, y_d = simulate(420, 2026, ['Hospital A', 'Hospital B', 'Hospital C'], 'NPH')
    val, lp_v, y_v = simulate(150, 4711, ['Hospital D'], 'NPHV', shift={'age': 2, 'log_duration': 0.15, 'desh': -0.05, 'intercept': -0.1})
    write('medical_shunt_cohort_large.csv', dev)
    write('medical_shunt_validation_cohort.csv', val)
    # Mini demo and its batch cases: one hospital, no Site column
    no_site = [c for c in COLS if c != 'Site']
    mini, lp_m, y_m = simulate(40, 404, ['Hospital A'], 'NPH')
    batch, lp_b, y_b = simulate(10, 1010, ['Hospital A'], 'NEW')
    # (blank cells are kept: the app fills them with the model's own missing-data method at prediction time)
    write('medical_shunt_cohort.csv', mini, no_site)
    write('medical_batch_unseen_cases.csv', batch, no_site)
    # Leakage example: gait improvement 6 months after shunting is how response is usually judged, so it is
    # (almost) a copy of the outcome and is not known when the decision to shunt is made. Responders improve by at
    # least 1 point (that is how response is judged); a few non-responders also improve by 1 point
    rng = np.random.default_rng(99)
    leak_cols = COLS[:-1] + ['Gait_Improvement_6mo_Points', 'Shunt_Response']
    leaky = []
    for r, y in zip(dev, y_d):
        imp = rng.normal(1.8, 0.6) if y else rng.normal(-0.3, 0.4)
        leaky.append({**r, 'Gait_Improvement_6mo_Points': int(np.clip(np.round(imp), -1 if not y else 1, 3 if y else 1))})
    write('medical_shunt_cohort_leakage.csv', leaky, leak_cols)
    for name, lp, y in [('development', lp_d, y_d), ('validation', lp_v, y_v), ('mini demo', lp_m, y_m)]:
        print(f'{name}: {len(y)} patients, {y.mean():.0%} responders, AUC of the true risk score {auc(lp, y):.3f}')
