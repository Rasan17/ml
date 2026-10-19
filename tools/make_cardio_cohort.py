"""Generates the simulated cardiometabolic risk demo cohorts (one-year cardiovascular event, yes/no).

universal_medical_cohort.csv        30 patients (development; deliberately small demo)
universal_medical_test_cohort.csv   8 further patients from the same model (built-in test cohort for Option B)

The patients are simulated, not real. Each predictor has a modest, clinically plausible association with the
event (older age, male sex, higher blood pressure, creatinine and HbA1c, worse heart-failure class, smoking,
ischaemia on the ECG and comorbidity raise risk), so that no single predictor separates patients perfectly and
the true risk score has an AUC of roughly 0.75–0.8, as in real cardiovascular risk models. A few values are
missing, as in practice.

Usage: python3 tools/make_cardio_cohort.py   (reproducible: fixed seeds)
"""
import csv
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
COLS = ['Patient_MRN', 'Age', 'Sex', 'BMI', 'Systolic_BP_mmHg', 'Diastolic_BP_mmHg', 'Serum_Creatinine_mg_dl',
        'HbA1c_Percent', 'NYHA_Heart_Failure_Class', 'Smoking_Status', 'ECG_Ischemia', 'Charlson_Comorbidity_Index',
        'Cardiovascular_Event_1Yr']


def simulate(n, seed, prefix, start, with_missing=True):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(64, 10, n), 40, 90)
    male = rng.random(n) < 0.53
    bmi = np.clip(rng.normal(28, 4, n), 18, 45)
    sbp = np.clip(rng.normal(132 + 0.4 * (age - 64) + 0.8 * (bmi - 28), 15), 95, 200)
    dbp = np.clip(rng.normal(80 + 0.35 * (sbp - 132), 7), 55, 115)
    creat = np.clip(np.exp(rng.normal(np.log(1.0) + 0.01 * (age - 64), 0.25)), 0.5, 3.5)
    hba1c = np.clip(rng.normal(6.3 + 0.05 * (bmi - 28), 1.0), 4.8, 11)
    nyha = np.clip(np.round(rng.normal(1.2 + 0.02 * (age - 64), 1.0)), 0, 4)
    smoke = rng.choice(['Never', 'Former', 'Current'], n, p=[0.45, 0.35, 0.20])
    charlson = np.clip(np.round(rng.normal(1.8 + 0.04 * (age - 64), 1.3)), 0, 8)
    ischaemia = rng.random(n) < np.clip(0.25 + 0.01 * (age - 64) + 0.1 * male, 0.05, 0.8)

    lp = (-1.0 + 0.035 * (age - 64) + 0.30 * male + 0.015 * (sbp - 132) + 0.55 * np.log(creat)
          + 0.20 * (hba1c - 6.3) + 0.30 * (nyha - 1.2) + 0.35 * (smoke == 'Current') + 0.15 * (smoke == 'Former')
          + 0.65 * ischaemia + 0.15 * (charlson - 1.8))
    event = rng.random(n) < 1 / (1 + np.exp(-lp))

    rows = []
    for i in range(n):
        r = {'Patient_MRN': f'{prefix}-{start + i}', 'Age': int(round(age[i])), 'Sex': 'Male' if male[i] else 'Female',
             'BMI': f'{bmi[i]:.1f}', 'Systolic_BP_mmHg': int(round(sbp[i])), 'Diastolic_BP_mmHg': int(round(dbp[i])),
             'Serum_Creatinine_mg_dl': f'{creat[i]:.1f}', 'HbA1c_Percent': f'{hba1c[i]:.1f}', 'NYHA_Heart_Failure_Class': int(nyha[i]),
             'Smoking_Status': smoke[i], 'ECG_Ischemia': 'Present' if ischaemia[i] else 'Absent',
             'Charlson_Comorbidity_Index': int(charlson[i]), 'Cardiovascular_Event_1Yr': 'Yes' if event[i] else 'No'}
        if with_missing:
            for c, p in [('BMI', 0.06), ('Diastolic_BP_mmHg', 0.04), ('Serum_Creatinine_mg_dl', 0.04), ('NYHA_Heart_Failure_Class', 0.04), ('ECG_Ischemia', 0.06)]:
                if rng.random() < p:
                    r[c] = ''
        rows.append(r)
    return rows, lp, event


def write(name, rows):
    with open(os.path.join(ROOT, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.sum()
    return (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


if __name__ == '__main__':
    dev, lp_d, y_d = simulate(30, 3030, 'MED', 1001)
    test, lp_t, y_t = simulate(8, 808, 'TEST-MED', 2001, with_missing=False)
    write('universal_medical_cohort.csv', dev)
    write('universal_medical_test_cohort.csv', test)
    big, lp_b, y_b = simulate(20000, 1, 'X', 0, with_missing=False)
    print(f'development: {len(y_d)} patients, {y_d.mean():.0%} with an event; test: {len(y_t)} patients, {y_t.mean():.0%} with an event')
    print(f'Large sample from the same model: {y_b.mean():.0%} with an event, AUC of the true risk score {auc(lp_b, y_b):.3f}')
