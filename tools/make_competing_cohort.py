"""Generates the simulated shunt-revision cohort with death as a competing event.

medical_shunt_revision_competing.csv   400 patients followed after shunting; outcome Revision, Death (before any
                                       revision) or "Alive, no revision" (censored at the last follow-up)

The patients are simulated, not real. Two cause-specific hazards act at the same time: revision (higher after
haemorrhage or trauma, with a fixed-pressure valve, after previous shunt surgery and with higher BMI) and death
(rising steeply with age). Older patients therefore die before they can need a revision, so treating death as
censoring overstates revision risk, as in real elderly NPH cohorts. Follow-up ends between 12 and 72 months.

Usage: python3 tools/make_competing_cohort.py   (reproducible: fixed seed)
"""
import csv
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
COLS = ['Patient_ID', 'Age', 'Sex', 'BMI', 'Aetiology', 'Valve_Type', 'Prior_Shunt_Surgery', 'Months_Followup', 'Outcome']


def simulate(n, seed):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(72, 9, n), 40, 92)
    male = rng.random(n) < 0.55
    bmi = np.clip(rng.normal(27, 4.5, n), 17, 45)
    aet = rng.choice(['Idiopathic NPH', 'Post-haemorrhagic', 'Post-traumatic'], n, p=[0.6, 0.25, 0.15])
    fixed = rng.random(n) < 0.4
    prior = rng.random(n) < 0.2
    lp_rev = (0.55 * (aet == 'Post-haemorrhagic') + 0.45 * (aet == 'Post-traumatic') + 0.40 * fixed + 0.55 * prior
              + 0.04 * (bmi - 27) - 0.01 * (age - 72))
    lp_death = 0.085 * (age - 72) + 0.25 * male + 0.30 * (aet == 'Post-haemorrhagic')
    t_rev = rng.exponential(1 / (0.011 * np.exp(lp_rev)))
    t_death = rng.exponential(1 / (0.0065 * np.exp(lp_death)))
    t_cens = rng.uniform(12, 72, n)
    t = np.minimum.reduce([t_rev, t_death, t_cens])
    outcome = np.where(t == t_cens, 'Alive, no revision', np.where(t_rev <= t_death, 'Revision', 'Death'))
    rows = []
    for i in range(n):
        r = {'Patient_ID': f'CR{i + 1:04d}', 'Age': int(round(age[i])), 'Sex': 'Male' if male[i] else 'Female', 'BMI': f'{bmi[i]:.1f}',
             'Aetiology': aet[i], 'Valve_Type': 'Fixed pressure' if fixed[i] else 'Programmable', 'Prior_Shunt_Surgery': 'Yes' if prior[i] else 'No',
             'Months_Followup': f'{max(0.1, t[i]):.1f}', 'Outcome': outcome[i]}
        if rng.random() < 0.05:
            r['BMI'] = ''
        rows.append(r)
    return rows


if __name__ == '__main__':
    rows = simulate(400, 2027)
    with open(os.path.join(ROOT, 'medical_shunt_revision_competing.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(Counter(r['Outcome'] for r in rows))
