"""Independent data and model checks with Deepchecks (open-source tabular suites).

Uses the outputs of tests/audit_app_runner.js: the app's own Option A split of the simulated iNPH cohort and
its trained models (called through tests/audit_predict.js, so Deepchecks scores the app's real preprocessing
and models). Runs Deepchecks' data-integrity, train-test-validation and model-evaluation suites, and a
repeated-patient demonstration (every patient twice, split by row and by patient).

Usage (Deepchecks needs its own environment with older libraries):
    python3 -m venv .venv-deepchecks
    .venv-deepchecks/bin/pip install "setuptools<81" deepchecks==0.19.1 scikit-learn==1.5.2 category_encoders==2.6.4 pandas==2.2.3 numpy==1.26.4
    node tests/audit_app_runner.js /tmp/ummp-audit
    .venv-deepchecks/bin/python tests/audit_deepchecks.py /tmp/ummp-audit
Writes tests/AUDIT_DEEPCHECKS.md and the full HTML reports into the audit folder.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import deepchecks
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestSamplesMix
from deepchecks.tabular.suites import data_integrity, model_evaluation, train_test_validation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
src = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ummp-audit'
res = json.load(open(os.path.join(src, 'app_results.json')))
LABEL = res['target']
NOT_FEATURES = ['Patient_ID', 'Visit']
CAT = ['Site', 'Sex', 'DESH_Sign', 'Tap_Test_Response']


def load(name):
    return pd.read_csv(os.path.join(src, name), keep_default_na=True)


def dataset(df):
    feats = [c for c in df.columns if c not in NOT_FEATURES + [LABEL]]
    return Dataset(df[feats + [LABEL]], label=LABEL, cat_features=[c for c in CAT if c in feats])


class AppModel:
    """The app's trained model (preprocessing + model), called through Node; predictions are cached."""
    def __init__(self, model_id):
        self.path = os.path.join(src, f'model_{model_id}.json')
        self.classes_ = np.array(['Non-Responder', 'Responder'])
        self.cache = {}
        self.calls = 0

    def predict_proba(self, X):
        key = hashlib.sha1(pd.util.hash_pandas_object(X, index=False).values.tobytes()).hexdigest()
        if key not in self.cache:
            rows = [{k: ('' if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)) for k, v in r.items()} for r in X.to_dict('records')]
            with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                json.dump(rows, f)
            out = subprocess.run(['node', os.path.join(HERE, 'audit_predict.js'), self.path, f.name], capture_output=True, text=True, cwd=ROOT)
            os.unlink(f.name)
            p1 = np.array(json.loads(out.stdout.strip().splitlines()[-1]))
            self.cache[key] = np.column_stack([1 - p1, p1])
            self.calls += 1
        return self.cache[key]

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1] >= 0.5, 'Responder', 'Non-Responder')

    def fit(self, X, y):
        # Already trained by the app; present so scikit-learn tools (permutation importance) accept the model
        return self

    def __sklearn_is_fitted__(self):
        return True


def summarise(suite_result, title, html_name):
    suite_result.save_as_html(os.path.join(src, html_name), as_widget=False)
    rows = []
    for r in suite_result.results:
        name = r.get_header() if hasattr(r, 'get_header') else getattr(r, 'header', str(r))
        conds = getattr(r, 'conditions_results', None)
        if conds is None:  # a check that could not run
            rows.append((name, 'not run', str(getattr(r, 'exception', ''))[:140]))
            continue
        if not conds:
            rows.append((name, 'info', 'no condition (displayed in the HTML report)'))
            continue
        for c in conds:
            rows.append((name, c.category.name if hasattr(c.category, 'name') else str(c.category), (c.details or c.name)[:180]))
    md = [f'## {title}', '', f'Full report: `{html_name}` in the audit folder.', '', '| Check | Result | Details |', '|---|---|---|']
    md += [f'| {n} | {s} | {d.replace("|", "/")} |' for n, s, d in rows]
    return md, rows


train, test = load('train.csv'), load('test.csv')
full = pd.concat([train, test], ignore_index=True)
report = [f'# Deepchecks audit', '',
          f'Deepchecks {deepchecks.__version__} (open-source tabular suites). Cohort: {res["cohort"]} (simulated), the app\'s Option A split '
          f'(seed 42): {len(train)} training and {len(test)} test patients. Models are the app\'s own (preprocessing and model, run through Node).', '',
          'Results are PASS, WARN (worth a look) or FAIL against Deepchecks\' default thresholds, which are generic: '
          'a WARN or FAIL is a prompt to look, not necessarily a defect.', '']

md, _ = summarise(data_integrity().run(dataset(full)), '1. Data integrity (whole cohort)', 'deepchecks_data_integrity.html')
report += md + ['']
md, ttv = summarise(train_test_validation().run(dataset(train), dataset(test)), '2. Train-test validation (the app\'s split)', 'deepchecks_train_test.html')
report += md + ['']
for mid in res['models']:
    model = AppModel(mid)
    md, _ = summarise(model_evaluation().run(dataset(train), dataset(test), model), f'3. Model evaluation: {mid}', f'deepchecks_model_{mid}.html')
    report += md + [f'', f'({model.calls} batches of predictions computed by the app\'s engine.)', '']

# Repeated patients: two visits per patient, split by row (a patient can be on both sides) or by patient
mix = {}
for kind in ['row', 'patient']:
    r = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0).run(dataset(load(f'visits_{kind}_train.csv')), dataset(load(f'visits_{kind}_test.csv')))
    c = r.conditions_results[0]
    mix[kind] = (c.category.name, c.details)
report += ['## 4. Repeated patients (every patient twice)', '',
           'Deepchecks\' Train Test Samples Mix looks for test records that also appear in the training set.', '',
           '| Split | Result | Details |', '|---|---|---|',
           f'| By row (a random split ignoring the patient) | {mix["row"][0]} | {mix["row"][1]} |',
           f'| By patient (the app\'s "keep each patient\'s records together") | {mix["patient"][0]} | {mix["patient"][1]} |', '']
open(os.path.join(HERE, 'AUDIT_DEEPCHECKS.md'), 'w').write('\n'.join(report) + '\n')
print('\n'.join(report))
