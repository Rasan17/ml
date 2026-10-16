"""
Benchmark of the app's engine against established Python libraries.

Compares the app (data-engine.js + models.js, real TensorFlow.js neural network on the CPU backend, run
in Node by benchmark_app_runner.js) with scikit-learn, PyTorch and imbalanced-learn on public datasets,
using identical seeded train/test splits, and writes tests/BENCHMARK_REPORT.md.

Usage (from the project folder; needs Node.js and Python with numpy, scipy, scikit-learn, statsmodels,
torch and imbalanced-learn):
    python3 tests/benchmark_vs_python.py

Acceptance criteria are fixed in CRITERIA below (set before running):
  * Deterministic components (median imputation + z-scoring, k-NN, Gaussian naive Bayes) must agree to
    numerical rounding.
  * Stochastic models (Random Forest, neural network, logistic/ridge regression, voting ensemble) are
    compared on paired splits. Verdict: PASS if the 95% CI of the mean paired difference (app - Python)
    lies inside +/- margin; FAIL if the CI excludes 0 and |mean| > margin; otherwise INCONCLUSIVE.
  * SMOTENC: synthetic-data distribution vs imbalanced-learn and downstream AUC on an untouched test set.
  * Calibration (binary task) is reported for information only.
"""
import os
# PyTorch and scikit-learn ship different OpenMP runtimes; on macOS loading both and running
# multi-threaded segfaults inside scikit-learn's neighbour search. Single-threaded is ample here.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import subprocess
import sys
import time
import warnings

import numpy as np
import statsmodels.api as sm
import torch
from imblearn.over_sampling import SMOTENC
from scipy import stats
from sklearn import metrics
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'bench_work')
SEEDS = list(range(1, 11))          # 10 paired train/test splits per dataset
MISSING_RATE = 0.05                 # MCAR blanks injected into predictors to exercise imputation

H = {  # shared hyper-parameters (app and Python)
    'knn_k': 5, 'rf_trees': 100, 'rf_depth': 8,
    'glm_lr': 0.05, 'glm_epochs': 150, 'glm_lambda': 0.01,           # app defaults
    'nn_hidden': [64, 32], 'nn_epochs': 100, 'nn_lr': 0.005, 'nn_batch': 16, 'nn_dropout': 0.1, 'nn_l2': 1e-4,
    'nn_val_fraction': 0.15,
    # Ensemble (app 'balanced' capacity): NN [48,24] x 35 epochs, RF 30 trees depth 6, GLM 150 epochs
    'ens_nn_hidden': [48, 24], 'ens_nn_epochs': 35, 'ens_rf_trees': 30, 'ens_rf_depth': 6,
    # Gradient-boosted trees (app GBM vs scikit-learn HistGradientBoosting, early stopping on 15% of train)
    'gbm_rounds': 200, 'gbm_lr': 0.1, 'gbm_depth': 3, 'gbm_l2': 1.0, 'gbm_bins': 32, 'gbm_min_leaf': 5,
}
CRITERIA = {
    'preprocessing_max_abs_diff': 1e-4,   # app stores z-scores to 5 decimal places
    'knn_agreement': 1.0, 'knn_prob_max_diff': 1e-3, 'knn_reg_max_rel_diff': 1e-3,
    'nb_agreement': 0.99, 'nb_prob_max_diff': 1e-3,
    'margin': {'auc': 0.02, 'accuracy': 0.03, 'macro_f1': 0.03, 'r2': 0.05},
    'smote_ks_max': 0.10, 'smote_tv_max': 0.10, 'smote_auc_margin': 0.02,
}


# ----------------------------------------------------------------------------------------------- data
def load_datasets():
    out = []
    rng = np.random.default_rng(2026)
    for name, loader, task in [('breast_cancer', load_breast_cancer, 'binary'),
                               ('wine', load_wine, 'multiclass'),
                               ('diabetes', load_diabetes, 'regression')]:
        d = loader()
        X = d.data.astype(float).copy()
        X[rng.random(X.shape) < MISSING_RATE] = np.nan
        y = d.target.astype(int) if task != 'regression' else d.target.astype(float)
        features = [f'f{j}' for j in range(X.shape[1])]
        splits = []
        for seed in SEEDS:
            idx = np.arange(len(y))
            tr, te = train_test_split(idx, test_size=0.2, random_state=seed,
                                      stratify=y if task != 'regression' else None)
            splits.append({'seed': seed, 'train': tr.tolist(), 'test': te.tolist()})
        out.append({'name': name, 'task': task, 'features': features, 'X': X, 'y': y, 'splits': splits})
    return out


def smotenc_jobs():
    jobs = []
    for seed in SEEDS[:5]:
        X, y = make_classification(n_samples=1000, n_features=4, n_informative=3, n_redundant=0,
                                   weights=[0.85], class_sep=1.0, random_state=100 + seed)
        rng = np.random.default_rng(seed)
        cat1 = np.where(X[:, 0] + rng.normal(0, 0.7, len(y)) > 0, 'a', np.where(rng.random(len(y)) < 0.5, 'b', 'c'))
        cat2 = np.where(X[:, 1] > 0.5, 'hi', 'lo')
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        jobs.append({'seed': seed, 'X': X, 'cats': np.column_stack([cat1, cat2]), 'y': y, 'train': tr, 'test': te})
    return jobs


# ------------------------------------------------------------------------------------- python models
def torch_mlp(Xtr, ytr, Xte, task, k, seed, hidden, epochs):
    """PyTorch MLP mirroring the app: Glorot init, ReLU, dropout, Adam, L2 on hidden kernels, mini-batches,
    early stopping on a stratified 15% validation split of the training data, best weights restored."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(ytr)
    val_idx, fit_idx = [], []
    if n >= 20:
        if task != 'regression':
            for c in np.unique(ytr):
                ids = rng.permutation(np.where(ytr == c)[0])
                nv = max(1, round(len(ids) * H['nn_val_fraction'])) if len(ids) >= 4 else 0
                val_idx += ids[:nv].tolist()
                fit_idx += ids[nv:].tolist()
        else:
            ids = rng.permutation(n)
            nv = max(1, round(n * H['nn_val_fraction']))
            val_idx, fit_idx = ids[:nv].tolist(), ids[nv:].tolist()
    else:
        fit_idx = list(range(n))
    out_dim = k if task != 'regression' else 1
    layers, prev, hidden_linears = [], Xtr.shape[1], []
    for u in hidden:
        lin = torch.nn.Linear(prev, u)
        hidden_linears.append(lin)
        layers += [lin, torch.nn.ReLU(), torch.nn.Dropout(H['nn_dropout'])]
        prev = u
    layers.append(torch.nn.Linear(prev, out_dim))
    net = torch.nn.Sequential(*layers)
    for mod in net:
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(mod.weight)
            torch.nn.init.zeros_(mod.bias)
    opt = torch.optim.Adam(net.parameters(), lr=H['nn_lr'])
    Xf = torch.tensor(Xtr[fit_idx], dtype=torch.float32)
    Xv = torch.tensor(Xtr[val_idx], dtype=torch.float32) if val_idx else None
    if task != 'regression':
        yf = torch.tensor(ytr[fit_idx], dtype=torch.long)
        yv = torch.tensor(ytr[val_idx], dtype=torch.long) if val_idx else None
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        yf = torch.tensor(ytr[fit_idx], dtype=torch.float32).unsqueeze(1)
        yv = torch.tensor(ytr[val_idx], dtype=torch.float32).unsqueeze(1) if val_idx else None
        loss_fn = torch.nn.MSELoss()
    patience = max(5, round(epochs * 0.2))
    best, best_state, wait = float('inf'), None, 0
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(fit_idx))
        for b in range(0, len(perm), H['nn_batch']):
            bi = perm[b:b + H['nn_batch']]
            opt.zero_grad()
            loss = loss_fn(net(Xf[bi]), yf[bi])
            loss = loss + H['nn_l2'] * sum((lin.weight ** 2).sum() for lin in hidden_linears)  # TF l2(1e-4)
            loss.backward()
            opt.step()
        if Xv is not None:
            net.eval()
            with torch.no_grad():
                vl = loss_fn(net(Xv), yv).item()
            if vl < best - 1e-6:
                best, wait = vl, 0
                best_state = {kk: v.clone() for kk, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        o = net(torch.tensor(Xte, dtype=torch.float32))
        return torch.softmax(o, 1).numpy() if task != 'regression' else o.squeeze(1).numpy()


def glm_python(Xtr, ytr, Xte, task, k):
    n = len(ytr)
    lam = H['glm_lambda']
    if task == 'regression':
        return Ridge(alpha=n * lam).fit(Xtr, ytr).predict(Xte)
    # app: softmax with (lambda/2)||w_c||^2 per class -> binary C = 2/(n*lambda); multinomial C = 1/(n*lambda)
    C = 2 / (n * lam) if k == 2 else 1 / (n * lam)
    return LogisticRegression(C=C, max_iter=10000, tol=1e-10).fit(Xtr, ytr).predict_proba(Xte)


# ---------------------------------------------------------------------------------------------- metrics
def score(task, y, out):
    """out: class-probability matrix (classification) or predictions (regression)."""
    if task == 'regression':
        return {'r2': metrics.r2_score(y, out), 'rmse': float(np.sqrt(metrics.mean_squared_error(y, out)))}
    # The app rounds probabilities to 4 dp, so rows can sum to 0.9999: renormalise (immaterial otherwise)
    out = np.asarray(out, dtype=float)
    sums = out.sum(1, keepdims=True)
    out = np.where(sums > 0, out / np.where(sums > 0, sums, 1), 1.0 / out.shape[1])
    pred = np.argmax(out, 1)
    s = {'accuracy': metrics.accuracy_score(y, pred), 'macro_f1': metrics.f1_score(y, pred, average='macro')}
    if out.shape[1] == 2:
        s['auc'] = metrics.roc_auc_score(y, out[:, 1])
        s['brier'] = metrics.brier_score_loss(y, out[:, 1])
    else:
        s['auc'] = metrics.roc_auc_score(y, out, multi_class='ovr', average='macro')
    return s


def paired_verdict(app_vals, py_vals, margin):
    d = np.array(app_vals) - np.array(py_vals)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    t = stats.t.ppf(0.975, len(d) - 1) if len(d) > 1 else 0.0
    lo, hi = m - t * se, m + t * se
    if lo >= -margin and hi <= margin:
        v = 'PASS'
    elif (lo > 0 or hi < 0) and abs(m) > margin:
        v = 'FAIL'
    else:
        v = 'INCONCLUSIVE'
    return m, lo, hi, v


def app_probs(app_model, k):
    return np.array([o['probs'] if o['probs'] is not None else np.eye(k)[o['pred']] for o in app_model])


# ------------------------------------------------------------------------------------------------- main
def main():
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()
    datasets = load_datasets()
    sjobs = smotenc_jobs()
    payload = {
        'hyperparams': H,
        'datasets': [{'name': d['name'], 'task': d['task'], 'features': d['features'],
                      'rows': [[None if np.isnan(v) else float(v) for v in row] for row in d['X']],
                      'y': d['y'].tolist(), 'splits': d['splits']} for d in datasets],
        'smotenc': [{'seed': j['seed'], 'continuous': ['c0', 'c1', 'c2', 'c3'], 'nominal': ['n0', 'n1'],
                     'trainRows': [[float(v) for v in j['X'][i]] + [str(c) for c in j['cats'][i]] + [str(int(j['y'][i]))]
                                   for i in j['train']]} for j in sjobs],
    }
    inp, outp = os.path.join(WORK, 'input.json'), os.path.join(WORK, 'app_output.json')
    with open(inp, 'w') as f:
        json.dump(payload, f)
    if '--reuse-app-output' in sys.argv and os.path.exists(outp):
        print('Reusing previous app output (--reuse-app-output)')
    else:
        print('Running the app engine in Node (real TF.js, CPU backend)...')
        subprocess.run(['node', os.path.join(HERE, 'benchmark_app_runner.js'), inp, outp], check=True)
    app = json.load(open(outp))

    checks = []      # (section, name, value, criterion, verdict)
    tables = {}      # dataset -> list of rows for the stochastic comparison table
    calib = {}       # model -> (y, p_app, p_py)

    for d in datasets:
        task, name = d['task'], d['name']
        k = len(np.unique(d['y'])) if task != 'regression' else 1
        runs = app['datasets'][name]
        pre_diff, knn_stats, nb_stats = 0.0, {}, {'agree': [], 'pdiff': 0.0}
        stoch = {m: {} for m in ['rf', 'nn', 'glm', 'ensemble', 'gbm']}
        fallback_used = False
        for r, sp in zip(runs, d['splits']):
            tr, te = np.array(sp['train']), np.array(sp['test'])
            Xtr_raw, Xte_raw = d['X'][tr], d['X'][te]
            ytr, yte = d['y'][tr], d['y'][te]
            # ---- preprocessing: SimpleImputer(median) + StandardScaler fitted on train
            imp = SimpleImputer(strategy='median').fit(Xtr_raw)
            sc = StandardScaler().fit(imp.transform(Xtr_raw))
            Xtr, Xte = sc.transform(imp.transform(Xtr_raw)), sc.transform(imp.transform(Xte_raw))
            aXtr, aXte = np.array(r['trainX']), np.array(r['testX'])
            pre_diff = max(pre_diff, np.abs(aXtr - Xtr).max(), np.abs(aXte - Xte).max())
            if task != 'regression':
                # app classes are sorted strings; map back to the original integer labels
                cls = [int(c) for c in r['classes']]
                assert cls == sorted(cls)
            yte_app = np.array(r['testY'])
            fallback_used |= r['nnInfo']['usedFallback']
            # ---- k-NN (exact) on the app's own encoded matrix, isolating the model
            for metric in ['euclidean', 'manhattan', 'chebyshev']:
                for w in ['uniform', 'distance']:
                    key = f'knn_{metric}_{w}'
                    Est = KNeighborsClassifier if task != 'regression' else KNeighborsRegressor
                    m = Est(n_neighbors=H['knn_k'], metric=metric, weights=w).fit(aXtr, np.array(r['trainY']))
                    st = knn_stats.setdefault(key, {'agree': [], 'pdiff': 0.0, 'rel': 0.0, 'tied': 0, 'n': 0})
                    # Rows whose k-th and (k+1)-th neighbour distances are exactly tied have an
                    # implementation-defined neighbour set (scikit-learn: tree order; app: seeded tie key),
                    # so they are excluded from the exact comparison and counted
                    Dm = (np.abs(aXte[:, None, :] - aXtr[None, :, :]).max(2) if metric == 'chebyshev' else
                          np.abs(aXte[:, None, :] - aXtr[None, :, :]).sum(2) if metric == 'manhattan' else
                          np.sqrt(((aXte[:, None, :] - aXtr[None, :, :]) ** 2).sum(2)))
                    Ds = np.sort(Dm, 1)
                    kk = H['knn_k']
                    untied = ~np.isclose(Ds[:, kk - 1], Ds[:, kk], rtol=0, atol=1e-9)
                    st['tied'] += int((~untied).sum())
                    st['n'] += len(untied)
                    if task != 'regression':
                        pp = m.predict_proba(aXte)[untied]
                        ap = app_probs(r['models'][key], k)[untied]
                        apred = np.array([o['pred'] for o in r['models'][key]])[untied]
                        st['agree'].append(np.mean(np.argmax(pp, 1) == apred) if untied.any() else 1.0)
                        st['pdiff'] = max(st['pdiff'], np.abs(pp - ap).max() if untied.any() else 0.0)
                    else:
                        pv = m.predict(aXte)[untied]
                        av = np.array([o['pred'] for o in r['models'][key]])[untied]
                        if untied.any():
                            st['rel'] = max(st['rel'], (np.abs(pv - av) / np.maximum(np.abs(pv), 1e-9)).max())
            # ---- Gaussian naive Bayes (exact on continuous features)
            if task != 'regression':
                g = GaussianNB(var_smoothing=1e-9).fit(aXtr, np.array(r['trainY']))
                gp = g.predict_proba(aXte)
                ap = app_probs(r['models']['nb'], k)
                nb_stats['agree'].append(np.mean(np.argmax(gp, 1) == np.argmax(ap, 1)))
                nb_stats['pdiff'] = max(nb_stats['pdiff'], np.abs(gp - ap).max())
            # ---- stochastic models: app vs Python, each with its own (verified-identical) preprocessing
            seed = sp['seed']
            py = {}
            if task != 'regression':
                py['rf'] = RandomForestClassifier(n_estimators=H['rf_trees'], max_depth=H['rf_depth'],
                                                  max_features='sqrt', random_state=seed).fit(Xtr, ytr).predict_proba(Xte)
            else:
                py['rf'] = RandomForestRegressor(n_estimators=H['rf_trees'], max_depth=H['rf_depth'],
                                                 max_features=1/3, random_state=seed).fit(Xtr, ytr).predict(Xte)
            py['nn'] = torch_mlp(Xtr, ytr, Xte, task, k, seed, H['nn_hidden'], H['nn_epochs'])
            py['glm'] = glm_python(Xtr, ytr, Xte, task, k)
            # Voting ensemble counterpart: equal-weight average of an MLP, a RF and the GLM (app 'balanced')
            e_nn = torch_mlp(Xtr, ytr, Xte, task, k, seed + 1007, H['ens_nn_hidden'], H['ens_nn_epochs'])
            if task != 'regression':
                e_rf = RandomForestClassifier(n_estimators=H['ens_rf_trees'], max_depth=H['ens_rf_depth'],
                                              max_features='sqrt', random_state=seed).fit(Xtr, ytr).predict_proba(Xte)
            else:
                e_rf = RandomForestRegressor(n_estimators=H['ens_rf_trees'], max_depth=H['ens_rf_depth'],
                                             max_features=1/3, random_state=seed).fit(Xtr, ytr).predict(Xte)
            py['ensemble'] = (e_nn + e_rf + py['glm']) / 3
            gb_args = dict(max_iter=H['gbm_rounds'], learning_rate=H['gbm_lr'], max_depth=H['gbm_depth'], max_leaf_nodes=None,
                           min_samples_leaf=H['gbm_min_leaf'], l2_regularization=H['gbm_l2'], max_bins=H['gbm_bins'],
                           early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=seed)
            py['gbm'] = (HistGradientBoostingClassifier(**gb_args).fit(Xtr, ytr).predict_proba(Xte) if task != 'regression'
                         else HistGradientBoostingRegressor(**gb_args).fit(Xtr, ytr).predict(Xte))
            for mname in stoch:
                if task != 'regression':
                    a_out = app_probs(r['models'][mname], k)
                else:
                    a_out = np.array([o['pred'] for o in r['models'][mname]])
                sa, sp_ = score(task, yte_app, a_out), score(task, yte, py[mname])
                for met in sa:
                    stoch[mname].setdefault(met, ([], []))
                    stoch[mname][met][0].append(sa[met])
                    stoch[mname][met][1].append(sp_[met])
                if task == 'binary':
                    c = calib.setdefault(mname, ([], [], []))
                    c[0].extend(yte.tolist()); c[1].extend(a_out[:, 1].tolist()); c[2].extend(py[mname][:, 1].tolist())

        checks.append(('Preprocessing', f'{name}: median imputation + z-scoring vs SimpleImputer + StandardScaler (max |diff|)',
                       f'{pre_diff:.2e}', f'<= {CRITERIA["preprocessing_max_abs_diff"]:.0e}',
                       'PASS' if pre_diff <= CRITERIA['preprocessing_max_abs_diff'] else 'FAIL'))
        for key, st in knn_stats.items():
            if task != 'regression':
                agree = float(np.mean(st['agree']))
                ok = agree >= CRITERIA['knn_agreement'] and st['pdiff'] <= CRITERIA['knn_prob_max_diff']
                checks.append(('k-NN (exact)', f'{name}: {key} vs scikit-learn', f'agreement {agree:.1%}, max prob diff {st["pdiff"]:.1e} '
                               f'({st["tied"]}/{st["n"]} rows with a tied k-th neighbour excluded)',
                               f'100%, <= {CRITERIA["knn_prob_max_diff"]:.0e}', 'PASS' if ok else 'FAIL'))
            else:
                checks.append(('k-NN (exact)', f'{name}: {key} vs scikit-learn', f'max rel diff {st["rel"]:.1e} '
                               f'({st["tied"]}/{st["n"]} rows with a tied k-th neighbour excluded)',
                               f'<= {CRITERIA["knn_reg_max_rel_diff"]:.0e}', 'PASS' if st['rel'] <= CRITERIA['knn_reg_max_rel_diff'] else 'FAIL'))
        if task != 'regression':
            agree = float(np.mean(nb_stats['agree']))
            ok = agree >= CRITERIA['nb_agreement'] and nb_stats['pdiff'] <= CRITERIA['nb_prob_max_diff']
            checks.append(('Naive Bayes (exact)', f'{name}: vs scikit-learn GaussianNB', f'agreement {agree:.1%}, max prob diff {nb_stats["pdiff"]:.1e}',
                           f'>= 99%, <= {CRITERIA["nb_prob_max_diff"]:.0e}', 'PASS' if ok else 'FAIL'))
        checks.append(('Neural network', f'{name}: app trained a real TF.js network (no linear fallback)',
                       'fallback used' if fallback_used else 'no fallback', 'no fallback', 'FAIL' if fallback_used else 'PASS'))
        rows = []
        for mname, mets in stoch.items():
            for met, (av, pv) in mets.items():
                if met not in CRITERIA['margin']:
                    continue
                m, lo, hi, v = paired_verdict(av, pv, CRITERIA['margin'][met])
                rows.append((mname, met, np.mean(av), np.std(av, ddof=1), np.mean(pv), np.std(pv, ddof=1), m, lo, hi, v))
        tables[name] = rows

    # ---- SMOTENC vs imbalanced-learn
    smote_rows = []
    app_syn_all, py_syn_all = [], []
    auc_app, auc_py = [], []
    bounds_ok = True
    for j, a in zip(sjobs, app['smotenc']):
        Xc, cats, y = j['X'], j['cats'], j['y']
        tr, te = j['train'], j['test']
        Xtr_mixed = np.column_stack([Xc[tr].astype(object), cats[tr]])
        sm_ = SMOTENC(categorical_features=[4, 5], random_state=j['seed'], k_neighbors=5)
        Xres, yres = sm_.fit_resample(Xtr_mixed, y[tr])
        py_syn = Xres[len(tr):]
        app_syn = np.array(a['synthetic'], dtype=object)
        n_min = int(min(np.bincount(y[tr])))
        n_maj = int(max(np.bincount(y[tr])))
        smote_rows.append((j['seed'], len(app_syn), len(py_syn), n_maj - n_min))
        app_syn_all.append(app_syn)
        py_syn_all.append(py_syn)
        mins = Xc[tr][y[tr] == 1].min(0)
        maxs = Xc[tr][y[tr] == 1].max(0)
        for syn in (app_syn, py_syn):
            v = syn[:, :4].astype(float)
            bounds_ok &= bool(np.all(v >= mins - 1e-9) and np.all(v <= maxs + 1e-9))

        def enc(Xm):
            num = Xm[:, :4].astype(float)
            c1 = np.column_stack([(Xm[:, 4] == v).astype(float) for v in ['a', 'b', 'c']])
            c2 = (Xm[:, 5] == 'hi').astype(float)[:, None]
            return np.hstack([num, c1, c2])
        app_train = np.vstack([Xtr_mixed, app_syn[:, :6]])
        app_y = np.concatenate([y[tr], np.array([int(t) for t in app_syn[:, 6]])])
        Xte_m = np.column_stack([Xc[te].astype(object), cats[te]])
        for Xb, yb, store in ((app_train, app_y, auc_app), (Xres, yres, auc_py)):
            rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=j['seed']).fit(enc(Xb), yb)
            store.append(metrics.roc_auc_score(y[te], rf.predict_proba(enc(Xte_m))[:, 1]))
    A = np.vstack(app_syn_all)
    P = np.vstack(py_syn_all)
    counts_ok = all(r[1] == r[2] == r[3] for r in smote_rows)
    checks.append(('SMOTENC', 'Synthetic count = majority − minority (app and imbalanced-learn)',
                   ', '.join(f'{r[1]}/{r[2]}' for r in smote_rows), 'equal', 'PASS' if counts_ok else 'FAIL'))
    checks.append(('SMOTENC', 'Synthetic continuous values within the minority training range (both)',
                   'yes' if bounds_ok else 'no', 'yes', 'PASS' if bounds_ok else 'FAIL'))
    for c in range(4):
        D = stats.ks_2samp(A[:, c].astype(float), P[:, c].astype(float)).statistic
        checks.append(('SMOTENC', f'Continuous feature c{c}: KS distance app vs imbalanced-learn synthetic', f'{D:.3f}',
                       f'<= {CRITERIA["smote_ks_max"]}', 'PASS' if D <= CRITERIA['smote_ks_max'] else 'FAIL'))
    for c, levels in ((4, ['a', 'b', 'c']), (5, ['hi', 'lo'])):
        pa = np.array([np.mean(A[:, c] == v) for v in levels])
        pp = np.array([np.mean(P[:, c] == v) for v in levels])
        tv = 0.5 * np.abs(pa - pp).sum()
        checks.append(('SMOTENC', f'Nominal feature n{c - 4}: total-variation distance', f'{tv:.3f}',
                       f'<= {CRITERIA["smote_tv_max"]}', 'PASS' if tv <= CRITERIA['smote_tv_max'] else 'FAIL'))
    m, lo, hi, v = paired_verdict(auc_app, auc_py, CRITERIA['smote_auc_margin'])
    checks.append(('SMOTENC', 'Downstream RF test AUC: app-balanced vs imblearn-balanced training data (paired, untouched test set)',
                   f'{np.mean(auc_app):.3f} vs {np.mean(auc_py):.3f}; diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}]',
                   f'95% CI within ±{CRITERIA["smote_auc_margin"]}', v))

    # ---- calibration (binary task, pooled over splits; informational)
    cal_rows = []
    for mname, (yy, pa, pp) in calib.items():
        yy, pa, pp = np.array(yy), np.clip(np.array(pa), 1e-6, 1 - 1e-6), np.clip(np.array(pp), 1e-6, 1 - 1e-6)
        row = [mname]
        for p in (pa, pp):
            lp = np.log(p / (1 - p))
            try:
                fit = sm.Logit(yy, sm.add_constant(lp)).fit(disp=0)
                slope = fit.params[1]
                citl = sm.GLM(yy, np.ones((len(yy), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
            except Exception:
                slope, citl = float('nan'), float('nan')
            row += [metrics.brier_score_loss(yy, p), citl, slope]
        cal_rows.append(row)

    write_report(datasets, app, checks, tables, cal_rows, time.time() - t0)
    n_fail = sum(1 for c in checks if c[4] == 'FAIL') + sum(1 for rows in tables.values() for r in rows if r[9] == 'FAIL')
    print(f'\nReport written to tests/BENCHMARK_REPORT.md  ({n_fail} FAIL)')
    sys.exit(1 if n_fail else 0)


def write_report(datasets, app, checks, tables, cal_rows, secs):
    L = []
    L.append('# Benchmark: app engine vs Python reference libraries\n')
    L.append(f'Generated by `tests/benchmark_vs_python.py` on {time.strftime("%Y-%m-%d %H:%M")} (runtime {secs:.0f} s). '
             f'App engine: `data-engine.js` + `models.js` with TensorFlow.js {app["tfjs"]} ({app["backend"]} backend). '
             'References: scikit-learn, PyTorch, imbalanced-learn, SciPy, statsmodels.\n')
    L.append('## Design\n')
    L.append(f'- Datasets (scikit-learn): breast cancer (binary, 569×30), wine (3-class, 178×13), diabetes (regression, 442×10). '
             f'{MISSING_RATE:.0%} of predictor values blanked at random (seeded) to exercise imputation.')
    L.append(f'- {len(SEEDS)} seeded 80/20 train/test splits per dataset (stratified for classification), identical for app and Python.')
    L.append('- App side follows Option A: median imputation and z-scoring fitted on the training split, applied to the test split.')
    L.append('- Python side: `SimpleImputer(median)` + `StandardScaler` fitted on the training split; k-NN, RandomForest, '
             'LogisticRegression/Ridge (equivalent penalty), GaussianNB from scikit-learn; a PyTorch MLP mirroring the app '
             '(Glorot init, ReLU, dropout, Adam, L2 on hidden kernels, mini-batches, early stopping on a stratified 15% '
             'validation split of the training data with best-weight restoration); SMOTENC from imbalanced-learn.')
    L.append('- Hyper-parameters (shared): ' + ', '.join(f'`{k}`={v}' for k, v in H.items()) + '.')
    L.append('- Metrics for both sides are computed by scikit-learn from the raw predictions.')
    L.append('- **Verdict rule for stochastic models** (fixed before running): paired difference app − Python over the splits; '
             f'PASS if the 95% CI lies within ±margin (AUC {CRITERIA["margin"]["auc"]}, accuracy {CRITERIA["margin"]["accuracy"]}, '
             f'macro-F1 {CRITERIA["margin"]["macro_f1"]}, R² {CRITERIA["margin"]["r2"]}); FAIL if the CI excludes 0 and '
             '|mean difference| exceeds the margin; otherwise INCONCLUSIVE (not shown to be equivalent at this sample size).\n')
    L.append('## Deterministic and component checks\n')
    L.append('| Section | Check | Result | Criterion | Verdict |')
    L.append('|---|---|---|---|---|')
    for c in checks:
        L.append(f'| {c[0]} | {c[1]} | {c[2]} | {c[3]} | **{c[4]}** |')
    L.append('\n## Stochastic models: app vs Python on paired splits\n')
    names = {'gbm': 'Gradient-boosted trees', 'rf': 'Random Forest', 'nn': 'Neural network (MLP)', 'glm': 'Logistic / ridge regression',
             'ensemble': 'Voting ensemble (soft)'}
    for ds, rows in tables.items():
        L.append(f'### {ds}\n')
        L.append('| Model | Metric | App mean (SD) | Python mean (SD) | Mean diff (app − Py) | 95% CI | Verdict |')
        L.append('|---|---|---|---|---|---|---|')
        for r in rows:
            L.append(f'| {names[r[0]]} | {r[1]} | {r[2]:.3f} ({r[3]:.3f}) | {r[4]:.3f} ({r[5]:.3f}) | {r[6]:+.3f} | '
                     f'[{r[7]:+.3f}, {r[8]:+.3f}] | **{r[9]}** |')
        L.append('')
    L.append('## Calibration on the binary task (informational; pooled test predictions over all splits)\n')
    L.append('Calibration-in-the-large: ideal 0. Calibration slope: ideal 1 (< 1 = over-confident, > 1 = under-confident).\n')
    L.append('| Model | App Brier | App CITL | App slope | Python Brier | Python CITL | Python slope |')
    L.append('|---|---|---|---|---|---|---|')
    for r in cal_rows:
        L.append(f'| {names[r[0]]} | {r[1]:.3f} | {r[2]:+.2f} | {r[3]:.2f} | {r[4]:.3f} | {r[5]:+.2f} | {r[6]:.2f} |')
    L.append('\n## Interpretation notes\n')
    L.append('- Random Forest and neural network results cannot match prediction-by-prediction: the libraries use different '
             'random-number streams, and the app\'s trees search ~10 candidate thresholds per feature rather than all of them. '
             'Agreement is therefore judged on performance across paired splits.')
    L.append('- The app\'s logistic/ridge model uses a fixed number of gradient-descent iterations (default 150), whereas '
             'scikit-learn solves to convergence; differences in that row reflect the iteration budget, not the formula '
             '(the formula itself is verified exactly in `tests/reference_check.py`).')
    L.append('- k-NN with distance weighting: the app uses 1/(d + 1e-5) rather than 1/d, so probabilities may differ in '
             'the 5th decimal; class predictions are expected to be identical.')
    L.append('- k-NN rows whose k-th and (k+1)-th neighbours are at exactly the same distance are excluded from the exact '
             'comparison: which tied case is "k-th" is implementation-defined (scikit-learn follows its tree traversal; '
             'the app uses a seeded tie key). Exact ties are common with the Chebyshev metric and with imputed values.')
    L.append('- Calibration: both the app and the Python references show the same pattern (logistic/ridge, Random Forest '
             'and the ensemble under-confident, slope > 1; the MLP over-confident, slope < 1), i.e. it is a property of the '
             'model types and settings, not of the implementation. Probabilities should not be read as calibrated risks.')
    with open(os.path.join(HERE, 'BENCHMARK_REPORT.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
