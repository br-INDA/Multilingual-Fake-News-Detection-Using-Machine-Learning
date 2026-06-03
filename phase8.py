"""
Phase 8 — Other Classifiers & Perceptron
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Labs 7 & 8: Other Classifiers, Perceptron
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.pipeline import Pipeline
import time
warnings.filterwarnings('ignore')

# ── optional XGBoost ─────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost not installed. Run: pip install xgboost")
    print("       Skipping XGBoost — all other models will still run.\n")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
P2_DIR     = "phase2_outputs"
P3_DIR     = "phase3_outputs"
P7_DIR     = "phase7_outputs"
OUTPUT_DIR = "phase8_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES    = ['Hindi', 'Marathi', 'Gujarati', 'Telugu']
RANDOM_STATE = 42
N_JOBS       = 1          # keep 1 to avoid Windows joblib crashes


# ─────────────────────────────────────────────────────────────────
# STEP 1  — Load Artefacts
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 8 — Other Classifiers & Perceptron")
print("=" * 60)

def load_npz(d, n): return sp.load_npz(os.path.join(d, n))
def load_npy(d, n): return np.load(os.path.join(d, n))
def load_csv(d, n): return pd.read_csv(os.path.join(d, n)).squeeze()

# Char TF-IDF (sparse, non-negative → good for NB)
X_train_char = load_npz(P2_DIR, "X_train_tfidf.npz")
X_val_char   = load_npz(P2_DIR, "X_val_tfidf.npz")
X_test_char  = load_npz(P2_DIR, "X_test_tfidf.npz")

# Word TF-IDF (sparse, non-negative)
X_train_word = load_npz(P2_DIR, "X_train_word.npz")
X_val_word   = load_npz(P2_DIR, "X_val_word.npz")
X_test_word  = load_npz(P2_DIR, "X_test_word.npz")

# SVD-50 dense features
X_train_svd  = load_npy(P3_DIR, "X_train_svd.npy")
X_val_svd    = load_npy(P3_DIR, "X_val_svd.npy")
X_test_svd   = load_npy(P3_DIR, "X_test_svd.npy")

# Distance features
dist_train = load_npy(P3_DIR, "dist_train.npy")
dist_val   = load_npy(P3_DIR, "dist_val.npy")
dist_test  = load_npy(P3_DIR, "dist_test.npy")

# Labels & language
y_train    = load_csv(P7_DIR, "y_train.csv").astype(int)
y_val      = load_csv(P7_DIR, "y_val.csv").astype(int)
y_test     = load_csv(P7_DIR, "y_test.csv").astype(int)
lang_train = load_csv(P7_DIR, "lang_train.csv")
lang_val   = load_csv(P7_DIR, "lang_val.csv")
lang_test  = load_csv(P7_DIR, "lang_test.csv")

# L2-normalised TF-IDF (for models that need it)
X_tr_norm = normalize(X_train_char, norm='l2')
X_va_norm = normalize(X_val_char,   norm='l2')
X_te_norm = normalize(X_test_char,  norm='l2')

# MinMax-scaled SVD (MLP & Perceptron prefer [0,1])
scaler      = MinMaxScaler()
X_tr_scaled = scaler.fit_transform(X_train_svd)
X_va_scaled = scaler.transform(X_val_svd)
X_te_scaled = scaler.transform(X_test_svd)

# Combined train+val
X_tv_char   = sp.vstack([X_train_char, X_val_char])
X_tv_word   = sp.vstack([X_train_word, X_val_word])
X_tv_svd    = np.vstack([X_train_svd,  X_val_svd])
X_tv_scaled = np.vstack([X_tr_scaled,  X_va_scaled])
X_tv_norm   = normalize(X_tv_char, norm='l2')
y_tv        = pd.concat([y_train, y_val]).reset_index(drop=True)
lang_tv     = pd.concat([lang_train, lang_val]).reset_index(drop=True)

# MinMax on combined for final training
scaler_tv      = MinMaxScaler()
X_tv_scaled_f  = scaler_tv.fit_transform(X_tv_svd)
X_te_scaled_f  = scaler_tv.transform(X_test_svd)

print(f"\n✓ Artefacts loaded")
print(f"  Char TF-IDF: {X_train_char.shape}  Word TF-IDF: {X_train_word.shape}")
print(f"  SVD: {X_train_svd.shape}  Distance: {dist_train.shape}")
print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
if XGBOOST_AVAILABLE:
    print("  XGBoost: available ✓")


# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob=None, label=''):
    return {
        'label':         label,
        'accuracy':      accuracy_score(y_true, y_pred),
        'precision':     precision_score(y_true, y_pred, zero_division=0),
        'recall':        recall_score(y_true, y_pred, zero_division=0),
        'f1':            f1_score(y_true, y_pred, zero_division=0),
        'roc_auc':       roc_auc_score(y_true, y_prob)
                         if y_prob is not None else float('nan'),
        'avg_precision': average_precision_score(y_true, y_prob)
                         if y_prob is not None else float('nan'),
    }

def print_eval(m):
    print(f"  Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  "
          f"AUC={m['roc_auc']:.4f}")

def tune_and_report(name, model_fn, param_grid,
                    X_tr, X_va, y_tr, y_va,
                    param_label_fn=None):
    """
    Simple validation-set tuning loop.
    model_fn(params) → fitted sklearn estimator
    param_grid: list of param dicts
    Returns best params dict and best val F1.
    """
    print(f"\n  {'Params':<45} {'Val F1':>8} {'Val AUC':>9}")
    print("  " + "─" * 65)
    results = []
    for params in param_grid:
        t0  = time.time()
        mdl = model_fn(params)
        mdl.fit(X_tr, y_tr)
        yp  = mdl.predict(X_va)
        ypr = (mdl.predict_proba(X_va)[:, 1]
               if hasattr(mdl, 'predict_proba') else None)
        m   = evaluate(y_va, yp, ypr, label=str(params))
        m.update(params)
        results.append({'params': params, 'metrics': m,
                        'time': time.time() - t0})
        label = (param_label_fn(params) if param_label_fn
                 else str(params))
        auc_str = f"{m['roc_auc']:>9.4f}" if ypr is not None else '       N/A'
        print(f"  {label:<45} {m['f1']:>8.4f} {auc_str}")
    best = max(results, key=lambda x: x['metrics']['f1'])
    print(f"\n  ✓ Best {name}: {best['params']}"
          f"  (Val F1={best['metrics']['f1']:.4f})")
    return best['params'], best['metrics']['f1']


# ═════════════════════════════════════════════════════════════════
#  PART A — RANDOM FOREST  (Lab 7)
# ═════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  PART A — RANDOM FOREST")
print("─" * 60)

# ─────────────────────────────────────────────────────────────────
# STEP 2  — Random Forest: n_estimators & max_features tuning
#           Uses SVD-50 — RF builds many DTs on random feature subsets
# ─────────────────────────────────────────────────────────────────
print("\n[Step 2] Random Forest tuning (SVD-50 features)...")

RF_GRID = [
    {'n_estimators': n, 'max_depth': d, 'max_features': mf}
    for n  in [50, 100, 200]
    for d  in [None, 10, 20]
    for mf in ['sqrt', 'log2']
]

def make_rf(p):
    return RandomForestClassifier(
        n_estimators=p['n_estimators'],
        max_depth=p['max_depth'],
        max_features=p['max_features'],
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

best_rf_params, _ = tune_and_report(
    'Random Forest', make_rf, RF_GRID,
    X_train_svd, X_val_svd, y_train, y_val,
    param_label_fn=lambda p: (
        f"n={p['n_estimators']}  depth={p['max_depth']}"
        f"  feats={p['max_features']}"
    )
)

# Final RF on test
final_rf = make_rf(best_rf_params)
final_rf.fit(X_tv_svd, y_tv)
y_pred_rf = final_rf.predict(X_test_svd)
y_prob_rf = final_rf.predict_proba(X_test_svd)[:, 1]
m_rf      = evaluate(y_test, y_pred_rf, y_prob_rf, label='Random Forest')
print(f"\n  Test Results — Random Forest:")
print_eval(m_rf)
print(classification_report(y_test, y_pred_rf,
      target_names=['Real','Fake'], digits=4))


# ═════════════════════════════════════════════════════════════════
#  PART B — NAIVE BAYES  (Lab 7)
# ═════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  PART B — NAIVE BAYES")
print("─" * 60)
print("\n[Step 3] Naive Bayes variants (word TF-IDF, non-negative)...")

# MinMax-scale word TF-IDF to [0,1] for MultinomialNB
scaler_nb    = MinMaxScaler()
X_tr_nb      = scaler_nb.fit_transform(X_train_word.toarray())
X_va_nb      = scaler_nb.transform(X_val_word.toarray())
X_te_nb      = scaler_nb.transform(X_test_word.toarray())
X_tv_nb      = scaler_nb.fit_transform(X_tv_word.toarray())
X_te_nb_f    = scaler_nb.transform(X_test_word.toarray())

NB_ALPHA = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
nb_results = []

print(f"\n  {'Model':<20} {'Alpha':>8} {'Val F1':>9} {'Val AUC':>9}")
print("  " + "─" * 50)

for alpha in NB_ALPHA:
    for NbClass, name, X_tr_use, X_va_use in [
        (MultinomialNB, 'MultinomialNB', X_tr_nb,      X_va_nb),
        (ComplementNB,  'ComplementNB',  X_tr_nb,      X_va_nb),
        (GaussianNB,    'GaussianNB',    X_train_svd,  X_val_svd),
    ]:
        if NbClass == GaussianNB and alpha != NB_ALPHA[0]:
            continue  # GaussianNB has no alpha
        try:
            nb = (NbClass(var_smoothing=1e-9)
                  if NbClass == GaussianNB
                  else NbClass(alpha=alpha))
            nb.fit(X_tr_use, y_train)
            yp  = nb.predict(X_va_use)
            ypr = nb.predict_proba(X_va_use)[:, 1]
            m   = evaluate(y_val, yp, ypr,
                           label=f'{name} alpha={alpha}')
            m['model'] = name; m['alpha'] = alpha
            nb_results.append({'model_obj': nb, 'metrics': m,
                                'X_tr': X_tr_use, 'X_va': X_va_use,
                                'name': name, 'alpha': alpha})
            a_str = 'N/A' if NbClass == GaussianNB else f'{alpha:.3f}'
            print(f"  {name:<20} {a_str:>8} "
                  f"{m['f1']:>9.4f} {m['roc_auc']:>9.4f}")
        except Exception as e:
            print(f"  {name:<20} {alpha:>8.3f}  ERROR: {e}")

best_nb_row  = max(nb_results, key=lambda x: x['metrics']['f1'])
best_nb_name = best_nb_row['name']
best_nb_alph = best_nb_row['alpha']
print(f"\n  ✓ Best NB: {best_nb_name}  alpha={best_nb_alph}"
      f"  (Val F1={best_nb_row['metrics']['f1']:.4f})")

# Final NB on test
best_nb_model = best_nb_row['model_obj']
X_te_nb_use   = (X_te_nb if best_nb_name in
                 ['MultinomialNB', 'ComplementNB'] else X_test_svd)
y_pred_nb = best_nb_model.predict(X_te_nb_use)
y_prob_nb = best_nb_model.predict_proba(X_te_nb_use)[:, 1]
m_nb      = evaluate(y_test, y_pred_nb, y_prob_nb,
                     label=f'{best_nb_name} (α={best_nb_alph})')
print(f"\n  Test Results — {best_nb_name}:")
print_eval(m_nb)
print(classification_report(y_test, y_pred_nb,
      target_names=['Real','Fake'], digits=4))


# ═════════════════════════════════════════════════════════════════
#  PART C — GRADIENT BOOSTING  (Lab 7)
# ═════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  PART C — GRADIENT BOOSTING")
print("─" * 60)
print("\n[Step 4] Gradient Boosting tuning (SVD-50 features)...")

GB_GRID = [
    {'n_estimators': n, 'learning_rate': lr, 'max_depth': d}
    for n  in [100, 200]
    for lr in [0.05, 0.1, 0.2]
    for d  in [3, 5]
]

def make_gb(p):
    return GradientBoostingClassifier(
        n_estimators=p['n_estimators'],
        learning_rate=p['learning_rate'],
        max_depth=p['max_depth'],
        random_state=RANDOM_STATE,
    )

best_gb_params, _ = tune_and_report(
    'GradBoost', make_gb, GB_GRID,
    X_train_svd, X_val_svd, y_train, y_val,
    param_label_fn=lambda p: (
        f"n={p['n_estimators']}  lr={p['learning_rate']}"
        f"  depth={p['max_depth']}"
    )
)

final_gb  = make_gb(best_gb_params)
final_gb.fit(X_tv_svd, y_tv)
y_pred_gb = final_gb.predict(X_test_svd)
y_prob_gb = final_gb.predict_proba(X_test_svd)[:, 1]
m_gb      = evaluate(y_test, y_pred_gb, y_prob_gb,
                     label='Gradient Boosting')
print(f"\n  Test Results — Gradient Boosting:")
print_eval(m_gb)
print(classification_report(y_test, y_pred_gb,
      target_names=['Real','Fake'], digits=4))


# ═════════════════════════════════════════════════════════════════
#  PART D — XGBoost  (Lab 7, optional)
# ═════════════════════════════════════════════════════════════════
m_xgb = None
if XGBOOST_AVAILABLE:
    print("\n" + "─" * 60)
    print("  PART D — XGBoost")
    print("─" * 60)
    print("\n[Step 5] XGBoost tuning (SVD-50 features)...")

    XGB_GRID = [
        {'n_estimators': n, 'learning_rate': lr,
         'max_depth': d, 'subsample': ss}
        for n  in [100, 200]
        for lr in [0.05, 0.1]
        for d  in [3, 6]
        for ss in [0.8, 1.0]
    ]

    def make_xgb(p):
        return XGBClassifier(
            n_estimators=p['n_estimators'],
            learning_rate=p['learning_rate'],
            max_depth=p['max_depth'],
            subsample=p['subsample'],
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
        )

    best_xgb_params, _ = tune_and_report(
        'XGBoost', make_xgb, XGB_GRID,
        X_train_svd, X_val_svd, y_train, y_val,
        param_label_fn=lambda p: (
            f"n={p['n_estimators']}  lr={p['learning_rate']}"
            f"  depth={p['max_depth']}  ss={p['subsample']}"
        )
    )

    final_xgb  = make_xgb(best_xgb_params)
    final_xgb.fit(X_tv_svd, y_tv)
    y_pred_xgb = final_xgb.predict(X_test_svd)
    y_prob_xgb = final_xgb.predict_proba(X_test_svd)[:, 1]
    m_xgb      = evaluate(y_test, y_pred_xgb, y_prob_xgb,
                           label='XGBoost')
    print(f"\n  Test Results — XGBoost:")
    print_eval(m_xgb)
    print(classification_report(y_test, y_pred_xgb,
          target_names=['Real','Fake'], digits=4))


# ═════════════════════════════════════════════════════════════════
#  PART E — PERCEPTRON  (Lab 8)
# ═════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  PART E — PERCEPTRON  (Lab 8)")
print("─" * 60)
print("\n[Step 6] Perceptron tuning (L2-normalised TF-IDF)...")
print("  Lab 8: single-layer linear threshold classifier")

PERC_GRID = [
    {'alpha': a, 'max_iter': it, 'penalty': pen}
    for a   in [0.0001, 0.001, 0.01]
    for it  in [100, 500, 1000]
    for pen in [None, 'l2', 'l1']
]

perc_results = []
print(f"\n  {'Alpha':>8} {'MaxIter':>8} {'Penalty':>8} "
      f"{'Val F1':>9} {'Val Acc':>9}")
print("  " + "─" * 50)

for p in PERC_GRID:
    perc = Perceptron(
        alpha=p['alpha'],
        max_iter=p['max_iter'],
        penalty=p['penalty'],
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    perc.fit(X_tr_norm, y_train)
    yp = perc.predict(X_va_norm)
    m  = evaluate(y_val, yp, label=str(p))
    m.update(p)
    perc_results.append({'params': p, 'metrics': m, 'model': perc})
    print(f"  {p['alpha']:>8.4f} {p['max_iter']:>8} "
          f"{str(p['penalty']):>8} "
          f"{m['f1']:>9.4f} {m['accuracy']:>9.4f}")

best_perc_row    = max(perc_results, key=lambda x: x['metrics']['f1'])
best_perc_params = best_perc_row['params']
print(f"\n  ✓ Best Perceptron: {best_perc_params}"
      f"  (Val F1={best_perc_row['metrics']['f1']:.4f})")

# Final Perceptron on test
final_perc = Perceptron(
    **{k: v for k, v in best_perc_params.items()},
    random_state=RANDOM_STATE, n_jobs=N_JOBS
)
final_perc.fit(X_tv_norm, y_tv)
y_pred_perc = final_perc.predict(X_te_norm)
m_perc      = evaluate(y_test, y_pred_perc,
                        label='Perceptron')
print(f"\n  Test Results — Perceptron:")
print(f"  Acc={m_perc['accuracy']:.4f}  F1={m_perc['f1']:.4f}  "
      f"Prec={m_perc['precision']:.4f}  Rec={m_perc['recall']:.4f}")
print(classification_report(y_test, y_pred_perc,
      target_names=['Real','Fake'], digits=4))


# ═════════════════════════════════════════════════════════════════
#  PART F — MLP (Multi-Layer Perceptron)  (Lab 8)
# ═════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  PART F — MLP (Multi-Layer Perceptron)  (Lab 8)")
print("─" * 60)
print("\n[Step 7] MLP tuning (MinMax-scaled SVD-50 features)...")
print("  Lab 8: multi-layer extension of the Perceptron")

MLP_GRID = [
    {'hidden_layer_sizes': h, 'alpha': a, 'activation': act}
    for h   in [(64,), (128,), (64, 32), (128, 64)]
    for a   in [0.0001, 0.001]
    for act in ['relu', 'tanh']
]

mlp_results = []
print(f"\n  {'Layers':<16} {'Alpha':>8} {'Act':>6} "
      f"{'Val F1':>9} {'Val AUC':>9}")
print("  " + "─" * 54)

for p in MLP_GRID:
    mlp = MLPClassifier(
        hidden_layer_sizes=p['hidden_layer_sizes'],
        alpha=p['alpha'],
        activation=p['activation'],
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_tr_scaled, y_train)
    yp  = mlp.predict(X_va_scaled)
    ypr = mlp.predict_proba(X_va_scaled)[:, 1]
    m   = evaluate(y_val, yp, ypr, label=str(p))
    m.update(p)
    mlp_results.append({'params': p, 'metrics': m, 'model': mlp})
    print(f"  {str(p['hidden_layer_sizes']):<16} {p['alpha']:>8.4f} "
          f"{p['activation']:>6} "
          f"{m['f1']:>9.4f} {m['roc_auc']:>9.4f}")

best_mlp_row    = max(mlp_results, key=lambda x: x['metrics']['f1'])
best_mlp_params = best_mlp_row['params']
print(f"\n  ✓ Best MLP: {best_mlp_params}"
      f"  (Val F1={best_mlp_row['metrics']['f1']:.4f})")

# Final MLP on test
final_mlp = MLPClassifier(
    hidden_layer_sizes=best_mlp_params['hidden_layer_sizes'],
    alpha=best_mlp_params['alpha'],
    activation=best_mlp_params['activation'],
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=RANDOM_STATE,
)
final_mlp.fit(X_tv_scaled_f, y_tv)
y_pred_mlp = final_mlp.predict(X_te_scaled_f)
y_prob_mlp = final_mlp.predict_proba(X_te_scaled_f)[:, 1]
m_mlp      = evaluate(y_test, y_pred_mlp, y_prob_mlp,
                       label='MLP')
print(f"\n  Test Results — MLP:")
print_eval(m_mlp)
print(classification_report(y_test, y_pred_mlp,
      target_names=['Real','Fake'], digits=4))


# ─────────────────────────────────────────────────────────────────
# STEP 8  — Per-Language Breakdown (best model per family)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 8] Per-language test breakdown...")

FINAL_MODELS = {
    'RandomForest': (final_rf,  X_test_svd,  None),
    'NaiveBayes':   (best_nb_model, X_te_nb_use, None),
    'GradBoost':    (final_gb,  X_test_svd,  None),
    'Perceptron':   (final_perc, X_te_norm,  None),
    'MLP':          (final_mlp, X_te_scaled_f, None),
}
if XGBOOST_AVAILABLE:
    FINAL_MODELS['XGBoost'] = (final_xgb, X_test_svd, None)

lang_results_all = []
print(f"\n  {'Language':<12}", end='')
for name in FINAL_MODELS:
    print(f"  {name:<14}", end='')
print()
print("  " + "─" * (12 + 16 * len(FINAL_MODELS)))

for lang in LANGUAGES:
    mask = (lang_test == lang).values
    if mask.sum() == 0:
        continue
    print(f"  {lang:<12}", end='')
    for model_name, (mdl, X_te_use, _) in FINAL_MODELS.items():
        try:
            yp   = mdl.predict(X_te_use[mask])
            m    = evaluate(y_test[mask], yp)
            m['language'] = lang; m['model'] = model_name
            lang_results_all.append(m)
            print(f"  {m['f1']:<14.4f}", end='')
        except Exception:
            print(f"  {'N/A':<14}", end='')
    print()

lang_df = pd.DataFrame(lang_results_all)


# ─────────────────────────────────────────────────────────────────
# STEP 9  — Full Model Comparison Table
# ─────────────────────────────────════════════════════════════════
print("\n[Step 9] Full model comparison...")

all_test_metrics = [m_rf, m_nb, m_gb, m_perc, m_mlp]
if m_xgb:
    all_test_metrics.append(m_xgb)

comp_df = pd.DataFrame(all_test_metrics)
cols    = ['label','accuracy','precision','recall','f1','roc_auc']
print(f"\n  {'Model':<30} {'Acc':>8} {'Prec':>8} "
      f"{'Rec':>8} {'F1':>8} {'AUC':>8}")
print("  " + "─" * 72)
for _, row in comp_df[cols].iterrows():
    auc_s = f"{row['roc_auc']:>8.4f}" if not np.isnan(row['roc_auc']) else '     N/A'
    print(f"  {row['label']:<30} {row['accuracy']:>8.4f} "
          f"{row['precision']:>8.4f} {row['recall']:>8.4f} "
          f"{row['f1']:>8.4f} {auc_s}")

best_phase8 = comp_df.loc[comp_df['f1'].idxmax(), 'label']
print(f"\n  ✓ Best model in Phase 8: {best_phase8}")


# ─────────────────────────────────────────────────────────────────
# STEP 10 — Save Artefacts
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 10] Saving artefacts...")

models_to_save = {
    'rf_final.pkl':    final_rf,
    'gb_final.pkl':    final_gb,
    'nb_final.pkl':    best_nb_model,
    'perc_final.pkl':  final_perc,
    'mlp_final.pkl':   final_mlp,
    'scaler_mlp.pkl':  scaler_tv,
}
if XGBOOST_AVAILABLE:
    models_to_save['xgb_final.pkl'] = final_xgb

for fname, obj in models_to_save.items():
    with open(os.path.join(OUTPUT_DIR, fname), 'wb') as f:
        pickle.dump(obj, f)

comp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"),      index=False)
lang_df.to_csv(os.path.join(OUTPUT_DIR, "per_language_results.csv"),  index=False)
pd.DataFrame([r['metrics'] for r in perc_results]).to_csv(
    os.path.join(OUTPUT_DIR, "perceptron_tuning.csv"), index=False)
pd.DataFrame([r['metrics'] for r in mlp_results]).to_csv(
    os.path.join(OUTPUT_DIR, "mlp_tuning.csv"), index=False)

config = {
    'rf':    {'params': best_rf_params,   'test_f1': round(m_rf['f1'], 4)},
    'nb':    {'model': best_nb_name,
              'alpha': best_nb_alph,      'test_f1': round(m_nb['f1'], 4)},
    'gb':    {'params': best_gb_params,   'test_f1': round(m_gb['f1'], 4)},
    'perc':  {'params': best_perc_params, 'test_f1': round(m_perc['f1'], 4)},
    'mlp':   {'params': best_mlp_params,  'test_f1': round(m_mlp['f1'], 4)},
    'best_model': best_phase8,
}
if m_xgb:
    config['xgb'] = {'params': best_xgb_params,
                     'test_f1': round(m_xgb['f1'], 4)}

with open(os.path.join(OUTPUT_DIR, "phase8_config.pkl"), 'wb') as f:
    pickle.dump(config, f)

for fname in ["y_train.csv", "y_val.csv", "y_test.csv",
              "lang_train.csv", "lang_val.csv", "lang_test.csv"]:
    pd.read_csv(os.path.join(P7_DIR, fname)).to_csv(
        os.path.join(OUTPUT_DIR, fname), index=False)

print(f"  ✓ Artefacts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────
# STEP 11 — Visualisations
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 11] Generating plots...")

COLORS = ['#4C72B0','#C44E52','#55A868','#DD8452','#9467BD','#8C564B']

fig = plt.figure(figsize=(18, 16))
fig.suptitle("Phase 8 — Other Classifiers & Perceptron",
             fontsize=15, fontweight='bold')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── Plot 1: Overall F1 bar — all models ─────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
all_names = [m['label']    for m in all_test_metrics]
all_f1    = [m['f1']       for m in all_test_metrics]
all_auc   = [m['roc_auc']  for m in all_test_metrics]
x = np.arange(len(all_names)); w = 0.38
bars = ax1.bar(x - w/2, all_f1,  w, color='#4C72B0', alpha=0.85, label='F1')
ax1.bar(x + w/2, [a if not np.isnan(a) else 0 for a in all_auc],
        w, color='#C44E52', alpha=0.85, label='AUC')
ax1.set_xticks(x)
ax1.set_xticklabels([n.split('(')[0][:12] for n in all_names],
                    rotation=20, ha='right', fontsize=8)
ax1.set_ylim(max(0, min(all_f1) - 0.08), 1.05)
ax1.set_ylabel("Score"); ax1.set_title("F1 & AUC — all models")
ax1.legend(fontsize=8)
for bar, v in zip(bars, all_f1):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f'{v:.3f}', ha='center', fontsize=7)

# ── Plot 2: ROC curves — all models with proba ──────────────────
ax2 = fig.add_subplot(gs[0, 1])
roc_models = [
    ('Random Forest', y_prob_rf,  '#4C72B0'),
    ('NaiveBayes',    y_prob_nb,  '#C44E52'),
    ('GradBoost',     y_prob_gb,  '#55A868'),
    ('MLP',           y_prob_mlp, '#DD8452'),
]
if m_xgb:
    roc_models.append(('XGBoost', y_prob_xgb, '#9467BD'))
for name, yprob, color in roc_models:
    fpr_c, tpr_c, _ = roc_curve(y_test, yprob)
    auc_c = roc_auc_score(y_test, yprob)
    ax2.plot(fpr_c, tpr_c, color=color, linewidth=1.8,
             label=f'{name} ({auc_c:.3f})')
ax2.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.4)
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("ROC Curves — probabilistic models")
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.25)

# ── Plot 3: RF feature importances ──────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
rf_imp   = final_rf.feature_importances_
top_rf   = rf_imp.argsort()[::-1][:15]
ax3.barh(range(15), rf_imp[top_rf][::-1], color='#4C72B0', alpha=0.8)
ax3.set_yticks(range(15))
ax3.set_yticklabels([f'SVD-{i}' for i in top_rf[::-1]], fontsize=8)
ax3.set_xlabel("Feature importance")
ax3.set_title("Random Forest: top 15 importances")

# ── Plot 4: MLP learning curve (loss) ───────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
if hasattr(final_mlp, 'loss_curve_'):
    ax4.plot(final_mlp.loss_curve_, color='#4C72B0', linewidth=2,
             label='Train loss')
    if hasattr(final_mlp, 'validation_scores_') and \
       final_mlp.validation_scores_ is not None:
        ax4.plot(final_mlp.validation_scores_, color='#C44E52',
                 linewidth=2, linestyle='--', label='Val score')
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Loss / Score")
ax4.set_title(f"MLP learning curve\n{best_mlp_params['hidden_layer_sizes']}"
              f" {best_mlp_params['activation']}")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.25)

# ── Plot 5: Perceptron val F1 across alpha & penalty ────────────
ax5 = fig.add_subplot(gs[1, 1])
perc_df = pd.DataFrame([{**r['params'], **r['metrics']}
                         for r in perc_results])
for pen, color in [(None, '#4C72B0'), ('l2', '#C44E52'), ('l1', '#55A868')]:
    sub = perc_df[perc_df['penalty'] == pen].groupby('alpha')['f1'].mean()
    if not sub.empty:
        ax5.semilogx(sub.index, sub.values, 'o-', color=color,
                     linewidth=2, markersize=6,
                     label=f'penalty={pen}')
ax5.set_xlabel("Alpha (log scale)"); ax5.set_ylabel("Mean Val F1")
ax5.set_title("Perceptron: alpha vs F1 by penalty")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.25)

# ── Plot 6: Confusion matrix — best model ───────────────────────
ax6 = fig.add_subplot(gs[1, 2])
best_pred = {
    'Random Forest':    y_pred_rf,
    'MLP':              y_pred_mlp,
    'Gradient Boosting':y_pred_gb,
    'XGBoost':          y_pred_xgb if XGBOOST_AVAILABLE else y_pred_rf,
}.get(best_phase8, y_pred_rf)
cm = confusion_matrix(y_test, best_pred)
im = ax6.imshow(cm, cmap='Blues')
ax6.set_title(f"Confusion — {best_phase8}", fontsize=9)
ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual")
ax6.set_xticks([0,1]); ax6.set_xticklabels(['Real','Fake'])
ax6.set_yticks([0,1]); ax6.set_yticklabels(['Real','Fake'])
for i in range(2):
    for j in range(2):
        ax6.text(j, i, f'{cm[i,j]}\n({cm[i,j]/cm.sum()*100:.1f}%)',
                 ha='center', va='center', fontsize=11,
                 color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax6)

# ── Plot 7: Per-language F1 heatmap ─────────────────────────────
ax7 = fig.add_subplot(gs[2, 0:2])
if not lang_df.empty:
    pivot = lang_df.pivot_table(
        index='language', columns='model', values='f1', aggfunc='mean')
    im7 = ax7.imshow(pivot.values.astype(float),
                     cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)
    ax7.set_xticks(range(len(pivot.columns)))
    ax7.set_xticklabels(pivot.columns, rotation=20, ha='right', fontsize=9)
    ax7.set_yticks(range(len(pivot.index)))
    ax7.set_yticklabels(pivot.index, fontsize=9)
    ax7.set_title("Per-language F1 heatmap — all Phase 8 models")
    plt.colorbar(im7, ax=ax7)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax7.text(j, i, f'{val:.3f}', ha='center',
                         va='center', fontsize=9, color='black')

# ── Plot 8: Metric radar across all models ───────────────────────
ax8 = fig.add_subplot(gs[2, 2])
metric_keys = ['accuracy','precision','recall','f1']
x = np.arange(len(metric_keys)); w = 1.0 / (len(all_test_metrics) + 1)
for i, (m, color) in enumerate(zip(all_test_metrics, COLORS)):
    vals = [m[k] for k in metric_keys]
    ax8.bar(x + (i - len(all_test_metrics)/2) * w, vals,
            w, color=color, alpha=0.8,
            label=m['label'].split('(')[0][:10])
ax8.set_xticks(x)
ax8.set_xticklabels(['Acc','Prec','Rec','F1'], fontsize=9)
ax8.set_ylim(max(0, min(
    [m[k] for m in all_test_metrics for k in metric_keys]) - 0.05), 1.05)
ax8.set_ylabel("Score")
ax8.set_title("Metric comparison — all models")
ax8.legend(fontsize=6, ncol=2)

plt.savefig(os.path.join(OUTPUT_DIR, "phase8_classifiers_results.png"),
            dpi=150, bbox_inches='tight')
print(f"  ✓ Saved → {OUTPUT_DIR}/phase8_classifiers_results.png")


# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 8 SUMMARY — Test Set Performance")
print("=" * 60)
print(f"  {'Model':<30} {'F1':>8} {'AUC':>8} {'Acc':>8}")
print("  " + "─" * 58)
for m in sorted(all_test_metrics, key=lambda x: x['f1'], reverse=True):
    auc_s = f"{m['roc_auc']:>8.4f}" if not np.isnan(m['roc_auc']) else '     N/A'
    print(f"  {m['label']:<30} {m['f1']:>8.4f} {auc_s} "
          f"{m['accuracy']:>8.4f}")
print(f"\n  ✓ Best model: {best_phase8}")
print(f"\nArtefacts in: ./{OUTPUT_DIR}/")
print("  rf_final.pkl  gb_final.pkl  nb_final.pkl")
print("  perc_final.pkl  mlp_final.pkl  scaler_mlp.pkl")
print("  model_comparison.csv  per_language_results.csv")
print("  phase8_config.pkl  phase8_classifiers_results.png")
print("\nReady for Phase 9 — Stacking Ensemble")
print("=" * 60)
