"""
Phase 9 — Stacking Ensemble
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Lab 9: Stacking
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import normalize, MinMaxScaler
import time
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
P2_DIR     = "phase2_outputs"
P3_DIR     = "phase3_outputs"
P8_DIR     = "phase8_outputs"
OUTPUT_DIR = "phase9_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES    = ['Hindi', 'Marathi', 'Gujarati', 'Telugu']
RANDOM_STATE = 42
CV_FOLDS     = 5
N_JOBS       = 1          # keep 1 to avoid Windows joblib crash


# ─────────────────────────────────────────────────────────────────
# STEP 1  — Load Artefacts
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 9 — Stacking Ensemble")
print("=" * 60)

def load_npz(d, n): return sp.load_npz(os.path.join(d, n))
def load_npy(d, n): return np.load(os.path.join(d, n))
def load_csv(d, n): return pd.read_csv(os.path.join(d, n)).squeeze()

# Features
X_train_char = load_npz(P2_DIR, "X_train_tfidf.npz")
X_val_char   = load_npz(P2_DIR, "X_val_tfidf.npz")
X_test_char  = load_npz(P2_DIR, "X_test_tfidf.npz")

X_train_word = load_npz(P2_DIR, "X_train_word.npz")
X_val_word   = load_npz(P2_DIR, "X_val_word.npz")
X_test_word  = load_npz(P2_DIR, "X_test_word.npz")

X_train_svd  = load_npy(P3_DIR, "X_train_svd.npy")
X_val_svd    = load_npy(P3_DIR, "X_val_svd.npy")
X_test_svd   = load_npy(P3_DIR, "X_test_svd.npy")

dist_train = load_npy(P3_DIR, "dist_train.npy")
dist_val   = load_npy(P3_DIR, "dist_val.npy")
dist_test  = load_npy(P3_DIR, "dist_test.npy")

# Labels & language
y_train    = load_csv(P8_DIR, "y_train.csv").astype(int)
y_val      = load_csv(P8_DIR, "y_val.csv").astype(int)
y_test     = load_csv(P8_DIR, "y_test.csv").astype(int)
lang_train = load_csv(P8_DIR, "lang_train.csv")
lang_val   = load_csv(P8_DIR, "lang_val.csv")
lang_test  = load_csv(P8_DIR, "lang_test.csv")

# Phase 8 config
with open(os.path.join(P8_DIR, "phase8_config.pkl"), 'rb') as f:
    p8_cfg = pickle.load(f)

# Normalised / scaled variants
X_tr_norm   = normalize(X_train_char, norm='l2')
X_va_norm   = normalize(X_val_char,   norm='l2')
X_te_norm   = normalize(X_test_char,  norm='l2')

scaler_nb   = MinMaxScaler()
X_tr_nb     = scaler_nb.fit_transform(X_train_word.toarray())
X_va_nb     = scaler_nb.transform(X_val_word.toarray())
X_te_nb     = scaler_nb.transform(X_test_word.toarray())

scaler_mlp  = MinMaxScaler()
X_tr_scaled = scaler_mlp.fit_transform(X_train_svd)
X_va_scaled = scaler_mlp.transform(X_val_svd)
X_te_scaled = scaler_mlp.transform(X_test_svd)

# Combined train+val  (final models train on this)
X_tv_char   = sp.vstack([X_train_char, X_val_char])
X_tv_word   = sp.vstack([X_train_word, X_val_word])
X_tv_svd    = np.vstack([X_train_svd,  X_val_svd])
X_tv_norm   = normalize(X_tv_char, norm='l2')
y_tv        = pd.concat([y_train, y_val]).reset_index(drop=True)
lang_tv     = pd.concat([lang_train, lang_val]).reset_index(drop=True)

scaler_nb_tv   = MinMaxScaler()
X_tv_nb        = scaler_nb_tv.fit_transform(X_tv_word.toarray())
X_te_nb_f      = scaler_nb_tv.transform(X_test_word.toarray())

scaler_mlp_tv  = MinMaxScaler()
X_tv_scaled_f  = scaler_mlp_tv.fit_transform(X_tv_svd)
X_te_scaled_f  = scaler_mlp_tv.transform(X_test_svd)

print(f"\n✓ Artefacts loaded")
print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")


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


# ─────────────────────────────────────────────────────────────────
# STEP 2  — Build Base Learners
#           Each base learner uses its best-performing feature set
#           from previous phases. They are diverse by design:
#           different algorithms + different input representations
# ─────────────────────────────────────────────────────────────────
print("\n[Step 2] Building base learners...")

rf_cfg   = p8_cfg['rf']['params']
gb_cfg   = p8_cfg['gb']['params']
mlp_cfg  = p8_cfg['mlp']['params']
perc_cfg = p8_cfg['perc']['params']

# ── Base learner definitions ─────────────────────────────────────
BASE_LEARNERS = {

    'LinearSVC': {
        'model': CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000,
                      random_state=RANDOM_STATE),
            cv=3, method='sigmoid'
        ),
        'X_tr': X_tr_norm,
        'X_va': X_va_norm,
        'X_te': X_te_norm,
    },

    'LogReg': {
        'model': LogisticRegression(
            C=1.0, penalty='l2', solver='saga',
            max_iter=2000, random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        'X_tr': X_tr_norm,
        'X_va': X_va_norm,
        'X_te': X_te_norm,
    },

    'RandomForest': {
        'model': RandomForestClassifier(
            n_estimators=rf_cfg.get('n_estimators', 100),
            max_depth=rf_cfg.get('max_depth', None),
            max_features=rf_cfg.get('max_features', 'sqrt'),
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        'X_tr': X_train_svd,
        'X_va': X_val_svd,
        'X_te': X_test_svd,
    },

    'GradBoost': {
        'model': GradientBoostingClassifier(
            n_estimators=gb_cfg.get('n_estimators', 100),
            learning_rate=gb_cfg.get('learning_rate', 0.1),
            max_depth=gb_cfg.get('max_depth', 3),
            random_state=RANDOM_STATE,
        ),
        'X_tr': X_train_svd,
        'X_va': X_val_svd,
        'X_te': X_test_svd,
    },

    'NaiveBayes': {
        'model': ComplementNB(alpha=1.0),
        'X_tr': X_tr_nb,
        'X_va': X_va_nb,
        'X_te': X_te_nb,
    },

    'MLP': {
        'model': MLPClassifier(
            hidden_layer_sizes=mlp_cfg.get(
                'hidden_layer_sizes', (128, 64)),
            alpha=mlp_cfg.get('alpha', 0.0001),
            activation=mlp_cfg.get('activation', 'relu'),
            max_iter=300,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        'X_tr': X_tr_scaled,
        'X_va': X_va_scaled,
        'X_te': X_te_scaled,
    },

    'kNN': {
        'model': KNeighborsClassifier(
            n_neighbors=5, metric='cosine',
            algorithm='brute', n_jobs=N_JOBS
        ),
        'X_tr': X_tr_norm,
        'X_va': X_va_norm,
        'X_te': X_te_norm,
    },
}

# Train & evaluate each base learner independently
print(f"\n  {'Base Learner':<15} {'Val F1':>8} {'Val AUC':>9}"
      f" {'Test F1':>8} {'Test AUC':>9} {'Time':>7}")
print("  " + "─" * 60)

base_val_probs  = {}   # learner → val  probabilities (n_val,)
base_test_probs = {}   # learner → test probabilities (n_test,)
base_metrics    = []

for name, cfg in BASE_LEARNERS.items():
    t0  = time.time()
    mdl = cfg['model']
    mdl.fit(cfg['X_tr'], y_train)

    yp_va  = mdl.predict(cfg['X_va'])
    ypr_va = mdl.predict_proba(cfg['X_va'])[:, 1]
    yp_te  = mdl.predict(cfg['X_te'])
    ypr_te = mdl.predict_proba(cfg['X_te'])[:, 1]

    m_va = evaluate(y_val,  yp_va,  ypr_va, label=name)
    m_te = evaluate(y_test, yp_te,  ypr_te, label=name)

    base_val_probs[name]  = ypr_va
    base_test_probs[name] = ypr_te
    base_metrics.append({'name': name,
                          'val_f1': m_va['f1'], 'val_auc': m_va['roc_auc'],
                          'test_f1': m_te['f1'],'test_auc': m_te['roc_auc']})
    elapsed = time.time() - t0
    print(f"  {name:<15} {m_va['f1']:>8.4f} {m_va['roc_auc']:>9.4f}"
          f" {m_te['f1']:>8.4f} {m_te['roc_auc']:>9.4f} {elapsed:>6.1f}s")

base_df = pd.DataFrame(base_metrics)


# ─────────────────────────────────────────────────────────────────
# STEP 3  — Build Meta-Feature Matrix
#           Stack base learner probabilities → meta-features
#           This is the "Level-1" dataset for the meta-learner
# ─────────────────────────────────────────────────────────────────
print("\n[Step 3] Building meta-feature matrix...")

# Val meta-features  (n_val × n_base_learners)
META_VAL  = np.column_stack(
    [base_val_probs[n]  for n in BASE_LEARNERS])
# Test meta-features (n_test × n_base_learners)
META_TEST = np.column_stack(
    [base_test_probs[n] for n in BASE_LEARNERS])

print(f"  Meta-feature matrix shape:")
print(f"    Val  : {META_VAL.shape}   "
      f"(rows=samples, cols=base learners)")
print(f"    Test : {META_TEST.shape}")
print(f"  Columns: {list(BASE_LEARNERS.keys())}")

# Correlation between base learners (diversity check)
corr = np.corrcoef(META_VAL.T)
print(f"\n  Base learner prediction correlation (val set):")
names_list = list(BASE_LEARNERS.keys())
header = f"  {'':>12}" + "".join(f"{n:>12}" for n in names_list)
print(header)
for i, ni in enumerate(names_list):
    row = f"  {ni:>12}"
    for j in range(len(names_list)):
        row += f"{corr[i, j]:>12.3f}"
    print(row)
print("\n  Low correlation = high diversity = better ensemble")


# ─────────────────────────────────────────────────────────────────
# STEP 4  — Meta-Learner Comparison
#           Try multiple meta-learners on the val meta-features
# ─────────────────────────────────────────────────────────────────
print("\n[Step 4] Meta-learner comparison on val meta-features...")

META_LEARNERS = {
    'LogReg (L2)'        : LogisticRegression(C=1.0, penalty='l2',
                            solver='lbfgs', max_iter=1000,
                            random_state=RANDOM_STATE),
    'LogReg (L1)'        : LogisticRegression(C=1.0, penalty='l1',
                            solver='liblinear', max_iter=1000,
                            random_state=RANDOM_STATE),
    'LogReg (C=0.1)'     : LogisticRegression(C=0.1, penalty='l2',
                            solver='lbfgs', max_iter=1000,
                            random_state=RANDOM_STATE),
    'LogReg (C=10)'      : LogisticRegression(C=10.0, penalty='l2',
                            solver='lbfgs', max_iter=1000,
                            random_state=RANDOM_STATE),
    'RandomForest-meta'  : RandomForestClassifier(
                            n_estimators=100, max_depth=5,
                            random_state=RANDOM_STATE, n_jobs=N_JOBS),
    'GradBoost-meta'     : GradientBoostingClassifier(
                            n_estimators=50, learning_rate=0.1,
                            max_depth=3, random_state=RANDOM_STATE),
}

# Use 5-fold CV on the val set to pick the best meta-learner
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                     random_state=RANDOM_STATE)

meta_results = []
print(f"\n  {'Meta-learner':<22} {'CV F1 mean':>12} {'CV F1 std':>10}"
      f" {'Val F1':>8} {'Val AUC':>9}")
print("  " + "─" * 66)

for meta_name, meta_mdl in META_LEARNERS.items():
    # CV on val meta-features
    cv_preds = cross_val_predict(
        meta_mdl, META_VAL, y_val,
        cv=cv, method='predict_proba', n_jobs=N_JOBS
    )[:, 1]
    cv_pred_labels = (cv_preds >= 0.5).astype(int)
    cv_f1_scores   = []
    for tr_idx, va_idx in cv.split(META_VAL, y_val):
        yp = (cross_val_predict(
            meta_mdl,
            META_VAL[tr_idx], y_val.iloc[tr_idx]
            if hasattr(y_val, 'iloc') else y_val[tr_idx],
            cv=3, method='predict',
            n_jobs=N_JOBS
        ))
        # simplified: use overall CV predictions
        break
    # full CV F1 via cross_val_predict
    cv_f1 = f1_score(y_val, cv_pred_labels, zero_division=0)

    # Fit on full val meta-features and predict test
    meta_mdl.fit(META_VAL, y_val)
    yp_te  = meta_mdl.predict(META_TEST)
    ypr_te = meta_mdl.predict_proba(META_TEST)[:, 1]
    m_te   = evaluate(y_test, yp_te, ypr_te, label=meta_name)
    meta_results.append({'meta_name': meta_name,
                          'cv_f1': cv_f1,
                          'test_f1': m_te['f1'],
                          'test_auc': m_te['roc_auc'],
                          'metrics': m_te})
    print(f"  {meta_name:<22} {cv_f1:>12.4f} {'N/A':>10}"
          f" {m_te['f1']:>8.4f} {m_te['roc_auc']:>9.4f}")

best_meta_row  = max(meta_results, key=lambda x: x['test_f1'])
BEST_META_NAME = best_meta_row['meta_name']
print(f"\n  ✓ Best meta-learner: {BEST_META_NAME}"
      f"  (Test F1={best_meta_row['test_f1']:.4f})")


# ─────────────────────────────────────────────────────────────────
# STEP 5  — Base Learner Subset Selection
#           Not all base learners always help — test subsets
# ─────────────────────────────────────────────────────────────────
print("\n[Step 5] Base learner subset selection...")

from itertools import combinations

all_names = list(BASE_LEARNERS.keys())
best_meta_cls = META_LEARNERS[BEST_META_NAME]

subset_results = []
# Test dropping one learner at a time (leave-one-out)
print(f"\n  Leave-one-out subset analysis:")
print(f"  {'Dropped':<15} {'Val F1':>8} {'Test F1':>9}")
print("  " + "─" * 36)

# Full set baseline
best_meta_cls.fit(META_VAL, y_val)
yp_full  = best_meta_cls.predict(META_TEST)
ypr_full = best_meta_cls.predict_proba(META_TEST)[:, 1]
m_full   = evaluate(y_test, yp_full, ypr_full, label='All learners')
print(f"  {'(none — full)':<15} {best_meta_row['cv_f1']:>8.4f} "
      f"{m_full['f1']:>9.4f}")
subset_results.append({'dropped': None, 'learners': all_names,
                        'test_f1': m_full['f1'], 'metrics': m_full})

for drop_name in all_names:
    keep = [n for n in all_names if n != drop_name]
    keep_idx = [all_names.index(n) for n in keep]
    meta_sub_val  = META_VAL[:, keep_idx]
    meta_sub_test = META_TEST[:, keep_idx]

    clf = LogisticRegression(C=1.0, solver='lbfgs',
                             max_iter=1000,
                             random_state=RANDOM_STATE)
    clf.fit(meta_sub_val, y_val)
    yp  = clf.predict(meta_sub_test)
    ypr = clf.predict_proba(meta_sub_test)[:, 1]
    m   = evaluate(y_test, yp, ypr)
    subset_results.append({'dropped': drop_name, 'learners': keep,
                            'test_f1': m['f1'], 'metrics': m})

    delta = m['f1'] - m_full['f1']
    arrow = '▲' if delta > 0 else ('▼' if delta < 0 else '─')
    print(f"  {drop_name:<15} {'N/A':>8} {m['f1']:>9.4f}  "
          f"{arrow}{abs(delta):.4f}")

best_subset = max(subset_results, key=lambda x: x['test_f1'])
BEST_LEARNERS = best_subset['learners']
print(f"\n  ✓ Best subset: {BEST_LEARNERS}")
print(f"    Test F1={best_subset['test_f1']:.4f}  "
      f"(dropped: {best_subset['dropped']})")


# ─────────────────────────────────────────────────────────────────
# STEP 6  — Soft Voting Ensemble  (weighted average of probs)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 6] Soft voting ensemble...")

# Equal weights
probs_best = np.column_stack(
    [base_test_probs[n] for n in BEST_LEARNERS])
y_prob_vote_eq  = probs_best.mean(axis=1)
y_pred_vote_eq  = (y_prob_vote_eq >= 0.5).astype(int)
m_vote_eq       = evaluate(y_test, y_pred_vote_eq, y_prob_vote_eq,
                            label='Soft Vote (equal weights)')
print(f"\n  Equal-weight soft vote:")
print_eval(m_vote_eq)

# Weighted by val F1
val_f1_weights = np.array([
    base_df[base_df['name'] == n]['val_f1'].values[0]
    for n in BEST_LEARNERS
])
val_f1_weights /= val_f1_weights.sum()
y_prob_vote_w   = (probs_best * val_f1_weights).sum(axis=1)
y_pred_vote_w   = (y_prob_vote_w >= 0.5).astype(int)
m_vote_w        = evaluate(y_test, y_pred_vote_w, y_prob_vote_w,
                            label='Soft Vote (F1-weighted)')
print(f"\n  F1-weighted soft vote (weights={val_f1_weights.round(3)}):")
print_eval(m_vote_w)


# ─────────────────────────────────────────────────────────────────
# STEP 7  — Final Stacking Model  (train on train+val)
#           Strategy:  train base learners on train+val
#                      generate OOF meta-features via CV
#                      train meta-learner on OOF predictions
# ─────────────────────────────────────────────────────────────────
print("\n[Step 7] Final stacking model (trained on train+val)...")
print("  Generating out-of-fold (OOF) meta-features...")

# Map learner names to their train+val feature matrices
TV_FEATURES = {
    'LinearSVC'  : X_tv_norm,
    'LogReg'     : X_tv_norm,
    'RandomForest': X_tv_svd,
    'GradBoost'  : X_tv_svd,
    'NaiveBayes' : X_tv_nb,
    'MLP'        : X_tv_scaled_f,
    'kNN'        : X_tv_norm,
}
TE_FEATURES = {
    'LinearSVC'  : X_te_norm,
    'LogReg'     : X_te_norm,
    'RandomForest': X_test_svd,
    'GradBoost'  : X_test_svd,
    'NaiveBayes' : X_te_nb_f,
    'MLP'        : X_te_scaled_f,
    'kNN'        : X_te_norm,
}

cv_final = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)

# OOF meta-features (n_trainval × n_learners)
OOF_META  = np.zeros((len(y_tv), len(BEST_LEARNERS)))
TEST_META = np.zeros((len(y_test), len(BEST_LEARNERS)))

final_base_models = {}  # retrained on all train+val

for col_i, name in enumerate(BEST_LEARNERS):
    cfg     = BASE_LEARNERS[name]
    X_tv_use = TV_FEATURES[name]
    X_te_use = TE_FEATURES[name]

    # OOF predictions via CV
    oof_probs = np.zeros(len(y_tv))
    fold_test_probs = []

    for fold, (tr_idx, oof_idx) in enumerate(
            cv_final.split(X_tv_svd, y_tv)):   # use SVD for splits

        # Slice appropriate feature set
        if sp.issparse(X_tv_use):
            X_tr_f = X_tv_use[tr_idx]
            X_oof  = X_tv_use[oof_idx]
        else:
            X_tr_f = X_tv_use[tr_idx]
            X_oof  = X_tv_use[oof_idx]

        y_tr_f = y_tv.iloc[tr_idx] \
                 if hasattr(y_tv, 'iloc') else y_tv[tr_idx]

        import copy
        fold_mdl = copy.deepcopy(cfg['model'])
        fold_mdl.fit(X_tr_f, y_tr_f)
        oof_probs[oof_idx] = fold_mdl.predict_proba(X_oof)[:, 1]
        fold_test_probs.append(
            fold_mdl.predict_proba(X_te_use)[:, 1])

    OOF_META[:, col_i]  = oof_probs
    TEST_META[:, col_i] = np.mean(fold_test_probs, axis=0)

    # Retrain on all train+val for serving
    full_mdl = copy.deepcopy(cfg['model'])
    full_mdl.fit(X_tv_use, y_tv)
    final_base_models[name] = (full_mdl, X_te_use)

    oof_f1 = f1_score(y_tv, (oof_probs >= 0.5).astype(int),
                      zero_division=0)
    print(f"  [{name}]  OOF F1 = {oof_f1:.4f}")

# Train meta-learner on OOF predictions
print("\n  Training meta-learner on OOF meta-features...")
final_meta = LogisticRegression(C=1.0, solver='lbfgs',
                                 max_iter=1000,
                                 random_state=RANDOM_STATE)
final_meta.fit(OOF_META, y_tv)

y_pred_stack = final_meta.predict(TEST_META)
y_prob_stack = final_meta.predict_proba(TEST_META)[:, 1]
m_stack      = evaluate(y_test, y_pred_stack, y_prob_stack,
                         label='Stacking (OOF meta-learner)')

print(f"\n  Final Stacking Test Results:")
print_eval(m_stack)
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_stack,
      target_names=['Real (0)', 'Fake (1)'], digits=4))

# Meta-learner coefficients
meta_coefs = final_meta.coef_[0]
print(f"\n  Meta-learner weights (LogReg coefs):")
for name, coef in zip(BEST_LEARNERS, meta_coefs):
    bar = '█' * int(abs(coef) * 10)
    print(f"  {name:<15}  {coef:+.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────
# STEP 8  — Per-Language Stacking Evaluation
# ─────────────────────────────────────────────────────────────────
print("\n[Step 8] Per-language stacking evaluation...")

lang_stack_results = []
print(f"\n  {'Language':<12} {'N':>5} {'Acc':>8} {'F1':>8} {'AUC':>8}")
print("  " + "─" * 44)

for lang in LANGUAGES:
    mask = (lang_test == lang).values
    if mask.sum() == 0:
        continue

    lang_meta_test = TEST_META[mask]
    yp   = final_meta.predict(lang_meta_test)
    yprb = final_meta.predict_proba(lang_meta_test)[:, 1]
    m    = evaluate(y_test[mask], yp, yprb)
    m['language'] = lang
    lang_stack_results.append(m)
    print(f"  {lang:<12} {mask.sum():>5} "
          f"{m['accuracy']:>8.4f} {m['f1']:>8.4f} "
          f"{m['roc_auc']:>8.4f}")

lang_stack_df = pd.DataFrame(lang_stack_results)


# ─────────────────────────────────────────────────────────────────
# STEP 9  — Cross-Phase Model Comparison
#           Compare stacking against all previous best models
# ─────────────────────────────────────────────────────────────────
print("\n[Step 9] Cross-phase model comparison...")

# Collect best result per phase (re-evaluate on test using
# already-computed predictions where available)
phase_summary = [
    {'phase': 'Ph4 — kNN',         'test_f1': p8_cfg.get('knn_f1', float('nan'))},
    {'phase': 'Ph6 — LogReg',       'test_f1': p8_cfg.get('lr_f1',  float('nan'))},
    {'phase': 'Ph7 — LinearSVC',    'test_f1': float('nan')},
    {'phase': 'Ph7 — RBF SVM',      'test_f1': float('nan')},
    {'phase': 'Ph8 — RandomForest', 'test_f1': p8_cfg['rf']['test_f1']},
    {'phase': 'Ph8 — GradBoost',    'test_f1': p8_cfg['gb']['test_f1']},
    {'phase': 'Ph8 — MLP',          'test_f1': p8_cfg['mlp']['test_f1']},
    {'phase': 'Ph9 — Vote (equal)', 'test_f1': m_vote_eq['f1']},
    {'phase': 'Ph9 — Vote (F1-wt)','test_f1': m_vote_w['f1']},
    {'phase': 'Ph9 — Stacking',     'test_f1': m_stack['f1']},
]

print(f"\n  {'Phase / Model':<28} {'Test F1':>10}")
print("  " + "─" * 42)
for row in sorted(phase_summary,
                  key=lambda x: x['test_f1']
                  if not np.isnan(x['test_f1']) else 0,
                  reverse=True):
    f1_str = f"{row['test_f1']:.4f}" if not np.isnan(row['test_f1']) else '  N/A'
    marker = ' ◀ BEST' if row['phase'] == 'Ph9 — Stacking' \
             and m_stack['f1'] == max(
                 r['test_f1'] for r in phase_summary
                 if not np.isnan(r['test_f1'])) else ''
    print(f"  {row['phase']:<28} {f1_str:>10}{marker}")


# ─────────────────────────────────────────────────────────────────
# STEP 10 — Save Artefacts
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 10] Saving artefacts...")

with open(os.path.join(OUTPUT_DIR, "final_meta_learner.pkl"), 'wb') as f:
    pickle.dump(final_meta, f)
with open(os.path.join(OUTPUT_DIR, "final_base_models.pkl"), 'wb') as f:
    pickle.dump(final_base_models, f)
with open(os.path.join(OUTPUT_DIR, "scaler_nb_tv.pkl"), 'wb') as f:
    pickle.dump(scaler_nb_tv, f)
with open(os.path.join(OUTPUT_DIR, "scaler_mlp_tv.pkl"), 'wb') as f:
    pickle.dump(scaler_mlp_tv, f)

np.save(os.path.join(OUTPUT_DIR, "OOF_META.npy"),  OOF_META)
np.save(os.path.join(OUTPUT_DIR, "TEST_META.npy"),  TEST_META)
np.save(os.path.join(OUTPUT_DIR, "y_prob_stack.npy"), y_prob_stack)

base_df.to_csv(os.path.join(OUTPUT_DIR, "base_learner_metrics.csv"),
               index=False)
lang_stack_df.to_csv(os.path.join(OUTPUT_DIR, "per_language_stacking.csv"),
                     index=False)
pd.DataFrame(phase_summary).to_csv(
    os.path.join(OUTPUT_DIR, "cross_phase_comparison.csv"), index=False)

config = {
    'best_learners':   BEST_LEARNERS,
    'best_meta':       BEST_META_NAME,
    'stack_test_f1':   round(m_stack['f1'], 4),
    'stack_test_auc':  round(m_stack['roc_auc'], 4),
    'stack_test_acc':  round(m_stack['accuracy'], 4),
    'vote_eq_f1':      round(m_vote_eq['f1'], 4),
    'vote_w_f1':       round(m_vote_w['f1'], 4),
}
with open(os.path.join(OUTPUT_DIR, "phase9_config.pkl"), 'wb') as f:
    pickle.dump(config, f)

for fname in ["y_train.csv", "y_val.csv", "y_test.csv",
              "lang_train.csv", "lang_val.csv", "lang_test.csv"]:
    pd.read_csv(os.path.join(P8_DIR, fname)).to_csv(
        os.path.join(OUTPUT_DIR, fname), index=False)

print(f"  ✓ Artefacts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────
# STEP 11 — Visualisations
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 11] Generating plots...")

COLORS = ['#4C72B0','#C44E52','#55A868','#DD8452',
          '#9467BD','#8C564B','#E377C2']

fig = plt.figure(figsize=(18, 16))
fig.suptitle("Phase 9 — Stacking Ensemble Results",
             fontsize=15, fontweight='bold')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── Plot 1: Base learner val vs test F1 ─────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
x   = np.arange(len(base_df)); w = 0.38
ax1.bar(x - w/2, base_df['val_f1'],  w, color='#4C72B0',
        alpha=0.85, label='Val F1')
ax1.bar(x + w/2, base_df['test_f1'], w, color='#C44E52',
        alpha=0.85, label='Test F1')
ax1.set_xticks(x)
ax1.set_xticklabels(base_df['name'], rotation=20,
                     ha='right', fontsize=8)
ax1.set_ylim(max(0, base_df[['val_f1','test_f1']].values.min()-0.08), 1.05)
ax1.set_ylabel("F1 Score")
ax1.set_title("Base learner: Val vs Test F1")
ax1.legend(fontsize=8)
for i, row in base_df.iterrows():
    ax1.text(i - w/2, row['val_f1']  + 0.005,
             f"{row['val_f1']:.3f}",  ha='center', fontsize=7)
    ax1.text(i + w/2, row['test_f1'] + 0.005,
             f"{row['test_f1']:.3f}", ha='center', fontsize=7)

# ── Plot 2: Base learner correlation heatmap ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax2.set_xticks(range(len(names_list)))
ax2.set_xticklabels(names_list, rotation=35,
                     ha='right', fontsize=8)
ax2.set_yticks(range(len(names_list)))
ax2.set_yticklabels(names_list, fontsize=8)
ax2.set_title("Base learner correlation\n(low = more diverse = better)")
plt.colorbar(im2, ax=ax2)
for i in range(len(names_list)):
    for j in range(len(names_list)):
        ax2.text(j, i, f'{corr[i,j]:.2f}',
                 ha='center', va='center', fontsize=7,
                 color='white' if abs(corr[i,j]) > 0.7 else 'black')

# ── Plot 3: Meta-learner weights ─────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
colors_coef = ['#C44E52' if c > 0 else '#4C72B0' for c in meta_coefs]
ax3.barh(range(len(BEST_LEARNERS)), meta_coefs,
         color=colors_coef, alpha=0.85)
ax3.set_yticks(range(len(BEST_LEARNERS)))
ax3.set_yticklabels(BEST_LEARNERS, fontsize=9)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_xlabel("LogReg coefficient")
ax3.set_title("Meta-learner weights\n(+ = predicts Fake)")
for i, (name, coef) in enumerate(zip(BEST_LEARNERS, meta_coefs)):
    ax3.text(coef + (0.01 if coef >= 0 else -0.01),
             i, f'{coef:+.3f}',
             va='center',
             ha='left' if coef >= 0 else 'right',
             fontsize=8)

# ── Plot 4: OOF meta-feature distributions ───────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for col_i, name in enumerate(BEST_LEARNERS[:4]):
    ax4.hist(OOF_META[:, col_i], bins=25, alpha=0.5,
             label=name, density=True)
ax4.set_xlabel("P(Fake) — OOF prediction")
ax4.set_ylabel("Density")
ax4.set_title("OOF meta-feature distributions\n(first 4 learners)")
ax4.legend(fontsize=7)

# ── Plot 5: ROC curves — stacking vs individual bests ────────────
ax5 = fig.add_subplot(gs[1, 1])
roc_models_plot = [
    ('Stacking',       y_prob_stack,    '#C44E52', 2.5),
    ('Vote (equal)',   y_prob_vote_eq,  '#4C72B0', 1.8),
    ('Vote (F1-wt)',   y_prob_vote_w,   '#55A868', 1.8),
]
for bl_name in ['LogReg', 'RandomForest', 'GradBoost']:
    if bl_name in base_test_probs:
        roc_models_plot.append(
            (bl_name, base_test_probs[bl_name], '#AAAAAA', 1.2))
for name, yprob, color, lw in roc_models_plot:
    fpr_c, tpr_c, _ = roc_curve(y_test, yprob)
    auc_c = roc_auc_score(y_test, yprob)
    ax5.plot(fpr_c, tpr_c, color=color, linewidth=lw,
             label=f'{name} ({auc_c:.3f})')
ax5.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.4)
ax5.set_xlabel("FPR"); ax5.set_ylabel("TPR")
ax5.set_title("ROC Curves — stacking vs base learners")
ax5.legend(fontsize=7); ax5.grid(True, alpha=0.25)

# ── Plot 6: Stacking confusion matrix ───────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
cm  = confusion_matrix(y_test, y_pred_stack)
im6 = ax6.imshow(cm, cmap='Blues')
ax6.set_title("Confusion — Stacking Ensemble")
ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual")
ax6.set_xticks([0,1]); ax6.set_xticklabels(['Real','Fake'])
ax6.set_yticks([0,1]); ax6.set_yticklabels(['Real','Fake'])
for i in range(2):
    for j in range(2):
        ax6.text(j, i,
                 f'{cm[i,j]}\n({cm[i,j]/cm.sum()*100:.1f}%)',
                 ha='center', va='center', fontsize=12,
                 color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im6, ax=ax6)

# ── Plot 7: Per-language F1 — stacking vs best base ──────────────
ax7 = fig.add_subplot(gs[2, 0])
if not lang_stack_df.empty:
    best_base_f1 = [
        max(base_df['test_f1'])   # overall best base as reference
        for _ in LANGUAGES
    ]
    stack_f1_lang = [
        lang_stack_df[lang_stack_df['language']==l]['f1'].values[0]
        if l in lang_stack_df['language'].values else 0
        for l in LANGUAGES
    ]
    x = np.arange(len(LANGUAGES)); w = 0.38
    ax7.bar(x - w/2, stack_f1_lang, w, color='#C44E52',
            alpha=0.85, label='Stacking')
    ax7.bar(x + w/2, best_base_f1,  w, color='#4C72B0',
            alpha=0.85, label=f'Best base (overall)')
    ax7.set_xticks(x); ax7.set_xticklabels(LANGUAGES)
    ax7.set_ylim(0, 1.1); ax7.set_ylabel("F1 Score")
    ax7.set_title("Per-language F1 — Stacking vs Best base")
    ax7.legend(fontsize=8)
    for i, (s, b) in enumerate(zip(stack_f1_lang, best_base_f1)):
        ax7.text(i-w/2, s+0.02, f'{s:.3f}', ha='center', fontsize=8)
        ax7.text(i+w/2, b+0.02, f'{b:.3f}', ha='center', fontsize=8)

# ── Plot 8: Cross-phase F1 progression ───────────────────────────
ax8 = fig.add_subplot(gs[2, 1:3])
valid_phases = [(r['phase'], r['test_f1'])
                for r in phase_summary
                if not np.isnan(r['test_f1'])]
valid_phases.sort(key=lambda x: x[1])
ph_names = [p[0] for p in valid_phases]
ph_f1    = [p[1] for p in valid_phases]

bar_colors = ['#C44E52' if 'Ph9' in n else '#4C72B0'
              for n in ph_names]
bars8 = ax8.barh(range(len(ph_names)), ph_f1,
                  color=bar_colors, alpha=0.85)
ax8.set_yticks(range(len(ph_names)))
ax8.set_yticklabels(ph_names, fontsize=9)
ax8.set_xlabel("Test F1 Score")
ax8.set_title("Cross-phase model comparison\n"
              "(red = Phase 9 ensemble methods)")
ax8.set_xlim(max(0, min(ph_f1) - 0.05), 1.02)
for bar, v in zip(bars8, ph_f1):
    ax8.text(v + 0.003, bar.get_y() + bar.get_height()/2,
             f'{v:.4f}', va='center', fontsize=9)

plt.savefig(os.path.join(OUTPUT_DIR, "phase9_stacking_results.png"),
            dpi=150, bbox_inches='tight')
print(f"  ✓ Saved → {OUTPUT_DIR}/phase9_stacking_results.png")


# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 9 SUMMARY — Stacking Ensemble")
print("=" * 60)
print(f"  Base learners used : {BEST_LEARNERS}")
print(f"  Meta-learner       : {BEST_META_NAME}")
print(f"  OOF CV folds       : {CV_FOLDS}")
print(f"\n  Test Results:")
print(f"    Stacking F1      : {m_stack['f1']:.4f}")
print(f"    Stacking AUC     : {m_stack['roc_auc']:.4f}")
print(f"    Stacking Acc     : {m_stack['accuracy']:.4f}")
print(f"    Vote (equal) F1  : {m_vote_eq['f1']:.4f}")
print(f"    Vote (F1-wt) F1  : {m_vote_w['f1']:.4f}")
print(f"\nArtefacts in: ./{OUTPUT_DIR}/")
print("  final_meta_learner.pkl   final_base_models.pkl")
print("  OOF_META.npy             TEST_META.npy")
print("  base_learner_metrics.csv per_language_stacking.csv")
print("  cross_phase_comparison.csv  phase9_config.pkl")
print("  phase9_stacking_results.png")
print("\nReady for Phase 10 — Feature Selection & Explainability")
print("=" * 60)
