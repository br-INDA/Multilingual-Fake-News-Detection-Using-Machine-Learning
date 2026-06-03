"""
Phase 10 — Feature Selection & Explainability
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Lab 10: Feature Selection and Explainability
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from sklearn.feature_selection import (
    chi2, mutual_info_classif,
    SelectKBest, SelectPercentile, RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report
)
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.pipeline import Pipeline
import time
warnings.filterwarnings('ignore')

# ── optional SHAP ────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[INFO] SHAP not installed. Run: pip install shap")
    print("       SHAP plots will be skipped.\n")

# ── optional LIME ────────────────────────────────────────────────
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[INFO] LIME not installed. Run: pip install lime")
    print("       LIME explanations will be skipped.\n")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
P2_DIR     = "phase2_outputs"
P3_DIR     = "phase3_outputs"
P9_DIR     = "phase9_outputs"
OUTPUT_DIR = "phase10_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES    = ['Hindi', 'Marathi', 'Gujarati', 'Telugu']
RANDOM_STATE = 42
N_JOBS       = 1
TOP_K_VALUES = [100, 500, 1000, 2000, 5000]    # feature counts to test


# ─────────────────────────────────────────────────────────────────
# STEP 1  — Load Artefacts
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 10 — Feature Selection & Explainability")
print("=" * 60)

def load_npz(d, n): return sp.load_npz(os.path.join(d, n))
def load_npy(d, n): return np.load(os.path.join(d, n))
def load_csv(d, n): return pd.read_csv(os.path.join(d, n)).squeeze()

# TF-IDF (sparse, non-negative — required for chi2)
X_train_char = load_npz(P2_DIR, "X_train_tfidf.npz")
X_val_char   = load_npz(P2_DIR, "X_val_tfidf.npz")
X_test_char  = load_npz(P2_DIR, "X_test_tfidf.npz")

X_train_word = load_npz(P2_DIR, "X_train_word.npz")
X_val_word   = load_npz(P2_DIR, "X_val_word.npz")
X_test_word  = load_npz(P2_DIR, "X_test_word.npz")

# SVD dense features
X_train_svd  = load_npy(P3_DIR, "X_train_svd.npy")
X_val_svd    = load_npy(P3_DIR, "X_val_svd.npy")
X_test_svd   = load_npy(P3_DIR, "X_test_svd.npy")

# Distance features
dist_train = load_npy(P3_DIR, "dist_train.npy")
dist_val   = load_npy(P3_DIR, "dist_val.npy")
dist_test  = load_npy(P3_DIR, "dist_test.npy")

# Labels & language
y_train    = load_csv(P9_DIR, "y_train.csv").astype(int)
y_val      = load_csv(P9_DIR, "y_val.csv").astype(int)
y_test     = load_csv(P9_DIR, "y_test.csv").astype(int)
lang_train = load_csv(P9_DIR, "lang_train.csv")
lang_val   = load_csv(P9_DIR, "lang_val.csv")
lang_test  = load_csv(P9_DIR, "lang_test.csv")

# Vocabulary from Phase 2
with open(os.path.join(P2_DIR, "tfidf_combined.pkl"), 'rb') as f:
    tfidf_char = pickle.load(f)
with open(os.path.join(P2_DIR, "tfidf_word.pkl"), 'rb') as f:
    tfidf_word = pickle.load(f)

char_features = np.array(tfidf_char.get_feature_names_out())
word_features = np.array(tfidf_word.get_feature_names_out())

# Text for LIME (from splits CSV)
splits_df  = pd.read_csv(os.path.join(P2_DIR, "splits.csv"),
                          encoding='utf-8-sig')
test_texts = splits_df[splits_df['split'] == 'test'][
    'text_clean'].reset_index(drop=True)

# Normalised + scaled variants
X_tr_norm   = normalize(X_train_char, norm='l2')
X_va_norm   = normalize(X_val_char,   norm='l2')
X_te_norm   = normalize(X_test_char,  norm='l2')

scaler_mm   = MinMaxScaler()
X_tr_mm     = scaler_mm.fit_transform(X_train_char.toarray())
X_va_mm     = scaler_mm.transform(X_val_char.toarray())
X_te_mm     = scaler_mm.transform(X_test_char.toarray())

# Combined train+val
X_tv_char   = sp.vstack([X_train_char, X_val_char])
X_tv_norm   = normalize(X_tv_char, norm='l2')
X_tv_mm     = scaler_mm.fit_transform(X_tv_char.toarray())
X_te_mm_f   = scaler_mm.transform(X_test_char.toarray())
y_tv        = pd.concat([y_train, y_val]).reset_index(drop=True)
lang_tv     = pd.concat([lang_train, lang_val]).reset_index(drop=True)

print(f"\n✓ Artefacts loaded")
print(f"  Char TF-IDF features : {X_train_char.shape[1]:,}")
print(f"  Word TF-IDF features : {X_train_word.shape[1]:,}")
print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")


# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob=None, label=''):
    return {
        'label':    label,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1':       f1_score(y_true, y_pred, zero_division=0),
        'roc_auc':  roc_auc_score(y_true, y_prob)
                    if y_prob is not None else float('nan'),
    }

def print_eval(m):
    print(f"  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
          f"AUC={m['roc_auc']:.4f}")


# ═════════════════════════════════════════════════════════════════
#  PART A — FEATURE SELECTION  (Lab 10)
# ═════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# STEP 2  — Chi-Squared Feature Selection
#           Lab 10: measures dependence between feature and label
#           Requires non-negative features → MinMax-scaled TF-IDF
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  PART A — FEATURE SELECTION")
print("─" * 60)
print("\n[Step 2] Chi-squared feature selection...")

# Compute chi2 scores for ALL features on training set
chi2_scores, chi2_pvals = chi2(X_tr_mm, y_train)
chi2_ranks = np.argsort(chi2_scores)[::-1]

print(f"\n  Top 20 features by chi-squared score:")
print(f"  {'Rank':<6} {'Feature':<35} {'Chi2':>10} {'p-value':>12}")
print("  " + "─" * 66)
for rank, idx in enumerate(chi2_ranks[:20]):
    print(f"  {rank+1:<6} {char_features[idx]:<35} "
          f"{chi2_scores[idx]:>10.2f} {chi2_pvals[idx]:>12.2e}")

# Test different k values with LogReg
print(f"\n  K-feature sweep (chi2 + LogReg, val set):")
print(f"  {'K':>6} {'Val F1':>9} {'Val AUC':>9} {'Train time':>12}")
print("  " + "─" * 42)

chi2_results = []
for k in TOP_K_VALUES:
    selector = SelectKBest(chi2, k=k)
    X_tr_sel = selector.fit_transform(X_tr_mm, y_train)
    X_va_sel = selector.transform(X_va_mm)
    X_te_sel = selector.transform(X_te_mm)

    t0  = time.time()
    clf = LogisticRegression(C=1.0, solver='saga', max_iter=1000,
                              random_state=RANDOM_STATE, n_jobs=N_JOBS)
    clf.fit(X_tr_sel, y_train)
    elapsed = time.time() - t0

    yp  = clf.predict(X_va_sel)
    ypr = clf.predict_proba(X_va_sel)[:, 1]
    m   = evaluate(y_val, yp, ypr, label=f'chi2 k={k}')
    m['k'] = k; m['method'] = 'chi2'
    chi2_results.append({'k': k, 'metrics': m, 'selector': selector,
                          'X_te_sel': X_te_sel})
    print(f"  {k:>6} {m['f1']:>9.4f} {m['roc_auc']:>9.4f} "
          f"{elapsed:>10.2f}s")

best_chi2   = max(chi2_results, key=lambda x: x['metrics']['f1'])
BEST_K_CHI2 = best_chi2['k']
print(f"\n  ✓ Best k (chi2): {BEST_K_CHI2}"
      f"  (Val F1={best_chi2['metrics']['f1']:.4f})")


# ─────────────────────────────────────────────────────────────────
# STEP 3  — Mutual Information Feature Selection
#           Lab 10: MI captures non-linear dependencies
#           Works on any feature type
# ─────────────────────────────────────────────────────────────────
print("\n[Step 3] Mutual information feature selection...")

mi_scores = mutual_info_classif(
    X_tr_norm, y_train,
    random_state=RANDOM_STATE
)
mi_ranks = np.argsort(mi_scores)[::-1]

print(f"\n  Top 20 features by mutual information:")
print(f"  {'Rank':<6} {'Feature':<35} {'MI Score':>10}")
print("  " + "─" * 54)
for rank, idx in enumerate(mi_ranks[:20]):
    print(f"  {rank+1:<6} {char_features[idx]:<35} "
          f"{mi_scores[idx]:>10.4f}")

print(f"\n  K-feature sweep (MI + LogReg, val set):")
print(f"  {'K':>6} {'Val F1':>9} {'Val AUC':>9}")
print("  " + "─" * 28)

mi_results = []
for k in TOP_K_VALUES:
    selector = SelectKBest(
        lambda X, y: (mi_scores, np.zeros_like(mi_scores)), k=k)
    # Use pre-computed MI scores — select top-k indices
    top_k_idx = mi_ranks[:k]
    if sp.issparse(X_train_char):
        X_tr_sel = X_tr_norm[:, top_k_idx]
        X_va_sel = X_va_norm[:, top_k_idx]
        X_te_sel = X_te_norm[:, top_k_idx]
    else:
        X_tr_sel = X_tr_norm[:, top_k_idx]
        X_va_sel = X_va_norm[:, top_k_idx]
        X_te_sel = X_te_norm[:, top_k_idx]

    clf = LogisticRegression(C=1.0, solver='saga', max_iter=1000,
                              random_state=RANDOM_STATE, n_jobs=N_JOBS)
    clf.fit(X_tr_sel, y_train)
    yp  = clf.predict(X_va_sel)
    ypr = clf.predict_proba(X_va_sel)[:, 1]
    m   = evaluate(y_val, yp, ypr, label=f'MI k={k}')
    m['k'] = k; m['method'] = 'MI'
    mi_results.append({'k': k, 'metrics': m,
                        'top_k_idx': top_k_idx,
                        'X_te_sel': X_te_sel})
    print(f"  {k:>6} {m['f1']:>9.4f} {m['roc_auc']:>9.4f}")

best_mi   = max(mi_results, key=lambda x: x['metrics']['f1'])
BEST_K_MI = best_mi['k']
print(f"\n  ✓ Best k (MI): {BEST_K_MI}"
      f"  (Val F1={best_mi['metrics']['f1']:.4f})")


# ─────────────────────────────────────────────────────────────────
# STEP 4  — Per-Language Feature Importance Analysis
#           Which features matter most for each language?
# ─────────────────────────────────────────────────────────────────
print("\n[Step 4] Per-language feature importance (chi2)...")

lang_top_features = {}
print(f"\n  Top 10 features per language (chi2):\n")

for lang in LANGUAGES:
    mask = (lang_train == lang).values
    if mask.sum() < 20:
        continue

    X_lang = X_tr_mm[mask]
    y_lang = y_train[mask]

    if len(np.unique(y_lang)) < 2:
        continue

    try:
        scores, _ = chi2(X_lang, y_lang)
        top_idx   = scores.argsort()[::-1][:10]
        lang_top_features[lang] = list(zip(
            char_features[top_idx],
            scores[top_idx].round(2)
        ))
        print(f"  [{lang}]")
        for i, (feat, score) in enumerate(lang_top_features[lang]):
            print(f"    {i+1:>2}. {feat:<30}  chi2={score:.2f}")
        print()
    except Exception as e:
        print(f"  [{lang}] Error: {e}")


# ─────────────────────────────────────────────────────────────────
# STEP 5  — L1 Logistic Regression Feature Selection
#           L1 penalty drives irrelevant weights to zero
#           Lab 10: embedded (model-based) feature selection
# ─────────────────────────────────────────────────────────────────
print("\n[Step 5] L1 LogReg embedded feature selection...")

l1_clf = LogisticRegression(
    penalty='l1', C=1.0, solver='saga',
    max_iter=3000, random_state=RANDOM_STATE, n_jobs=N_JOBS
)
l1_clf.fit(X_tr_norm, y_train)

l1_coef       = l1_clf.coef_[0]
nonzero_mask  = (l1_coef != 0)
n_selected    = nonzero_mask.sum()
print(f"\n  L1 selected {n_selected:,} / {len(l1_coef):,} features "
      f"({n_selected/len(l1_coef)*100:.1f}%)")

top_fake_idx  = l1_coef.argsort()[::-1][:15]
top_real_idx  = l1_coef.argsort()[:15]

print(f"\n  Top 15 features → FAKE  (positive L1 coef):")
print(f"  {'Rank':<5} {'Feature':<35} {'Coef':>8}")
print("  " + "─" * 52)
for i, idx in enumerate(top_fake_idx):
    print(f"  {i+1:<5} {char_features[idx]:<35} "
          f"{l1_coef[idx]:>+8.4f}")

print(f"\n  Top 15 features → REAL  (negative L1 coef):")
print(f"  {'Rank':<5} {'Feature':<35} {'Coef':>8}")
print("  " + "─" * 52)
for i, idx in enumerate(top_real_idx):
    print(f"  {i+1:<5} {char_features[idx]:<35} "
          f"{l1_coef[idx]:>+8.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 6  — Feature Selection Comparison
#           Full features vs chi2-selected vs MI-selected vs L1
# ─────────────────────────────────────────────────────────────────
print("\n[Step 6] Feature selection method comparison (test set)...")

# Best chi2 selector → test
chi2_sel  = best_chi2['selector']
X_te_chi2 = best_chi2['X_te_sel']
clf_chi2  = LogisticRegression(C=1.0, solver='saga', max_iter=1000,
                                random_state=RANDOM_STATE, n_jobs=N_JOBS)
clf_chi2.fit(chi2_sel.transform(X_tv_mm), y_tv)
yp_chi2  = clf_chi2.predict(X_te_chi2)
ypr_chi2 = clf_chi2.predict_proba(X_te_chi2)[:, 1]
m_chi2   = evaluate(y_test, yp_chi2, ypr_chi2,
                     label=f'Chi2 k={BEST_K_CHI2}')

# Best MI selector → test
mi_idx    = best_mi['top_k_idx']
X_tv_mi   = X_tv_norm[:, mi_idx]
X_te_mi   = X_te_norm[:, mi_idx]
clf_mi    = LogisticRegression(C=1.0, solver='saga', max_iter=1000,
                                random_state=RANDOM_STATE, n_jobs=N_JOBS)
clf_mi.fit(X_tv_mi, y_tv)
yp_mi  = clf_mi.predict(X_te_mi)
ypr_mi = clf_mi.predict_proba(X_te_mi)[:, 1]
m_mi   = evaluate(y_test, yp_mi, ypr_mi,
                   label=f'MI k={BEST_K_MI}')

# L1 embedded → test (use non-zero features)
X_tv_l1 = X_tv_norm[:, nonzero_mask]
X_te_l1 = X_te_norm[:, nonzero_mask]
clf_l1  = LogisticRegression(C=1.0, solver='saga', max_iter=1000,
                               random_state=RANDOM_STATE, n_jobs=N_JOBS)
clf_l1.fit(X_tv_l1, y_tv)
yp_l1  = clf_l1.predict(X_te_l1)
ypr_l1 = clf_l1.predict_proba(X_te_l1)[:, 1]
m_l1   = evaluate(y_test, yp_l1, ypr_l1,
                   label=f'L1 ({n_selected} feats)')

# Full features baseline
clf_full = LogisticRegression(C=1.0, solver='saga', max_iter=1000,
                               random_state=RANDOM_STATE, n_jobs=N_JOBS)
clf_full.fit(X_tv_norm, y_tv)
yp_full  = clf_full.predict(X_te_norm)
ypr_full = clf_full.predict_proba(X_te_norm)[:, 1]
m_full   = evaluate(y_test, yp_full, ypr_full,
                     label='Full TF-IDF (10k feats)')

fs_comp = [m_full, m_chi2, m_mi, m_l1]
print(f"\n  {'Method':<30} {'Features':>10} {'F1':>8} "
      f"{'AUC':>8} {'Acc':>8}")
print("  " + "─" * 68)
for m, n_feats in zip(fs_comp,
                       [X_train_char.shape[1], BEST_K_CHI2,
                        BEST_K_MI, n_selected]):
    print(f"  {m['label']:<30} {n_feats:>10,} {m['f1']:>8.4f} "
          f"{m['roc_auc']:>8.4f} {m['accuracy']:>8.4f}")


# ═════════════════════════════════════════════════════════════════
#  PART B — EXPLAINABILITY  (Lab 10)
# ═════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# STEP 7  — SHAP Values  (global + local explanations)
#           Lab 10: understand WHY the model predicts Fake/Real
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  PART B — EXPLAINABILITY")
print("─" * 60)

shap_values_arr   = None
shap_feature_names = None

if SHAP_AVAILABLE:
    print("\n[Step 7] SHAP explanation (LinearExplainer on LogReg)...")

    # Use L1-selected features for SHAP — dense, manageable size
    X_shap_train = X_tv_l1.toarray() if sp.issparse(X_tv_l1) \
                   else X_tv_l1
    X_shap_test  = X_te_l1.toarray()  if sp.issparse(X_te_l1) \
                   else X_te_l1
    shap_feature_names = char_features[nonzero_mask]

    # LinearExplainer — exact for linear models, fast
    explainer  = shap.LinearExplainer(
        clf_l1, X_shap_train,
        feature_perturbation='interventional'
    )

    # Explain a sample of test instances
    sample_n   = min(200, len(X_shap_test))
    rng        = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_shap_test), sample_n, replace=False)
    X_shap_sample = X_shap_test[sample_idx]

    shap_values_arr = explainer.shap_values(X_shap_sample)
    print(f"  ✓ SHAP values computed for {sample_n} test samples")
    print(f"    Shape: {shap_values_arr.shape}")

    # Global: mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
    top_shap_idx  = mean_abs_shap.argsort()[::-1][:20]

    print(f"\n  Top 20 features by mean |SHAP|:")
    print(f"  {'Rank':<5} {'Feature':<35} {'Mean |SHAP|':>12}")
    print("  " + "─" * 55)
    for i, idx in enumerate(top_shap_idx):
        print(f"  {i+1:<5} {shap_feature_names[idx]:<35} "
              f"{mean_abs_shap[idx]:>12.4f}")

    # Save SHAP values
    np.save(os.path.join(OUTPUT_DIR, "shap_values.npy"), shap_values_arr)
    np.save(os.path.join(OUTPUT_DIR, "shap_sample_idx.npy"), sample_idx)
    with open(os.path.join(OUTPUT_DIR, "shap_feature_names.pkl"), 'wb') as f:
        pickle.dump(shap_feature_names, f)
    print(f"  ✓ SHAP values saved")

else:
    print("\n[Step 7] SHAP — skipped (not installed).")
    print("  Install with: pip install shap")


# ─────────────────────────────────────────────────────────────────
# STEP 8  — LIME Explanations  (local, instance-level)
#           Lab 10: explain individual predictions
# ─────────────────────────────────────────────────────────────────
lime_explanations = []

if LIME_AVAILABLE and len(test_texts) > 0:
    print("\n[Step 8] LIME explanations (local instance explanations)...")

    # LIME needs a text-based classifier pipeline
    from sklearn.pipeline import Pipeline as SKPipeline
    lime_pipeline = SKPipeline([
        ('tfidf', tfidf_char),
        ('norm',  # can't put normalize in Pipeline directly — use a custom step
         type('Normalizer', (), {
             'fit': lambda self, X, y=None: self,
             'transform': lambda self, X: normalize(X, norm='l2'),
             'fit_transform': lambda self, X, y=None:
                 normalize(X, norm='l2'),
         })()
         ),
    ])

    # Simpler: re-fit a LogReg pipeline directly on raw text
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    # Rebuild a word-level TF-IDF + LogReg for LIME (text-compatible)
    lime_clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2,
            analyzer='char_wb'
        )),
        ('clf', LogisticRegression(
            C=1.0, solver='saga', max_iter=1000,
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        )),
    ])

    train_texts_all = pd.read_csv(
        os.path.join(P2_DIR, "splits.csv"), encoding='utf-8-sig'
    )
    train_subset = train_texts_all[
        train_texts_all['split'].isin(['train', 'val'])
    ].reset_index(drop=True)

    lime_clf_pipeline.fit(
        train_subset['text_clean'].tolist(),
        train_subset['label'].tolist()
    )

    explainer_lime = LimeTextExplainer(
        class_names=['Real', 'Fake'],
        random_state=RANDOM_STATE
    )

    N_LIME         = 5        # explain 5 test samples
    lime_sample_n  = min(N_LIME, len(test_texts))

    # Pick mix of Fake & Real and correct & incorrect
    y_pred_test_lime = lime_clf_pipeline.predict(test_texts.tolist())
    lime_indices     = []

    for target_label in [0, 1]:
        correct_mask = (y_test.values == target_label) & \
                       (y_pred_test_lime == target_label)
        idxs = np.where(correct_mask)[0]
        if len(idxs) > 0:
            lime_indices.append(idxs[0])

    wrong_mask = (y_test.values != y_pred_test_lime)
    wrong_idxs = np.where(wrong_mask)[0]
    if len(wrong_idxs) > 0:
        lime_indices.append(wrong_idxs[0])

    lime_indices = lime_indices[:lime_sample_n]

    print(f"  Explaining {len(lime_indices)} test samples with LIME...")
    lime_report_rows = []

    for i, idx in enumerate(lime_indices):
        text      = test_texts.iloc[idx]
        true_lbl  = y_test.iloc[idx]
        pred_lbl  = y_pred_test_lime[idx]
        prob      = lime_clf_pipeline.predict_proba([text])[0]

        exp = explainer_lime.explain_instance(
            text,
            lime_clf_pipeline.predict_proba,
            num_features=10,
            num_samples=500,
        )

        top_feats = exp.as_list()
        lime_explanations.append({
            'sample_idx': idx,
            'true_label': 'Fake' if true_lbl == 1 else 'Real',
            'pred_label': 'Fake' if pred_lbl == 1 else 'Real',
            'p_real': round(prob[0], 4),
            'p_fake': round(prob[1], 4),
            'correct': true_lbl == pred_lbl,
            'top_features': top_feats,
        })

        print(f"\n  Sample {i+1} (idx={idx}):")
        print(f"  True: {'Fake' if true_lbl==1 else 'Real'}  "
              f"Pred: {'Fake' if pred_lbl==1 else 'Real'}  "
              f"P(Fake)={prob[1]:.4f}  "
              f"{'✓ Correct' if true_lbl==pred_lbl else '✗ Wrong'}")
        print(f"  Text (first 100 chars): {str(text)[:100]}...")
        print(f"  Top LIME features:")
        for feat, weight in top_feats[:5]:
            direction = '→ Fake' if weight > 0 else '→ Real'
            print(f"    '{feat}'  weight={weight:+.4f}  {direction}")

        # Save individual LIME explanation as HTML
        lime_html_path = os.path.join(
            OUTPUT_DIR, f"lime_sample_{i+1}.html")
        exp.save_to_file(lime_html_path)

    lime_df = pd.DataFrame([{
        'sample_idx': r['sample_idx'],
        'true_label': r['true_label'],
        'pred_label': r['pred_label'],
        'p_fake': r['p_fake'],
        'correct': r['correct'],
        'top_feature_1': r['top_features'][0][0]
                         if r['top_features'] else '',
        'weight_1': r['top_features'][0][1]
                    if r['top_features'] else 0,
    } for r in lime_explanations])
    lime_df.to_csv(os.path.join(OUTPUT_DIR, "lime_explanations.csv"),
                   index=False)
    print(f"\n  ✓ LIME HTML reports saved to {OUTPUT_DIR}/")

else:
    if not LIME_AVAILABLE:
        print("\n[Step 8] LIME — skipped (not installed).")
        print("  Install with: pip install lime")
    else:
        print("\n[Step 8] LIME — skipped (no test texts available).")


# ─────────────────────────────────────────────────────────────────
# STEP 9  — Language Bias Analysis
#           Does the model rely on language-specific artifacts
#           rather than semantic fake-news signals?
# ─────────────────────────────────────────────────────────────────
print("\n[Step 9] Language bias analysis...")

# Compare: feature overlap between per-language top chi2 features
print("\n  Feature overlap between languages (top 100 chi2 features):")
lang_top100 = {}
for lang in LANGUAGES:
    mask = (lang_train == lang).values
    if mask.sum() < 20:
        continue
    X_lang = X_tr_mm[mask]
    y_lang = y_train[mask]
    if len(np.unique(y_lang)) < 2:
        continue
    try:
        scores, _ = chi2(X_lang, y_lang)
        top100     = set(char_features[scores.argsort()[::-1][:100]])
        lang_top100[lang] = top100
    except Exception:
        pass

if len(lang_top100) > 1:
    print(f"\n  {'':>12}", end='')
    for lang in lang_top100:
        print(f"  {lang:>12}", end='')
    print()
    for lang_i in lang_top100:
        print(f"  {lang_i:>12}", end='')
        for lang_j in lang_top100:
            overlap = len(lang_top100[lang_i] & lang_top100[lang_j])
            print(f"  {overlap:>12}", end='')
        print()
    print("\n  (Diagonal=100; off-diagonal = shared features)")
    print("  High off-diagonal overlap → cross-lingual signal")
    print("  Low off-diagonal overlap  → language-specific artifacts")

# Error rate analysis: is one language significantly harder?
print(f"\n  Per-language error breakdown (full LogReg, test set):")
print(f"  {'Language':<12} {'N':>5} {'Acc':>8} {'F1':>8} "
      f"{'FPR':>8} {'FNR':>8}")
print("  " + "─" * 54)

bias_rows = []
for lang in LANGUAGES:
    mask = (lang_test == lang).values
    if mask.sum() == 0:
        continue
    yp  = clf_full.predict(X_te_norm[mask])
    yt  = y_test[mask]
    acc = accuracy_score(yt, yp)
    f1  = f1_score(yt, yp, zero_division=0)
    # FPR / FNR
    from sklearn.metrics import confusion_matrix as cm_fn
    try:
        tn, fp, fn, tp = cm_fn(yt, yp).ravel()
        fpr = fp / (fp + tn + 1e-10)
        fnr = fn / (fn + tp + 1e-10)
    except Exception:
        fpr = fnr = float('nan')

    bias_rows.append({'language': lang, 'n': mask.sum(),
                       'accuracy': acc, 'f1': f1,
                       'fpr': fpr, 'fnr': fnr})
    print(f"  {lang:<12} {mask.sum():>5} {acc:>8.4f} "
          f"{f1:>8.4f} {fpr:>8.4f} {fnr:>8.4f}")

bias_df = pd.DataFrame(bias_rows)


# ─────────────────────────────────────────────────────────────────
# STEP 10 — Save Artefacts
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 10] Saving artefacts...")

pd.DataFrame([m for m in [m_full, m_chi2, m_mi, m_l1]]).to_csv(
    os.path.join(OUTPUT_DIR, "feature_selection_comparison.csv"),
    index=False)
pd.DataFrame([r['metrics'] for r in chi2_results]).to_csv(
    os.path.join(OUTPUT_DIR, "chi2_k_sweep.csv"), index=False)
pd.DataFrame([r['metrics'] for r in mi_results]).to_csv(
    os.path.join(OUTPUT_DIR, "mi_k_sweep.csv"), index=False)
bias_df.to_csv(os.path.join(OUTPUT_DIR, "language_bias.csv"), index=False)

# Feature importance master table
feat_table = pd.DataFrame({
    'feature':   char_features,
    'chi2_score': chi2_scores,
    'mi_score':   mi_scores,
    'l1_coef':    l1_coef,
})
feat_table['chi2_rank'] = feat_table['chi2_score'].rank(
    ascending=False).astype(int)
feat_table['mi_rank']   = feat_table['mi_score'].rank(
    ascending=False).astype(int)
feat_table.sort_values('chi2_rank').to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance_table.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "clf_full.pkl"),   'wb') as f:
    pickle.dump(clf_full, f)
with open(os.path.join(OUTPUT_DIR, "clf_l1.pkl"),     'wb') as f:
    pickle.dump(clf_l1, f)

print(f"  ✓ Artefacts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────
# STEP 11 — Visualisations
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 11] Generating plots...")

fig = plt.figure(figsize=(18, 18))
fig.suptitle("Phase 10 — Feature Selection & Explainability",
             fontsize=15, fontweight='bold')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.40)

COLORS = ['#4C72B0','#C44E52','#55A868','#DD8452','#9467BD']

# ── Plot 1: Chi2 k vs F1 ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
chi2_ks  = [r['k']           for r in chi2_results]
chi2_f1s = [r['metrics']['f1'] for r in chi2_results]
mi_ks    = [r['k']             for r in mi_results]
mi_f1s   = [r['metrics']['f1'] for r in mi_results]
ax1.semilogx(chi2_ks, chi2_f1s, 'o-', color='#4C72B0',
             linewidth=2, markersize=7, label='Chi-squared')
ax1.semilogx(mi_ks,   mi_f1s,   's-', color='#C44E52',
             linewidth=2, markersize=7, label='Mutual Info')
ax1.axhline(m_full['f1'], color='grey', linestyle='--',
            linewidth=1.5, label=f'Full (F1={m_full["f1"]:.3f})')
ax1.set_xlabel("k features (log scale)")
ax1.set_ylabel("Validation F1")
ax1.set_title("Feature selection: k vs F1")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

# ── Plot 2: Feature selection method comparison ──────────────────
ax2 = fig.add_subplot(gs[0, 1])
methods   = [m['label'].split(' (')[0][:18] for m in fs_comp]
f1_vals   = [m['f1']       for m in fs_comp]
auc_vals  = [m['roc_auc']  for m in fs_comp]
n_feats   = [X_train_char.shape[1], BEST_K_CHI2, BEST_K_MI, n_selected]
x = np.arange(len(methods)); w = 0.35
ax2.bar(x - w/2, f1_vals,  w, color='#4C72B0', alpha=0.85, label='F1')
ax2.bar(x + w/2, auc_vals, w, color='#C44E52', alpha=0.85, label='AUC')
ax2.set_xticks(x)
ax2.set_xticklabels(methods, rotation=15, ha='right', fontsize=8)
ax2.set_ylim(max(0, min(f1_vals) - 0.05), 1.05)
ax2.set_ylabel("Score"); ax2.set_title("Feature selection comparison")
ax2.legend(fontsize=8)
for i, (f, n) in enumerate(zip(f1_vals, n_feats)):
    ax2.text(i - w/2, f + 0.005, f'{f:.3f}', ha='center', fontsize=7)
    ax2.text(i + w/2, auc_vals[i] + 0.005,
             f'{auc_vals[i]:.3f}', ha='center', fontsize=7)

# ── Plot 3: Top chi2 features (overall) ─────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
top20_idx = chi2_ranks[:20]
top20_f   = [char_features[i][:25] for i in top20_idx]
top20_s   = chi2_scores[top20_idx]
ax3.barh(range(20), top20_s[::-1], color='#55A868', alpha=0.8)
ax3.set_yticks(range(20))
ax3.set_yticklabels(top20_f[::-1], fontsize=7)
ax3.set_xlabel("Chi2 score")
ax3.set_title("Top 20 features — Chi-squared")

# ── Plot 4: L1 top Fake features ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
top15_fake = top_fake_idx[:15]
ax4.barh(range(15), l1_coef[top15_fake][::-1],
         color='#C44E52', alpha=0.85)
ax4.set_yticks(range(15))
ax4.set_yticklabels([char_features[i][:25]
                     for i in top15_fake[::-1]], fontsize=7)
ax4.axvline(0, color='black', linewidth=0.8)
ax4.set_xlabel("L1 coefficient")
ax4.set_title("Top 15 features → FAKE (L1 coef)")

# ── Plot 5: L1 top Real features ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
top15_real = top_real_idx[:15]
ax5.barh(range(15), l1_coef[top15_real],
         color='#4C72B0', alpha=0.85)
ax5.set_yticks(range(15))
ax5.set_yticklabels([char_features[i][:25]
                     for i in top15_real], fontsize=7)
ax5.axvline(0, color='black', linewidth=0.8)
ax5.set_xlabel("L1 coefficient")
ax5.set_title("Top 15 features → REAL (L1 coef)")

# ── Plot 6: Chi2 vs MI score correlation ────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
sample_mask = chi2_scores > np.percentile(chi2_scores, 80)
ax6.scatter(chi2_scores[sample_mask],
            mi_scores[sample_mask],
            alpha=0.3, s=10, color='#4C72B0')
corr_val = np.corrcoef(chi2_scores, mi_scores)[0, 1]
ax6.set_xlabel("Chi2 score")
ax6.set_ylabel("Mutual Information score")
ax6.set_title(f"Chi2 vs MI correlation\n(r={corr_val:.3f}, top 20% feats)")
ax6.grid(True, alpha=0.25)

# ── Plot 7: SHAP summary (if available) ─────────────────────────
ax7 = fig.add_subplot(gs[2, 0:2])
if SHAP_AVAILABLE and shap_values_arr is not None:
    mean_abs = np.abs(shap_values_arr).mean(axis=0)
    top20    = mean_abs.argsort()[::-1][:20]
    colors_shap = ['#C44E52' if l1_coef[nonzero_mask][i] > 0
                   else '#4C72B0' for i in top20]
    ax7.barh(range(20), mean_abs[top20][::-1], color=colors_shap[::-1])
    ax7.set_yticks(range(20))
    ax7.set_yticklabels([shap_feature_names[i][:30]
                         for i in top20[::-1]], fontsize=7)
    ax7.set_xlabel("Mean |SHAP| value")
    ax7.set_title("SHAP feature importance (red=→Fake, blue=→Real)")
    from matplotlib.patches import Patch
    ax7.legend(handles=[
        Patch(color='#C44E52', label='→ Fake'),
        Patch(color='#4C72B0', label='→ Real'),
    ], fontsize=8)
else:
    ax7.text(0.5, 0.5,
             "SHAP not available\n\npip install shap",
             ha='center', va='center',
             transform=ax7.transAxes, fontsize=14,
             bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax7.set_title("SHAP Feature Importance")
    ax7.axis('off')

# ── Plot 8: Language bias — FPR & FNR ───────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
if not bias_df.empty:
    x = np.arange(len(bias_df)); w = 0.35
    ax8.bar(x - w/2, bias_df['fpr'], w, color='#C44E52',
            alpha=0.85, label='FPR (real→fake)')
    ax8.bar(x + w/2, bias_df['fnr'], w, color='#4C72B0',
            alpha=0.85, label='FNR (fake→real)')
    ax8.set_xticks(x)
    ax8.set_xticklabels(bias_df['language'])
    ax8.set_ylabel("Error rate"); ax8.set_title("Language bias: FPR & FNR")
    ax8.legend(fontsize=8)
    for i, row in bias_df.iterrows():
        ax8.text(i-w/2, row['fpr']+0.005,
                 f"{row['fpr']:.3f}", ha='center', fontsize=8)
        ax8.text(i+w/2, row['fnr']+0.005,
                 f"{row['fnr']:.3f}", ha='center', fontsize=8)

# ── Plot 9: Per-language F1 with feature selection ───────────────
ax9 = fig.add_subplot(gs[3, 0:3])
if not bias_df.empty:
    x = np.arange(len(bias_df)); w = 0.2
    labels_meth = ['Full TF-IDF', f'Chi2 k={BEST_K_CHI2}',
                   f'MI k={BEST_K_MI}', f'L1 ({n_selected} feats)']
    models_map  = [
        (clf_full,  X_te_norm,     'L2-norm TF-IDF'),
        (clf_chi2,  X_te_chi2,     'Chi2 selected'),
        (clf_mi,    X_te_mi,       'MI selected'),
        (clf_l1,    X_te_l1,       'L1 selected'),
    ]
    for j, (mdl, X_te_use, lbl) in enumerate(models_map):
        lang_f1s = []
        for lang in bias_df['language']:
            mask = (lang_test == lang).values
            if mask.sum() == 0:
                lang_f1s.append(0); continue
            X_te_use_lang = (X_te_use[mask].toarray()
                             if sp.issparse(X_te_use)
                             else X_te_use[mask])
            yp = mdl.predict(X_te_use_lang)
            lang_f1s.append(
                f1_score(y_test[mask], yp, zero_division=0))
        ax9.bar(x + (j-1.5)*w, lang_f1s, w,
                color=COLORS[j], alpha=0.85, label=lbl)

    ax9.set_xticks(x)
    ax9.set_xticklabels(bias_df['language'].tolist())
    ax9.set_ylim(0, 1.15); ax9.set_ylabel("F1 Score")
    ax9.set_title("Per-language F1: Full vs Feature-selected models")
    ax9.legend(fontsize=8, loc='upper right')

plt.savefig(os.path.join(OUTPUT_DIR, "phase10_explainability.png"),
            dpi=150, bbox_inches='tight')
print(f"  ✓ Saved → {OUTPUT_DIR}/phase10_explainability.png")


# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 10 SUMMARY — Feature Selection & Explainability")
print("=" * 60)
print(f"\n  Feature Selection Results (test set):")
print(f"  {'Method':<30} {'Features':>10} {'F1':>8} {'AUC':>8}")
print("  " + "─" * 60)
for m, n_f in zip(fs_comp, [X_train_char.shape[1], BEST_K_CHI2,
                              BEST_K_MI, n_selected]):
    print(f"  {m['label']:<30} {n_f:>10,} "
          f"{m['f1']:>8.4f} {m['roc_auc']:>8.4f}")

print(f"\n  Best chi2 k : {BEST_K_CHI2}")
print(f"  Best MI   k : {BEST_K_MI}")
print(f"  L1 selected : {n_selected:,} features")
print(f"\n  SHAP available : {SHAP_AVAILABLE}")
print(f"  LIME available : {LIME_AVAILABLE}")
if LIME_AVAILABLE:
    print(f"  LIME samples   : {len(lime_explanations)}")
print(f"\nArtefacts in: ./{OUTPUT_DIR}/")
print("  feature_selection_comparison.csv")
print("  chi2_k_sweep.csv      mi_k_sweep.csv")
print("  feature_importance_table.csv")
print("  language_bias.csv")
print("  clf_full.pkl          clf_l1.pkl")
if SHAP_AVAILABLE:
    print("  shap_values.npy       shap_feature_names.pkl")
if LIME_AVAILABLE:
    print("  lime_explanations.csv lime_sample_*.html")
print("  phase10_explainability.png")
print("\n" + "=" * 60)
print("ALL PHASES COMPLETE — Ready to write your report!")
print("=" * 60)
