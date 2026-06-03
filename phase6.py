"""
Phase 6 — Linear Baseline & Logistic Regression
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Lab 5: Linear Regression (applied as Logistic Regression for classification)
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV, SGDClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import time
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
P2_DIR     = "phase2_outputs"
P3_DIR     = "phase3_outputs"
P4_DIR     = "phase4_outputs"
P5_DIR     = "phase5_outputs"
OUTPUT_DIR = "phase6_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES    = ['Hindi', 'Marathi', 'Gujarati', 'Telugu']
RANDOM_STATE = 42
CV_FOLDS     = 5


# ─────────────────────────────────────────────────────────────────
# STEP 1  — Load Artefacts
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 6 — Linear Baseline & Logistic Regression")
print("=" * 60)

def load_npz(d, name): return sp.load_npz(os.path.join(d, name))
def load_csv(d, name): return pd.read_csv(os.path.join(d, name)).squeeze()

# TF-IDF char (primary) and word features
X_train_char = load_npz(P2_DIR, "X_train_tfidf.npz")
X_val_char   = load_npz(P2_DIR, "X_val_tfidf.npz")
X_test_char  = load_npz(P2_DIR, "X_test_tfidf.npz")

X_train_word = load_npz(P2_DIR, "X_train_word.npz")
X_val_word   = load_npz(P2_DIR, "X_val_word.npz")
X_test_word  = load_npz(P2_DIR, "X_test_word.npz")

# SVD dense features
X_train_svd  = np.load(os.path.join(P3_DIR, "X_train_svd.npy"))
X_val_svd    = np.load(os.path.join(P3_DIR, "X_val_svd.npy"))
X_test_svd   = np.load(os.path.join(P3_DIR, "X_test_svd.npy"))

# Distance features
dist_train = np.load(os.path.join(P3_DIR, "dist_train.npy"))
dist_val   = np.load(os.path.join(P3_DIR, "dist_val.npy"))
dist_test  = np.load(os.path.join(P3_DIR, "dist_test.npy"))

# Labels & language
y_train    = load_csv(P5_DIR, "y_train.csv").astype(int)
y_val      = load_csv(P5_DIR, "y_val.csv").astype(int)
y_test     = load_csv(P5_DIR, "y_test.csv").astype(int)
lang_train = load_csv(P5_DIR, "lang_train.csv")
lang_val   = load_csv(P5_DIR, "lang_val.csv")
lang_test  = load_csv(P5_DIR, "lang_test.csv")

# L2-normalised TF-IDF
X_tr_norm = normalize(X_train_char, norm='l2')
X_va_norm = normalize(X_val_char,   norm='l2')
X_te_norm = normalize(X_test_char,  norm='l2')

X_tr_word_norm = normalize(X_train_word, norm='l2')
X_va_word_norm = normalize(X_val_word,   norm='l2')
X_te_word_norm = normalize(X_test_word,  norm='l2')

# Combined train+val for final models
X_trainval_char = sp.vstack([X_train_char, X_val_char])
X_trainval_norm = normalize(X_trainval_char, norm='l2')
X_trainval_word = normalize(sp.vstack([X_train_word, X_val_word]), norm='l2')
y_trainval      = pd.concat([y_train, y_val]).reset_index(drop=True)
lang_trainval   = pd.concat([lang_train, lang_val]).reset_index(drop=True)

print(f"\n✓ Artefacts loaded")
print(f"  Char TF-IDF  — {X_train_char.shape[1]:,} features")
print(f"  Word TF-IDF  — {X_train_word.shape[1]:,} features")
print(f"  SVD          — {X_train_svd.shape[1]} components")
print(f"  Train: {len(y_train):,}   Val: {len(y_val):,}   Test: {len(y_test):,}")


# ─────────────────────────────────────────────────────────────────
# HELPER — Evaluate any classifier
# ─────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob=None, label=''):
    m = {
        'label':         label,
        'accuracy':      accuracy_score(y_true, y_pred),
        'precision':     precision_score(y_true, y_pred, zero_division=0),
        'recall':        recall_score(y_true, y_pred, zero_division=0),
        'f1':            f1_score(y_true, y_pred, zero_division=0),
        'roc_auc':       roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan'),
        'avg_precision': average_precision_score(y_true, y_prob) if y_prob is not None else float('nan'),
    }
    return m

def print_eval(m):
    print(f"  Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  "
          f"AUC={m['roc_auc']:.4f}  AP={m['avg_precision']:.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 2  — Regularisation Search
#           L1 (Lasso), L2 (Ridge), ElasticNet  ×  C values
#           Lab 5: equivalent to alpha tuning in linear regression
# ─────────────────────────────────────────────────────────────────
print("\n[Step 2] Regularisation search (val set, char TF-IDF)...")

C_VALUES   = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
PENALTIES  = ['l2', 'l1']     # ElasticNet added separately
SOLVERS    = {'l2': 'saga', 'l1': 'saga'}

reg_results = []
print(f"\n  {'Penalty':<6} {'C':>8} {'Val Acc':>9} {'Val F1':>8} {'Val AUC':>9}")
print("  " + "─" * 46)

for penalty in PENALTIES:
    for C in C_VALUES:
        lr = LogisticRegression(
            penalty=penalty, C=C,
            solver=SOLVERS[penalty],
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        lr.fit(X_tr_norm, y_train)
        y_pred = lr.predict(X_va_norm)
        y_prob = lr.predict_proba(X_va_norm)[:, 1]
        m = evaluate(y_val, y_pred, y_prob,
                     label=f'{penalty.upper()} C={C}')
        m['penalty'] = penalty
        m['C']       = C
        reg_results.append(m)
        print(f"  {penalty.upper():<6} {C:>8.3f} "
              f"{m['accuracy']:>9.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}")

# ElasticNet
for C in [0.1, 1.0, 10.0]:
    lr = LogisticRegression(
        penalty='elasticnet', C=C, l1_ratio=0.5,
        solver='saga', max_iter=2000,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    lr.fit(X_tr_norm, y_train)
    y_pred = lr.predict(X_va_norm)
    y_prob = lr.predict_proba(X_va_norm)[:, 1]
    m = evaluate(y_val, y_pred, y_prob, label=f'ElasticNet C={C}')
    m['penalty'] = 'elasticnet'; m['C'] = C
    reg_results.append(m)
    print(f"  {'EN':<6} {C:>8.3f} "
          f"{m['accuracy']:>9.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}")

reg_df   = pd.DataFrame(reg_results)
best_reg = reg_df.loc[reg_df['f1'].idxmax()]
BEST_C   = best_reg['C']
BEST_PEN = best_reg['penalty']
print(f"\n  ✓ Best: penalty={BEST_PEN.upper()}  C={BEST_C}"
      f"  (Val F1={best_reg['f1']:.4f})")


# ─────────────────────────────────────────────────────────────────
# STEP 3  — Feature Set Comparison
# ─────────────────────────────────────────────────────────────────
print("\n[Step 3] Feature set comparison (best regularisation)...")

def make_lr(penalty=BEST_PEN, C=BEST_C):
    kwargs = dict(penalty=penalty, C=C, solver='saga',
                  max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1)
    if penalty == 'elasticnet':
        kwargs['l1_ratio'] = 0.5
    return LogisticRegression(**kwargs)

FEATURE_SETS = {
    'Char TF-IDF (L2-norm)':  (X_tr_norm,      X_va_norm,      X_te_norm),
    'Word TF-IDF (L2-norm)':  (X_tr_word_norm,  X_va_word_norm, X_te_word_norm),
    'SVD-50':                  (X_train_svd,     X_val_svd,      X_test_svd),
    'Distance features':       (dist_train,      dist_val,       dist_test),
}

fs_results = []
for fs_name, (X_tr, X_va, X_te) in FEATURE_SETS.items():
    lr = make_lr()
    lr.fit(X_tr, y_train)
    y_pred = lr.predict(X_va)
    y_prob = lr.predict_proba(X_va)[:, 1]
    m = evaluate(y_val, y_pred, y_prob, label=fs_name)
    fs_results.append(m)
    print(f"  {fs_name:<35}")
    print_eval(m)

best_fs = max(fs_results, key=lambda x: x['f1'])['label']
print(f"\n  ✓ Best feature set: {best_fs}")


# ─────────────────────────────────────────────────────────────────
# STEP 4  — Stratified k-Fold Cross-Validation  (on train set)
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 4] {CV_FOLDS}-fold stratified cross-validation...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                     random_state=RANDOM_STATE)
lr_cv = make_lr()
cv_results = cross_validate(
    lr_cv, X_train_svd, y_train, cv=cv,          # SVD-50 instead of full TF-IDF
    scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
    n_jobs=1,                                      # no parallel workers on Windows
)

print(f"\n  {CV_FOLDS}-Fold CV Results on Training Set:")
print(f"  {'Metric':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("  " + "─" * 50)
for key in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
    scores = cv_results[f'test_{key}']
    print(f"  {key:<15} {scores.mean():>8.4f} {scores.std():>8.4f} "
          f"{scores.min():>8.4f} {scores.max():>8.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 5  — Per-Language Logistic Regression  (monolingual)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 5] Per-language monolingual Logistic Regression...")

lang_results  = []
lang_lr_models = {}

for lang in LANGUAGES:
    mask_tr = (lang_train == lang).values
    mask_te = (lang_test  == lang).values
    mask_va = (lang_val   == lang).values

    if mask_tr.sum() < 20:
        print(f"  [{lang}] Not enough samples — skipping.")
        continue

    lr_lang = make_lr()
    lr_lang.fit(X_tr_norm[mask_tr], y_train[mask_tr])
    lang_lr_models[lang] = lr_lang

    # Validation
    y_pred_va = lr_lang.predict(X_va_norm[mask_va])
    y_prob_va = lr_lang.predict_proba(X_va_norm[mask_va])[:, 1]
    m_va = evaluate(y_val[mask_va], y_pred_va, y_prob_va,
                    label=f'{lang} val')

    # Test
    y_pred_te = lr_lang.predict(X_te_norm[mask_te])
    y_prob_te = lr_lang.predict_proba(X_te_norm[mask_te])[:, 1]
    m_te = evaluate(y_test[mask_te], y_pred_te, y_prob_te,
                    label=f'{lang} test')

    m_va['language'] = m_te['language'] = lang
    m_va['split']    = 'val'
    m_te['split']    = 'test'
    lang_results.extend([m_va, m_te])

    print(f"  [{lang}]  Val  F1={m_va['f1']:.4f}  AUC={m_va['roc_auc']:.4f}  |  "
          f"Test F1={m_te['f1']:.4f}  AUC={m_te['roc_auc']:.4f}")

lang_lr_df = pd.DataFrame(lang_results)


# ─────────────────────────────────────────────────────────────────
# STEP 6  — Top Predictive Features per Class  (L1/L2 weights)
#           Lab 5: inspect learned coefficients like linear model
# ─────────────────────────────────────────────────────────────────
print("\n[Step 6] Top coefficient features per class...")

lr_for_coef = LogisticRegression(
    penalty='l1', C=BEST_C, solver='saga',
    max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1
)
lr_for_coef.fit(X_tr_norm, y_train)

with open(os.path.join(P2_DIR, "tfidf_combined.pkl"), 'rb') as f:
    tfidf_vocab = pickle.load(f)

feature_names = np.array(tfidf_vocab.get_feature_names_out())
coef          = lr_for_coef.coef_[0]
TOP_N         = 15

top_fake_idx = coef.argsort()[::-1][:TOP_N]   # most positive → predicts fake
top_real_idx = coef.argsort()[:TOP_N]          # most negative → predicts real

print(f"\n  Top {TOP_N} features → FAKE (positive coef):")
for i, idx in enumerate(top_fake_idx):
    print(f"  {i+1:>3}. {feature_names[idx]:<30}  coef={coef[idx]:+.4f}")

print(f"\n  Top {TOP_N} features → REAL (negative coef):")
for i, idx in enumerate(top_real_idx):
    print(f"  {i+1:>3}. {feature_names[idx]:<30}  coef={coef[idx]:+.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 7  — Final Model: Best Config on TEST Set
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 7] Final evaluation on held-out TEST set...")
print(f"  Config: penalty={BEST_PEN.upper()}  C={BEST_C}\n")

final_lr = make_lr()
final_lr.fit(X_trainval_norm, y_trainval)

y_pred_test = final_lr.predict(X_te_norm)
y_prob_test = final_lr.predict_proba(X_te_norm)[:, 1]
final_m     = evaluate(y_test, y_pred_test, y_prob_test,
                       label=f'LogReg {BEST_PEN.upper()} C={BEST_C}')

print(f"  Overall Test Results:")
print_eval(final_m)

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_test,
      target_names=['Real (0)', 'Fake (1)'], digits=4))

cm = confusion_matrix(y_test, y_pred_test)
print(f"  Confusion Matrix:")
print(f"                Predicted Real  Predicted Fake")
print(f"  Actual Real   {cm[0,0]:>12}  {cm[0,1]:>14}")
print(f"  Actual Fake   {cm[1,0]:>12}  {cm[1,1]:>14}")

print(f"\n  Per-language test results:")
print(f"  {'Language':<12} {'N':>5} {'Acc':>8} {'F1':>8} {'AUC':>8}")
print("  " + "─" * 44)
for lang in LANGUAGES:
    mask = (lang_test == lang).values
    if mask.sum() == 0:
        continue
    yp   = final_lr.predict(X_te_norm[mask])
    yprb = final_lr.predict_proba(X_te_norm[mask])[:, 1]
    m    = evaluate(y_test[mask], yp, yprb)
    print(f"  {lang:<12} {mask.sum():>5} "
          f"{m['accuracy']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>8.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 8  — Save Artefacts
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 8] Saving artefacts...")

with open(os.path.join(OUTPUT_DIR, "lr_final.pkl"),        'wb') as f:
    pickle.dump(final_lr, f)
with open(os.path.join(OUTPUT_DIR, "lr_lang_models.pkl"),  'wb') as f:
    pickle.dump(lang_lr_models, f)
with open(os.path.join(OUTPUT_DIR, "lr_coef_model.pkl"),   'wb') as f:
    pickle.dump(lr_for_coef, f)

reg_df.to_csv(os.path.join(OUTPUT_DIR, "regularisation_search.csv"),  index=False)
pd.DataFrame(fs_results).to_csv(os.path.join(OUTPUT_DIR, "feature_set_results.csv"), index=False)
lang_lr_df.to_csv(os.path.join(OUTPUT_DIR, "per_language_results.csv"), index=False)

cv_df = pd.DataFrame({k.replace('test_',''):v
                      for k,v in cv_results.items() if k.startswith('test_')})
cv_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)

config = {
    'best_C': BEST_C, 'best_penalty': BEST_PEN,
    'best_feature_set': best_fs,
    'test_f1': round(final_m['f1'], 4),
    'test_acc': round(final_m['accuracy'], 4),
    'test_auc': round(final_m['roc_auc'], 4),
}
with open(os.path.join(OUTPUT_DIR, "lr_config.pkl"), 'wb') as f:
    pickle.dump(config, f)

for fname in ["y_train.csv", "y_val.csv", "y_test.csv",
              "lang_train.csv", "lang_val.csv", "lang_test.csv"]:
    pd.read_csv(os.path.join(P5_DIR, fname)).to_csv(
        os.path.join(OUTPUT_DIR, fname), index=False)

print(f"  ✓ Artefacts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────
# STEP 9  — Visualisations
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 9] Generating plots...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Phase 6 — Logistic Regression Results", fontsize=15, fontweight='bold')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)
COLORS = ['#4C72B0', '#C44E52', '#55A868', '#DD8452', '#9467BD']

# ── Plot 1: C vs F1 for L1 and L2 ───────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for pen, color in [('l2', '#4C72B0'), ('l1', '#C44E52')]:
    sub = reg_df[reg_df['penalty'] == pen].sort_values('C')
    ax1.semilogx(sub['C'], sub['f1'], 'o-', color=color,
                 linewidth=2, markersize=6, label=pen.upper())
ax1.axvline(BEST_C, color='grey', linestyle='--',
            linewidth=1.5, label=f'Best C={BEST_C}')
ax1.set_xlabel("C (inverse regularisation strength)")
ax1.set_ylabel("Validation F1")
ax1.set_title("Regularisation strength vs F1")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

# ── Plot 2: C vs AUC for L1 and L2 ──────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for pen, color in [('l2', '#4C72B0'), ('l1', '#C44E52')]:
    sub = reg_df[reg_df['penalty'] == pen].sort_values('C')
    ax2.semilogx(sub['C'], sub['roc_auc'], 'o-', color=color,
                 linewidth=2, markersize=6, label=pen.upper())
ax2.axvline(BEST_C, color='grey', linestyle='--', linewidth=1.5)
ax2.set_xlabel("C"); ax2.set_ylabel("Validation AUC")
ax2.set_title("Regularisation strength vs AUC")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

# ── Plot 3: Feature set F1 comparison ───────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
fs_names  = [r['label'] for r in fs_results]
fs_f1     = [r['f1']    for r in fs_results]
fs_auc    = [r['roc_auc'] for r in fs_results]
x = np.arange(len(fs_names)); w = 0.35
ax3.bar(x - w/2, fs_f1,  w, color='#4C72B0', alpha=0.85, label='F1')
ax3.bar(x + w/2, fs_auc, w, color='#55A868', alpha=0.85, label='AUC')
ax3.set_xticks(x)
short = [n.split('(')[0].strip()[:18] for n in fs_names]
ax3.set_xticklabels(short, rotation=20, ha='right', fontsize=8)
ax3.set_ylim(max(0, min(fs_f1+fs_auc) - 0.05), 1.05)
ax3.set_title("Feature set comparison")
ax3.set_ylabel("Score"); ax3.legend(fontsize=8)
for i, (f, a) in enumerate(zip(fs_f1, fs_auc)):
    ax3.text(i-w/2, f+0.005, f'{f:.3f}', ha='center', fontsize=7)
    ax3.text(i+w/2, a+0.005, f'{a:.3f}', ha='center', fontsize=7)

# ── Plot 4: CV fold F1 distribution ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
cv_f1_scores = cv_results['test_f1']
ax4.bar(range(1, CV_FOLDS+1), cv_f1_scores,
        color='#4C72B0', alpha=0.8, edgecolor='white')
ax4.axhline(cv_f1_scores.mean(), color='#C44E52', linestyle='--',
            linewidth=2, label=f'Mean={cv_f1_scores.mean():.4f}')
ax4.fill_between(
    [0.4, CV_FOLDS+0.6],
    cv_f1_scores.mean() - cv_f1_scores.std(),
    cv_f1_scores.mean() + cv_f1_scores.std(),
    alpha=0.15, color='#C44E52', label=f'±1 std={cv_f1_scores.std():.4f}'
)
ax4.set_xlabel("Fold"); ax4.set_ylabel("F1 Score")
ax4.set_title(f"{CV_FOLDS}-Fold CV F1 Distribution")
ax4.set_xticks(range(1, CV_FOLDS+1))
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.25, axis='y')

# ── Plot 5: ROC curve ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
fpr_c, tpr_c, _ = roc_curve(y_test, y_prob_test)
auc_val = roc_auc_score(y_test, y_prob_test)
ax5.plot(fpr_c, tpr_c, color='#4C72B0', linewidth=2.5,
         label=f'LogReg (AUC={auc_val:.4f})')
ax5.fill_between(fpr_c, tpr_c, alpha=0.1, color='#4C72B0')
ax5.plot([0,1], [0,1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax5.set_xlabel("FPR"); ax5.set_ylabel("TPR")
ax5.set_title("ROC Curve (test set)")
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.25)

# ── Plot 6: Confusion matrix ─────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
im6 = ax6.imshow(cm, cmap='Blues')
ax6.set_title(f"Confusion Matrix — LogReg\n({BEST_PEN.upper()}, C={BEST_C})")
ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual")
ax6.set_xticks([0,1]); ax6.set_xticklabels(['Real','Fake'])
ax6.set_yticks([0,1]); ax6.set_yticklabels(['Real','Fake'])
for i in range(2):
    for j in range(2):
        ax6.text(j, i, f'{cm[i,j]}\n({cm[i,j]/cm.sum()*100:.1f}%)',
                 ha='center', va='center', fontsize=12,
                 color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im6, ax=ax6)

# ── Plot 7: Top coefficients — Fake ─────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
coef     = lr_for_coef.coef_[0]
top_idx  = coef.argsort()[::-1][:12]
ax7.barh(range(12), coef[top_idx][::-1], color='#C44E52', alpha=0.8)
ax7.set_yticks(range(12))
ax7.set_yticklabels([feature_names[i][:25] for i in top_idx[::-1]], fontsize=8)
ax7.set_xlabel("Coefficient value")
ax7.set_title("Top 12 features → FAKE")
ax7.axvline(0, color='black', linewidth=0.8)

# ── Plot 8: Top coefficients — Real ─────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
bot_idx  = coef.argsort()[:12]
ax8.barh(range(12), coef[bot_idx], color='#4C72B0', alpha=0.8)
ax8.set_yticks(range(12))
ax8.set_yticklabels([feature_names[i][:25] for i in bot_idx], fontsize=8)
ax8.set_xlabel("Coefficient value")
ax8.set_title("Top 12 features → REAL")
ax8.axvline(0, color='black', linewidth=0.8)

# ── Plot 9: Per-language test F1 ────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
if not lang_lr_df.empty:
    te_df  = lang_lr_df[lang_lr_df['split'] == 'test']
    val_df = lang_lr_df[lang_lr_df['split'] == 'val']
    langs_p = te_df['language'].tolist()
    x = np.arange(len(langs_p)); w = 0.35
    ax9.bar(x - w/2, val_df['f1'].values, w, color='#DD8452',
            alpha=0.85, label='Val F1')
    ax9.bar(x + w/2, te_df['f1'].values,  w, color='#4C72B0',
            alpha=0.85, label='Test F1')
    ax9.set_xticks(x); ax9.set_xticklabels(langs_p)
    ax9.set_ylim(0, 1.1); ax9.set_ylabel("F1 Score")
    ax9.set_title("Per-language F1 — monolingual LogReg")
    ax9.legend(fontsize=8)
    for i, (v, t) in enumerate(zip(val_df['f1'].values, te_df['f1'].values)):
        ax9.text(i-w/2, v+0.02, f'{v:.2f}', ha='center', fontsize=8)
        ax9.text(i+w/2, t+0.02, f'{t:.2f}', ha='center', fontsize=8)

plt.savefig(os.path.join(OUTPUT_DIR, "phase6_logreg_results.png"),
            dpi=150, bbox_inches='tight')
print(f"  ✓ Saved → {OUTPUT_DIR}/phase6_logreg_results.png")


# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6 SUMMARY — Test Set Performance")
print("=" * 60)
print(f"  Penalty      : {BEST_PEN.upper()}")
print(f"  Best C       : {BEST_C}")
print(f"  Feature set  : {best_fs}")
print(f"  Accuracy     : {final_m['accuracy']:.4f}")
print(f"  Precision    : {final_m['precision']:.4f}")
print(f"  Recall       : {final_m['recall']:.4f}")
print(f"  F1           : {final_m['f1']:.4f}")
print(f"  ROC-AUC      : {final_m['roc_auc']:.4f}")
print(f"  Avg Prec     : {final_m['avg_precision']:.4f}")
print(f"\nArtefacts in: ./{OUTPUT_DIR}/")
print("  lr_final.pkl               lr_lang_models.pkl")
print("  lr_coef_model.pkl          lr_config.pkl")
print("  regularisation_search.csv  feature_set_results.csv")
print("  per_language_results.csv   cv_results.csv")
print("  phase6_logreg_results.png")
print("\nReady for Phase 7 — Decision Tree & SVM")
print("=" * 60)
