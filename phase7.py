"""
Phase 7 — Decision Tree & SVM
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Lab 6: Decision Tree, SVM
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
import time
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
P2_DIR     = "phase2_outputs"
P3_DIR     = "phase3_outputs"
P6_DIR     = "phase6_outputs"
OUTPUT_DIR = "phase7_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES    = ['Hindi', 'Marathi', 'Gujarati', 'Telugu']
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────
# STEP 1  — Load Artefacts
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 7 — Decision Tree & SVM")
print("=" * 60)

def load_npz(d, n): return sp.load_npz(os.path.join(d, n))
def load_npy(d, n): return np.load(os.path.join(d, n))
def load_csv(d, n): return pd.read_csv(os.path.join(d, n)).squeeze()

# TF-IDF char features
X_train_char = load_npz(P2_DIR, "X_train_tfidf.npz")
X_val_char   = load_npz(P2_DIR, "X_val_tfidf.npz")
X_test_char  = load_npz(P2_DIR, "X_test_tfidf.npz")

# SVD dense (Decision Tree needs dense; SVM works on both)
X_train_svd = load_npy(P3_DIR, "X_train_svd.npy")
X_val_svd   = load_npy(P3_DIR, "X_val_svd.npy")
X_test_svd  = load_npy(P3_DIR, "X_test_svd.npy")

# Distance features
dist_train = load_npy(P3_DIR, "dist_train.npy")
dist_val   = load_npy(P3_DIR, "dist_val.npy")
dist_test  = load_npy(P3_DIR, "dist_test.npy")

# Labels & language
y_train    = load_csv(P6_DIR, "y_train.csv").astype(int)
y_val      = load_csv(P6_DIR, "y_val.csv").astype(int)
y_test     = load_csv(P6_DIR, "y_test.csv").astype(int)
lang_train = load_csv(P6_DIR, "lang_train.csv")
lang_val   = load_csv(P6_DIR, "lang_val.csv")
lang_test  = load_csv(P6_DIR, "lang_test.csv")

# L2-normalised TF-IDF for SVM
X_tr_norm = normalize(X_train_char, norm='l2')
X_va_norm = normalize(X_val_char,   norm='l2')
X_te_norm = normalize(X_test_char,  norm='l2')

# Combined train+val for final models
X_tv_svd  = np.vstack([X_train_svd, X_val_svd])
X_tv_norm = normalize(sp.vstack([X_train_char, X_val_char]), norm='l2')
y_tv      = pd.concat([y_train, y_val]).reset_index(drop=True)
lang_tv   = pd.concat([lang_train, lang_val]).reset_index(drop=True)

# Combined SVD + distance features for Decision Tree
X_train_dt = np.hstack([X_train_svd, dist_train])
X_val_dt   = np.hstack([X_val_svd,   dist_val])
X_test_dt  = np.hstack([X_test_svd,  dist_test])
X_tv_dt    = np.hstack([X_tv_svd, np.vstack([dist_train, dist_val])])

DT_FEAT_NAMES = (
    [f'SVD_{i}' for i in range(X_train_svd.shape[1])] +
    ['cos_sim_fake', 'cos_sim_real', 'euc_dist_fake', 'euc_dist_real',
     'man_dist_fake', 'man_dist_real', 'cos_diff']
)

print(f"\n✓ Artefacts loaded")
print(f"  TF-IDF : {X_train_char.shape}  SVD: {X_train_svd.shape}  DT input: {X_train_dt.shape}")
print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")


# ─────────────────────────────────────────────────────────────────
# HELPER — evaluate any classifier
# ─────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob=None, label=''):
    return {
        'label':         label,
        'accuracy':      accuracy_score(y_true, y_pred),
        'precision':     precision_score(y_true, y_pred, zero_division=0),
        'recall':        recall_score(y_true, y_pred, zero_division=0),
        'f1':            f1_score(y_true, y_pred, zero_division=0),
        'roc_auc':       roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan'),
        'avg_precision': average_precision_score(y_true, y_prob) if y_prob is not None else float('nan'),
    }

def print_eval(m):
    print(f"  Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  "
          f"AUC={m['roc_auc']:.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 2  — Decision Tree: Depth Tuning
#           Lab 6: understand tree complexity vs generalisation
# ─────────────────────────────────────────────────────────────────
print("\n[Step 2] Decision Tree — depth tuning...")

MAX_DEPTHS    = [2, 3, 5, 7, 10, 15, 20, None]
CRITERIONS    = ['gini', 'entropy']
dt_results    = []

print(f"\n  {'Criterion':<10} {'MaxDepth':>9} {'ValAcc':>8} "
      f"{'ValF1':>8} {'ValAUC':>8} {'#Leaves':>8}")
print("  " + "─" * 56)

for criterion in CRITERIONS:
    for max_depth in MAX_DEPTHS:
        dt = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=RANDOM_STATE,
        )
        dt.fit(X_train_dt, y_train)
        y_pred = dt.predict(X_val_dt)
        y_prob = dt.predict_proba(X_val_dt)[:, 1]
        m = evaluate(y_val, y_pred, y_prob,
                     label=f'{criterion} depth={max_depth}')
        m['criterion']  = criterion
        m['max_depth']  = max_depth if max_depth else 999
        m['n_leaves']   = dt.get_n_leaves()
        m['tree_depth'] = dt.get_depth()
        dt_results.append(m)
        depth_str = str(max_depth) if max_depth else 'None'
        print(f"  {criterion:<10} {depth_str:>9} {m['accuracy']:>8.4f} "
              f"{m['f1']:>8.4f} {m['roc_auc']:>8.4f} {dt.get_n_leaves():>8}")

dt_df    = pd.DataFrame(dt_results)
best_dt  = dt_df.loc[dt_df['f1'].idxmax()]
BEST_DT_DEPTH     = None if best_dt['max_depth'] == 999 else int(best_dt['max_depth'])
BEST_DT_CRITERION = best_dt['criterion']
print(f"\n  ✓ Best DT: criterion={BEST_DT_CRITERION}  "
      f"max_depth={BEST_DT_DEPTH}  (Val F1={best_dt['f1']:.4f})")


# ─────────────────────────────────────────────────────────────────
# STEP 3  — Decision Tree: Feature Importance
# ─────────────────────────────────────────────────────────────────
print("\n[Step 3] Decision Tree — feature importance...")

best_dt_model = DecisionTreeClassifier(
    max_depth=BEST_DT_DEPTH,
    criterion=BEST_DT_CRITERION,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=RANDOM_STATE,
)
best_dt_model.fit(X_train_dt, y_train)
importances  = best_dt_model.feature_importances_
feat_imp_df  = pd.DataFrame({
    'feature':    DT_FEAT_NAMES,
    'importance': importances,
}).sort_values('importance', ascending=False)

print(f"\n  Top 15 features by Gini importance:")
print(f"  {'Rank':<5} {'Feature':<25} {'Importance':>12}")
print("  " + "─" * 44)
for i, row in feat_imp_df.head(15).iterrows():
    print(f"  {feat_imp_df.index.get_loc(i)+1:<5} "
          f"{row['feature']:<25} {row['importance']:>12.5f}")


# ─────────────────────────────────────────────────────────────────
# STEP 4  — SVM: Kernel & C Tuning
#           LinearSVC on TF-IDF (fast), calibrated for probabilities
#           Lab 6: SVM decision boundary analysis
# ─────────────────────────────────────────────────────────────────
print("\n[Step 4] SVM — C tuning (LinearSVC on TF-IDF)...")

C_VALUES   = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
svm_results = []

print(f"\n  {'C':>8} {'ValAcc':>9} {'ValF1':>8} {'ValAUC':>9} {'Time(s)':>9}")
print("  " + "─" * 48)

for C in C_VALUES:
    t0  = time.time()
    svm = LinearSVC(C=C, max_iter=5000, random_state=RANDOM_STATE)
    cal = CalibratedClassifierCV(svm, cv=3, method='sigmoid')
    cal.fit(X_tr_norm, y_train)
    y_pred = cal.predict(X_va_norm)
    y_prob = cal.predict_proba(X_va_norm)[:, 1]
    elapsed = time.time() - t0
    m = evaluate(y_val, y_pred, y_prob, label=f'LinearSVC C={C}')
    m['C'] = C
    svm_results.append(m)
    print(f"  {C:>8.3f} {m['accuracy']:>9.4f} {m['f1']:>8.4f} "
          f"{m['roc_auc']:>9.4f} {elapsed:>9.2f}")

svm_df   = pd.DataFrame(svm_results)
best_svm = svm_df.loc[svm_df['f1'].idxmax()]
BEST_SVM_C = float(best_svm['C'])
print(f"\n  ✓ Best LinearSVC: C={BEST_SVM_C}  (Val F1={best_svm['f1']:.4f})")


# ─────────────────────────────────────────────────────────────────
# STEP 5  — SVM Kernel Comparison  (on SVD-50 — RBF needs dense)
#           LinearSVC vs RBF SVC vs Poly SVC
# ─────────────────────────────────────────────────────────────────
print("\n[Step 5] SVM kernel comparison (SVD-50 features)...")

KERNELS = {
    'linear': SVC(kernel='linear', C=BEST_SVM_C,
                  probability=True, random_state=RANDOM_STATE),
    'rbf':    SVC(kernel='rbf',    C=BEST_SVM_C, gamma='scale',
                  probability=True, random_state=RANDOM_STATE),
    'poly':   SVC(kernel='poly',   C=BEST_SVM_C, degree=3,
                  probability=True, random_state=RANDOM_STATE),
}

kernel_results = []
for kname, svc in KERNELS.items():
    t0 = time.time()
    svc.fit(X_train_svd, y_train)
    y_pred = svc.predict(X_val_svd)
    y_prob = svc.predict_proba(X_val_svd)[:, 1]
    elapsed = time.time() - t0
    m = evaluate(y_val, y_pred, y_prob, label=f'SVC {kname}')
    m['kernel'] = kname
    kernel_results.append(m)
    print(f"  [{kname:<6}]  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  "
          f"AUC={m['roc_auc']:.4f}  ({elapsed:.1f}s)")

best_kernel = max(kernel_results, key=lambda x: x['f1'])['kernel']
print(f"\n  ✓ Best kernel: {best_kernel}")


# ─────────────────────────────────────────────────────────────────
# STEP 6  — Per-Language DT & SVM  (monolingual)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 6] Per-language Decision Tree & SVM...")

lang_results  = []
lang_models   = {}

for lang in LANGUAGES:
    mask_tr = (lang_train == lang).values
    mask_va = (lang_val   == lang).values
    mask_te = (lang_test  == lang).values

    if mask_tr.sum() < 20:
        print(f"  [{lang}] Not enough samples — skipping.")
        continue

    # Decision Tree
    dt_lang = DecisionTreeClassifier(
        max_depth=BEST_DT_DEPTH, criterion=BEST_DT_CRITERION,
        min_samples_leaf=5, min_samples_split=10,
        random_state=RANDOM_STATE,
    )
    dt_lang.fit(X_train_dt[mask_tr], y_train[mask_tr])

    # SVM
    svm_lang = CalibratedClassifierCV(
        LinearSVC(C=BEST_SVM_C, max_iter=5000, random_state=RANDOM_STATE),
        cv=3, method='sigmoid'
    )
    svm_lang.fit(X_tr_norm[mask_tr], y_train[mask_tr])

    lang_models[lang] = {'dt': dt_lang, 'svm': svm_lang}

    for model_name, model, X_va_m, X_te_m in [
        ('DT',  dt_lang,  X_val_dt[mask_va],   X_test_dt[mask_te]),
        ('SVM', svm_lang, X_va_norm[mask_va],   X_te_norm[mask_te]),
    ]:
        for split, X_m, y_m, mask_m in [
            ('val',  X_va_m, y_val[mask_va],   mask_va),
            ('test', X_te_m, y_test[mask_te],  mask_te),
        ]:
            if mask_m.sum() == 0:
                continue
            yp   = model.predict(X_m)
            yprb = model.predict_proba(X_m)[:, 1]
            m    = evaluate(y_test[mask_te] if split == 'test'
                           else y_val[mask_va], yp, yprb)
            m['language'] = lang
            m['model']    = model_name
            m['split']    = split
            lang_results.append(m)

    # Print summary for this language
    lang_sub = [r for r in lang_results if r['language'] == lang]
    dt_te  = next((r for r in lang_sub if r['model']=='DT'  and r['split']=='test'), {})
    svm_te = next((r for r in lang_sub if r['model']=='SVM' and r['split']=='test'), {})
    print(f"  [{lang}]  DT  Test F1={dt_te.get('f1', float('nan')):.4f}  |  "
          f"SVM Test F1={svm_te.get('f1', float('nan')):.4f}")

lang_results_df = pd.DataFrame(lang_results)


# ─────────────────────────────────────────────────────────────────
# STEP 7  — Final Models on TEST Set
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 7] Final evaluation on held-out TEST set...")

# Final Decision Tree
final_dt = DecisionTreeClassifier(
    max_depth=BEST_DT_DEPTH, criterion=BEST_DT_CRITERION,
    min_samples_leaf=5, min_samples_split=10,
    random_state=RANDOM_STATE,
)
final_dt.fit(X_tv_dt, y_tv)
y_pred_dt = final_dt.predict(X_test_dt)
y_prob_dt = final_dt.predict_proba(X_test_dt)[:, 1]
m_dt = evaluate(y_test, y_pred_dt, y_prob_dt,
                label=f'DT {BEST_DT_CRITERION} depth={BEST_DT_DEPTH}')

# Final SVM (LinearSVC + calibration)
final_svm = CalibratedClassifierCV(
    LinearSVC(C=BEST_SVM_C, max_iter=5000, random_state=RANDOM_STATE),
    cv=3, method='sigmoid'
)
final_svm.fit(X_tv_norm, y_tv)
y_pred_svm = final_svm.predict(X_te_norm)
y_prob_svm = final_svm.predict_proba(X_te_norm)[:, 1]
m_svm = evaluate(y_test, y_pred_svm, y_prob_svm,
                 label=f'LinearSVC C={BEST_SVM_C}')

print(f"\n  ── Decision Tree (depth={BEST_DT_DEPTH}, {BEST_DT_CRITERION}) ──")
print_eval(m_dt)
print(f"\n  ── LinearSVC (C={BEST_SVM_C}) ──")
print_eval(m_svm)

# Classification reports
for name, y_pred in [('Decision Tree', y_pred_dt), ('LinearSVC', y_pred_svm)]:
    print(f"\n  [{name}] Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Real (0)', 'Fake (1)'], digits=4))

# Per-language test results
print(f"\n  Per-language test results:")
print(f"  {'Language':<12} {'N':>5}  "
      f"{'DT Acc':>7} {'DT F1':>7}  "
      f"{'SVM Acc':>8} {'SVM F1':>7}")
print("  " + "─" * 55)
for lang in LANGUAGES:
    mask = (lang_test == lang).values
    if mask.sum() == 0:
        continue
    dt_pred  = final_dt.predict(X_test_dt[mask])
    svm_pred = final_svm.predict(X_te_norm[mask])
    dt_f1    = f1_score(y_test[mask], dt_pred,  zero_division=0)
    svm_f1   = f1_score(y_test[mask], svm_pred, zero_division=0)
    dt_acc   = accuracy_score(y_test[mask], dt_pred)
    svm_acc  = accuracy_score(y_test[mask], svm_pred)
    print(f"  {lang:<12} {mask.sum():>5}  "
          f"{dt_acc:>7.4f} {dt_f1:>7.4f}  "
          f"{svm_acc:>8.4f} {svm_f1:>7.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 8  — Save Artefacts
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 8] Saving artefacts...")

with open(os.path.join(OUTPUT_DIR, "dt_final.pkl"),         'wb') as f:
    pickle.dump(final_dt, f)
with open(os.path.join(OUTPUT_DIR, "svm_final.pkl"),        'wb') as f:
    pickle.dump(final_svm, f)
with open(os.path.join(OUTPUT_DIR, "lang_models_dt_svm.pkl"), 'wb') as f:
    pickle.dump(lang_models, f)

dt_df.to_csv(os.path.join(OUTPUT_DIR, "dt_depth_tuning.csv"),    index=False)
svm_df.to_csv(os.path.join(OUTPUT_DIR, "svm_c_tuning.csv"),      index=False)
pd.DataFrame(kernel_results).to_csv(
    os.path.join(OUTPUT_DIR, "svm_kernel_results.csv"),           index=False)
lang_results_df.to_csv(
    os.path.join(OUTPUT_DIR, "per_language_results.csv"),         index=False)
feat_imp_df.to_csv(
    os.path.join(OUTPUT_DIR, "dt_feature_importance.csv"),        index=False)

config = {
    'best_dt_depth': BEST_DT_DEPTH, 'best_dt_criterion': BEST_DT_CRITERION,
    'best_svm_c': BEST_SVM_C,       'best_kernel': best_kernel,
    'dt_test_f1': round(m_dt['f1'], 4),  'dt_test_acc': round(m_dt['accuracy'], 4),
    'svm_test_f1': round(m_svm['f1'], 4),'svm_test_acc': round(m_svm['accuracy'], 4),
}
with open(os.path.join(OUTPUT_DIR, "phase7_config.pkl"), 'wb') as f:
    pickle.dump(config, f)

for fname in ["y_train.csv", "y_val.csv", "y_test.csv",
              "lang_train.csv", "lang_val.csv", "lang_test.csv"]:
    pd.read_csv(os.path.join(P6_DIR, fname)).to_csv(
        os.path.join(OUTPUT_DIR, fname), index=False)

print(f"  ✓ Artefacts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────
# STEP 9  — Visualisations
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 9] Generating plots...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Phase 7 — Decision Tree & SVM Results",
             fontsize=15, fontweight='bold')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)
COLORS = ['#4C72B0', '#C44E52', '#55A868', '#DD8452']

# ── Plot 1: DT depth vs F1 ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for crit, color in [('gini', '#4C72B0'), ('entropy', '#C44E52')]:
    sub = dt_df[dt_df['criterion'] == crit].sort_values('max_depth')
    depths = sub['max_depth'].replace(999, 25).astype(int)
    ax1.plot(depths, sub['f1'], 'o-', color=color,
             linewidth=2, markersize=6, label=crit.capitalize())
ax1.set_xlabel("Max depth  (25 = None/unlimited)")
ax1.set_ylabel("Validation F1")
ax1.set_title("Decision Tree — depth vs F1")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)

# ── Plot 2: DT depth vs leaves (overfitting signal) ─────────────
ax2 = fig.add_subplot(gs[0, 1])
for crit, color in [('gini', '#4C72B0'), ('entropy', '#C44E52')]:
    sub = dt_df[dt_df['criterion'] == crit].sort_values('max_depth')
    depths = sub['max_depth'].replace(999, 25).astype(int)
    ax2.plot(depths, sub['n_leaves'], 'o-', color=color,
             linewidth=2, markersize=6, label=crit.capitalize())
ax2.set_xlabel("Max depth  (25 = None/unlimited)")
ax2.set_ylabel("Number of leaves")
ax2.set_title("Decision Tree — depth vs leaves")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)

# ── Plot 3: SVM C vs F1 / AUC ───────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.semilogx(svm_df['C'], svm_df['f1'], 'o-', color='#4C72B0',
             linewidth=2, markersize=6, label='F1')
ax3.semilogx(svm_df['C'], svm_df['roc_auc'], 's-', color='#55A868',
             linewidth=2, markersize=6, label='ROC-AUC')
ax3.axvline(BEST_SVM_C, color='grey', linestyle='--',
            linewidth=1.5, label=f'Best C={BEST_SVM_C}')
ax3.set_xlabel("C  (log scale)"); ax3.set_ylabel("Score")
ax3.set_title("LinearSVC — C tuning")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25)

# ── Plot 4: Kernel comparison bar ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
k_names = [r['label'] for r in kernel_results]
k_f1    = [r['f1']    for r in kernel_results]
k_auc   = [r['roc_auc'] for r in kernel_results]
x = np.arange(len(k_names)); w = 0.35
ax4.bar(x - w/2, k_f1,  w, color='#4C72B0', alpha=0.85, label='F1')
ax4.bar(x + w/2, k_auc, w, color='#55A868', alpha=0.85, label='AUC')
ax4.set_xticks(x)
ax4.set_xticklabels([n.replace('SVC ', '') for n in k_names], fontsize=9)
ax4.set_ylim(max(0, min(k_f1+k_auc) - 0.05), 1.05)
ax4.set_title("SVM kernel comparison (SVD-50)")
ax4.set_ylabel("Score"); ax4.legend(fontsize=9)
for i, (f, a) in enumerate(zip(k_f1, k_auc)):
    ax4.text(i-w/2, f+0.005, f'{f:.3f}', ha='center', fontsize=8)
    ax4.text(i+w/2, a+0.005, f'{a:.3f}', ha='center', fontsize=8)

# ── Plot 5: ROC curves — DT vs SVM ──────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
for name, y_prob, color in [
    (f'DT (depth={BEST_DT_DEPTH})', y_prob_dt,  '#4C72B0'),
    (f'SVM (C={BEST_SVM_C})',       y_prob_svm, '#C44E52'),
]:
    fpr_c, tpr_c, _ = roc_curve(y_test, y_prob)
    auc_c = roc_auc_score(y_test, y_prob)
    ax5.plot(fpr_c, tpr_c, color=color, linewidth=2,
             label=f'{name} (AUC={auc_c:.3f})')
ax5.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5, label='Random')
ax5.set_xlabel("FPR"); ax5.set_ylabel("TPR")
ax5.set_title("ROC Curves — DT vs SVM")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.25)

# ── Plot 6: Confusion matrices side by side ──────────────────────
for col_idx, (name, y_pred, cmap) in enumerate([
    (f'DT depth={BEST_DT_DEPTH}', y_pred_dt,  'Blues'),
    (f'LinearSVC C={BEST_SVM_C}', y_pred_svm, 'Oranges'),
]):
    ax = fig.add_subplot(gs[1, 2] if col_idx == 1 else gs[1, 2])
    if col_idx == 0:
        ax = fig.add_subplot(gs[1, 2])
    else:
        break  # handled below
    cm_m = confusion_matrix(y_test, y_pred)
    im   = ax.imshow(cm_m, cmap=cmap)
    ax.set_title(f"CM — {name}", fontsize=9)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Real','Fake'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['Real','Fake'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm_m[i,j]}\n({cm_m[i,j]/cm_m.sum()*100:.1f}%)',
                    ha='center', va='center', fontsize=10,
                    color='white' if cm_m[i,j] > cm_m.max()/2 else 'black')
    plt.colorbar(im, ax=ax)

# ── Plot 6b: DT confusion matrix in its own cell ────────────────
ax6b = fig.add_subplot(gs[2, 0])
cm_dt = confusion_matrix(y_test, y_pred_dt)
im6b  = ax6b.imshow(cm_dt, cmap='Blues')
ax6b.set_title(f"CM — DT (depth={BEST_DT_DEPTH})")
ax6b.set_xticks([0,1]); ax6b.set_xticklabels(['Real','Fake'])
ax6b.set_yticks([0,1]); ax6b.set_yticklabels(['Real','Fake'])
for i in range(2):
    for j in range(2):
        ax6b.text(j, i,
                  f'{cm_dt[i,j]}\n({cm_dt[i,j]/cm_dt.sum()*100:.1f}%)',
                  ha='center', va='center', fontsize=10,
                  color='white' if cm_dt[i,j] > cm_dt.max()/2 else 'black')
plt.colorbar(im6b, ax=ax6b)

# ── Plot 7: Top-20 DT feature importances ───────────────────────
ax7 = fig.add_subplot(gs[2, 1])
top20 = feat_imp_df.head(20)
ax7.barh(range(len(top20)), top20['importance'].values[::-1],
         color='#4C72B0', alpha=0.8)
ax7.set_yticks(range(len(top20)))
ax7.set_yticklabels([f[:22] for f in top20['feature'].values[::-1]],
                    fontsize=7)
ax7.set_xlabel("Gini importance")
ax7.set_title("Top-20 DT feature importances")
ax7.grid(True, alpha=0.2, axis='x')

# ── Plot 8: Per-language test F1 — DT vs SVM ────────────────────
ax8 = fig.add_subplot(gs[2, 2])
if not lang_results_df.empty:
    dt_te  = lang_results_df[(lang_results_df['model']=='DT')  &
                              (lang_results_df['split']=='test')]
    svm_te = lang_results_df[(lang_results_df['model']=='SVM') &
                              (lang_results_df['split']=='test')]
    langs_p = dt_te['language'].tolist()
    x = np.arange(len(langs_p)); w = 0.35
    ax8.bar(x - w/2, dt_te['f1'].values,  w, color='#4C72B0',
            alpha=0.85, label=f'DT depth={BEST_DT_DEPTH}')
    ax8.bar(x + w/2, svm_te['f1'].values, w, color='#C44E52',
            alpha=0.85, label=f'SVM C={BEST_SVM_C}')
    ax8.set_xticks(x); ax8.set_xticklabels(langs_p)
    ax8.set_ylim(0, 1.1); ax8.set_ylabel("F1 Score")
    ax8.set_title("Per-language test F1 — DT vs SVM")
    ax8.legend(fontsize=8)
    for i, (d, s) in enumerate(zip(dt_te['f1'].values, svm_te['f1'].values)):
        ax8.text(i-w/2, d+0.02, f'{d:.2f}', ha='center', fontsize=8)
        ax8.text(i+w/2, s+0.02, f'{s:.2f}', ha='center', fontsize=8)

plt.savefig(os.path.join(OUTPUT_DIR, "phase7_dt_svm_results.png"),
            dpi=150, bbox_inches='tight')
print(f"  ✓ Saved → {OUTPUT_DIR}/phase7_dt_svm_results.png")


# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 7 SUMMARY — Test Set Performance")
print("=" * 60)
print(f"  {'Model':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
print("  " + "─" * 58)
for m in [m_dt, m_svm]:
    print(f"  {m['label']:<30} {m['accuracy']:>8.4f} "
          f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
          f"{m['f1']:>8.4f} {m['roc_auc']:>8.4f}")
print(f"\nArtefacts in: ./{OUTPUT_DIR}/")
print("  dt_final.pkl            svm_final.pkl")
print("  dt_depth_tuning.csv     svm_c_tuning.csv")
print("  svm_kernel_results.csv  dt_feature_importance.csv")
print("  per_language_results.csv  phase7_config.pkl")
print("  phase7_dt_svm_results.png")
print("\nReady for Phase 8 — Other Classifiers & Perceptron")
print("=" * 60)
