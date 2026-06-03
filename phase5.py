"""
Phase 5 — Evaluation Framework
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Lab 4: Classification Performance Matrices
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, matthews_corrcoef, cohen_kappa_score,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import normalize, label_binarize
from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_DIR  = "phase4_outputs"
P2_DIR     = "phase2_outputs"
P3_DIR     = "phase3_outputs"
OUTPUT_DIR = "phase5_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES    = ['Hindi', 'Marathi', 'Gujarati', 'Telugu']
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────
# STEP 1  — Load Artefacts
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 5 — Evaluation Framework")
print("=" * 60)

# TF-IDF features
X_train_tfidf = sp.load_npz(os.path.join(P2_DIR, "X_train_tfidf.npz"))
X_val_tfidf   = sp.load_npz(os.path.join(P2_DIR, "X_val_tfidf.npz"))
X_test_tfidf  = sp.load_npz(os.path.join(P2_DIR, "X_test_tfidf.npz"))

# SVD features
X_train_svd = np.load(os.path.join(P3_DIR, "X_train_svd.npy"))
X_val_svd   = np.load(os.path.join(P3_DIR, "X_val_svd.npy"))
X_test_svd  = np.load(os.path.join(P3_DIR, "X_test_svd.npy"))

# Labels & language
def load_csv(d, name):
    return pd.read_csv(os.path.join(d, name)).squeeze()

y_train    = load_csv(INPUT_DIR, "y_train.csv").astype(int)
y_val      = load_csv(INPUT_DIR, "y_val.csv").astype(int)
y_test     = load_csv(INPUT_DIR, "y_test.csv").astype(int)
lang_train = load_csv(INPUT_DIR, "lang_train.csv")
lang_val   = load_csv(INPUT_DIR, "lang_val.csv")
lang_test  = load_csv(INPUT_DIR, "lang_test.csv")

# Best kNN config from Phase 4
with open(os.path.join(INPUT_DIR, "knn_config.pkl"), 'rb') as f:
    knn_config = pickle.load(f)
BEST_K = knn_config['best_k']

# Normalise TF-IDF for cosine kNN
X_tr_norm = normalize(X_train_tfidf, norm='l2')
X_va_norm = normalize(X_val_tfidf,   norm='l2')
X_te_norm = normalize(X_test_tfidf,  norm='l2')

# Combine train+val for final models
X_trainval_norm = sp.vstack([X_tr_norm, X_va_norm])
y_trainval      = pd.concat([y_train, y_val]).reset_index(drop=True)
lang_trainval   = pd.concat([lang_train, lang_val]).reset_index(drop=True)

print(f"\n✓ Artefacts loaded  |  Best k from Phase 4: {BEST_K}")
print(f"  Train: {X_train_tfidf.shape}  Val: {X_val_tfidf.shape}  Test: {X_test_tfidf.shape}")


# ─────────────────────────────────────────────────────────────────
# HELPER — Full Metrics Suite
# ─────────────────────────────────────────────────────────────────
def full_metrics(y_true, y_pred, y_prob=None, label=''):
    """
    Compute the complete Lab 4 metric set:
      Accuracy, Precision, Recall, F1, Specificity,
      FPR, FNR, MCC, Cohen's Kappa, ROC-AUC, Avg Precision
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)    # sensitivity / TPR
    f1    = f1_score(y_true, y_pred, zero_division=0)
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0        # specificity / TNR
    fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0.0        # false positive rate
    fnr   = fn / (fn + tp) if (fn + tp) > 0 else 0.0        # false negative rate
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_prob)    if y_prob is not None else float('nan')
    ap    = average_precision_score(y_true, y_prob) if y_prob is not None else float('nan')

    return {
        'label': label, 'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'specificity': spec,
        'fpr': fpr, 'fnr': fnr, 'mcc': mcc, 'kappa': kappa,
        'roc_auc': auc, 'avg_precision': ap,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }


def print_full_metrics(m):
    print(f"  Accuracy    : {m['accuracy']:.4f}")
    print(f"  Precision   : {m['precision']:.4f}   (of all predicted Fake, how many are Fake)")
    print(f"  Recall/TPR  : {m['recall']:.4f}   (of all actual Fake, how many caught)")
    print(f"  Specificity : {m['specificity']:.4f}   (of all actual Real, how many correctly rejected)")
    print(f"  F1 Score    : {m['f1']:.4f}   (harmonic mean of precision & recall)")
    print(f"  FPR         : {m['fpr']:.4f}   (real news wrongly flagged as fake)")
    print(f"  FNR         : {m['fnr']:.4f}   (fake news missed)")
    print(f"  MCC         : {m['mcc']:.4f}   (balanced; +1 perfect, 0 random, -1 inverse)")
    print(f"  Cohen Kappa : {m['kappa']:.4f}   (agreement beyond chance)")
    print(f"  ROC-AUC     : {m['roc_auc']:.4f}")
    print(f"  Avg Prec    : {m['avg_precision']:.4f}   (area under PR curve)")
    print(f"  Confusion   : TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")


# ─────────────────────────────────────────────────────────────────
# STEP 2  — Train Evaluation Models
# ─────────────────────────────────────────────────────────────────
print("\n[Step 2] Training models for evaluation...")

# Model 1: kNN (from Phase 4)
knn = KNeighborsClassifier(
    n_neighbors=BEST_K, metric='cosine',
    algorithm='brute', n_jobs=-1
)
knn.fit(X_trainval_norm, y_trainval)
y_pred_knn  = knn.predict(X_te_norm)
y_prob_knn  = knn.predict_proba(X_te_norm)[:, 1]

# Model 2: Logistic Regression (quick additional model for comparison)
lr = LogisticRegression(
    C=1.0, max_iter=1000, solver='saga',
    random_state=RANDOM_STATE, n_jobs=-1
)
lr.fit(X_trainval_norm, y_trainval)
y_pred_lr   = lr.predict(X_te_norm)
y_prob_lr   = lr.predict_proba(X_te_norm)[:, 1]

print(f"  ✓ kNN (k={BEST_K}, cosine) trained")
print(f"  ✓ Logistic Regression (C=1.0, saga) trained")


# ─────────────────────────────────────────────────────────────────
# STEP 3  — Full Metric Evaluation  (overall)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 3] Full metric evaluation on test set...")

metrics_knn = full_metrics(y_test, y_pred_knn, y_prob_knn, label=f'kNN (k={BEST_K})')
metrics_lr  = full_metrics(y_test, y_pred_lr,  y_prob_lr,  label='Logistic Regression')

print(f"\n  ── kNN (k={BEST_K}) ──")
print_full_metrics(metrics_knn)

print(f"\n  ── Logistic Regression ──")
print_full_metrics(metrics_lr)

print(f"\n  Full Classification Reports:")
for name, y_pred in [(f'kNN k={BEST_K}', y_pred_knn), ('Logistic Regression', y_pred_lr)]:
    print(f"\n  [{name}]")
    print(classification_report(y_test, y_pred,
          target_names=['Real (0)', 'Fake (1)'], digits=4))


# ─────────────────────────────────────────────────────────────────
# STEP 4  — Per-Language Metric Breakdown
# ─────────────────────────────────────────────────────────────────
print("\n[Step 4] Per-language metric breakdown...")

lang_metrics = []
METRIC_COLS  = ['accuracy', 'precision', 'recall', 'f1',
                'specificity', 'fpr', 'mcc', 'roc_auc']

for lang in LANGUAGES:
    mask = (lang_test == lang).values
    if mask.sum() == 0:
        continue

    for model_name, y_pred, y_prob in [
        (f'kNN(k={BEST_K})', y_pred_knn, y_prob_knn),
        ('LogReg',           y_pred_lr,  y_prob_lr),
    ]:
        if len(np.unique(y_test[mask])) < 2:
            continue
        m = full_metrics(y_test[mask], y_pred[mask],
                         y_prob[mask], label=f'{lang} — {model_name}')
        m['language'] = lang
        m['model']    = model_name
        lang_metrics.append(m)

lang_df = pd.DataFrame(lang_metrics)

print(f"\n  {'Language':<12} {'Model':<16} {'Acc':>6} {'Prec':>6} "
      f"{'Rec':>6} {'F1':>6} {'Spec':>6} {'MCC':>6} {'AUC':>6}")
print("  " + "─" * 72)
for _, row in lang_df.iterrows():
    print(f"  {row['language']:<12} {row['model']:<16} "
          f"{row['accuracy']:>6.3f} {row['precision']:>6.3f} "
          f"{row['recall']:>6.3f} {row['f1']:>6.3f} "
          f"{row['specificity']:>6.3f} {row['mcc']:>6.3f} "
          f"{row['roc_auc']:>6.3f}")


# ─────────────────────────────────────────────────────────────────
# STEP 5  — Threshold Analysis
#           Default threshold = 0.5 but optimal may differ
#           Lab 4: understand precision-recall trade-off
# ─────────────────────────────────────────────────────────────────
print("\n[Step 5] Threshold analysis (kNN)...")

thresholds   = np.arange(0.1, 0.91, 0.05)
thresh_rows  = []
for t in thresholds:
    y_pred_t = (y_prob_knn >= t).astype(int)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec  = recall_score(y_test, y_pred_t, zero_division=0)
    f1   = f1_score(y_test, y_pred_t, zero_division=0)
    acc  = accuracy_score(y_test, y_pred_t)
    thresh_rows.append({'threshold': round(t, 2), 'precision': prec,
                        'recall': rec, 'f1': f1, 'accuracy': acc})

thresh_df  = pd.DataFrame(thresh_rows)
best_thresh = thresh_df.loc[thresh_df['f1'].idxmax(), 'threshold']
print(f"  Best threshold by F1: {best_thresh}  "
      f"(F1={thresh_df.loc[thresh_df['f1'].idxmax(), 'f1']:.4f})")
print(f"  Default (0.5)   F1  : "
      f"{thresh_df[thresh_df['threshold']==0.5]['f1'].values[0]:.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 6  — Error Analysis  (what did the model get wrong?)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 6] Error analysis...")

splits_df = pd.read_csv(os.path.join(P2_DIR, "splits.csv"), encoding='utf-8-sig')
test_df   = splits_df[splits_df['split'] == 'test'].reset_index(drop=True)

if len(test_df) == len(y_pred_knn):
    test_df['pred_knn']  = y_pred_knn
    test_df['prob_fake'] = y_prob_knn
    test_df['correct']   = (test_df['pred_knn'] == test_df['label'])
    test_df['error_type'] = 'correct'
    test_df.loc[(test_df['label']==1) & (test_df['pred_knn']==0), 'error_type'] = 'FN (missed fake)'
    test_df.loc[(test_df['label']==0) & (test_df['pred_knn']==1), 'error_type'] = 'FP (wrongly flagged real)'

    print("\n  Error distribution:")
    print(test_df['error_type'].value_counts().to_string())

    print("\n  Error rate per language:")
    for lang in LANGUAGES:
        sub = test_df[test_df['language'] == lang]
        if sub.empty:
            continue
        err_rate = 1 - sub['correct'].mean()
        fn_rate  = (sub['error_type'] == 'FN (missed fake)').sum()
        fp_rate  = (sub['error_type'] == 'FP (wrongly flagged real)').sum()
        print(f"  {lang:<12}  Error={err_rate:.3f}  FN={fn_rate}  FP={fp_rate}")

    # Save misclassified examples
    errors = test_df[~test_df['correct']].sort_values('prob_fake', ascending=False)
    errors.to_csv(os.path.join(OUTPUT_DIR, "misclassified_samples.csv"),
                  index=False, encoding='utf-8-sig')
    print(f"\n  ✓ Saved {len(errors)} misclassified samples → misclassified_samples.csv")
else:
    print("  [Warning] Test split size mismatch — skipping error analysis.")


# ─────────────────────────────────────────────────────────────────
# STEP 7  — Save All Artefacts
# ─────────────────────────────────────────────────────────────────
print("\n[Step 7] Saving artefacts...")

summary = pd.DataFrame([metrics_knn, metrics_lr])
summary.to_csv(os.path.join(OUTPUT_DIR, "overall_metrics.csv"), index=False)
lang_df.to_csv(os.path.join(OUTPUT_DIR, "per_language_metrics.csv"), index=False)
thresh_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_analysis.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "eval_models.pkl"), 'wb') as f:
    pickle.dump({'knn': knn, 'lr': lr}, f)

# Copy labels & splits forward
for fname in ["y_train.csv", "y_val.csv", "y_test.csv",
              "lang_train.csv", "lang_val.csv", "lang_test.csv"]:
    pd.read_csv(os.path.join(INPUT_DIR, fname)).to_csv(
        os.path.join(OUTPUT_DIR, fname), index=False)

print(f"  ✓ Artefacts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────
# STEP 8  — Visualisations
# ─────────────────────────────────────────────────────────────────
print("\n[Step 8] Generating plots...")

COLORS   = ['#4C72B0', '#C44E52', '#55A868', '#DD8452', '#9467BD']
LANG_CLR = {'Hindi': '#4C72B0', 'Marathi': '#DD8452',
            'Gujarati': '#55A868', 'Telugu': '#C44E52'}

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Phase 5 — Evaluation Framework", fontsize=15, fontweight='bold')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── Plot 1: Confusion matrix — kNN ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
cm_knn = confusion_matrix(y_test, y_pred_knn)
im1 = ax1.imshow(cm_knn, cmap='Blues')
ax1.set_title(f"Confusion matrix — kNN (k={BEST_K})")
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(['Real', 'Fake'])
ax1.set_yticks([0, 1]); ax1.set_yticklabels(['Real', 'Fake'])
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f'{cm_knn[i,j]}\n({cm_knn[i,j]/cm_knn.sum()*100:.1f}%)',
                 ha='center', va='center', fontsize=11,
                 color='white' if cm_knn[i,j] > cm_knn.max()/2 else 'black')
plt.colorbar(im1, ax=ax1)

# ── Plot 2: Confusion matrix — Logistic Regression ──────────────
ax2 = fig.add_subplot(gs[0, 1])
cm_lr = confusion_matrix(y_test, y_pred_lr)
im2 = ax2.imshow(cm_lr, cmap='Oranges')
ax2.set_title("Confusion matrix — Logistic Regression")
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
ax2.set_xticks([0, 1]); ax2.set_xticklabels(['Real', 'Fake'])
ax2.set_yticks([0, 1]); ax2.set_yticklabels(['Real', 'Fake'])
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{cm_lr[i,j]}\n({cm_lr[i,j]/cm_lr.sum()*100:.1f}%)',
                 ha='center', va='center', fontsize=11,
                 color='white' if cm_lr[i,j] > cm_lr.max()/2 else 'black')
plt.colorbar(im2, ax=ax2)

# ── Plot 3: Metric radar / bar comparison ───────────────────────
ax3 = fig.add_subplot(gs[0, 2])
metric_keys  = ['accuracy', 'precision', 'recall', 'f1',
                'specificity', 'mcc', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1',
                 'Specificity', 'MCC', 'ROC-AUC']
knn_vals = [metrics_knn[k] for k in metric_keys]
lr_vals  = [metrics_lr[k]  for k in metric_keys]
x = np.arange(len(metric_keys))
w = 0.35
ax3.bar(x - w/2, knn_vals, w, label=f'kNN k={BEST_K}',
        color='#4C72B0', alpha=0.85)
ax3.bar(x + w/2, lr_vals,  w, label='LogReg',
        color='#C44E52', alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(metric_labels, rotation=30, ha='right', fontsize=8)
ax3.set_ylim(0, 1.1); ax3.set_ylabel("Score")
ax3.set_title("Metric comparison")
ax3.legend(fontsize=8)

# ── Plot 4: ROC curves ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for name, y_prob, color in [
    (f'kNN k={BEST_K}', y_prob_knn, '#4C72B0'),
    ('LogReg',          y_prob_lr,  '#C44E52'),
]:
    fpr_c, tpr_c, _ = roc_curve(y_test, y_prob)
    auc_c = roc_auc_score(y_test, y_prob)
    ax4.plot(fpr_c, tpr_c, color=color, linewidth=2,
             label=f'{name} (AUC={auc_c:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax4.fill_between(*roc_curve(y_test, y_prob_knn)[:2], alpha=0.08, color='#4C72B0')
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.set_title("ROC Curve")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.25)

# ── Plot 5: Precision-Recall curves ─────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
for name, y_prob, color in [
    (f'kNN k={BEST_K}', y_prob_knn, '#4C72B0'),
    ('LogReg',          y_prob_lr,  '#C44E52'),
]:
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    ap_c = average_precision_score(y_test, y_prob)
    ax5.plot(rec_c, prec_c, color=color, linewidth=2,
             label=f'{name} (AP={ap_c:.3f})')
baseline = y_test.mean()
ax5.axhline(baseline, color='grey', linestyle='--',
            linewidth=1, label=f'Baseline ({baseline:.3f})')
ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
ax5.set_title("Precision-Recall Curve")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.25)

# ── Plot 6: Threshold vs Precision / Recall / F1 ────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(thresh_df['threshold'], thresh_df['precision'],
         'o-', color='#4C72B0', label='Precision', linewidth=2, markersize=5)
ax6.plot(thresh_df['threshold'], thresh_df['recall'],
         's-', color='#C44E52', label='Recall',    linewidth=2, markersize=5)
ax6.plot(thresh_df['threshold'], thresh_df['f1'],
         '^-', color='#55A868', label='F1',         linewidth=2, markersize=5)
ax6.axvline(best_thresh, color='grey', linestyle='--',
            linewidth=1.5, label=f'Best t={best_thresh}')
ax6.set_xlabel("Decision threshold"); ax6.set_ylabel("Score")
ax6.set_title("Threshold vs Precision / Recall / F1")
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.25)

# ── Plot 7: Per-language F1 grouped bar ─────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
knn_f1  = []
lr_f1   = []
langs_p = []
for lang in LANGUAGES:
    sub_knn = lang_df[(lang_df['language']==lang) & (lang_df['model'].str.contains('kNN'))]
    sub_lr  = lang_df[(lang_df['language']==lang) & (lang_df['model']=='LogReg')]
    if sub_knn.empty and sub_lr.empty:
        continue
    langs_p.append(lang)
    knn_f1.append(sub_knn['f1'].values[0] if not sub_knn.empty else 0)
    lr_f1.append( sub_lr['f1'].values[0]  if not sub_lr.empty  else 0)

x  = np.arange(len(langs_p)); w = 0.35
ax7.bar(x - w/2, knn_f1, w, label=f'kNN k={BEST_K}', color='#4C72B0', alpha=0.85)
ax7.bar(x + w/2, lr_f1,  w, label='LogReg',           color='#C44E52', alpha=0.85)
ax7.set_xticks(x); ax7.set_xticklabels(langs_p)
ax7.set_ylim(0, 1.1); ax7.set_ylabel("F1 Score")
ax7.set_title("Per-language F1 (test set)")
ax7.legend(fontsize=8)
for xi, (kf, lf) in enumerate(zip(knn_f1, lr_f1)):
    ax7.text(xi - w/2, kf + 0.02, f'{kf:.2f}', ha='center', fontsize=8)
    ax7.text(xi + w/2, lf + 0.02, f'{lf:.2f}', ha='center', fontsize=8)

# ── Plot 8: Per-language FPR & FNR heatmap ──────────────────────
ax8 = fig.add_subplot(gs[2, 1])
err_metrics = ['fpr', 'fnr', 'specificity', 'mcc']
err_labels  = ['FPR', 'FNR', 'Specificity', 'MCC']
knn_lang_df = lang_df[lang_df['model'].str.contains('kNN')]
if not knn_lang_df.empty:
    heat_data = knn_lang_df.set_index('language')[err_metrics].T
    im8 = ax8.imshow(heat_data.values.astype(float), cmap='RdYlGn',
                     aspect='auto', vmin=-0.2, vmax=1.0)
    ax8.set_xticks(range(len(heat_data.columns)))
    ax8.set_xticklabels(heat_data.columns, fontsize=9)
    ax8.set_yticks(range(len(err_labels)))
    ax8.set_yticklabels(err_labels, fontsize=9)
    ax8.set_title(f"Error metrics per language — kNN k={BEST_K}")
    plt.colorbar(im8, ax=ax8)
    for i in range(len(err_labels)):
        for j in range(len(heat_data.columns)):
            ax8.text(j, i, f'{heat_data.values[i,j]:.2f}',
                     ha='center', va='center', fontsize=9,
                     color='black')

# ── Plot 9: Score distribution (prob_fake) ───────────────────────
ax9 = fig.add_subplot(gs[2, 2])
ax9.hist(y_prob_knn[y_test == 0], bins=30, alpha=0.6,
         color='#4C72B0', label='Real news', density=True)
ax9.hist(y_prob_knn[y_test == 1], bins=30, alpha=0.6,
         color='#C44E52', label='Fake news', density=True)
ax9.axvline(0.5, color='black', linestyle='--', linewidth=1.5,
            label='Threshold=0.5')
ax9.axvline(best_thresh, color='#55A868', linestyle='--',
            linewidth=1.5, label=f'Best t={best_thresh}')
ax9.set_xlabel("P(Fake) score"); ax9.set_ylabel("Density")
ax9.set_title("Score distribution — kNN")
ax9.legend(fontsize=8)

plt.savefig(os.path.join(OUTPUT_DIR, "phase5_evaluation.png"),
            dpi=150, bbox_inches='tight')
print(f"  ✓ Saved → {OUTPUT_DIR}/phase5_evaluation.png")


# ─────────────────────────────────────────────────────────────────
# Final Summary Table
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 5 SUMMARY — Test Set Performance")
print("=" * 60)
cols = ['label', 'accuracy', 'precision', 'recall',
        'f1', 'specificity', 'mcc', 'roc_auc']
summary_print = summary[cols].copy()
summary_print.columns = ['Model', 'Accuracy', 'Precision',
                          'Recall', 'F1', 'Specificity', 'MCC', 'AUC']
print(summary_print.to_string(index=False, float_format='{:.4f}'.format))
print(f"\n  Best threshold (by F1) : {best_thresh}")
print(f"\nArtefacts in: ./{OUTPUT_DIR}/")
print("  overall_metrics.csv       per_language_metrics.csv")
print("  threshold_analysis.csv    misclassified_samples.csv")
print("  eval_models.pkl           phase5_evaluation.png")
print("\nReady for Phase 6 — Logistic Regression")
print("=" * 60)
