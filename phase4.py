"""
Phase 4 — kNN Baseline Classifier
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Lab 3: kNN
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import normalize
import time
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_DIR  = "phase3_outputs"
OUTPUT_DIR = "phase4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES    = ['Hindi', 'Marathi', 'Gujarati', 'Telugu']
RANDOM_STATE = 42

# kNN search space
K_VALUES       = [1, 3, 5, 7, 9, 11, 15, 21]
METRICS        = ['cosine', 'euclidean', 'manhattan']
ALGORITHM      = 'brute'      # required for cosine metric
N_JOBS         = -1           # use all CPU cores


# ─────────────────────────────────────────────────────────────────
# STEP 1  — Load Phase 3 Artefacts
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 4 — kNN Baseline Classifier")
print("=" * 60)

def load_npz(name):
    return sp.load_npz(os.path.join(INPUT_DIR, name))

def load_npy(name):
    return np.load(os.path.join(INPUT_DIR, name))

def load_csv(name):
    return pd.read_csv(os.path.join(INPUT_DIR, name)).squeeze()

# ── TF-IDF char features (primary) ──────────────────────────────
X_train_tfidf = sp.load_npz(os.path.join("phase2_outputs", "X_train_tfidf.npz"))
X_val_tfidf   = sp.load_npz(os.path.join("phase2_outputs", "X_val_tfidf.npz"))
X_test_tfidf  = sp.load_npz(os.path.join("phase2_outputs", "X_test_tfidf.npz"))

# ── SVD dense features (faster kNN) ────────────────────────────
X_train_svd = load_npy("X_train_svd.npy")
X_val_svd   = load_npy("X_val_svd.npy")
X_test_svd  = load_npy("X_test_svd.npy")

# ── Distance features from Phase 3 ──────────────────────────────
dist_train = load_npy("dist_train.npy")
dist_val   = load_npy("dist_val.npy")
dist_test  = load_npy("dist_test.npy")

# ── Labels & language column ─────────────────────────────────────
y_train    = load_csv("y_train.csv").astype(int)
y_val      = load_csv("y_val.csv").astype(int)
y_test     = load_csv("y_test.csv").astype(int)
lang_train = load_csv("lang_train.csv")
lang_val   = load_csv("lang_val.csv")
lang_test  = load_csv("lang_test.csv")

print(f"\n✓ Loaded artefacts")
print(f"  TF-IDF  — Train: {X_train_tfidf.shape}  Val: {X_val_tfidf.shape}  Test: {X_test_tfidf.shape}")
print(f"  SVD     — Train: {X_train_svd.shape}    Val: {X_val_svd.shape}    Test: {X_test_svd.shape}")


# ─────────────────────────────────────────────────────────────────
# HELPER — Evaluation
# ─────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob=None, label=''):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')
    return {'label': label, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'roc_auc': auc}

def print_metrics(m):
    print(f"  Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 2  — Feature Sets to Compare
# ─────────────────────────────────────────────────────────────────
# L2-normalise TF-IDF for cosine kNN (dot product = cosine on unit vectors)
X_tr_norm = normalize(X_train_tfidf, norm='l2')
X_va_norm = normalize(X_val_tfidf,   norm='l2')
X_te_norm = normalize(X_test_tfidf,  norm='l2')

# SVD features normalised
X_tr_svd_norm = normalize(X_train_svd, norm='l2')
X_va_svd_norm = normalize(X_val_svd,   norm='l2')
X_te_svd_norm = normalize(X_test_svd,  norm='l2')

FEATURE_SETS = {
    'TF-IDF (char, cosine)': {
        'train': X_tr_norm,  'val': X_va_norm, 'test': X_te_norm,
        'metric': 'cosine',  'sparse': True,
    },
    'SVD-50 (cosine)': {
        'train': X_tr_svd_norm, 'val': X_va_svd_norm, 'test': X_te_svd_norm,
        'metric': 'cosine',     'sparse': False,
    },
    'SVD-50 (euclidean)': {
        'train': X_train_svd, 'val': X_val_svd, 'test': X_test_svd,
        'metric': 'euclidean', 'sparse': False,
    },
    'Distance features (euclidean)': {
        'train': dist_train, 'val': dist_val, 'test': dist_test,
        'metric': 'euclidean', 'sparse': False,
    },
}


# ─────────────────────────────────────────────────────────────────
# STEP 3  — k Tuning on Validation Set  (TF-IDF, cosine)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 3] Tuning k on validation set (TF-IDF char + cosine)...")
print(f"  Testing k ∈ {K_VALUES}\n")

k_results = []
for k in K_VALUES:
    t0 = time.time()
    knn = KNeighborsClassifier(
        n_neighbors=k, metric='cosine',
        algorithm=ALGORITHM, n_jobs=N_JOBS
    )
    knn.fit(X_tr_norm, y_train)
    y_pred = knn.predict(X_va_norm)
    y_prob = knn.predict_proba(X_va_norm)[:, 1]
    elapsed = time.time() - t0

    m = evaluate(y_val, y_pred, y_prob, label=f'k={k}')
    m['time_s'] = round(elapsed, 2)
    k_results.append(m)
    print(f"  k={k:>2}  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  "
          f"AUC={m['roc_auc']:.4f}  ({elapsed:.1f}s)")

k_df = pd.DataFrame(k_results)
best_k_row = k_df.loc[k_df['f1'].idxmax()]
BEST_K = int(best_k_row['label'].split('=')[1])
print(f"\n  ✓ Best k = {BEST_K}  (Val F1 = {best_k_row['f1']:.4f})")


# ─────────────────────────────────────────────────────────────────
# STEP 4  — Distance Metric Comparison  (best k, SVD features)
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 4] Distance metric comparison (k={BEST_K}, SVD-50 features)...")
metric_results = []
for metric in METRICS:
    knn = KNeighborsClassifier(
        n_neighbors=BEST_K, metric=metric,
        algorithm=ALGORITHM, n_jobs=N_JOBS
    )
    X_tr_m = X_tr_svd_norm if metric == 'cosine' else X_train_svd
    X_va_m = X_va_svd_norm if metric == 'cosine' else X_val_svd
    knn.fit(X_tr_m, y_train)
    y_pred = knn.predict(X_va_m)
    y_prob = knn.predict_proba(X_va_m)[:, 1]
    m = evaluate(y_val, y_pred, y_prob, label=metric)
    metric_results.append(m)
    print(f"  {metric:<12}  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  AUC={m['roc_auc']:.4f}")

best_metric = max(metric_results, key=lambda x: x['f1'])['label']
print(f"\n  ✓ Best distance metric: {best_metric}")


# ─────────────────────────────────────────────────────────────────
# STEP 5  — Feature Set Comparison  (best k, each feature set)
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 5] Feature set comparison (k={BEST_K})...")
feature_results = []
for fs_name, fs in FEATURE_SETS.items():
    knn = KNeighborsClassifier(
        n_neighbors=BEST_K, metric=fs['metric'],
        algorithm=ALGORITHM, n_jobs=N_JOBS
    )
    knn.fit(fs['train'], y_train)
    y_pred = knn.predict(fs['val'])
    y_prob = knn.predict_proba(fs['val'])[:, 1]
    m = evaluate(y_val, y_pred, y_prob, label=fs_name)
    feature_results.append(m)
    print(f"  {fs_name:<35}  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")

best_fs_name = max(feature_results, key=lambda x: x['f1'])['label']
best_fs      = FEATURE_SETS[best_fs_name]
print(f"\n  ✓ Best feature set: {best_fs_name}")


# ─────────────────────────────────────────────────────────────────
# STEP 6  — Per-Language kNN  (monolingual models)
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 6] Per-language monolingual kNN (k={BEST_K}, cosine, TF-IDF)...")

lang_results = []
lang_models  = {}

for lang in LANGUAGES:
    mask_tr = (lang_train == lang).values
    mask_va = (lang_val   == lang).values
    mask_te = (lang_test  == lang).values

    if mask_tr.sum() < BEST_K + 1:
        print(f"  [{lang}] Not enough samples — skipping.")
        continue

    knn = KNeighborsClassifier(
        n_neighbors=BEST_K, metric='cosine',
        algorithm=ALGORITHM, n_jobs=N_JOBS
    )
    knn.fit(X_tr_norm[mask_tr], y_train[mask_tr])

    # Validation
    if mask_va.sum() > 0:
        y_pred_va = knn.predict(X_va_norm[mask_va])
        y_prob_va = knn.predict_proba(X_va_norm[mask_va])[:, 1]
        m_va = evaluate(y_val[mask_va], y_pred_va, y_prob_va, label=f'{lang} (val)')
    else:
        m_va = {'label': f'{lang} (val)', 'f1': float('nan')}

    # Test
    if mask_te.sum() > 0:
        y_pred_te = knn.predict(X_te_norm[mask_te])
        y_prob_te = knn.predict_proba(X_te_norm[mask_te])[:, 1]
        m_te = evaluate(y_test[mask_te], y_pred_te, y_prob_te, label=f'{lang} (test)')
    else:
        m_te = {'label': f'{lang} (test)', 'f1': float('nan')}

    lang_models[lang] = knn
    lang_results.append({**m_va, 'split': 'val',  'language': lang})
    lang_results.append({**m_te, 'split': 'test', 'language': lang})

    print(f"  [{lang}]  Val  F1={m_va.get('f1', float('nan')):.4f}  "
          f"Acc={m_va.get('accuracy', float('nan')):.4f}  |  "
          f"Test F1={m_te.get('f1', float('nan')):.4f}  "
          f"Acc={m_te.get('accuracy', float('nan')):.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 7  — Final Model: Best Config on TEST Set
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 7] Final evaluation on held-out TEST set...")
print(f"  Config: k={BEST_K}, metric=cosine, features=TF-IDF char\n")

final_knn = KNeighborsClassifier(
    n_neighbors=BEST_K, metric='cosine',
    algorithm=ALGORITHM, n_jobs=N_JOBS
)
# Train on train+val combined for final model
X_trainval = sp.vstack([X_train_tfidf, X_val_tfidf])
X_trainval_norm = normalize(X_trainval, norm='l2')
y_trainval = pd.concat([y_train, y_val]).reset_index(drop=True)

final_knn.fit(X_trainval_norm, y_trainval)
y_pred_test = final_knn.predict(X_te_norm)
y_prob_test = final_knn.predict_proba(X_te_norm)[:, 1]

final_metrics = evaluate(y_test, y_pred_test, y_prob_test, label='Final kNN (test)')
print(f"  Overall Test Results:")
print_metrics(final_metrics)

# Full classification report
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_test,
                            target_names=['Real (0)', 'Fake (1)'],
                            digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print(f"  Confusion Matrix:")
print(f"                Predicted Real  Predicted Fake")
print(f"  Actual Real   {cm[0,0]:>12}  {cm[0,1]:>14}")
print(f"  Actual Fake   {cm[1,0]:>12}  {cm[1,1]:>14}")

# Per-language on test
print(f"\n  Per-language test results:")
print(f"  {'Language':<12} {'Samples':>8} {'Accuracy':>10} {'F1':>8} {'AUC':>8}")
print("  " + "─" * 50)
for lang in LANGUAGES:
    mask = (lang_test == lang).values
    if mask.sum() == 0:
        continue
    yp = final_knn.predict(X_te_norm[mask])
    yprob = final_knn.predict_proba(X_te_norm[mask])[:, 1]
    m = evaluate(y_test[mask], yp, yprob)
    print(f"  {lang:<12} {mask.sum():>8} {m['accuracy']:>10.4f} {m['f1']:>8.4f} {m['roc_auc']:>8.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 8  — Save Artefacts
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 8] Saving artefacts...")

with open(os.path.join(OUTPUT_DIR, "knn_final.pkl"), 'wb') as f:
    pickle.dump(final_knn, f)
with open(os.path.join(OUTPUT_DIR, "knn_lang_models.pkl"), 'wb') as f:
    pickle.dump(lang_models, f)

# Save all results as CSV
k_df.to_csv(os.path.join(OUTPUT_DIR, "k_tuning_results.csv"), index=False)
pd.DataFrame(feature_results).to_csv(os.path.join(OUTPUT_DIR, "feature_set_results.csv"), index=False)
pd.DataFrame(metric_results).to_csv(os.path.join(OUTPUT_DIR, "metric_results.csv"), index=False)
pd.DataFrame(lang_results).to_csv(os.path.join(OUTPUT_DIR, "per_language_results.csv"), index=False)

# Save config for downstream phases
config = {
    'best_k': BEST_K,
    'best_metric': 'cosine',
    'best_feature_set': best_fs_name,
    'final_test_f1': round(final_metrics['f1'], 4),
    'final_test_acc': round(final_metrics['accuracy'], 4),
    'final_test_auc': round(final_metrics['roc_auc'], 4),
}
with open(os.path.join(OUTPUT_DIR, "knn_config.pkl"), 'wb') as f:
    pickle.dump(config, f)

# Copy labels forward
for fname in ["y_train.csv", "y_val.csv", "y_test.csv",
              "lang_train.csv", "lang_val.csv", "lang_test.csv"]:
    pd.read_csv(os.path.join(INPUT_DIR, fname)).to_csv(
        os.path.join(OUTPUT_DIR, fname), index=False)

print(f"  ✓ Artefacts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────
# STEP 9  — Visualisations
# ─────────────────────────────────────────────────────────────────
print(f"\n[Step 9] Generating plots...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Phase 4 — kNN Classifier Results", fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#9467BD', '#8C564B']

# ── Plot 1: k vs F1 on validation set ───────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(k_df['label'].str.replace('k=', '').astype(int),
         k_df['f1'], 'o-', color='#4C72B0', linewidth=2, markersize=7)
ax1.axvline(BEST_K, color='#C44E52', linestyle='--', linewidth=1.5,
            label=f'Best k={BEST_K}')
ax1.set_title("k vs Validation F1")
ax1.set_xlabel("k (number of neighbours)")
ax1.set_ylabel("F1 score")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: k vs Accuracy & AUC ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ks = k_df['label'].str.replace('k=', '').astype(int)
ax2.plot(ks, k_df['accuracy'], 's-', color='#55A868', linewidth=2,
         markersize=7, label='Accuracy')
ax2.plot(ks, k_df['roc_auc'],  '^-', color='#DD8452', linewidth=2,
         markersize=7, label='ROC-AUC')
ax2.axvline(BEST_K, color='#C44E52', linestyle='--', linewidth=1.5)
ax2.set_title("k vs Accuracy & AUC")
ax2.set_xlabel("k")
ax2.set_ylabel("Score")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── Plot 3: Feature set comparison bar ──────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
fs_names  = [r['label'] for r in feature_results]
fs_f1     = [r['f1']    for r in feature_results]
short_names = [n.split('(')[0].strip() for n in fs_names]
bars = ax3.bar(short_names, fs_f1, color=COLORS[:len(fs_names)])
ax3.set_title("Feature set F1 comparison")
ax3.set_ylabel("Validation F1")
ax3.set_ylim(max(0, min(fs_f1) - 0.05), min(1.0, max(fs_f1) + 0.05))
ax3.tick_params(axis='x', rotation=20)
for bar, v in zip(bars, fs_f1):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005, f'{v:.3f}', ha='center', fontsize=9)

# ── Plot 4: Confusion matrix ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
ax4.set_title(f"Confusion matrix (test, k={BEST_K})")
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
ax4.set_xticks([0, 1]); ax4.set_xticklabels(['Real', 'Fake'])
ax4.set_yticks([0, 1]); ax4.set_yticklabels(['Real', 'Fake'])
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax4)

# ── Plot 5: Per-language test F1 ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
lang_test_df = pd.DataFrame(lang_results)
if not lang_test_df.empty:
    test_only = lang_test_df[lang_test_df['split'] == 'test'].copy()
    bars = ax5.bar(test_only['language'], test_only['f1'],
                   color=COLORS[:len(test_only)])
    ax5.set_title("Per-language test F1 (monolingual kNN)")
    ax5.set_ylabel("F1 score")
    ax5.set_ylim(0, 1.05)
    for bar, v in zip(bars, test_only['f1']):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01, f'{v:.3f}',
                 ha='center', fontsize=10)

# ── Plot 6: Distance metric comparison ──────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
met_names = [r['label']    for r in metric_results]
met_f1    = [r['f1']       for r in metric_results]
met_acc   = [r['accuracy'] for r in metric_results]
x = np.arange(len(met_names))
w = 0.35
ax6.bar(x - w/2, met_f1,  w, label='F1',       color='#4C72B0', alpha=0.85)
ax6.bar(x + w/2, met_acc, w, label='Accuracy', color='#55A868', alpha=0.85)
ax6.set_title("Distance metric comparison")
ax6.set_xticks(x)
ax6.set_xticklabels(met_names)
ax6.set_ylabel("Score")
ax6.set_ylim(max(0, min(met_f1 + met_acc) - 0.05), 1.0)
ax6.legend()
for i, (f, a) in enumerate(zip(met_f1, met_acc)):
    ax6.text(i - w/2, f  + 0.005, f'{f:.3f}', ha='center', fontsize=8)
    ax6.text(i + w/2, a + 0.005, f'{a:.3f}', ha='center', fontsize=8)

plt.savefig(os.path.join(OUTPUT_DIR, "phase4_knn_results.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved → {OUTPUT_DIR}/phase4_knn_results.png")


# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 4 complete. Summary:")
print(f"  Best k          : {BEST_K}")
print(f"  Best metric     : cosine")
print(f"  Test Accuracy   : {final_metrics['accuracy']:.4f}")
print(f"  Test F1         : {final_metrics['f1']:.4f}")
print(f"  Test ROC-AUC    : {final_metrics['roc_auc']:.4f}")
print(f"\nArtefacts in: ./{OUTPUT_DIR}/")
print("  knn_final.pkl          knn_lang_models.pkl")
print("  k_tuning_results.csv   feature_set_results.csv")
print("  metric_results.csv     per_language_results.csv")
print("  knn_config.pkl         phase4_knn_results.png")
print("\nReady for Phase 5 — Evaluation Framework")
print("=" * 60)
