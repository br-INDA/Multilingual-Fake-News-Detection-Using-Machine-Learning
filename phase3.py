"""
Phase 3 — Similarity & Distance Features
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Lab 2: Matrices – Similarities & Distances
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics.pairwise import (
    cosine_similarity, euclidean_distances, manhattan_distances
)
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_DIR  = "phase2_outputs"
OUTPUT_DIR = "phase3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES    = ['Hindi', 'Marathi', 'Gujarati', 'Telugu']
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────
# STEP 1  — Load Phase 2 Artefacts
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 3 — Similarity & Distance Features")
print("=" * 60)

def load(name):
    return sp.load_npz(os.path.join(INPUT_DIR, name))

X_train = load("X_train_tfidf.npz")
X_val   = load("X_val_tfidf.npz")
X_test  = load("X_test_tfidf.npz")

X_train_word = load("X_train_word.npz")
X_val_word   = load("X_val_word.npz")
X_test_word  = load("X_test_word.npz")

y_train = pd.read_csv(os.path.join(INPUT_DIR, "y_train.csv")).squeeze()
y_val   = pd.read_csv(os.path.join(INPUT_DIR, "y_val.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(INPUT_DIR, "y_test.csv")).squeeze()

lang_train = pd.read_csv(os.path.join(INPUT_DIR, "lang_train.csv")).squeeze()
lang_val   = pd.read_csv(os.path.join(INPUT_DIR, "lang_val.csv")).squeeze()
lang_test  = pd.read_csv(os.path.join(INPUT_DIR, "lang_test.csv")).squeeze()

splits_df = pd.read_csv(os.path.join(INPUT_DIR, "splits.csv"), encoding='utf-8-sig')

print(f"\n✓ Loaded artefacts from '{INPUT_DIR}/'")
print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")


# ─────────────────────────────────────────────────────────────────
# STEP 2  — Build Class Centroids  (mean TF-IDF vector per class)
#           Lab 2: Matrix operations — column-wise mean
# ─────────────────────────────────────────────────────────────────
print("\n[Step 2] Building class centroids from training set...")

def build_centroids(X, y, label_map={0: 'Real', 1: 'Fake'}):
    """Compute L2-normalised mean TF-IDF vector for each class."""
    centroids = {}
    for label, name in label_map.items():
        mask = (y == label).values
        subset = X[mask]
        centroid = subset.mean(axis=0)                    # (1 × features) matrix
        centroid = np.asarray(centroid).flatten()
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)  # L2 normalise
        centroids[name] = centroid
        print(f"  {name} centroid — non-zero dims: "
              f"{(centroid > 0).sum():,} / {len(centroid):,}")
    return centroids

centroids = build_centroids(X_train, y_train)

# Inter-class cosine similarity between Real and Fake centroids
inter_sim = np.dot(centroids['Real'], centroids['Fake'])
print(f"\n  Real ↔ Fake centroid cosine similarity: {inter_sim:.4f}")
print(f"  (closer to 0 = more separable; closer to 1 = harder to distinguish)")


# ─────────────────────────────────────────────────────────────────
# STEP 3  — Per-Language Centroids
# ─────────────────────────────────────────────────────────────────
print("\n[Step 3] Per-language × per-class centroids...")

lang_centroids = {}   # {lang: {label: centroid_vector}}
for lang in LANGUAGES:
    mask = (lang_train == lang)
    if mask.sum() == 0:
        continue
    lang_centroids[lang] = build_centroids(
        X_train[mask.values], y_train[mask]
    )

# Cross-language centroid similarity matrix
print("\n  Cross-language Real-centroid cosine similarity:")
header = f"{'':>12}" + "".join(f"{l:>12}" for l in lang_centroids)
print(header)
for lang_i in lang_centroids:
    row = f"{lang_i:>12}"
    for lang_j in lang_centroids:
        sim = np.dot(lang_centroids[lang_i].get('Real', np.zeros(1)),
                     lang_centroids[lang_j].get('Real', np.zeros(1)))
        row += f"{sim:>12.4f}"
    print(row)


# ─────────────────────────────────────────────────────────────────
# STEP 4  — Distance Features for Every Sample
#           For each sample compute:
#             (a) Cosine similarity  to Fake centroid
#             (b) Cosine similarity  to Real centroid
#             (c) Euclidean distance to Fake centroid
#             (d) Euclidean distance to Real centroid
#             (e) Manhattan distance to Fake centroid
#             (f) Manhattan distance to Real centroid
#             (g) Centroid similarity difference (Fake - Real)
#           Lab 2: product price determination analogy — distances
#           measure how close a sample is to a reference vector
# ─────────────────────────────────────────────────────────────────
print("\n[Step 4] Computing distance features for all splits...")

CENTROID_FAKE = centroids['Fake'].reshape(1, -1)
CENTROID_REAL = centroids['Real'].reshape(1, -1)


def compute_distance_features(X_sparse, label=''):
    """
    Compute 7 distance/similarity features per sample.
    Returns a (n_samples × 7) dense numpy array.
    """
    X_dense = np.asarray(X_sparse.todense())   # needed for Euclidean/Manhattan

    cos_fake = cosine_similarity(X_sparse, CENTROID_FAKE).flatten()
    cos_real = cosine_similarity(X_sparse, CENTROID_REAL).flatten()
    euc_fake = euclidean_distances(X_dense, CENTROID_FAKE).flatten()
    euc_real = euclidean_distances(X_dense, CENTROID_REAL).flatten()
    man_fake = manhattan_distances(X_dense, CENTROID_FAKE).flatten()
    man_real = manhattan_distances(X_dense, CENTROID_REAL).flatten()
    diff_cos = cos_fake - cos_real         # positive → leans fake

    features = np.column_stack([
        cos_fake, cos_real, euc_fake, euc_real,
        man_fake, man_real, diff_cos
    ])
    print(f"  {label}: {features.shape}")
    return features

FEATURE_NAMES_DIST = [
    'cos_sim_fake', 'cos_sim_real',
    'euc_dist_fake', 'euc_dist_real',
    'man_dist_fake', 'man_dist_real',
    'cos_diff_fake_minus_real',
]

dist_train = compute_distance_features(X_train, 'Train')
dist_val   = compute_distance_features(X_val,   'Val  ')
dist_test  = compute_distance_features(X_test,  'Test ')


# ─────────────────────────────────────────────────────────────────
# STEP 5  — Jaccard Similarity  (token overlap — word TF-IDF)
#           Approximated via binarised sparse matrices
# ─────────────────────────────────────────────────────────────────
print("\n[Step 5] Jaccard-based token overlap features...")

def jaccard_to_centroid(X_sparse, centroid_sparse_bin):
    """
    Approximate per-sample Jaccard to a binary centroid.
    J(A,B) = |A∩B| / |A∪B|
    """
    X_bin = (X_sparse > 0).astype(np.float32)
    intersection = X_bin.dot(centroid_sparse_bin.T).toarray().flatten()
    size_a       = np.asarray(X_bin.sum(axis=1)).flatten()
    size_b       = float(centroid_sparse_bin.nnz)
    union        = size_a + size_b - intersection
    jaccard      = intersection / (union + 1e-10)
    return jaccard


def word_centroid_binary(X_word, y, label_val):
    mask = (y == label_val).values
    centroid = X_word[mask].mean(axis=0)
    centroid_bin = (centroid > centroid.mean()).astype(np.float32)
    return sp.csr_matrix(centroid_bin)

word_centroid_fake = word_centroid_binary(X_train_word, y_train, 1)
word_centroid_real = word_centroid_binary(X_train_word, y_train, 0)

jac_fake_train = jaccard_to_centroid(X_train_word, word_centroid_fake)
jac_real_train = jaccard_to_centroid(X_train_word, word_centroid_real)
jac_fake_val   = jaccard_to_centroid(X_val_word,   word_centroid_fake)
jac_real_val   = jaccard_to_centroid(X_val_word,   word_centroid_real)
jac_fake_test  = jaccard_to_centroid(X_test_word,  word_centroid_fake)
jac_real_test  = jaccard_to_centroid(X_test_word,  word_centroid_real)

print(f"  Jaccard fake mean (train) — Fake samples: "
      f"{jac_fake_train[y_train==1].mean():.4f}  "
      f"Real samples: {jac_fake_train[y_train==0].mean():.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 6  — Intra-Class Variance  (spread inside each class)
#           High variance → noisy class  |  Lab 2: matrix analysis
# ─────────────────────────────────────────────────────────────────
print("\n[Step 6] Intra-class variance (spread analysis)...")

for lang in ['all'] + LANGUAGES:
    if lang == 'all':
        X_sub, y_sub = X_train, y_train
        label = 'All languages'
    else:
        mask = (lang_train == lang).values
        if mask.sum() == 0:
            continue
        X_sub, y_sub = X_train[mask], y_train[mask]
        label = lang

    row = f"  {label:<15}"
    for lval, lname in [(0, 'Real'), (1, 'Fake')]:
        lmask = (y_sub == lval).values
        if lmask.sum() == 0:
            row += f"  {lname}: N/A"
            continue
        subset = X_sub[lmask]
        centroid = np.asarray(subset.mean(axis=0))          # (1 × features)
        # convert subset to dense then compute variance
        subset_dense = np.asarray(subset.todense())          # (n × features)
        diffs = subset_dense - centroid                      # broadcast subtract
        variance = np.sqrt((diffs ** 2).sum()) / lmask.sum()
        row += f"  {lname}: {variance:.4f}"
    print(row)


# ─────────────────────────────────────────────────────────────────
# STEP 7  — Dimensionality Reduction for Visualisation
#           SVD (LSA) → 2D projection  (Lab 2: matrix decomposition)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 7] SVD dimensionality reduction for visualisation...")

svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
X_train_svd = svd.fit_transform(X_train)
X_val_svd   = svd.transform(X_val)
X_test_svd  = svd.transform(X_test)

explained = svd.explained_variance_ratio_.cumsum()
print(f"  50 SVD components explain {explained[-1]*100:.1f}% variance")
print(f"  2  SVD components explain {svd.explained_variance_ratio_[:2].sum()*100:.1f}% variance")


# ─────────────────────────────────────────────────────────────────
# STEP 8  — Combine ALL Features into Final Feature Matrix
#           [TF-IDF chars | TF-IDF word | dist features | jaccard | SVD]
# ─────────────────────────────────────────────────────────────────
print("\n[Step 8] Building final combined feature matrix...")

def build_final_features(X_char, X_word, dist_feats, jac_fake, jac_real, X_svd):
    """Horizontally stack all feature groups."""
    extra = np.column_stack([dist_feats, jac_fake, jac_real])
    return sp.hstack([
        X_char,                         # 10,000 char TF-IDF features
        X_word,                         # 10,000 word TF-IDF features
        sp.csr_matrix(extra),           # 9 distance/Jaccard features
        sp.csr_matrix(X_svd),           # 50 SVD/LSA features
    ], format='csr')

X_train_final = build_final_features(
    X_train, X_train_word, dist_train, jac_fake_train, jac_real_train, X_train_svd)
X_val_final   = build_final_features(
    X_val,   X_val_word,   dist_val,   jac_fake_val,   jac_real_val,   X_val_svd)
X_test_final  = build_final_features(
    X_test,  X_test_word,  dist_test,  jac_fake_test,  jac_real_test,  X_test_svd)

print(f"  Final feature dimensions:")
print(f"    Train : {X_train_final.shape}")
print(f"    Val   : {X_val_final.shape}")
print(f"    Test  : {X_test_final.shape}")
print(f"  Feature groups:")
print(f"    Char TF-IDF : {X_train.shape[1]:>6} features")
print(f"    Word TF-IDF : {X_train_word.shape[1]:>6} features")
print(f"    Distance    : {len(FEATURE_NAMES_DIST):>6} features  (cosine, euclidean, manhattan)")
print(f"    Jaccard     : {2:>6} features  (fake centroid, real centroid)")
print(f"    SVD/LSA     : {X_train_svd.shape[1]:>6} features")
print(f"    ─────────────────────────")
print(f"    Total       : {X_train_final.shape[1]:>6} features")


# ─────────────────────────────────────────────────────────────────
# STEP 9  — Save All Artefacts
# ─────────────────────────────────────────────────────────────────
print("\n[Step 9] Saving artefacts...")

sp.save_npz(os.path.join(OUTPUT_DIR, "X_train_final.npz"), X_train_final)
sp.save_npz(os.path.join(OUTPUT_DIR, "X_val_final.npz"),   X_val_final)
sp.save_npz(os.path.join(OUTPUT_DIR, "X_test_final.npz"),  X_test_final)

np.save(os.path.join(OUTPUT_DIR, "X_train_svd.npy"), X_train_svd)
np.save(os.path.join(OUTPUT_DIR, "X_val_svd.npy"),   X_val_svd)
np.save(os.path.join(OUTPUT_DIR, "X_test_svd.npy"),  X_test_svd)

np.save(os.path.join(OUTPUT_DIR, "dist_train.npy"), dist_train)
np.save(os.path.join(OUTPUT_DIR, "dist_val.npy"),   dist_val)
np.save(os.path.join(OUTPUT_DIR, "dist_test.npy"),  dist_test)

with open(os.path.join(OUTPUT_DIR, "centroids.pkl"), 'wb') as f:
    pickle.dump({'global': centroids, 'per_language': lang_centroids}, f)
with open(os.path.join(OUTPUT_DIR, "svd_model.pkl"), 'wb') as f:
    pickle.dump(svd, f)

# Copy labels and language columns forward (phases 4+ need them)
for fname in ["y_train.csv", "y_val.csv", "y_test.csv",
              "lang_train.csv", "lang_val.csv", "lang_test.csv",
              "splits.csv"]:
    src = pd.read_csv(os.path.join(INPUT_DIR, fname))
    src.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

print(f"  ✓ All artefacts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────
# STEP 10 — Visualisations
# ─────────────────────────────────────────────────────────────────
print("\n[Step 10] Generating plots...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Phase 3 — Similarity & Distance Feature Analysis", fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

COLORS = {0: '#4C72B0', 1: '#C44E52'}
LNAMES = {0: 'Real', 1: 'Fake'}

# ── Plot 1: Cosine similarity to Fake centroid (train) ──────────
ax1 = fig.add_subplot(gs[0, 0])
for lval in [0, 1]:
    mask = (y_train == lval).values
    ax1.hist(dist_train[mask, 0], bins=40, alpha=0.6,
             color=COLORS[lval], label=LNAMES[lval], density=True)
ax1.set_title("Cosine sim to Fake centroid")
ax1.set_xlabel("Cosine similarity")
ax1.set_ylabel("Density")
ax1.legend()

# ── Plot 2: Euclidean distance to Fake centroid ─────────────────
ax2 = fig.add_subplot(gs[0, 1])
for lval in [0, 1]:
    mask = (y_train == lval).values
    ax2.hist(dist_train[mask, 2], bins=40, alpha=0.6,
             color=COLORS[lval], label=LNAMES[lval], density=True)
ax2.set_title("Euclidean dist to Fake centroid")
ax2.set_xlabel("Distance")
ax2.set_ylabel("Density")
ax2.legend()

# ── Plot 3: Cosine diff (Fake − Real) — discrimination signal ───
ax3 = fig.add_subplot(gs[0, 2])
for lval in [0, 1]:
    mask = (y_train == lval).values
    ax3.hist(dist_train[mask, 6], bins=40, alpha=0.6,
             color=COLORS[lval], label=LNAMES[lval], density=True)
ax3.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
ax3.set_title("Cosine diff (Fake − Real sim)")
ax3.set_xlabel("Score (+ → fake, − → real)")
ax3.set_ylabel("Density")
ax3.legend()

# ── Plot 4: SVD 2D scatter (first 2 components) ─────────────────
ax4 = fig.add_subplot(gs[1, 0])
sample_idx = np.random.RandomState(42).choice(len(X_train_svd),
                                               size=min(800, len(X_train_svd)),
                                               replace=False)
for lval in [0, 1]:
    mask = (y_train.values[sample_idx] == lval)
    ax4.scatter(X_train_svd[sample_idx][mask, 0],
                X_train_svd[sample_idx][mask, 1],
                c=COLORS[lval], label=LNAMES[lval],
                alpha=0.4, s=12, edgecolors='none')
ax4.set_title("SVD 2D projection (LSA)")
ax4.set_xlabel("SVD component 1")
ax4.set_ylabel("SVD component 2")
ax4.legend()

# ── Plot 5: Per-language cosine diff boxplot ─────────────────────
ax5 = fig.add_subplot(gs[1, 1])
box_data, box_labels, box_colors = [], [], []
for lang in LANGUAGES:
    for lval, lname in [(0, 'Real'), (1, 'Fake')]:
        mask_l = (lang_train == lang).values
        mask_c = (y_train == lval).values
        combined_mask = mask_l & mask_c
        if combined_mask.sum() == 0:
            continue
        box_data.append(dist_train[combined_mask, 6])
        box_labels.append(f"{lang}\n{lname}")
        box_colors.append(COLORS[lval])

bp = ax5.boxplot(box_data, patch_artist=True,
                 medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax5.set_xticks(range(1, len(box_labels) + 1))
ax5.set_xticklabels(box_labels, fontsize=7)
ax5.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax5.set_title("Cosine diff by language & class")
ax5.set_ylabel("Cosine diff score")

# ── Plot 6: SVD explained variance ──────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
cumvar = svd.explained_variance_ratio_.cumsum() * 100
ax6.plot(range(1, len(cumvar) + 1), cumvar, color='#4C72B0', linewidth=2)
ax6.axhline(80, color='grey', linestyle='--', linewidth=1, alpha=0.7)
ax6.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.15, color='#4C72B0')
ax6.set_title("SVD cumulative explained variance")
ax6.set_xlabel("Number of components")
ax6.set_ylabel("Variance explained (%)")
ax6.set_ylim(0, 100)

plt.savefig(os.path.join(OUTPUT_DIR, "phase3_overview.png"), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved → {OUTPUT_DIR}/phase3_overview.png")


# ─────────────────────────────────────────────────────────────────
# Quick feature discrimination report
# ─────────────────────────────────────────────────────────────────
print("\n[Feature Discrimination Summary — Train Set]")
print(f"{'Feature':<30} {'Fake mean':>12} {'Real mean':>12} {'Difference':>12}")
print("─" * 68)
feat_labels = FEATURE_NAMES_DIST + ['jaccard_fake', 'jaccard_real']
all_dist = np.column_stack([
    dist_train, jac_fake_train, jac_real_train
])
for i, fname in enumerate(feat_labels):
    fake_mean = all_dist[y_train == 1, i].mean()
    real_mean = all_dist[y_train == 0, i].mean()
    diff = fake_mean - real_mean
    marker = ' ◀' if abs(diff) > 0.01 else ''
    print(f"  {fname:<28} {fake_mean:>12.5f} {real_mean:>12.5f} {diff:>+12.5f}{marker}")

print("\n  ◀ = features with notable separation between Fake and Real")

print("\n" + "=" * 60)
print("Phase 3 complete. Artefacts ready for Phase 4 (kNN Classifier).")
print(f"Outputs in: ./{OUTPUT_DIR}/")
print("  X_train_final.npz  X_val_final.npz  X_test_final.npz")
print("  X_train_svd.npy    X_val_svd.npy    X_test_svd.npy")
print("  dist_train.npy     dist_val.npy     dist_test.npy")
print("  centroids.pkl      svd_model.pkl")
print("  phase3_overview.png")
print("=" * 60)
