"""
Phase 2 — Multilingual Preprocessing & Vectorization
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Lab 1: Data Handling  |  Lab 2: Similarities & Distances
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_FILE   = "cleaned_multilingual_dataset.csv"   # output from Phase 1
OUTPUT_DIR   = "phase2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES    = ['Hindi', 'Marathi', 'Gujarati', 'Telugu']
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# TF-IDF settings
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE  = (1, 2)      # unigrams + bigrams


# ─────────────────────────────────────────────────────────────────
# STEP 1  — Load cleaned dataset (from Phase 1)
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 2 — Preprocessing & Vectorization")
print("=" * 60)

df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
print(f"\n✓ Loaded {len(df):,} rows from '{INPUT_FILE}'")
print(f"  Columns : {df.columns.tolist()}")
print(f"  Languages: {df['language'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────
# STEP 2  — Text Cleaning  (language-aware)
# ─────────────────────────────────────────────────────────────────

# Unicode ranges for Indic scripts
SCRIPT_RANGES = {
    'Hindi':    r'[\u0900-\u097F]',   # Devanagari  (also covers Marathi)
    'Marathi':  r'[\u0900-\u097F]',   # Devanagari
    'Gujarati': r'[\u0A80-\u0AFF]',   # Gujarati
    'Telugu':   r'[\u0C00-\u0C7F]',   # Telugu
}

# Common stopwords per language (extend as needed)
STOPWORDS = {
    'Hindi': {
        'है', 'हैं', 'था', 'थे', 'थी', 'और', 'का', 'की', 'के', 'में',
        'से', 'पर', 'को', 'ने', 'एक', 'यह', 'वह', 'जो', 'कि', 'या',
        'हो', 'तो', 'भी', 'इस', 'उस', 'लिए', 'अब', 'बाद', 'बहुत',
        'सकता', 'सकती', 'करना', 'कर', 'किया', 'हुए', 'रहे', 'गए',
    },
    'Marathi': {
        'आहे', 'आहेत', 'होते', 'होती', 'होता', 'आणि', 'का', 'की', 'च्या',
        'मध्ये', 'त्या', 'एक', 'या', 'त्यांनी', 'करण्यात', 'असल्याचे',
        'सांगितले', 'झाले', 'येथे', 'केले', 'असून', 'तर', 'व', 'ते',
    },
    'Gujarati': {
        'છે', 'હતું', 'હતી', 'હતો', 'અને', 'ની', 'ના', 'ને', 'માં',
        'એ', 'આ', 'એક', 'પણ', 'કે', 'સાથે', 'પર', 'થઈ', 'થયું',
        'કર્યું', 'જ', 'તો', 'ત્યારે', 'કહ્યું', 'લઈ', 'વધુ',
    },
    'Telugu': {
        'ఉంది', 'ఉన్నారు', 'అయింది', 'అయ్యింది', 'మరియు', 'కి', 'కు',
        'లో', 'నుండి', 'ఒక', 'ఈ', 'ఆ', 'అని', 'కానీ', 'అయితే', 'తో',
        'వారు', 'చేసారు', 'చేశారు', 'అయిన', 'కూడా', 'అయినప్పుడు',
    },
}


def clean_text(text: str, language: str) -> str:
    """
    Clean a single news text:
    1. Lowercase (Latin chars)
    2. Remove URLs, emails, mentions, hashtags
    3. Keep only script-appropriate Unicode + spaces + digits
    4. Collapse whitespace
    5. Remove stopwords
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Lowercase Latin portion
    text = text.lower()

    # Remove noise
    text = re.sub(r'http\S+|www\.\S+', ' ', text)          # URLs
    text = re.sub(r'\S+@\S+', ' ', text)                    # emails
    text = re.sub(r'[@#]\w+', ' ', text)                    # mentions / hashtags
    text = re.sub(r'\d{10,}', ' ', text)                    # long number strings (phone/ID)

    # Keep: Indic script chars + common Latin + spaces + digits
    script = SCRIPT_RANGES.get(language, r'[\u0900-\u0C7F]')
    keep_pattern = rf'[^{script[1:-1]}a-z0-9\s।॥,.!?]'
    text = re.sub(keep_pattern, ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords (token-level)
    stops = STOPWORDS.get(language, set())
    tokens = text.split()
    tokens = [t for t in tokens if t not in stops and len(t) > 1]
    return ' '.join(tokens)


print("\n[Step 2] Cleaning text...")
df['text_clean'] = df.apply(
    lambda row: clean_text(row['text'], row['language']), axis=1
)

# Drop rows where cleaning left nothing
before = len(df)
df = df[df['text_clean'].str.strip().str.len() > 5].reset_index(drop=True)
print(f"  Dropped {before - len(df)} empty rows after cleaning. Remaining: {len(df):,}")

# Quick sanity check — show one cleaned example per language
print("\n  Sample cleaned texts:")
for lang in LANGUAGES:
    sample = df[df['language'] == lang]['text_clean'].iloc[0] if lang in df['language'].values else 'N/A'
    print(f"  [{lang}] {sample[:120]}...")


# ─────────────────────────────────────────────────────────────────
# STEP 3  — Token Statistics  (before vectorization)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 3] Token statistics per language...")

token_stats = []
for lang in LANGUAGES:
    sub = df[df['language'] == lang]['text_clean']
    if sub.empty:
        continue
    lengths = sub.str.split().str.len()
    all_tokens = ' '.join(sub).split()
    vocab_size = len(set(all_tokens))
    token_stats.append({
        'Language':     lang,
        'Samples':      len(sub),
        'Avg tokens':   round(lengths.mean(), 1),
        'Min tokens':   lengths.min(),
        'Max tokens':   lengths.max(),
        'Vocab size':   vocab_size,
    })

stats_df = pd.DataFrame(token_stats).set_index('Language')
print(f"\n{stats_df.to_string()}")


# ─────────────────────────────────────────────────────────────────
# STEP 4  — Train / Validation / Test Split  (stratified)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 4] Train/Val/Test split (60/20/20, stratified)...")

X = df['text_clean']
y = df['label'].astype(int)
lang_col = df['language']

# 60 train / 20 val / 20 test
X_train, X_temp, y_train, y_temp, lang_train, lang_temp = train_test_split(
    X, y, lang_col,
    test_size=0.40,
    stratify=y,
    random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test, lang_val, lang_test = train_test_split(
    X_temp, y_temp, lang_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"  Train : {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
print(f"  Train label balance — Fake: {y_train.sum()}  Real: {(y_train==0).sum()}")
print(f"  Test  label balance — Fake: {y_test.sum()}   Real: {(y_test==0).sum()}")


# ─────────────────────────────────────────────────────────────────
# STEP 5  — TF-IDF Vectorization  (3 strategies)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 5] TF-IDF Vectorization...")


# ── 5A. Combined multilingual TF-IDF ──────────────────────────────
print("  5A. Combined multilingual TF-IDF...")
tfidf_combined = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
    sublinear_tf=True,             # log(1+tf) — better for long texts
    min_df=2,                      # ignore terms that appear in < 2 docs
    analyzer='char_wb',            # character n-grams — language-agnostic
)
X_train_tfidf = tfidf_combined.fit_transform(X_train)
X_val_tfidf   = tfidf_combined.transform(X_val)
X_test_tfidf  = tfidf_combined.transform(X_test)

print(f"    Combined TF-IDF shape: {X_train_tfidf.shape}")


# ── 5B. Word-level TF-IDF ─────────────────────────────────────────
print("  5B. Word-level TF-IDF...")
tfidf_word = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    analyzer='word',
)
X_train_word = tfidf_word.fit_transform(X_train)
X_val_word   = tfidf_word.transform(X_val)
X_test_word  = tfidf_word.transform(X_test)

print(f"    Word TF-IDF shape: {X_train_word.shape}")


# ── 5C. Per-language TF-IDF (separate model per language) ─────────
print("  5C. Per-language TF-IDF...")
lang_tfidf_models  = {}
lang_train_vectors = {}
lang_test_vectors  = {}

for lang in LANGUAGES:
    mask_train = lang_train == lang
    mask_test  = lang_test  == lang

    if mask_train.sum() < 10:
        print(f"    [{lang}] not enough samples, skipping.")
        continue

    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        analyzer='char_wb',
    )
    X_tr = vec.fit_transform(X_train[mask_train])
    X_te = vec.transform(X_test[mask_test]) if mask_test.sum() > 0 else None

    lang_tfidf_models[lang]  = vec
    lang_train_vectors[lang] = (X_tr, y_train[mask_train])
    lang_test_vectors[lang]  = (X_te, y_test[mask_test])

    print(f"    [{lang}] train: {X_tr.shape}  test: {X_te.shape if X_te is not None else 'N/A'}")


# ─────────────────────────────────────────────────────────────────
# STEP 6  — Top TF-IDF Terms per Language & Class
#           (Lab 2: understand feature distributions)
# ─────────────────────────────────────────────────────────────────
print("\n[Step 6] Top TF-IDF terms per language and class...")

TOP_N = 10
top_terms_report = {}

for lang in LANGUAGES:
    mask = (lang_train == lang)
    if mask.sum() == 0:
        continue

    lang_X     = X_train[mask]
    lang_y     = y_train[mask]
    lang_vocab = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, analyzer='char_wb'
    )
    lang_mat = lang_vocab.fit_transform(lang_X)
    feat_names = np.array(lang_vocab.get_feature_names_out())

    results = {}
    for label_val, label_name in [(0, 'Real'), (1, 'Fake')]:
        idx = (lang_y == label_val).values
        if idx.sum() == 0:
            continue
        mean_tfidf = lang_mat[idx].mean(axis=0).A1
        top_idx    = mean_tfidf.argsort()[::-1][:TOP_N]
        results[label_name] = list(zip(feat_names[top_idx], mean_tfidf[top_idx].round(4)))

    top_terms_report[lang] = results
    print(f"\n  [{lang}]")
    for cls, terms in results.items():
        term_str = ', '.join([f"{t}({s:.3f})" for t, s in terms[:5]])
        print(f"    {cls}: {term_str}")


# ─────────────────────────────────────────────────────────────────
# STEP 7  — Optional: Sentence Embeddings  (LaBSE via sentence-transformers)
#           Uncomment if you have sentence-transformers installed
# ─────────────────────────────────────────────────────────────────
ENABLE_EMBEDDINGS = False   # set True if sentence-transformers is available

if ENABLE_EMBEDDINGS:
    print("\n[Step 7] Generating LaBSE sentence embeddings...")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('sentence-transformers/LaBSE')

        print("  Encoding train set (this may take a few minutes)...")
        X_train_emb = model.encode(X_train.tolist(), batch_size=64,
                                   show_progress_bar=True, normalize_embeddings=True)
        X_val_emb   = model.encode(X_val.tolist(),   batch_size=64,
                                   show_progress_bar=True, normalize_embeddings=True)
        X_test_emb  = model.encode(X_test.tolist(),  batch_size=64,
                                   show_progress_bar=True, normalize_embeddings=True)

        np.save(os.path.join(OUTPUT_DIR, "X_train_emb.npy"), X_train_emb)
        np.save(os.path.join(OUTPUT_DIR, "X_val_emb.npy"),   X_val_emb)
        np.save(os.path.join(OUTPUT_DIR, "X_test_emb.npy"),  X_test_emb)
        print(f"  Embedding shape: {X_train_emb.shape}  (saved to {OUTPUT_DIR}/)")
    except ImportError:
        print("  sentence-transformers not installed. Run: pip install sentence-transformers")
else:
    print("\n[Step 7] Skipping LaBSE embeddings (ENABLE_EMBEDDINGS=False).")
    print("  To enable: pip install sentence-transformers  then set ENABLE_EMBEDDINGS=True")


# ─────────────────────────────────────────────────────────────────
# STEP 8  — Save All Artefacts for Phase 3+
# ─────────────────────────────────────────────────────────────────
print("\n[Step 8] Saving artefacts...")

import scipy.sparse as sp

sp.save_npz(os.path.join(OUTPUT_DIR, "X_train_tfidf.npz"), X_train_tfidf)
sp.save_npz(os.path.join(OUTPUT_DIR, "X_val_tfidf.npz"),   X_val_tfidf)
sp.save_npz(os.path.join(OUTPUT_DIR, "X_test_tfidf.npz"),  X_test_tfidf)

sp.save_npz(os.path.join(OUTPUT_DIR, "X_train_word.npz"), X_train_word)
sp.save_npz(os.path.join(OUTPUT_DIR, "X_val_word.npz"),   X_val_word)
sp.save_npz(os.path.join(OUTPUT_DIR, "X_test_word.npz"),  X_test_word)

y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_val.to_csv(os.path.join(OUTPUT_DIR,   "y_val.csv"),   index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR,  "y_test.csv"),  index=False)

lang_train.to_csv(os.path.join(OUTPUT_DIR, "lang_train.csv"), index=False)
lang_val.to_csv(os.path.join(OUTPUT_DIR,   "lang_val.csv"),   index=False)
lang_test.to_csv(os.path.join(OUTPUT_DIR,  "lang_test.csv"),  index=False)

with open(os.path.join(OUTPUT_DIR, "tfidf_combined.pkl"), 'wb') as f:
    pickle.dump(tfidf_combined, f)
with open(os.path.join(OUTPUT_DIR, "tfidf_word.pkl"), 'wb') as f:
    pickle.dump(tfidf_word, f)
with open(os.path.join(OUTPUT_DIR, "lang_tfidf_models.pkl"), 'wb') as f:
    pickle.dump(lang_tfidf_models, f)

# Save cleaned split texts for Phase 3
split_df = pd.DataFrame({
    'text_clean': pd.concat([X_train, X_val, X_test]),
    'label':      pd.concat([y_train, y_val, y_test]),
    'language':   pd.concat([lang_train, lang_val, lang_test]),
    'split':      (['train'] * len(X_train) + ['val'] * len(X_val) + ['test'] * len(X_test))
})
split_df.to_csv(os.path.join(OUTPUT_DIR, "splits.csv"), index=False, encoding='utf-8-sig')

print(f"  ✓ All artefacts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────
# STEP 9  — Visualisations
# ─────────────────────────────────────────────────────────────────
print("\n[Step 9] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 2 — Preprocessing & Vectorization Overview", fontsize=14, fontweight='bold')

# ── Plot 1: Token length distribution per language ──────────────
ax = axes[0, 0]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
for lang, color in zip(LANGUAGES, colors):
    sub = df[df['language'] == lang]['text_clean']
    if sub.empty:
        continue
    lengths = sub.str.split().str.len()
    ax.hist(lengths, bins=30, alpha=0.55, label=lang, color=color, density=True)
ax.set_title("Token length distribution")
ax.set_xlabel("Tokens per article")
ax.set_ylabel("Density")
ax.legend()

# ── Plot 2: Vocabulary size per language ────────────────────────
ax = axes[0, 1]
langs_plot = [r['Language'] for r in token_stats]
vocab_vals = [r['Vocab size'] for r in token_stats]
bars = ax.bar(langs_plot, vocab_vals, color=colors[:len(langs_plot)])
ax.set_title("Vocabulary size per language")
ax.set_ylabel("Unique tokens")
for bar, v in zip(bars, vocab_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{v:,}', ha='center', fontsize=9)

# ── Plot 3: TF-IDF feature density (sparsity) ───────────────────
ax = axes[1, 0]
densities = [
    X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]),
    X_train_word.nnz  / (X_train_word.shape[0]  * X_train_word.shape[1]),
]
labels = ['Char n-gram\nTF-IDF', 'Word n-gram\nTF-IDF']
ax.bar(labels, [d * 100 for d in densities], color=['#4C72B0', '#55A868'])
ax.set_title("TF-IDF matrix density (%)")
ax.set_ylabel("Non-zero entries (%)")
for i, d in enumerate(densities):
    ax.text(i, d * 100 + 0.002, f'{d*100:.3f}%', ha='center', fontsize=10)

# ── Plot 4: Train/Val/Test split per language ───────────────────
ax = axes[1, 1]
split_counts = pd.DataFrame({
    'Train': lang_train.value_counts(),
    'Val':   lang_val.value_counts(),
    'Test':  lang_test.value_counts(),
}).fillna(0).astype(int)
split_counts.plot(kind='bar', ax=ax, color=['#4C72B0', '#F0C040', '#C44E52'])
ax.set_title("Train / Val / Test split per language")
ax.set_ylabel("Samples")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
ax.legend(loc='upper right')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "phase2_overview.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved plot → {plot_path}")

# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 2 complete. Artefacts ready for Phase 3 (kNN + Similarity).")
print(f"Outputs in: ./{OUTPUT_DIR}/")
print("  X_train_tfidf.npz  X_val_tfidf.npz  X_test_tfidf.npz")
print("  X_train_word.npz   X_val_word.npz   X_test_word.npz")
print("  y_train.csv  y_val.csv  y_test.csv")
print("  lang_train.csv  lang_val.csv  lang_test.csv")
print("  tfidf_combined.pkl  tfidf_word.pkl  lang_tfidf_models.pkl")
print("  splits.csv  phase2_overview.png")
print("=" * 60)
