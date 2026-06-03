"""
Phase 1 — Data Collection & Handling
Multilingual Fake News Detection (Marathi, Gujarati, Telugu, Hindi)
Lab 1: Data Handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'   # safe for terminal; swap for a Unicode font if needed
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. File paths — update these to your actual paths
# ─────────────────────────────────────────────
FILES = {
    'Marathi':  'Marathi_news.xlsx',    
    'Gujarati': 'Gujarati_news.xlsx',
    'Telugu':   'Telugu_news.xlsx',
    'Hindi':    'Hindi_news.xlsx',
}

# ─────────────────────────────────────────────
# 2. Smart loader — handles CSV, Excel, JSON
# ─────────────────────────────────────────────
def load_file(path: str) -> pd.DataFrame:
    """Auto-detect format and load into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    elif ext == '.json':
        return pd.read_json(path)
    else:                          # default: CSV (also works for no extension)
        # Try common encodings for Indic scripts
        for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                return pd.read_csv(path, encoding=enc)
            except (UnicodeDecodeError, Exception):
                continue
        raise ValueError(f"Could not read {path} with any known encoding.")


# ─────────────────────────────────────────────
# 3. Standardise column names
#    Expected columns: 'text' (news content) + 'label' (fake/real)
#    Adjust the mapping below to match YOUR column names
# ─────────────────────────────────────────────
COLUMN_MAP = {
    # common variants → standard name
    'news':      'text',
    'content':   'text',
    'headline':  'text',
    'title':     'text',
    'article':   'text',
    'News':      'text',
    'Text':      'text',

    'Label':     'label',
    'class':     'label',
    'Category':  'label',
    'target':    'label',
}

LABEL_MAP = {
    # normalise label values → 0 (real) / 1 (fake)
    'fake': 1, 'Fake': 1, 'FAKE': 1, 'false': 1, 'False': 1,
    'real': 0, 'Real': 0, 'REAL': 0, 'true':  0, 'True':  0,
    0: 0, 1: 1,
}


def standardise(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Rename columns and normalise labels."""
    df = df.rename(columns=COLUMN_MAP)

    if 'text' not in df.columns:
        # Fall back: use first string column as text
        str_cols = df.select_dtypes(include='object').columns.tolist()
        if str_cols:
            df = df.rename(columns={str_cols[0]: 'text'})
            print(f"  [{lang}] 'text' column not found — using '{str_cols[0]}' instead.")
        else:
            raise KeyError(f"[{lang}] No text column found. Columns: {df.columns.tolist()}")

    if 'label' not in df.columns:
        str_cols = [c for c in df.select_dtypes(include='object').columns if c != 'text']
        if str_cols:
            df = df.rename(columns={str_cols[0]: 'label'})
            print(f"  [{lang}] 'label' column not found — using '{str_cols[0]}' instead.")

    df['language'] = lang

    if 'label' in df.columns:
        df['label'] = df['label'].map(LABEL_MAP).fillna(df['label'])

    return df


# ─────────────────────────────────────────────
# 4. Load all datasets
# ─────────────────────────────────────────────
print("=" * 60)
print("  PHASE 1 — Data Collection & Handling")
print("=" * 60)

dfs = {}
for lang, path in FILES.items():
    if not os.path.exists(path):
        print(f"\n[WARNING] File not found: {path}  (skipping)")
        continue
    try:
        df = load_file(path)
        df = standardise(df, lang)
        dfs[lang] = df
        print(f"\n✓ {lang}: {df.shape[0]:,} rows × {df.shape[1]} cols  |  columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"\n[ERROR] {lang}: {e}")

if not dfs:
    raise SystemExit("No files loaded. Check your file paths above.")


# ─────────────────────────────────────────────
# 5. Per-language inspection
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("PER-LANGUAGE SUMMARY")
print("─" * 60)

for lang, df in dfs.items():
    print(f"\n── {lang} ──")

    # Sample rows
    print(df[['text', 'label']].head(3).to_string(index=False))

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n  Missing values:\n{missing[missing > 0]}")
    else:
        print("\n  No missing values.")

    # Class distribution
    if 'label' in df.columns:
        counts = df['label'].value_counts().sort_index()
        label_names = {0: 'Real (0)', 1: 'Fake (1)'}
        print("\n  Class distribution:")
        for k, v in counts.items():
            pct = v / len(df) * 100
            print(f"    {label_names.get(k, k)}: {v:,}  ({pct:.1f}%)")
    
    # Text length stats
    df['text_len'] = df['text'].astype(str).str.len()
    print(f"\n  Text length (chars) — mean: {df['text_len'].mean():.0f}, "
          f"min: {df['text_len'].min()}, max: {df['text_len'].max()}")


# ─────────────────────────────────────────────
# 6. Merge all languages
# ─────────────────────────────────────────────
combined = pd.concat(dfs.values(), ignore_index=True)
print("\n" + "─" * 60)
print(f"COMBINED DATASET: {len(combined):,} rows  |  languages: {combined['language'].unique().tolist()}")
print("─" * 60)


# ─────────────────────────────────────────────
# 7. Data Quality Checks
# ─────────────────────────────────────────────
print("\n[Data Quality]")

# 7a. Duplicates
n_dup = combined.duplicated(subset='text').sum()
print(f"  Duplicate texts: {n_dup}  ({n_dup/len(combined)*100:.1f}%)")
combined = combined.drop_duplicates(subset='text').reset_index(drop=True)
print(f"  After deduplication: {len(combined):,} rows")

# 7b. Empty/very short texts
too_short = (combined['text'].astype(str).str.strip().str.len() < 10).sum()
print(f"  Texts < 10 chars (likely garbage): {too_short}")
combined = combined[combined['text'].astype(str).str.strip().str.len() >= 10].reset_index(drop=True)

# 7c. Missing labels
if 'label' in combined.columns:
    n_null_labels = combined['label'].isnull().sum()
    print(f"  Rows with missing label: {n_null_labels}")
    combined = combined.dropna(subset=['label']).reset_index(drop=True)

print(f"\n  Final clean dataset: {len(combined):,} rows")


# ─────────────────────────────────────────────
# 8. Cross-language class balance overview
# ─────────────────────────────────────────────
print("\n[Cross-Language Class Balance]")
balance = combined.groupby(['language', 'label']).size().unstack(fill_value=0)
balance.columns = ['Real', 'Fake'] if len(balance.columns) == 2 else balance.columns
balance['Total'] = balance.sum(axis=1)
balance['Fake%'] = (balance.get('Fake', 0) / balance['Total'] * 100).round(1)
print(balance.to_string())


# ─────────────────────────────────────────────
# 9. Visualisations  (saved as PNG)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase 1 — Dataset Overview", fontsize=14, fontweight='bold')

# 9a. Sample count per language
counts_lang = combined['language'].value_counts()
axes[0].bar(counts_lang.index, counts_lang.values,
            color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
axes[0].set_title("Samples per language")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts_lang.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontsize=10)

# 9b. Class balance stacked bar per language
if 'label' in combined.columns:
    pivot = combined.groupby(['language', 'label']).size().unstack(fill_value=0)
    pivot.plot(kind='bar', stacked=True, ax=axes[1],
               color=['#4C72B0', '#C44E52'], legend=True)
    axes[1].set_title("Fake vs Real per language")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("")
    axes[1].legend(['Real (0)', 'Fake (1)'])
    axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig("phase1_overview.png", dpi=150, bbox_inches='tight')
print("\n[Saved] phase1_overview.png")


# ─────────────────────────────────────────────
# 10. Save cleaned combined dataset
# ─────────────────────────────────────────────
combined.to_csv("cleaned_multilingual_dataset.csv", index=False, encoding='utf-8-sig')
print("[Saved] cleaned_multilingual_dataset.csv")

print("\n" + "=" * 60)
print("Phase 1 complete. Ready for Phase 2 (Preprocessing & Vectorization).")
print("=" * 60)
