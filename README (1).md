# 🔍 Multilingual Fake News Detection Using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?style=for-the-badge&logo=flask&logoColor=white)
![MuRIL](https://img.shields.io/badge/MuRIL-768dim-green?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A 10-phase end-to-end machine learning pipeline for detecting fake news
across four Indian languages — Hindi, Marathi, Gujarati, and Telugu —
with stacking ensemble, SHAP/LIME explainability, and a Flask-based
Chrome extension API.**

[Overview](#-overview) •
[Dataset](#-dataset) •
[Pipeline](#-pipeline) •
[Results](#-results) •
[Installation](#-installation) •
[Usage](#-usage) •
[Explainability](#-explainability) •
[API](#-flask-rest-api)

</div>

---

## 📌 Overview

Fake news in Indian regional languages is a rapidly growing problem
with limited computational solutions.
This project builds a **comprehensive, reproducible ML pipeline** that:

- Processes **76,949 news articles** across 4 Indian languages
- Extracts features using **TF-IDF** (character and word n-grams) and
  **MuRIL transformer embeddings** (768-dim)
- Trains and evaluates **10+ classifiers** including ensemble methods
- Achieves a **stacking ensemble F1 of 0.9965** and **AUC of 0.9999**
  on a held-out test set
- Explains predictions using **SHAP** and **LIME**
- Deploys as a **Flask REST API** powering a Chrome browser extension

---

## 📂 Repository Structure

```
multilingual-fake-news/
│
├── phase1.py                  # Data collection & cleaning
├── phase2.py                  # Preprocessing & TF-IDF vectorisation
├── phase3.py                  # Naive Bayes baseline
├── phase4.py                  # k-NN classifier
├── phase5.py                  # Logistic Regression
├── phase6.py                  # SVM (Linear & RBF)
├── phase7.py                  # Random Forest & Extra Trees
├── phase8.py                  # Gradient Boosting, Perceptron & MLP
├── phase9.py                  # Stacking Ensemble
├── phase10.py                 # Feature Selection & Explainability
│
├── muril_embeddings.ipynb     # MuRIL (768-dim) embedding extraction
├── app.py                     # Flask REST API
│
├── phase8_outputs/            # Saved models & results (Phase 8)
│   ├── rf_final.pkl
│   ├── gb_final.pkl
│   ├── nb_final.pkl
│   ├── perc_final.pkl
│   ├── mlp_final.pkl
│   ├── scaler_mlp.pkl
│   ├── model_comparison.csv
│   └── phase8_classifiers_results.png
│
├── phase9_outputs/            # Stacking ensemble artefacts (Phase 9)
│   ├── final_meta_learner.pkl
│   ├── final_base_models.pkl
│   ├── OOF_META.npy
│   ├── TEST_META.npy
│   ├── base_learner_metrics.csv
│   ├── per_language_stacking.csv
│   └── phase9_stacking_results.png
│
├── phase10_outputs/           # Feature selection & SHAP/LIME (Phase 10)
│   ├── shap_values.npy
│   ├── lime_explanations.csv
│   ├── feature_importance_table.csv
│   ├── language_bias.csv
│   └── phase10_explainability.png
│
├── cleaned_multilingual_dataset.csv
├── muril1.npy                 # MuRIL embeddings (76949 × 768)
└── requirements.txt
```

---

## 📊 Dataset

### Sources
News articles collected across four Indian languages from publicly
available datasets and web sources.

### Raw Statistics

| Language | Articles | Real (0) | Fake (1) | Avg Length |
|----------|----------|----------|----------|------------|
| Hindi    | 20,593   | 10,293 (50.0%) | 10,300 (50.0%) | 3,665 chars |
| Telugu   | 20,001   | 10,000 (50.0%) | 10,001 (50.0%) | 2,517 chars |
| Marathi  | 18,696   |  8,696 (46.5%) | 10,000 (53.5%) | 2,794 chars |
| Gujarati | 17,659   |  8,859 (50.2%) |  8,800 (49.8%) | 2,696 chars |
| **Total**| **76,949** | | | |

### After Cleaning

| Step | Count |
|------|-------|
| Raw combined | 76,949 |
| After deduplication (27,279 removed) | 49,670 |
| After garbage removal (< 10 chars) | **49,669** |

### Cross-Language Balance (Post-cleaning)

| Language | Real  | Fake  | Total  | Fake % |
|----------|-------|-------|--------|--------|
| Hindi    | 7,501 | 7,599 | 15,100 | 50.3%  |
| Gujarati | 8,687 | 6,145 | 14,832 | 41.4%  |
| Telugu   | 6,664 | 4,795 | 11,459 | 41.8%  |
| Marathi  | 7,571 |   707 |  8,278 |  8.5%  |

> ⚠️ **Note:** Marathi has a severe class imbalance (only 8.5% Fake)
> which was addressed through stratified splitting and
> complementary classifier selection.

### Train / Validation / Test Split

```
Total : 49,669
Train : 29,801  (60%)
Val   :  9,934  (20%)
Test  :  9,934  (20%)
Stratified by both language and label
```

### Token Statistics

| Language | Avg Tokens | Min | Max   | Vocab Size |
|----------|-----------|-----|-------|-----------|
| Hindi    | 478.9     | 3   | 5,400 | 179,808   |
| Marathi  | 333.4     | 6   | 2,496 | 216,541   |
| Gujarati | 335.6     | 2   | 2,905 | 353,244   |
| Telugu   | 299.9     | 2   |   957 | 328,664   |

---

## 🔧 Pipeline

The project is structured as **10 sequential phases:**

```
Phase 1  ─── Data Collection & Cleaning
Phase 2  ─── Preprocessing & TF-IDF Vectorisation
Phase 3  ─── Naive Bayes (Complement NB) Baseline
Phase 4  ─── k-Nearest Neighbours
Phase 5  ─── Logistic Regression
Phase 6  ─── Support Vector Machine (Linear & RBF)
Phase 7  ─── Random Forest & Extra Trees
Phase 8  ─── Gradient Boosting + Perceptron + MLP
Phase 9  ─── Stacking Ensemble (7 base learners)
Phase 10 ─── Feature Selection & SHAP/LIME Explainability
```

### Feature Representations

| Type | Method | Dimensions |
|------|--------|-----------|
| Character n-gram TF-IDF | Combined multilingual | 10,000 |
| Word-level TF-IDF | Combined multilingual | 10,000 |
| Per-language TF-IDF | Language-specific | 3,947–5,000 |
| SVD (LSA) | Dimensionality reduction | 50 |
| **MuRIL embeddings** | Google multilingual BERT | **768** |

---

## 📈 Results

### Phase 8 — Individual Classifier Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| 🏆 Perceptron | **0.9967** | 0.9933 | 0.9982 | **0.9957** | N/A |
| Gradient Boosting | 0.9957 | 0.9922 | 0.9966 | 0.9944 | 0.9995 |
| Random Forest | 0.9953 | 0.9920 | 0.9958 | 0.9939 | **0.9999** |
| MLP (128→64, ReLU) | 0.9934 | 0.9899 | 0.9930 | 0.9914 | 0.9998 |
| ComplementNB (α=2.0) | 0.9476 | 0.9941 | 0.8698 | 0.9278 | 0.9960 |

### Phase 8 — Per-language Accuracy

| Language | Random Forest | NaiveBayes | GradBoost | Perceptron | MLP |
|----------|-------------|------------|-----------|------------|-----|
| Telugu   | **0.9990** | 0.9632 | **0.9990** | 0.9984 | 0.9937 |
| Gujarati | 0.9960 | 0.9468 | 0.9960 | **0.9968** | 0.9940 |
| Marathi  | 0.9959 | 0.5896 | 0.9959 | 0.9919 | 0.9918 |
| Hindi    | 0.9889 | 0.9093 | 0.9902 | **0.9935** | 0.9879 |

> ⚠️ NaiveBayes on Marathi = 0.5896 — barely above random, due to extreme class imbalance (8.5% fake).

### Phase 9 — Stacking Ensemble

**Base learners (7 models):**

| Model | Val F1 | Test F1 | Test AUC |
|-------|--------|---------|----------|
| LinearSVC | 0.9940 | 0.9957 | 0.9999 |
| LogReg | 0.9910 | 0.9915 | 0.9997 |
| RandomForest | 0.9918 | 0.9931 | 0.9999 |
| GradBoost | 0.9925 | 0.9943 | 0.9995 |
| NaiveBayes | 0.9289 | 0.9275 | 0.9962 |
| MLP | 0.9909 | 0.9921 | 0.9998 |
| kNN | 0.9861 | 0.9882 | 0.9979 |

**Meta-learner:** Logistic Regression (C=10)

**Meta-learner weights (trust assigned to each base learner):**

```
LinearSVC     +5.22  ██████████████████████████████████████████████████
kNN           +4.90  █████████████████████████████████████████████████
GradBoost     +2.22  ██████████████████████
RandomForest  +2.00  ████████████████████
LogReg        +0.90  █████████
NaiveBayes    +0.18  █
MLP           -1.55  ███████████████  (penalised)
```

**Final Stacking Test Results:**

```
Accuracy  : 0.9973
F1 Score  : 0.9965
AUC-ROC   : 0.9999
```

**Per-language stacking results:**

| Language | Samples | Accuracy | F1 | AUC |
|----------|---------|----------|-----|-----|
| Telugu   | 2,276 | **0.9996** | **0.9995** | 1.0000 |
| Marathi  | 1,640 | 0.9994 | 0.9959 | 1.0000 |
| Gujarati | 2,951 | 0.9980 | 0.9976 | 0.9999 |
| Hindi    | 3,067 | 0.9938 | 0.9938 | 0.9998 |

### Cross-Phase Model Ranking

| Rank | Phase / Model | Test F1 |
|------|--------------|---------|
| 🥇 | **Phase 9 — Stacking Ensemble** | **0.9965** |
| 🥈 | Phase 9 — Soft Vote (equal) | 0.9956 |
| 🥉 | Phase 8 — Gradient Boosting | 0.9944 |
| 4 | Phase 8 — Random Forest | 0.9939 |
| 5 | Phase 8 — MLP | 0.9914 |

### Phase 10 — Feature Selection Comparison

| Method | Features Used | F1 | AUC |
|--------|-------------|-----|-----|
| 🏆 Chi-squared k=5000 | **5,000** | **0.9964** | **1.0000** |
| Full TF-IDF | 10,000 | 0.9921 | 0.9997 |
| Mutual Info k=5000 | 5,000 | 0.9915 | 0.9997 |
| L1 Embedded | 185 | 0.9875 | 0.9994 |

> 💡 Chi2 with 5,000 features **outperforms** full 10,000 — 
> feature selection improved both F1 and AUC.

---

## 🔍 Explainability

### SHAP — Top 20 Global Features

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|--------------|----------------|
| 1 | `2` | 0.2931 | Digit — year/number patterns in fake content |
| 2 | `0` | 0.2669 | Digit pattern |
| 3 | `1` | 0.2339 | Digit pattern |
| 4 | `ి.` | 0.1944 | Telugu character sequence |
| 5 | `ं` | 0.1933 | Hindi nasal marker |
| 6 | `।` | 0.1693 | Devanagari full stop |
| 7 | `ీడ` | 0.1606 | Telugu bigram |
| 8 | `रल` | 0.1527 | Hindi/Marathi bigram |

### LIME — Sample Local Explanations

**Sample 1 — Marathi article (True: Real, Pred: Real ✅)**
```
P(Fake) = 0.0005  →  Very confident REAL prediction
Key features pushing toward Real:
  'aircraft'  weight = -0.0172
  'झ'         weight = -0.0146
  'आ'         weight = -0.0124
```

**Sample 2 — Marathi article (True: Fake, Pred: Fake ✅)**
```
P(Fake) = 0.9267  →  Strongly predicted FAKE
Key features pushing toward Fake:
  'ऱ'     weight = +0.0411
  '२०२३'  weight = +0.0257  (year 2023 appears in fake news)
```

**Sample 3 — Gujarati article (True: Real, Pred: Fake ❌)**
```
P(Fake) = 0.5107  →  Barely over threshold — uncertain
Gujarati text caused cross-language confusion
Most uncertain prediction observed
```

### Language Bias Analysis

```
Feature overlap between languages (top 100 Chi2 features):

                Hindi    Marathi  Gujarati  Telugu
Hindi            100       100       100       0
Marathi          100       100       100       0
Gujarati         100       100       100       0
Telugu             0         0         0     100
```

> **Finding:** Hindi, Marathi and Gujarati share 100% of top features
> — they are related Indo-Aryan languages with shared scripts.
> Telugu (Dravidian) is completely isolated — zero feature overlap.
> This explains why Telugu achieves the highest per-language accuracy
> (0.9990): its features are distinctive and unambiguous.

---

## 🛠 Installation

### Requirements

```bash
Python >= 3.9
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.25.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
flask>=3.0.0
flask-cors>=4.0.0
sentence-transformers>=2.2.0
transformers>=4.35.0
torch>=2.0.0
shap>=0.43.0
lime>=0.2.0.1
openpyxl>=3.1.0
joblib>=1.3.0
```

---

## 🚀 Usage

### Run phases sequentially

```bash
# Phase 1 — Data loading and cleaning
python phase1.py

# Phase 2 — Preprocessing and vectorisation
python phase2.py

# Phases 3-7 — Individual classifiers
python phase3.py   # Naive Bayes
python phase4.py   # kNN
python phase5.py   # Logistic Regression
python phase6.py   # SVM
python phase7.py   # Random Forest & Extra Trees

# Phase 8 — Gradient Boosting, Perceptron, MLP
python phase8.py

# Phase 9 — Stacking Ensemble
python phase9.py

# Phase 10 — Feature Selection & Explainability
python phase10.py
```

### Extract MuRIL embeddings (Google Colab recommended)

```bash
# Open in Colab — requires GPU for speed
jupyter notebook muril_embeddings.ipynb
```

### Run Flask API

```bash
python app.py
# API runs at http://localhost:5000
```

---

## 🌐 Flask REST API

The trained stacking ensemble is deployed as a REST API to power a **Chrome browser extension** for real-time fake news detection.

### Endpoints

```
POST /predict
Content-Type: application/json

Request body:
{
  "text": "Article text here...",
  "language": "hindi"   # hindi | marathi | gujarati | telugu
}

Response:
{
  "prediction": "Fake",
  "confidence": 0.9267,
  "language_detected": "marathi",
  "model": "stacking_ensemble"
}
```

### Supported Languages

| Code | Language | Script |
|------|----------|--------|
| `hindi` | Hindi | Devanagari |
| `marathi` | Marathi | Devanagari |
| `gujarati` | Gujarati | Gujarati |
| `telugu` | Telugu | Telugu |

---

## 🏗 Model Architecture

### Stacking Ensemble Architecture

```
Input Text
    │
    ▼
TF-IDF Vectorisation (10,000 features)
    │
    ├──► LinearSVC    ──┐
    ├──► Logistic Reg ──┤
    ├──► Random Forest──┤
    ├──► GradBoost    ──┼──► Meta-feature matrix (N × 7)
    ├──► Naive Bayes  ──┤         │
    ├──► MLP          ──┤         ▼
    └──► kNN          ──┘   LogReg Meta-learner (C=10)
                                  │
                                  ▼
                         Final Prediction
                         (Real / Fake + Probability)
```

### MuRIL Embedding Pipeline

```
Input Text (any of 4 languages)
    │
    ▼
MuRIL Tokenizer (Google multilingual BERT)
    │
    ▼
12 Transformer Layers × 12 Attention Heads
    │
    ▼
CLS Token → 768-dimensional embedding
    │
    ▼
Classifier (per-language SVM / LogReg)
    │
    ▼
Real / Fake
```

---

## 📋 Saved Artefacts

### Phase 8 outputs (`phase8_outputs/`)
| File | Description |
|------|-------------|
| `rf_final.pkl` | Trained Random Forest |
| `gb_final.pkl` | Trained Gradient Boosting |
| `nb_final.pkl` | Trained Complement NB |
| `perc_final.pkl` | Trained Perceptron |
| `mlp_final.pkl` | Trained MLP |
| `scaler_mlp.pkl` | MinMax scaler for MLP |
| `model_comparison.csv` | All metrics |
| `phase8_classifiers_results.png` | Comparison plot |

### Phase 9 outputs (`phase9_outputs/`)
| File | Description |
|------|-------------|
| `final_meta_learner.pkl` | Trained LogReg meta-learner |
| `final_base_models.pkl` | All 7 base learners |
| `OOF_META.npy` | Out-of-fold meta-features |
| `TEST_META.npy` | Test set meta-features |
| `per_language_stacking.csv` | Per-language results |
| `cross_phase_comparison.csv` | All phases ranked |

### Phase 10 outputs (`phase10_outputs/`)
| File | Description |
|------|-------------|
| `shap_values.npy` | SHAP values (200 samples × 185 features) |
| `shap_feature_names.pkl` | Feature names for SHAP |
| `lime_explanations.csv` | LIME weights per sample |
| `lime_sample_*.html` | Interactive LIME HTML reports |
| `feature_importance_table.csv` | All methods ranked |
| `language_bias.csv` | Cross-language feature overlap |

---

## 🔬 Key Findings

1. **Stacking ensemble beats all individual models** —
   F1 of 0.9965 vs best single model (Perceptron) at 0.9957.

2. **Feature selection improves performance** —
   Chi2 with 5,000 features achieves F1=0.9964 vs
   full 10,000 features at F1=0.9921.
   Fewer, better features outperform more features.

3. **Telugu is linguistically isolated** —
   Zero feature overlap with Hindi/Marathi/Gujarati
   but achieves the highest per-language accuracy (0.9996).
   Dravidian vs Indo-Aryan script difference is a feature, not a bug.

4. **NaiveBayes fails on Marathi** —
   F1=0.5896, almost random guessing, due to 8.5% Fake class
   imbalance — NB is the most sensitive classifier to class priors.

5. **Digit n-grams are the most important features** —
   SHAP ranks `"2"`, `"0"`, `"1"` as top 3 features —
   fake news articles contain more specific year/number references
   (e.g., `२०२३`) than real news.

6. **LinearSVC is most trusted by the meta-learner** —
   Weight = +5.22, highest among all base learners,
   suggesting it captures patterns others miss.

---

## 👥 Team

| Name | ID |
|------|----|
| E Sai Brinda | BL.SC.U4AIE24114 |
| G Sreehitha | BL.SC.U4AIE24117 |
| H Harshith | BL.SC.U4AIE24120 |

**Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham, Bengaluru
**Course:** Intelligent Bioinformatics Systems / Machine Learning Lab
**Year:** 2025

---

## 📚 References

1. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional
   Transformers for Language Understanding. *NAACL-HLT*.

2. Khandelwal et al. (2020). MuRIL: Multilingual Representations
   for Indian Languages. *arXiv:2103.10730*.

3. Ribeiro et al. (2016). "Why Should I Trust You?": Explaining
   the Predictions of Any Classifier. *KDD 2016*.

4. Lundberg & Lee (2017). A Unified Approach to Interpreting
   Model Predictions. *NeurIPS 30*.

5. Breiman (2001). Random Forests. *Machine Learning*, 45, 5–32.

6. Friedman (2001). Greedy Function Approximation: A Gradient
   Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.

---

## 📄 License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ for Indian language NLP

⭐ Star this repo if you found it useful!

</div>
