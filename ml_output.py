Python 3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.

========== RESTART: C:\Users\saibr\Downloads\language_excel\phase1.py ==========
============================================================
  PHASE 1 — Data Collection & Handling
============================================================

✓ Marathi: 18,696 rows × 4 cols  |  columns: ['filename', 'label', 'text', 'language']

✓ Gujarati: 17,659 rows × 4 cols  |  columns: ['filename', 'label', 'text', 'language']

✓ Telugu: 20,001 rows × 4 cols  |  columns: ['filename', 'label', 'text', 'language']

✓ Hindi: 20,593 rows × 4 cols  |  columns: ['filename', 'label', 'text', 'language']

────────────────────────────────────────────────────────────
PER-LANGUAGE SUMMARY
────────────────────────────────────────────────────────────

── Marathi ──


  No missing values.

  Class distribution:
    Real (0): 8,696  (46.5%)
    Fake (1): 10,000  (53.5%)

  Text length (chars) — mean: 2794, min: 8, max: 18765

── Gujarati ──


  No missing values.

  Class distribution:
    Real (0): 8,859  (50.2%)
    Fake (1): 8,800  (49.8%)

  Text length (chars) — mean: 2696, min: 16, max: 19418

── Telugu ──


  No missing values.

  Class distribution:
    Real (0): 10,000  (50.0%)
    Fake (1): 10,001  (50.0%)

  Text length (chars) — mean: 2517, min: 10, max: 8494

── Hindi ──


  No missing values.

  Class distribution:
    Real (0): 10,293  (50.0%)
    Fake (1): 10,300  (50.0%)

  Text length (chars) — mean: 3665, min: 16, max: 32767

────────────────────────────────────────────────────────────
COMBINED DATASET: 76,949 rows  |  languages: ['Marathi', 'Gujarati', 'Telugu', 'Hindi']
────────────────────────────────────────────────────────────

[Data Quality]
  Duplicate texts: 27279  (35.5%)
  After deduplication: 49,670 rows
  Texts < 10 chars (likely garbage): 1
  Rows with missing label: 0

  Final clean dataset: 49,669 rows

[Cross-Language Class Balance]
          Real  Fake  Total  Fake%
language                          
Gujarati  8687  6145  14832   41.4
Hindi     7501  7599  15100   50.3
Marathi   7571   707   8278    8.5
Telugu    6664  4795  11459   41.8

[Saved] phase1_overview.png
[Saved] cleaned_multilingual_dataset.csv

============================================================
Phase 1 complete. Ready for Phase 2 (Preprocessing & Vectorization).
============================================================

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase2.py ==============================================
============================================================
  PHASE 2 — Preprocessing & Vectorization
============================================================

✓ Loaded 49,669 rows from 'cleaned_multilingual_dataset.csv'
  Columns : ['filename', 'label', 'text', 'language', 'text_len']
  Languages: {'Hindi': 15100, 'Gujarati': 14832, 'Telugu': 11459, 'Marathi': 8278}

[Step 2] Cleaning text...
  Dropped 0 empty rows after cleaning. Remaining: 49,669

  Sample cleaned texts:
  [Hindi] बीबीसी द्वारा युगांडा इंटरनेट प्रचार नेटवर्क पर्दाफाश बीबीसी द्वारा युगांडा इंटरनेट प्रचार नेटवर्क खुलासा21 जनवरी प्रकाश...
  [Marathi] ज्या देशांमध्ये रस्ते स्वच्छ किंवा बहुधा ज्या देशात रस्ते स्वच्छ किंवा बहुधा 29 जानेवारी 2018 प्रकाशित गेले जपानला शेअरक...
  [Gujarati] બીબીસીએ કરણ થાપરને સંબંધિત નકલી સમાચારમાં જવાબ આપ્યો કરણ થાપરને લગતા બનાવટી ન્યૂઝ કેસમાં, બીબીસીએ નવેમ્બરમાં જવાબ આપ્યો ...
  [Telugu] లేదు, సిఎ భవానీ దేవి ఇంకా పారిస్ ఒలింపిక్స్ అర్హత సాధించలేదు ఏడాది జూలై పారిస్ ఒలింపిక్స్ భారతీయ ఫీచర్ సిఎ భవానీ దేవి ఇప...

[Step 3] Token statistics per language...

          Samples  Avg tokens  Min tokens  Max tokens  Vocab size
Language                                                         
Hindi       15100       478.9           3        5400      179808
Marathi      8278       333.4           6        2496      216541
Gujarati    14832       335.6           2        2905      353244
Telugu      11459       299.9           2         957      328664

[Step 4] Train/Val/Test split (60/20/20, stratified)...
  Train : 29,801  |  Val: 9,934  |  Test: 9,934
  Train label balance — Fake: 11547  Real: 18254
  Test  label balance — Fake: 3849   Real: 6085

[Step 5] TF-IDF Vectorization...
  5A. Combined multilingual TF-IDF...
    Combined TF-IDF shape: (29801, 10000)
  5B. Word-level TF-IDF...
    Word TF-IDF shape: (29801, 10000)
  5C. Per-language TF-IDF...
    [Hindi] train: (8914, 5000)  test: (3067, 5000)
    [Marathi] train: (4980, 4577)  test: (1640, 4577)
    [Gujarati] train: (8997, 4690)  test: (2951, 4690)
    [Telugu] train: (6910, 3947)  test: (2276, 3947)

[Step 6] Top TF-IDF terms per language and class...
Traceback (most recent call last):
  File "C:/Users/saibr/Downloads/language_excel/phase2.py", line 293, in <module>
    mean_tfidf = lang_mat[idx].mean(axis=0).A1
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\scipy\sparse\_index.py", line 30, in __getitem__
    index, new_shape, _, _ = _validate_indices(key, self.shape, self.format)
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\scipy\sparse\_index.py", line 401, in _validate_indices
    index.extend(idx.nonzero())
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\pandas\core\generic.py", line 6321, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'Series' object has no attribute 'nonzero'

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase2.py ==============================================
============================================================
  PHASE 2 — Preprocessing & Vectorization
============================================================

✓ Loaded 49,669 rows from 'cleaned_multilingual_dataset.csv'
  Columns : ['filename', 'label', 'text', 'language', 'text_len']
  Languages: {'Hindi': 15100, 'Gujarati': 14832, 'Telugu': 11459, 'Marathi': 8278}

[Step 2] Cleaning text...
  Dropped 0 empty rows after cleaning. Remaining: 49,669

  Sample cleaned texts:
  [Hindi] बीबीसी द्वारा युगांडा इंटरनेट प्रचार नेटवर्क पर्दाफाश बीबीसी द्वारा युगांडा इंटरनेट प्रचार नेटवर्क खुलासा21 जनवरी प्रकाश...
  [Marathi] ज्या देशांमध्ये रस्ते स्वच्छ किंवा बहुधा ज्या देशात रस्ते स्वच्छ किंवा बहुधा 29 जानेवारी 2018 प्रकाशित गेले जपानला शेअरक...
  [Gujarati] બીબીસીએ કરણ થાપરને સંબંધિત નકલી સમાચારમાં જવાબ આપ્યો કરણ થાપરને લગતા બનાવટી ન્યૂઝ કેસમાં, બીબીસીએ નવેમ્બરમાં જવાબ આપ્યો ...
  [Telugu] లేదు, సిఎ భవానీ దేవి ఇంకా పారిస్ ఒలింపిక్స్ అర్హత సాధించలేదు ఏడాది జూలై పారిస్ ఒలింపిక్స్ భారతీయ ఫీచర్ సిఎ భవానీ దేవి ఇప...

[Step 3] Token statistics per language...

          Samples  Avg tokens  Min tokens  Max tokens  Vocab size
Language                                                         
Hindi       15100       478.9           3        5400      179808
Marathi      8278       333.4           6        2496      216541
Gujarati    14832       335.6           2        2905      353244
Telugu      11459       299.9           2         957      328664

[Step 4] Train/Val/Test split (60/20/20, stratified)...
  Train : 29,801  |  Val: 9,934  |  Test: 9,934
  Train label balance — Fake: 11547  Real: 18254
  Test  label balance — Fake: 3849   Real: 6085

[Step 5] TF-IDF Vectorization...
  5A. Combined multilingual TF-IDF...
    Combined TF-IDF shape: (29801, 10000)
  5B. Word-level TF-IDF...
    Word TF-IDF shape: (29801, 10000)
  5C. Per-language TF-IDF...
    [Hindi] train: (8914, 5000)  test: (3067, 5000)
    [Marathi] train: (4980, 4577)  test: (1640, 4577)
    [Gujarati] train: (8997, 4690)  test: (2951, 4690)
    [Telugu] train: (6910, 3947)  test: (2276, 3947)

[Step 6] Top TF-IDF terms per language and class...

  [Hindi]
    Real:  (0.089), ा(0.072), र(0.067), ्(0.065), ि(0.062)
    Fake:  (0.097), ा(0.077), र(0.073), ्(0.069), े(0.069)

  [Marathi]
    Real:  (0.092), ा(0.079), ्(0.074), र(0.070), े(0.069)
    Fake:  (0.082), ा(0.071), ्(0.065), ं (0.065), र(0.063)

  [Gujarati]
    Real:  (0.089), ા(0.073), ર(0.068), ે(0.067), ્(0.066)
    Fake:  (0.101), ા(0.084), ર(0.077), ે(0.076), ન(0.075)

  [Telugu]
    Real:  (0.089), ్(0.077), ి(0.073), ా(0.070), ర(0.069)
    Fake:  (0.096), ్(0.081), ి(0.079), ు(0.077), ా(0.077)

[Step 7] Skipping LaBSE embeddings (ENABLE_EMBEDDINGS=False).
  To enable: pip install sentence-transformers  then set ENABLE_EMBEDDINGS=True

[Step 8] Saving artefacts...
  ✓ All artefacts saved to 'phase2_outputs/'

[Step 9] Generating plots...
  ✓ Saved plot → phase2_outputs\phase2_overview.png

============================================================
Phase 2 complete. Artefacts ready for Phase 3 (kNN + Similarity).
Outputs in: ./phase2_outputs/
  X_train_tfidf.npz  X_val_tfidf.npz  X_test_tfidf.npz
  X_train_word.npz   X_val_word.npz   X_test_word.npz
  y_train.csv  y_val.csv  y_test.csv
  lang_train.csv  lang_val.csv  lang_test.csv
  tfidf_combined.pkl  tfidf_word.pkl  lang_tfidf_models.pkl
  splits.csv  phase2_overview.png
============================================================

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase3.py ==============================================
============================================================
  PHASE 3 — Similarity & Distance Features
============================================================

✓ Loaded artefacts from 'phase2_outputs/'
  Train: (29801, 10000)  |  Val: (9934, 10000)  |  Test: (9934, 10000)

[Step 2] Building class centroids from training set...
  Real centroid — non-zero dims: 9,487 / 10,000
  Fake centroid — non-zero dims: 8,410 / 10,000

  Real ↔ Fake centroid cosine similarity: 0.9337
  (closer to 0 = more separable; closer to 1 = harder to distinguish)

[Step 3] Per-language × per-class centroids...
  Real centroid — non-zero dims: 4,581 / 10,000
  Fake centroid — non-zero dims: 3,217 / 10,000
  Real centroid — non-zero dims: 4,157 / 10,000
  Fake centroid — non-zero dims: 2,608 / 10,000
  Real centroid — non-zero dims: 4,122 / 10,000
  Fake centroid — non-zero dims: 2,930 / 10,000
  Real centroid — non-zero dims: 3,627 / 10,000
  Fake centroid — non-zero dims: 2,094 / 10,000

  Cross-language Real-centroid cosine similarity:
                   Hindi     Marathi    Gujarati      Telugu
       Hindi      1.0000      0.8300      0.0270      0.0206
     Marathi      0.8300      1.0000      0.0259      0.0187
    Gujarati      0.0270      0.0259      1.0000      0.0197
      Telugu      0.0206      0.0187      0.0197      1.0000

[Step 4] Computing distance features for all splits...
  Train: (29801, 7)
  Val  : (9934, 7)
  Test : (9934, 7)

[Step 5] Jaccard-based token overlap features...
  Jaccard fake mean (train) — Fake samples: 0.0555  Real samples: 0.0357

[Step 6] Intra-class variance (spread analysis)...
Traceback (most recent call last):
  File "C:/Users/saibr/Downloads/language_excel/phase3.py", line 240, in <module>
    variance = np.sqrt((diffs.multiply(diffs)).sum()) / lmask.sum()
AttributeError: 'matrix' object has no attribute 'multiply'

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase3.py ==============================================
============================================================
  PHASE 3 — Similarity & Distance Features
============================================================

✓ Loaded artefacts from 'phase2_outputs/'
  Train: (29801, 10000)  |  Val: (9934, 10000)  |  Test: (9934, 10000)

[Step 2] Building class centroids from training set...
  Real centroid — non-zero dims: 9,487 / 10,000
  Fake centroid — non-zero dims: 8,410 / 10,000

  Real ↔ Fake centroid cosine similarity: 0.9337
  (closer to 0 = more separable; closer to 1 = harder to distinguish)

[Step 3] Per-language × per-class centroids...
  Real centroid — non-zero dims: 4,581 / 10,000
  Fake centroid — non-zero dims: 3,217 / 10,000
  Real centroid — non-zero dims: 4,157 / 10,000
  Fake centroid — non-zero dims: 2,608 / 10,000
  Real centroid — non-zero dims: 4,122 / 10,000
  Fake centroid — non-zero dims: 2,930 / 10,000
  Real centroid — non-zero dims: 3,627 / 10,000
  Fake centroid — non-zero dims: 2,094 / 10,000

  Cross-language Real-centroid cosine similarity:
                   Hindi     Marathi    Gujarati      Telugu
       Hindi      1.0000      0.8300      0.0270      0.0206
     Marathi      0.8300      1.0000      0.0259      0.0187
    Gujarati      0.0270      0.0259      1.0000      0.0197
      Telugu      0.0206      0.0187      0.0197      1.0000

[Step 4] Computing distance features for all splits...
  Train: (29801, 7)
  Val  : (9934, 7)
  Test : (9934, 7)

[Step 5] Jaccard-based token overlap features...
  Jaccard fake mean (train) — Fake samples: 0.0555  Real samples: 0.0357

[Step 6] Intra-class variance (spread analysis)...
  All languages    Real: 0.0065  Fake: 0.0082
  Hindi            Real: 0.0092  Fake: 0.0091
  Marathi          Real: 0.0090  Fake: 0.0304
  Gujarati         Real: 0.0082  Fake: 0.0098
  Telugu           Real: 0.0086  Fake: 0.0100

[Step 7] SVD dimensionality reduction for visualisation...
  50 SVD components explain 62.3% variance
  2  SVD components explain 34.0% variance

[Step 8] Building final combined feature matrix...
  Final feature dimensions:
    Train : (29801, 20059)
    Val   : (9934, 20059)
    Test  : (9934, 20059)
  Feature groups:
    Char TF-IDF :  10000 features
    Word TF-IDF :  10000 features
    Distance    :      7 features  (cosine, euclidean, manhattan)
    Jaccard     :      2 features  (fake centroid, real centroid)
    SVD/LSA     :     50 features
    ─────────────────────────
    Total       :  20059 features

[Step 9] Saving artefacts...
  ✓ All artefacts saved to 'phase3_outputs/'

[Step 10] Generating plots...
  ✓ Saved → phase3_outputs/phase3_overview.png

[Feature Discrimination Summary — Train Set]
Feature                           Fake mean    Real mean   Difference
────────────────────────────────────────────────────────────────────
  cos_sim_fake                      0.47438      0.45348     +0.02090 ◀
  cos_sim_real                      0.44295      0.48566     -0.04271 ◀
  euc_dist_fake                     1.02258      1.04355     -0.02097 ◀
  euc_dist_real                     1.04992      1.00792     +0.04200 ◀
  man_dist_fake                    44.40402     45.88507     -1.48105 ◀
  man_dist_real                    47.63162     46.17684     +1.45478 ◀
  cos_diff_fake_minus_real          0.03143     -0.03218     +0.06361 ◀
  jaccard_fake                      0.05551      0.03572     +0.01979 ◀
  jaccard_real                      0.03838      0.05404     -0.01566 ◀

  ◀ = features with notable separation between Fake and Real

============================================================
Phase 3 complete. Artefacts ready for Phase 4 (kNN Classifier).
Outputs in: ./phase3_outputs/
  X_train_final.npz  X_val_final.npz  X_test_final.npz
  X_train_svd.npy    X_val_svd.npy    X_test_svd.npy
  dist_train.npy     dist_val.npy     dist_test.npy
  centroids.pkl      svd_model.pkl
  phase3_overview.png
============================================================

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase4.py ==============================================
============================================================
  PHASE 4 — kNN Baseline Classifier
============================================================

✓ Loaded artefacts
  TF-IDF  — Train: (29801, 10000)  Val: (9934, 10000)  Test: (9934, 10000)
  SVD     — Train: (29801, 50)    Val: (9934, 50)    Test: (9934, 50)

[Step 3] Tuning k on validation set (TF-IDF char + cosine)...
  Testing k ∈ [1, 3, 5, 7, 9, 11, 15, 21]

  k= 1  F1=0.9839  Acc=0.9876  AUC=0.9851  (73.9s)
  k= 3  F1=0.9865  Acc=0.9896  AUC=0.9962  (76.6s)
  k= 5  F1=0.9861  Acc=0.9893  AUC=0.9980  (78.0s)
  k= 7  F1=0.9865  Acc=0.9896  AUC=0.9987  (136.1s)
  k= 9  F1=0.9869  Acc=0.9899  AUC=0.9991  (80.1s)
  k=11  F1=0.9857  Acc=0.9890  AUC=0.9992  (92.8s)
  k=15  F1=0.9849  Acc=0.9884  AUC=0.9994  (84.1s)
  k=21  F1=0.9844  Acc=0.9880  AUC=0.9993  (86.4s)

  ✓ Best k = 9  (Val F1 = 0.9869)

[Step 4] Distance metric comparison (k=9, SVD-50 features)...
  cosine        F1=0.9938  Acc=0.9952  AUC=0.9991
  euclidean     F1=0.9942  Acc=0.9955  AUC=0.9989
  manhattan     F1=0.9940  Acc=0.9954  AUC=0.9990

  ✓ Best distance metric: euclidean

[Step 5] Feature set comparison (k=9)...
  TF-IDF (char, cosine)                F1=0.9869  Acc=0.9899
  SVD-50 (cosine)                      F1=0.9938  Acc=0.9952
  SVD-50 (euclidean)                   F1=0.9942  Acc=0.9955
  Distance features (euclidean)        F1=0.9571  Acc=0.9666

  ✓ Best feature set: SVD-50 (euclidean)

[Step 6] Per-language monolingual kNN (k=9, cosine, TF-IDF)...
  [Hindi]  Val  F1=0.9836  Acc=0.9836  |  Test F1=0.9865  Acc=0.9866
  [Marathi]  Val  F1=0.9728  Acc=0.9952  |  Test F1=0.9748  Acc=0.9963
  [Gujarati]  Val  F1=0.9869  Acc=0.9893  |  Test F1=0.9879  Acc=0.9898
  [Telugu]  Val  F1=0.9947  Acc=0.9956  |  Test F1=0.9910  Acc=0.9925

[Step 7] Final evaluation on held-out TEST set...
  Config: k=9, metric=cosine, features=TF-IDF char

  Overall Test Results:
  Acc=0.9915  Prec=0.9955  Rec=0.9826  F1=0.9890  AUC=0.9987

  Classification Report:
              precision    recall  f1-score   support

    Real (0)     0.9891    0.9972    0.9931      6085
    Fake (1)     0.9955    0.9826    0.9890      3849

    accuracy                         0.9915      9934
   macro avg     0.9923    0.9899    0.9911      9934
weighted avg     0.9916    0.9915    0.9915      9934

  Confusion Matrix:
                Predicted Real  Predicted Fake
  Actual Real           6068              17
  Actual Fake             67            3782

  Per-language test results:
  Language      Samples   Accuracy       F1      AUC
  ──────────────────────────────────────────────────
  Hindi            3067     0.9883   0.9882   0.9983
  Marathi          1640     0.9976   0.9833   0.9918
  Gujarati         2951     0.9902   0.9883   0.9988
  Telugu           2276     0.9934   0.9921   0.9994

[Step 8] Saving artefacts...
  ✓ Artefacts saved to 'phase4_outputs/'

[Step 9] Generating plots...
  ✓ Saved → phase4_outputs/phase4_knn_results.png

============================================================
Phase 4 complete. Summary:
  Best k          : 9
  Best metric     : cosine
  Test Accuracy   : 0.9915
  Test F1         : 0.9890
  Test ROC-AUC    : 0.9987

Artefacts in: ./phase4_outputs/
  knn_final.pkl          knn_lang_models.pkl
  k_tuning_results.csv   feature_set_results.csv
  metric_results.csv     per_language_results.csv
  knn_config.pkl         phase4_knn_results.png

Ready for Phase 5 — Evaluation Framework
============================================================

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase5.py ==============================================
============================================================
  PHASE 5 — Evaluation Framework
============================================================

✓ Artefacts loaded  |  Best k from Phase 4: 9
  Train: (29801, 10000)  Val: (9934, 10000)  Test: (9934, 10000)

[Step 2] Training models for evaluation...
  ✓ kNN (k=9, cosine) trained
  ✓ Logistic Regression (C=1.0, saga) trained

[Step 3] Full metric evaluation on test set...

  ── kNN (k=9) ──
  Accuracy    : 0.9915
  Precision   : 0.9955   (of all predicted Fake, how many are Fake)
  Recall/TPR  : 0.9826   (of all actual Fake, how many caught)
  Specificity : 0.9972   (of all actual Real, how many correctly rejected)
  F1 Score    : 0.9890   (harmonic mean of precision & recall)
  FPR         : 0.0028   (real news wrongly flagged as fake)
  FNR         : 0.0174   (fake news missed)
  MCC         : 0.9822   (balanced; +1 perfect, 0 random, -1 inverse)
  Cohen Kappa : 0.9821   (agreement beyond chance)
  ROC-AUC     : 0.9987
  Avg Prec    : 0.9980   (area under PR curve)
  Confusion   : TP=3782  TN=6068  FP=17  FN=67

  ── Logistic Regression ──
  Accuracy    : 0.9939
  Precision   : 0.9876   (of all predicted Fake, how many are Fake)
  Recall/TPR  : 0.9966   (of all actual Fake, how many caught)
  Specificity : 0.9921   (of all actual Real, how many correctly rejected)
  F1 Score    : 0.9921   (harmonic mean of precision & recall)
  FPR         : 0.0079   (real news wrongly flagged as fake)
  FNR         : 0.0034   (fake news missed)
  MCC         : 0.9871   (balanced; +1 perfect, 0 random, -1 inverse)
  Cohen Kappa : 0.9871   (agreement beyond chance)
  ROC-AUC     : 0.9997
  Avg Prec    : 0.9996   (area under PR curve)
  Confusion   : TP=3836  TN=6037  FP=48  FN=13

  Full Classification Reports:

  [kNN k=9]
              precision    recall  f1-score   support

    Real (0)     0.9891    0.9972    0.9931      6085
    Fake (1)     0.9955    0.9826    0.9890      3849

    accuracy                         0.9915      9934
   macro avg     0.9923    0.9899    0.9911      9934
weighted avg     0.9916    0.9915    0.9915      9934


  [Logistic Regression]
              precision    recall  f1-score   support

    Real (0)     0.9979    0.9921    0.9950      6085
    Fake (1)     0.9876    0.9966    0.9921      3849

    accuracy                         0.9939      9934
   macro avg     0.9927    0.9944    0.9935      9934
weighted avg     0.9939    0.9939    0.9939      9934


[Step 4] Per-language metric breakdown...

  Language     Model               Acc   Prec    Rec     F1   Spec    MCC    AUC
  ────────────────────────────────────────────────────────────────────────
  Hindi        kNN(k=9)          0.988  0.995  0.982  0.988  0.995  0.977  0.998
  Hindi        LogReg            0.990  0.985  0.994  0.990  0.985  0.979  0.999
  Marathi      kNN(k=9)          0.998  1.000  0.967  0.983  1.000  0.982  0.992
  Marathi      LogReg            0.999  1.000  0.992  0.996  1.000  0.996  1.000
  Gujarati     kNN(k=9)          0.990  0.995  0.982  0.988  0.996  0.980  0.999
  Gujarati     LogReg            0.994  0.986  0.999  0.992  0.989  0.987  0.999
  Telugu       kNN(k=9)          0.993  0.997  0.987  0.992  0.998  0.986  0.999
  Telugu       LogReg            0.996  0.993  0.998  0.995  0.995  0.992  1.000

[Step 5] Threshold analysis (kNN)...
  Best threshold by F1: 0.25  (F1=0.9925)
  Default (0.5)   F1  : 0.9890

[Step 6] Error analysis...

  Error distribution:
error_type
correct                      9850
FN (missed fake)               67
FP (wrongly flagged real)      17

  Error rate per language:
  Hindi         Error=0.012  FN=28  FP=8
  Marathi       Error=0.002  FN=4  FP=0
  Gujarati      Error=0.010  FN=23  FP=6
  Telugu        Error=0.007  FN=12  FP=3

  ✓ Saved 84 misclassified samples → misclassified_samples.csv

[Step 7] Saving artefacts...
  ✓ Artefacts saved to 'phase5_outputs/'

[Step 8] Generating plots...
  ✓ Saved → phase5_outputs/phase5_evaluation.png

============================================================
PHASE 5 SUMMARY — Test Set Performance
============================================================
              Model  Accuracy  Precision  Recall     F1  Specificity    MCC    AUC
          kNN (k=9)    0.9915     0.9955  0.9826 0.9890       0.9972 0.9822 0.9987
Logistic Regression    0.9939     0.9876  0.9966 0.9921       0.9921 0.9871 0.9997

  Best threshold (by F1) : 0.25

Artefacts in: ./phase5_outputs/
  overall_metrics.csv       per_language_metrics.csv
  threshold_analysis.csv    misclassified_samples.csv
  eval_models.pkl           phase5_evaluation.png

Ready for Phase 6 — Logistic Regression
============================================================

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase6.py ==============================================
============================================================
  PHASE 6 — Linear Baseline & Logistic Regression
============================================================

✓ Artefacts loaded
  Char TF-IDF  — 10,000 features
  Word TF-IDF  — 10,000 features
  SVD          — 50 components
  Train: 29,801   Val: 9,934   Test: 9,934

[Step 2] Regularisation search (val set, char TF-IDF)...

  Penalty        C   Val Acc   Val F1   Val AUC
  ──────────────────────────────────────────────
  L2        0.001    0.6124   0.0000    0.9328
  L2        0.010    0.9471   0.9270    0.9869
  L2        0.100    0.9862   0.9822    0.9983
  L2        0.500    0.9918   0.9895    0.9991
  L2        1.000    0.9931   0.9910    0.9993
  L2        5.000    0.9948   0.9933    0.9997
  L2       10.000    0.9956   0.9943    0.9998
  L2       50.000    0.9963   0.9952    0.9999
  L1        0.001    0.6124   0.0000    0.5000
  L1        0.010    0.6124   0.0000    0.8240
  L1        0.100    0.9705   0.9624    0.9916
  L1        0.500    0.9871   0.9834    0.9987
  L1        1.000    0.9908   0.9882    0.9992
  L1        5.000    0.9946   0.9930    0.9998
  L1       10.000    0.9952   0.9938    0.9998
  L1       50.000    0.9957   0.9944    0.9998
  EN        0.100    0.9796   0.9737    0.9968
  EN        1.000    0.9921   0.9899    0.9993
  EN       10.000    0.9954   0.9940    0.9998

  ✓ Best: penalty=L2  C=50.0  (Val F1=0.9952)

[Step 3] Feature set comparison (best regularisation)...
  Char TF-IDF (L2-norm)              
  Acc=0.9963  Prec=0.9935  Rec=0.9969  F1=0.9952  AUC=0.9999  AP=0.9998
  Word TF-IDF (L2-norm)              
  Acc=0.9937  Prec=0.9904  Rec=0.9932  F1=0.9918  AUC=0.9997  AP=0.9996
  SVD-50                             
  Acc=0.9906  Prec=0.9875  Rec=0.9883  F1=0.9879  AUC=0.9990  AP=0.9981
  Distance features                  
  Acc=0.9039  Prec=0.8426  Rec=0.9247  F1=0.8817  AUC=0.9561  AP=0.9468

  ✓ Best feature set: Char TF-IDF (L2-norm)

[Step 4] 5-fold stratified cross-validation...
Traceback (most recent call last):
  File "C:/Users/saibr/Downloads/language_excel/phase6.py", line 228, in <module>
    cv_results = cross_validate(
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\utils\_param_validation.py", line 218, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\model_selection\_validation.py", line 373, in cross_validate
    results = parallel(
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\utils\parallel.py", line 91, in __call__
    return super().__call__(iterable_with_config_and_warning_filters)
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\joblib\parallel.py", line 2072, in __call__
    return output if self.return_generator else list(output)
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\joblib\parallel.py", line 1682, in _get_outputs
    yield from self._retrieve()
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\joblib\parallel.py", line 1784, in _retrieve
    self._raise_error_fast()
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\joblib\parallel.py", line 1859, in _raise_error_fast
    error_job.get_result(self.timeout)
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\joblib\parallel.py", line 758, in get_result
    return self._return_or_raise()
  File "C:\Users\saibr\AppData\Local\Programs\Python\Python314\Lib\site-packages\joblib\parallel.py", line 773, in _return_or_raise
    raise self._result
joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker.

Detailed tracebacks of the workers should have been printed to stderr in the executor process if faulthandler was not disabled.

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase6.py ==============================================
============================================================
  PHASE 6 — Linear Baseline & Logistic Regression
============================================================

✓ Artefacts loaded
  Char TF-IDF  — 10,000 features
  Word TF-IDF  — 10,000 features
  SVD          — 50 components
  Train: 29,801   Val: 9,934   Test: 9,934

[Step 2] Regularisation search (val set, char TF-IDF)...

  Penalty        C   Val Acc   Val F1   Val AUC
  ──────────────────────────────────────────────
  L2        0.001    0.6124   0.0000    0.9328
  L2        0.010    0.9471   0.9270    0.9869
  L2        0.100    0.9862   0.9822    0.9983
  L2        0.500    0.9918   0.9895    0.9991
  L2        1.000    0.9931   0.9910    0.9993
  L2        5.000    0.9948   0.9933    0.9997
  L2       10.000    0.9956   0.9943    0.9998
  L2       50.000    0.9963   0.9952    0.9999
  L1        0.001    0.6124   0.0000    0.5000
  L1        0.010    0.6124   0.0000    0.8240
  L1        0.100    0.9705   0.9624    0.9916
  L1        0.500    0.9871   0.9834    0.9987
  L1        1.000    0.9908   0.9882    0.9992
  L1        5.000    0.9946   0.9930    0.9998
  L1       10.000    0.9952   0.9938    0.9998
  L1       50.000    0.9957   0.9944    0.9998
  EN        0.100    0.9796   0.9737    0.9968
  EN        1.000    0.9921   0.9899    0.9993
  EN       10.000    0.9954   0.9940    0.9998

  ✓ Best: penalty=L2  C=50.0  (Val F1=0.9952)

[Step 3] Feature set comparison (best regularisation)...
  Char TF-IDF (L2-norm)              
  Acc=0.9963  Prec=0.9935  Rec=0.9969  F1=0.9952  AUC=0.9999  AP=0.9998
  Word TF-IDF (L2-norm)              
  Acc=0.9937  Prec=0.9904  Rec=0.9932  F1=0.9918  AUC=0.9997  AP=0.9996
  SVD-50                             
  Acc=0.9906  Prec=0.9875  Rec=0.9883  F1=0.9879  AUC=0.9990  AP=0.9981
  Distance features                  
  Acc=0.9039  Prec=0.8426  Rec=0.9247  F1=0.8817  AUC=0.9561  AP=0.9468

  ✓ Best feature set: Char TF-IDF (L2-norm)

[Step 4] 5-fold stratified cross-validation...

  5-Fold CV Results on Training Set:
  Metric              Mean      Std      Min      Max
  ──────────────────────────────────────────────────
  accuracy          0.9913   0.0013   0.9888   0.9923
  f1                0.9888   0.0017   0.9855   0.9900
  roc_auc           0.9993   0.0001   0.9992   0.9994
  precision         0.9872   0.0018   0.9854   0.9900
  recall            0.9903   0.0030   0.9848   0.9935

[Step 5] Per-language monolingual Logistic Regression...
  [Hindi]  Val  F1=0.9937  AUC=0.9997  |  Test F1=0.9925  AUC=0.9998
  [Marathi]  Val  F1=0.9967  AUC=1.0000  |  Test F1=1.0000  AUC=1.0000
  [Gujarati]  Val  F1=0.9941  AUC=0.9998  |  Test F1=0.9968  AUC=0.9999
  [Telugu]  Val  F1=1.0000  AUC=1.0000  |  Test F1=0.9990  AUC=1.0000

[Step 6] Top coefficient features per class...

  Top 15 features → FAKE (positive coef):
    1. ు.                              coef=+63.6340
    2. ి.                              coef=+61.2422
    3. ે.                              coef=+59.9450
    4. .                               coef=+57.8764
    5. हय                              coef=+33.7055
    6. ૉ                               coef=+30.8633
    7. ઃ                               coef=+28.8189
    8. ే.                              coef=+27.0390
    9. ऱ                               coef=+26.8721
   10. ऱ्                              coef=+26.7264
   11. ।                               coef=+24.3743
   12. િક                              coef=+24.1599
   13. नि                              coef=+22.2381
   14. ૉલ                              coef=+20.5847
   15. ़                               coef=+20.2932

  Top 15 features → REAL (negative coef):
    1. 1                               coef=-51.3775
    2. ूम                              coef=-44.8123
    3. ीड                              coef=-42.2977
    4. a                               coef=-36.9952
    5. 2                               coef=-36.2136
    6. యొ                              coef=-36.2109
    7. त,                              coef=-35.1939
    8. ણો                              coef=-33.8627
    9. િઓ                              coef=-32.5849
   10. દા                              coef=-31.9504
   11. खब                              coef=-30.0572
   12. इम                              coef=-29.9626
   13. .ત                              coef=-28.7852
   14. బడ                              coef=-28.5838
   15. o                               coef=-28.3353

[Step 7] Final evaluation on held-out TEST set...
  Config: penalty=L2  C=50.0

  Overall Test Results:
  Acc=0.9975  Prec=0.9951  Rec=0.9984  F1=0.9968  AUC=1.0000  AP=0.9999

  Classification Report:
              precision    recall  f1-score   support

    Real (0)     0.9990    0.9969    0.9979      6085
    Fake (1)     0.9951    0.9984    0.9968      3849

    accuracy                         0.9975      9934
   macro avg     0.9970    0.9977    0.9974      9934
weighted avg     0.9975    0.9975    0.9975      9934

  Confusion Matrix:
                Predicted Real  Predicted Fake
  Actual Real           6066              19
  Actual Fake              6            3843

  Per-language test results:
  Language         N      Acc       F1      AUC
  ────────────────────────────────────────────
  Hindi         3067   0.9948   0.9948   0.9998
  Marathi       1640   1.0000   1.0000   1.0000
  Gujarati      2951   0.9976   0.9972   1.0000
  Telugu        2276   0.9991   0.9990   1.0000

[Step 8] Saving artefacts...
  ✓ Artefacts saved to 'phase6_outputs/'

[Step 9] Generating plots...
  ✓ Saved → phase6_outputs/phase6_logreg_results.png

============================================================
PHASE 6 SUMMARY — Test Set Performance
============================================================
  Penalty      : L2
  Best C       : 50.0
  Feature set  : Char TF-IDF (L2-norm)
  Accuracy     : 0.9975
  Precision    : 0.9951
  Recall       : 0.9984
  F1           : 0.9968
  ROC-AUC      : 1.0000
  Avg Prec     : 0.9999

Artefacts in: ./phase6_outputs/
  lr_final.pkl               lr_lang_models.pkl
  lr_coef_model.pkl          lr_config.pkl
  regularisation_search.csv  feature_set_results.csv
  per_language_results.csv   cv_results.csv
  phase6_logreg_results.png

Ready for Phase 7 — Decision Tree & SVM
============================================================

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase7.py ==============================================
============================================================
  PHASE 7 — Decision Tree & SVM
============================================================

✓ Artefacts loaded
  TF-IDF : (29801, 10000)  SVD: (29801, 50)  DT input: (29801, 57)
  Train: 29,801  Val: 9,934  Test: 9,934

[Step 2] Decision Tree — depth tuning...

  Criterion   MaxDepth   ValAcc    ValF1   ValAUC  #Leaves
  ────────────────────────────────────────────────────────
  gini               2   0.9036   0.8787   0.9273        4
  gini               3   0.9580   0.9445   0.9617        7
  gini               5   0.9789   0.9723   0.9795       22
  gini               7   0.9825   0.9771   0.9850       51
  gini              10   0.9854   0.9810   0.9860       93
  gini              15   0.9889   0.9857   0.9922      133
  gini              20   0.9887   0.9854   0.9924      148
  gini            None   0.9894   0.9863   0.9940      161
  entropy            2   0.8949   0.8725   0.9334        4
  entropy            3   0.9560   0.9419   0.9732        7
  entropy            5   0.9764   0.9690   0.9937       24
  entropy            7   0.9830   0.9778   0.9952       61
  entropy           10   0.9886   0.9853   0.9938      109
  entropy           15   0.9891   0.9860   0.9923      137
  entropy           20   0.9891   0.9860   0.9923      137
  entropy         None   0.9891   0.9860   0.9923      137

  ✓ Best DT: criterion=gini  max_depth=None  (Val F1=0.9863)

[Step 3] Decision Tree — feature importance...

  Top 15 features by Gini importance:
  Rank  Feature                     Importance
  ────────────────────────────────────────────
  1     cos_diff                       0.57523
  2     SVD_1                          0.25150
  3     SVD_8                          0.04572
  4     SVD_9                          0.02033
  5     SVD_5                          0.01751
  6     SVD_10                         0.01614
  7     SVD_16                         0.01441
  8     SVD_22                         0.01291
  9     SVD_0                          0.00761
  10    SVD_34                         0.00510
  11    SVD_3                          0.00499
  12    SVD_7                          0.00478
  13    SVD_13                         0.00360
  14    SVD_6                          0.00231
  15    SVD_24                         0.00204

[Step 4] SVM — C tuning (LinearSVC on TF-IDF)...

         C    ValAcc    ValF1    ValAUC   Time(s)
  ────────────────────────────────────────────────
     0.001    0.9627   0.9515    0.9824      5.82
     0.010    0.9848   0.9804    0.9981      9.91
     0.050    0.9925   0.9902    0.9991      9.21
     0.100    0.9935   0.9916    0.9993      9.56
     0.500    0.9948   0.9932    0.9997      8.32
     1.000    0.9954   0.9940    0.9998      7.19
     5.000    0.9960   0.9948    0.9999      4.44
    10.000    0.9960   0.9948    0.9999      4.08

  ✓ Best LinearSVC: C=5.0  (Val F1=0.9948)

[Step 5] SVM kernel comparison (SVD-50 features)...
  [linear]  F1=0.9882  Acc=0.9908  AUC=0.9988  (35.2s)
  [rbf   ]  F1=0.9930  Acc=0.9946  AUC=0.9990  (26.9s)
  [poly  ]  F1=0.9929  Acc=0.9945  AUC=0.9992  (24.4s)

  ✓ Best kernel: rbf

[Step 6] Per-language Decision Tree & SVM...
  [Hindi]  DT  Test F1=0.9761  |  SVM Test F1=0.9938
  [Marathi]  DT  Test F1=0.9876  |  SVM Test F1=1.0000
  [Gujarati]  DT  Test F1=0.9916  |  SVM Test F1=0.9980
  [Telugu]  DT  Test F1=0.9963  |  SVM Test F1=0.9990

[Step 7] Final evaluation on held-out TEST set...

  ── Decision Tree (depth=None, gini) ──
  Acc=0.9908  Prec=0.9886  Rec=0.9878  F1=0.9882  AUC=0.9943

  ── LinearSVC (C=5.0) ──
  Acc=0.9977  Prec=0.9961  Rec=0.9979  F1=0.9970  AUC=1.0000

  [Decision Tree] Classification Report:
              precision    recall  f1-score   support

    Real (0)     0.9923    0.9928    0.9925      6085
    Fake (1)     0.9886    0.9878    0.9882      3849

    accuracy                         0.9908      9934
   macro avg     0.9904    0.9903    0.9903      9934
weighted avg     0.9908    0.9908    0.9908      9934


  [LinearSVC] Classification Report:
              precision    recall  f1-score   support

    Real (0)     0.9987    0.9975    0.9981      6085
    Fake (1)     0.9961    0.9979    0.9970      3849

    accuracy                         0.9977      9934
   macro avg     0.9974    0.9977    0.9976      9934
weighted avg     0.9977    0.9977    0.9977      9934


  Per-language test results:
  Language         N   DT Acc   DT F1   SVM Acc  SVM F1
  ───────────────────────────────────────────────────────
  Hindi         3067   0.9785  0.9784    0.9954  0.9954
  Marathi       1640   0.9957  0.9712    0.9994  0.9959
  Gujarati      2951   0.9949  0.9940    0.9980  0.9976
  Telugu        2276   0.9987  0.9984    0.9991  0.9990

[Step 8] Saving artefacts...
  ✓ Artefacts saved to 'phase7_outputs/'

[Step 9] Generating plots...
  ✓ Saved → phase7_outputs/phase7_dt_svm_results.png

============================================================
PHASE 7 SUMMARY — Test Set Performance
============================================================
  Model                               Acc     Prec      Rec       F1      AUC
  ──────────────────────────────────────────────────────────
  DT gini depth=None               0.9908   0.9886   0.9878   0.9882   0.9943
  LinearSVC C=5.0                  0.9977   0.9961   0.9979   0.9970   1.0000

Artefacts in: ./phase7_outputs/
  dt_final.pkl            svm_final.pkl
  dt_depth_tuning.csv     svm_c_tuning.csv
  svm_kernel_results.csv  dt_feature_importance.csv
  per_language_results.csv  phase7_config.pkl
  phase7_dt_svm_results.png

Ready for Phase 8 — Other Classifiers & Perceptron
============================================================

=============================================== RESTART: C:/Users/saibr/Downloads/language_excel/phase8.py ==============================================
[INFO] XGBoost not installed. Run: pip install xgboost
       Skipping XGBoost — all other models will still run.

============================================================
  PHASE 8 — Other Classifiers & Perceptron
============================================================

✓ Artefacts loaded
  Char TF-IDF: (29801, 10000)  Word TF-IDF: (29801, 10000)
  SVD: (29801, 50)  Distance: (29801, 7)
  Train: 29,801  Val: 9,934  Test: 9,934

────────────────────────────────────────────────────────────
  PART A — RANDOM FOREST
────────────────────────────────────────────────────────────

[Step 2] Random Forest tuning (SVD-50 features)...

  Params                                          Val F1   Val AUC
  ─────────────────────────────────────────────────────────────────
  n=50  depth=None  feats=sqrt                    0.9917    0.9993
  n=50  depth=None  feats=log2                    0.9913    0.9992
  n=50  depth=10  feats=sqrt                      0.9887    0.9992
  n=50  depth=10  feats=log2                      0.9886    0.9990
  n=50  depth=20  feats=sqrt                      0.9915    0.9993
  n=50  depth=20  feats=log2                      0.9913    0.9992
  n=100  depth=None  feats=sqrt                   0.9912    0.9993
  n=100  depth=None  feats=log2                   0.9914    0.9993
  n=100  depth=10  feats=sqrt                     0.9891    0.9992
  n=100  depth=10  feats=log2                     0.9883    0.9991
  n=100  depth=20  feats=sqrt                     0.9917    0.9993
  n=100  depth=20  feats=log2                     0.9913    0.9993
  n=200  depth=None  feats=sqrt                   0.9918    0.9993
  n=200  depth=None  feats=log2                   0.9913    0.9993
  n=200  depth=10  feats=sqrt                     0.9896    0.9992
  n=200  depth=10  feats=log2                     0.9882    0.9991
  n=200  depth=20  feats=sqrt                     0.9913    0.9993
  n=200  depth=20  feats=log2                     0.9912    0.9993

  ✓ Best Random Forest: {'n_estimators': 200, 'max_depth': None, 'max_features': 'sqrt'}  (Val F1=0.9918)

  Test Results — Random Forest:
  Acc=0.9953  Prec=0.9920  Rec=0.9958  F1=0.9939  AUC=0.9999
              precision    recall  f1-score   support

        Real     0.9974    0.9949    0.9961      6085
        Fake     0.9920    0.9958    0.9939      3849

    accuracy                         0.9953      9934
   macro avg     0.9947    0.9954    0.9950      9934
weighted avg     0.9953    0.9953    0.9953      9934


────────────────────────────────────────────────────────────
  PART B — NAIVE BAYES
────────────────────────────────────────────────────────────

[Step 3] Naive Bayes variants (word TF-IDF, non-negative)...

  Model                   Alpha    Val F1   Val AUC
  ──────────────────────────────────────────────────
  MultinomialNB           0.001    0.9231    0.9971
  ComplementNB            0.001    0.9298    0.9971
  GaussianNB                N/A    0.8490    0.9565
  MultinomialNB           0.010    0.9231    0.9968
  ComplementNB            0.010    0.9297    0.9968
  MultinomialNB           0.100    0.9225    0.9964
  ComplementNB            0.100    0.9291    0.9964
  MultinomialNB           0.500    0.9219    0.9959
  ComplementNB            0.500    0.9288    0.9959
  MultinomialNB           1.000    0.9219    0.9956
  ComplementNB            1.000    0.9289    0.9956
  MultinomialNB           2.000    0.9219    0.9952
  ComplementNB            2.000    0.9300    0.9952
  MultinomialNB           5.000    0.9221    0.9946
  ComplementNB            5.000    0.9287    0.9946

  ✓ Best NB: ComplementNB  alpha=2.0  (Val F1=0.9300)

  Test Results — ComplementNB:
  Acc=0.9476  Prec=0.9941  Rec=0.8698  F1=0.9278  AUC=0.9960
              precision    recall  f1-score   support

        Real     0.9237    0.9967    0.9588      6085
        Fake     0.9941    0.8698    0.9278      3849

    accuracy                         0.9476      9934
   macro avg     0.9589    0.9333    0.9433      9934
weighted avg     0.9510    0.9476    0.9468      9934


────────────────────────────────────────────────────────────
  PART C — GRADIENT BOOSTING
────────────────────────────────────────────────────────────

[Step 4] Gradient Boosting tuning (SVD-50 features)...

  Params                                          Val F1   Val AUC
  ─────────────────────────────────────────────────────────────────
  n=100  lr=0.05  depth=3                         0.9809    0.9983
  n=100  lr=0.05  depth=5                         0.9875    0.9989
  n=100  lr=0.1  depth=3                          0.9869    0.9988
  n=100  lr=0.1  depth=5                          0.9900    0.9991
  n=100  lr=0.2  depth=3                          0.9900    0.9987
  n=100  lr=0.2  depth=5                          0.9917    0.9990
  n=200  lr=0.05  depth=3                         0.9871    0.9987
  n=200  lr=0.05  depth=5                         0.9905    0.9992
  n=200  lr=0.1  depth=3                          0.9899    0.9992
  n=200  lr=0.1  depth=5                          0.9919    0.9993
  n=200  lr=0.2  depth=3                          0.9910    0.9985
  n=200  lr=0.2  depth=5                          0.9925    0.9992

  ✓ Best GradBoost: {'n_estimators': 200, 'learning_rate': 0.2, 'max_depth': 5}  (Val F1=0.9925)

  Test Results — Gradient Boosting:
  Acc=0.9957  Prec=0.9922  Rec=0.9966  F1=0.9944  AUC=0.9995
              precision    recall  f1-score   support

        Real     0.9979    0.9951    0.9965      6085
        Fake     0.9922    0.9966    0.9944      3849

    accuracy                         0.9957      9934
   macro avg     0.9950    0.9958    0.9954      9934
weighted avg     0.9957    0.9957    0.9957      9934


────────────────────────────────────────────────────────────
  PART E — PERCEPTRON  (Lab 8)
────────────────────────────────────────────────────────────

[Step 6] Perceptron tuning (L2-normalised TF-IDF)...
  Lab 8: single-layer linear threshold classifier

     Alpha  MaxIter  Penalty    Val F1   Val Acc
  ──────────────────────────────────────────────────
    0.0001      100     None    0.9920    0.9938
    0.0001      100       l2    0.9790    0.9834
    0.0001      100       l1    0.9664    0.9737
    0.0001      500     None    0.9920    0.9938
    0.0001      500       l2    0.9790    0.9834
    0.0001      500       l1    0.9664    0.9737
    0.0001     1000     None    0.9920    0.9938
    0.0001     1000       l2    0.9790    0.9834
    0.0001     1000       l1    0.9664    0.9737
    0.0010      100     None    0.9920    0.9938
    0.0010      100       l2    0.7040    0.7658
    0.0010      100       l1    0.8619    0.8810
    0.0010      500     None    0.9920    0.9938
    0.0010      500       l2    0.7040    0.7658
    0.0010      500       l1    0.8619    0.8810
    0.0010     1000     None    0.9920    0.9938
    0.0010     1000       l2    0.7040    0.7658
    0.0010     1000       l1    0.8619    0.8810
    0.0100      100     None    0.9920    0.9938
    0.0100      100       l2    0.5947    0.6344
    0.0100      100       l1    0.0000    0.6124
    0.0100      500     None    0.9920    0.9938
    0.0100      500       l2    0.5947    0.6344
    0.0100      500       l1    0.0000    0.6124
    0.0100     1000     None    0.9920    0.9938
    0.0100     1000       l2    0.5947    0.6344
    0.0100     1000       l1    0.0000    0.6124

  ✓ Best Perceptron: {'alpha': 0.0001, 'max_iter': 100, 'penalty': None}  (Val F1=0.9920)

  Test Results — Perceptron:
  Acc=0.9967  F1=0.9957  Prec=0.9933  Rec=0.9982
              precision    recall  f1-score   support

        Real     0.9988    0.9957    0.9973      6085
        Fake     0.9933    0.9982    0.9957      3849

    accuracy                         0.9967      9934
   macro avg     0.9961    0.9970    0.9965      9934
weighted avg     0.9967    0.9967    0.9967      9934


────────────────────────────────────────────────────────────
  PART F — MLP (Multi-Layer Perceptron)  (Lab 8)
────────────────────────────────────────────────────────────

[Step 7] MLP tuning (MinMax-scaled SVD-50 features)...
  Lab 8: multi-layer extension of the Perceptron

  Layers              Alpha    Act    Val F1   Val AUC
  ──────────────────────────────────────────────────────
  (64,)              0.0001   relu    0.9875    0.9989
  (64,)              0.0001   tanh    0.9870    0.9989
  (64,)              0.0010   relu    0.9879    0.9990
  (64,)              0.0010   tanh    0.9870    0.9989
  (128,)             0.0001   relu    0.9879    0.9990
  (128,)             0.0001   tanh    0.9839    0.9988
  (128,)             0.0010   relu    0.9879    0.9990
  (128,)             0.0010   tanh    0.9839    0.9988
  (64, 32)           0.0001   relu    0.9856    0.9990
  (64, 32)           0.0001   tanh    0.9861    0.9988
  (64, 32)           0.0010   relu    0.9859    0.9990
  (64, 32)           0.0010   tanh    0.9861    0.9988
  (128, 64)          0.0001   relu    0.9909    0.9992
  (128, 64)          0.0001   tanh    0.9881    0.9989
  (128, 64)          0.0010   relu    0.9907    0.9993
  (128, 64)          0.0010   tanh    0.9881    0.9989

  ✓ Best MLP: {'hidden_layer_sizes': (128, 64), 'alpha': 0.0001, 'activation': 'relu'}  (Val F1=0.9909)

  Test Results — MLP:
  Acc=0.9934  Prec=0.9899  Rec=0.9930  F1=0.9914  AUC=0.9998
              precision    recall  f1-score   support

        Real     0.9956    0.9936    0.9946      6085
        Fake     0.9899    0.9930    0.9914      3849

    accuracy                         0.9934      9934
   macro avg     0.9927    0.9933    0.9930      9934
weighted avg     0.9934    0.9934    0.9934      9934


[Step 8] Per-language test breakdown...

  Language      RandomForest    NaiveBayes      GradBoost       Perceptron      MLP           
  ────────────────────────────────────────────────────────────────────────────────────────────
  Hindi         0.9889          0.9093          0.9902          0.9935          0.9879        
  Marathi       0.9959          0.5896          0.9959          0.9919          0.9918        
  Gujarati      0.9960          0.9468          0.9960          0.9968          0.9940        
  Telugu        0.9990          0.9632          0.9990          0.9984          0.9937        

[Step 9] Full model comparison...

  Model                               Acc     Prec      Rec       F1      AUC
  ────────────────────────────────────────────────────────────────────────
  Random Forest                    0.9953   0.9920   0.9958   0.9939   0.9999
  ComplementNB (α=2.0)             0.9476   0.9941   0.8698   0.9278   0.9960
  Gradient Boosting                0.9957   0.9922   0.9966   0.9944   0.9995
  Perceptron                       0.9967   0.9933   0.9982   0.9957      N/A
  MLP                              0.9934   0.9899   0.9930   0.9914   0.9998

  ✓ Best model in Phase 8: Perceptron

[Step 10] Saving artefacts...
  ✓ Artefacts saved to 'phase8_outputs/'

[Step 11] Generating plots...
  ✓ Saved → phase8_outputs/phase8_classifiers_results.png

============================================================
PHASE 8 SUMMARY — Test Set Performance
============================================================
  Model                                F1      AUC      Acc
  ──────────────────────────────────────────────────────────
  Perceptron                       0.9957      N/A   0.9967
  Gradient Boosting                0.9944   0.9995   0.9957
  Random Forest                    0.9939   0.9999   0.9953
  MLP                              0.9914   0.9998   0.9934
  ComplementNB (α=2.0)             0.9278   0.9960   0.9476

  ✓ Best model: Perceptron

Artefacts in: ./phase8_outputs/
  rf_final.pkl  gb_final.pkl  nb_final.pkl
  perc_final.pkl  mlp_final.pkl  scaler_mlp.pkl
  model_comparison.csv  per_language_results.csv
  phase8_config.pkl  phase8_classifiers_results.png

Ready for Phase 9 — Stacking Ensemble
============================================================
    
