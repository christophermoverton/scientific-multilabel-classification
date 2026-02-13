#  Scientific Multi-Label Classification

### LDA vs Classical Machine Learning vs SciBERT

##  Overview

This project investigates multi-label classification of scientific research articles across six academic disciplines:

* Computer Science
* Physics
* Mathematics
* Statistics
* Quantitative Biology
* Quantitative Finance

The dataset exhibits significant class imbalance and moderate interdisciplinary vocabulary overlap. This study compares unsupervised topic modeling with classical machine learning and transformer-based approaches to evaluate performance under label sparsity.

---

##  Dataset

Source:
Blesson Densil. *Topic Modeling for Research Articles*. Kaggle.
[https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles](https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles)

The dataset contains research article titles and abstracts with binary multi-label annotations across six disciplines.

Key characteristics:

* ~75–80% single-label papers
* Significant class imbalance
* Sparse minority classes (Quantitative Biology, Quantitative Finance)

---

##  Methodology

###  Exploratory Data Analysis

* Label distribution analysis
* Multi-label frequency inspection
* Correlation heatmap
* Vocabulary overlap investigation

### Unsupervised Topic Modeling

* Latent Dirichlet Allocation (LDA)
* Topic-label alignment analysis
* Evaluation of topic purity

**Finding:**
LDA recovered dominant disciplinary clusters (e.g., Physics, Mathematics) but failed to isolate sparse minority fields.

---

### Classical Baseline

**TF-IDF (1–2 grams) + One-vs-Rest Logistic Regression**

* Class weighting
* Per-label probability threshold tuning
* Macro & Micro F1 evaluation

**Result:**
Macro F1 ≈ 0.76–0.79

---

### Transformer Model

**SciBERT (allenai/scibert_scivocab_uncased)**

* Fine-tuned for multi-label classification
* GPU training (Colab)
* Binary Cross-Entropy loss
* Per-label threshold calibration

---

##  Results

| Model                          | Micro F1   | Macro F1   |
| ------------------------------ | ---------- | ---------- |
| TF-IDF + Logistic              | ~0.82      | ~0.76–0.79 |
| SciBERT (default threshold)    | ~0.84      | ~0.78      |
| **SciBERT (tuned thresholds)** | **0.8466** | **0.8051** |

### Minority Class Improvements

| Class                | Logistic | SciBERT (Tuned) |
| -------------------- | -------- | --------------- |
| Quantitative Biology | 0.54     | **0.63**        |
| Quantitative Finance | 0.73     | **0.80**        |

---

##  Key Insights

* The dataset behaves largely as quasi multi-class with moderate exclusivity among major disciplines.
* Minority class difficulty is driven primarily by **data sparsity**, not strong label entanglement.
* Classical ML remains competitive when properly tuned.
* Transformer-based contextual embeddings improve minority detection.
* Threshold calibration is critical in multi-label classification.
* Model capacity alone does not solve imbalance challenges.

---

##  Tech Stack

* Python
* pandas / NumPy
* scikit-learn
* spaCy
* matplotlib / seaborn
* HuggingFace Transformers
* PyTorch

---

##  Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Open and run:

```
notebooks/scientific_multilabel_classification_comparative_study.ipynb
```

GPU recommended for SciBERT training.

---

##  Project Takeaways

This study demonstrates structured experimentation across:

* Unsupervised learning
* Classical supervised learning
* Transformer fine-tuning
* Class imbalance mitigation
* Probability calibration

It highlights the importance of evaluation strategy and data distribution awareness in real-world NLP classification tasks.

---

##  Author

Christopher Overton
Applied Data Science | Machine Learning | NLP


