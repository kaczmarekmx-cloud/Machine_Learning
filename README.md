# Machine Learning Projects

A collection of machine learning projects exploring the full ML pipeline — from raw data to critical interpretation — across both **regression** and **classification** tasks.

**Author:** Maksymilian Kaczmarek
**Institution:** Collegium Da Vinci, Poznań · 2025/2026

---

## Overview

| Project | Type | Goal | Dataset |
|---|---|---|---|
| 🏊‍♂️🚴‍♂️🏃‍♂️ **Ironman Performance Predictor** | Regression | Predict Ironman finish time from athlete profile | ~1.44M race records (70.3 & 140.6) |
| 🍷 **Wine Classifier** | Classification | Identify a wine's vineyard from its chemistry | 178 samples, 13 chemical features |

Both projects share the same philosophy: **don't just train a model — understand why it behaves the way it does.**

---

## 🏊‍♂️🚴‍♂️🏃‍♂️ Ironman Performance Predictor

Predicting the total finish time of an Ironman triathlon from information known *before* the race: gender, age group, country and distance.

### Pipeline
- **Data:** Two Kaggle datasets (Ironman 140.6 and 70.3) merged into ~1.44M cleaned records.
- **Cleaning:** Converting `HH:MM:SS` times to seconds, unifying categories (gender, age groups), and detecting a real data-quality bug in the source (records where `runTime` equalled `overallTime`).
- **Models:** Linear Regression, Random Forest, XGBoost, MLP — compared head to head.
- **Metrics:** MAE, RMSE, R².

### Key finding — the confounding variable
All four models reached **R² ≈ 0.89**, which looked impressive. But a controlled experiment told a different story:

| Configuration | R² |
|---|---|
| Full model (with `distance`) | **0.89** |
| Profile only, distance removed | **0.23** |

The high score came almost entirely from knowing the **distance** (70.3 vs 140.6 is a huge time gap), not from the athlete's profile. Within a single distance, demographic profile explains only ~23% of the variance.

This is a textbook **confounding variable** / **Simpson's paradox**: the swim–finish correlation appeared strong (r = 0.91) on the combined data but dropped to r ≈ 0.60 once the distances were analysed separately.

### Takeaway
Four very different algorithms converged on the same result — a sign that the ceiling is set by the **data**, not the choice of model. Predicting individual performance needs training data (FTP, weekly volume, experience) that demographics simply don't capture.

---

## 🍷 Wine Classifier

Classifying which of three vineyards a wine comes from, based on 13 chemical measurements — paired with a deliberate experiment on **when model choice actually matters**.

### Part 1 — Wine (real data)
- **Data:** UCI Wine dataset — 178 samples, 13 chemical features, 3 classes.
- **Models:** Logistic Regression, Random Forest, XGBoost, MLP (all classifiers).
- **Validation:** train/test split *and* 5-fold cross-validation.
- **Result:** The problem is **linearly separable** (visible in the PCA projection), so every model scores 95–99%. The simplest model — **Logistic Regression — actually wins** in cross-validation (98.9%) and is the most stable.

### Part 2 — Two Moons (synthetic)
A non-linear dataset of two interleaving crescents, used to show the opposite case:

| Model | Wine (CV) | Two Moons |
|---|---|---|
| Logistic Regression | **98.9%** (best) | **88.0%** (worst) |
| MLP | 97.2% | 98.0% |
| Random Forest | 97.2% | 97.6% |
| XGBoost | 95.0% | 97.6% |

The **same model** is the best on Wine and the worst on Two Moons. Decision-boundary plots make the reason visual: Logistic Regression can only draw a straight line, while trees and the neural net learn the curves.

### Takeaway
There is no universally "best" model. On simple, linearly separable data a simple model wins (and is more stable); on complex, non-linear data it loses badly. **Model choice should follow the shape of the data, not a trend.**

---

## Techniques used

- **Data wrangling:** pandas, feature engineering, one-hot encoding, outlier detection, data-quality auditing
- **Models:** Linear/Logistic Regression, Random Forest, XGBoost, MLP (scikit-learn, xgboost)
- **Validation:** train/test split, stratification, 5-fold cross-validation
- **Dimensionality reduction:** PCA
- **Visualization:** Plotly (interactive charts, confusion matrices, feature importance), matplotlib (decision boundaries)
- **Interpretation:** correlation analysis, confounding variables, Simpson's paradox, the bias toward simpler models (Occam's razor)

---

## Repository structure

```
Machine_Learning/
├── Ironman_Performance_Predictor.ipynb   # Regression project
├── Wine_Classifier.ipynb                 # Classification project
└── README.md
```

## Running the notebooks

The notebooks are built for **Google Colab**. Each downloads its data automatically via `kagglehub`, so no manual file handling is needed — open in Colab and run all cells.

```python
import kagglehub
path = kagglehub.dataset_download("<dataset-slug>")
```

---

*These projects were developed as coursework for a Machine Learning module, with an emphasis on critical interpretation of results rather than chasing accuracy alone.*
