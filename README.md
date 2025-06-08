# QRT Stock Return Prediction

## Introduction

This project is part of our student initiative in the **Executive Master of Artificial Intelligence and Data Science at Dauphine PSL**, participating in the **Qube Research & Technologies (QRT)** stock return prediction challenge. The goal is to predict the **sign of residual stock returns**—returns adjusted to remove market-wide influences—based on 20 business days of historical data.

With over **600,000 samples**, the challenge mirrors real-world quantitative finance problems where the **signal-to-noise ratio is exceptionally low**. Our project explores both classical and deep learning models in this noisy time-series environment.

🔗 **Official Challenge**: [QRT Stock Return Prediction on Challenge Data](https://challengedata.ens.fr/participants/challenges/23/#)

---

## Dataset Description

- **Training Set**: 418,595 samples
- **Test Set**: 198,429 samples
- **Features**:
  - `RET_1` to `RET_20`: 20-day historical residual returns
  - `VOLUME_1` to `VOLUME_20`: 20-day relative volumes
  - Categorical: `STOCK`, `SECTOR`, `INDUSTRY`, `INDUSTRY_GROUP`, `SUB_INDUSTRY`
  - Target: `RET` (binary, return sign)

---

## Pipeline Overview

### 1. Data Cleaning

- **Missing Returns**: Rows missing any `RET_1` to `RET_20` were dropped:  
  - 6,126 rows from training  
  - 4,494 rows from test
- **Missing Volumes**: Imputed using **industry group and date-wise means**.
- **Categorical Alignment**: Verified consistency across `STOCK`, `SECTOR`, `INDUSTRY`, and `SUB_INDUSTRY`.

### 2. Feature Engineering

New and enhanced features:
- **Moving Averages (MA)** and **Exponential Moving Averages (EMA)**
- **MACD** indicators
- **Volatility**: Std dev of return history
- **Group Aggregations**: Mean return per `SECTOR` and `INDUSTRY_GROUP` by date

All features are standardized via `StandardScaler`.

---

## Models and Evaluation

We developed and validated the following models using time-based cross-validation (KFold over `DATE`):

- **Random Forest**: 500 trees, depth=10  
- **XGBoost**: 100 estimators, depth=4  
- **CatBoost**: 200 iterations  
- **Neural Network**: 4-layer MLP with dropout and batch norm

### Evaluation Metrics
- **Accuracy**
- **ROC AUC**
- **Classification Report**
- **Confusion Matrix**

---

## Ensemble Strategy

We built a **numerically optimized weighted ensemble** of the four base models:

- Final weights: `[ 2.45, -0.33, -0.78, -0.33 ]` for **RF, XGB, CAT, NN**
- Applied **per-date median thresholding** for label binarization, conforming to challenge logic.

### Final Training Results

- **Accuracy**: `61.55%`
- **ROC AUC**: `0.6709`

**Classification Report:**
```
          precision    recall  f1-score   support
       0     0.6148    0.6246    0.6196    206807
       1     0.6163    0.6064    0.6113    205662
```

This is a large improvement over the baseline (51.31%).

---

## Final Submission

- Predictions generated from the ensemble model
- Thresholding applied per date
- Output format: `qrt_output_stock_return_prediction.csv`

---

## Usage

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost catboost tensorflow seaborn matplotlib shap
```

### Run

Open the notebook:
```
QRT Stock Return Prediction Notebook.ipynb
```

Make sure the data files (`x_train.csv`, `y_train.csv`, `x_test.csv`) are in the same directory.

---

## Conclusion

This project provides a full-stack machine learning pipeline for financial time-series classification, incorporating both engineered and deep features, hybrid modeling, and a data-driven ensemble strategy. The approach significantly outperforms the baseline, highlighting the benefits of structured feature processing and ensemble learning in noisy data environments.