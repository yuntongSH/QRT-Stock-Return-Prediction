
# QRT Stock Return Prediction

## Introduction

This project is part of our student initiative to tackle the **Qube Research & Technologies (QRT)** stock return prediction challenge. The task is to predict the **sign of residual returns** (returns stripped of market-wide effects) of U.S. equities using 20 business days of historical data. With over **600,000 samples**, this challenge is representative of real-world quantitative finance problems where the **signal-to-noise ratio is extremely low**.

We approach this challenge through an end-to-end machine learning pipeline, including data cleaning, feature engineering, modeling with multiple algorithms, and ensemble learning for final prediction.

🔗 **Official Challenge**: [QRT Stock Return Prediction on Challenge Data](https://challengedata.ens.fr/participants/challenges/23/#)

---

## Dataset Description

- **Training Set**: 418,595 samples
- **Test Set**: 198,429 samples
- **Features**:
  - 20-day historical residual returns: `RET_1` to `RET_20`
  - 20-day relative volumes: `VOLUME_1` to `VOLUME_20`
  - Categorical: `STOCK`, `SECTOR`, `INDUSTRY`, `INDUSTRY_GROUP`, `SUB_INDUSTRY`
  - Target: `RET` (binary, top 50% residual return → 1)

---

## Pipeline Overview

### 1. Data Cleaning

- **Missing Returns**: Rows with any missing values across the full 20-day window (`RET_1` to `RET_20`) were removed from both datasets to ensure complete return history. This resulted in the removal of **6,126 rows from the training set (1.46%)** and **4,494 rows from the test set (2.26%)**.
- **Missing Volumes**: Missing values in `VOLUME_1` to `VOLUME_20` were imputed using the **mean value of the same `INDUSTRY_GROUP` and `DATE`**. This preserves contextual consistency while reducing imputation bias.
- **Categorical Consistency**: Categorical identifiers such as `STOCK`, `SECTOR`, `INDUSTRY`, and `INDUSTRY_GROUP` were preserved and checked to ensure alignment between training and test sets, avoiding category leakage or mismatches during encoding.

### 2. Feature Engineering

We engineered advanced features to enrich the signal:

- **Moving Averages (MA)**: 5-day and 10-day rolling averages over return windows.
- **Exponential Moving Averages (EMA)**: Smoothed features with decayed emphasis on recent days.
- **MACD**: Difference between short-term and long-term EMAs, capturing momentum shifts.
- **Volatility**: Standard deviation of 20-day return history.
- **Group-Based Aggregations**: Mean returns per `SECTOR` and `INDUSTRY_GROUP` per date.

All features were standardized using `StandardScaler` to ensure uniform input to machine learning models.

### 3. Feature Selection

Selected features include:
- Historical: `RET_1` to `RET_5`, `VOLUME_1` to `VOLUME_5`
- Engineered: `VOLATILITY`, `RET_MA*`, `RET_EMA*`, `RET_MACD*`, `RET_IN_SECTOR*`
- Total number of features used: *varies based on data quality*

---

## Models and Evaluation

We trained and validated four models using time-based cross-validation (KFold on `DATE`):

- **Random Forest**: 500 trees, max depth 10
- **XGBoost**: 100 trees, depth 4, learning rate 0.05
- **CatBoost**: 200 iterations, depth 4
- **Neural Network**: 4-layer MLP with dropout and batch normalization

Each model was evaluated using:
- **Accuracy**
- **ROC AUC**
- **Classification Report**
- **Confusion Matrix**

---
## Ensemble Performance

To maximize prediction accuracy, we constructed an **optimized weighted ensemble** by combining outputs from four base models:

- **Random Forest**
- **XGBoost**
- **CatBoost**
- **Neural Network**

Instead of assigning equal weights, we used a **numerical optimization method** to identify the best-performing set of ensemble weights. The optimal weights found were:

- **[ 2.45, -0.33, -0.78, -0.33 ]** for **RF, XGB, CAT, NN**, respectively

> Note: Negative weights indicate that certain models may contribute inverse predictive signal, and their inclusion improves overall ensemble calibration.

We applied a **per-date median thresholding** strategy to convert predicted probabilities into binary class labels. This ensures that, for each trading day, roughly half the stocks are classified as having positive residual return signals, as expected by the challenge design.

### Final Results on the Training Set:

- **Ensemble Accuracy**: `61.55%`  
- **Ensemble ROC AUC**: `0.6709`

**Classification Report:**
          precision    recall  f1-score   support

       0     0.6148    0.6246    0.6196    206807
       1     0.6163    0.6064    0.6113    205662

accuracy                         0.6155    412469

These results mark a significant improvement over the baseline (51.31%) and earlier ensemble versions, confirming that learned weighting schemes can outperform uniform ensembling in noisy financial classification tasks.

## Final Submission

- Test predictions were generated using the final ensemble model.
- A threshold per date was applied to assign class labels.
- Output was saved in submission format as: `qrt_output_stock_return_prediction.csv`.

---

## Usage

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost catboost tensorflow seaborn matplotlib
```

### Run
Open the notebook:
```
QRT Stock Return Prediction Notebook.ipynb
```
and execute the cells in sequence.

Place the required files (`x_train.csv`, `y_train.csv`, `x_test.csv`) in the notebook directory.

---

## Conclusion

This project demonstrates the complete development of a multi-model machine learning system for noisy financial time-series classification. Through structured preprocessing, targeted feature engineering, and robust ensembling, we surpassed the baseline benchmark.