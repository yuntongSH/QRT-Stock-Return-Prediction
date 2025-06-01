
# QRT Stock Return Prediction

## Introduction
The QRT Stock Return Prediction project is a machine learning challenge focused on predicting the sign of residual stock returns in the U.S. equity market using 20 days of historical data. Residual returns are computed after removing market-wide effects, making the problem highly relevant to real-world quantitative investment strategies where signal-to-noise ratios are low.

Our team implemented and evaluated several classification models—including Random Forest, XGBoost, CatBoost, and a Neural Network—and combined them using a weighted ensemble to improve predictive performance.

🔗 **Challenge Link**: https://challengedata.ens.fr/participants/challenges/23/#

## Objectives
- Load and preprocess historical stock data.
- Engineer informative features using time-series and group-based techniques.
- Train and evaluate multiple machine learning models.
- Implement an ensemble model for improved robustness and accuracy.
- Analyze performance using accuracy, AUC, and classification metrics.

## Files
- `Benchmark QRT.ipynb`: Main notebook containing the full pipeline: data cleaning, feature engineering, model training, evaluation, and ensemble learning.

## Usage Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/QRT-Stock-Return-Prediction.git
cd QRT-Stock-Return-Prediction
```

### 2. Install Required Libraries
```bash
pip install pandas numpy scikit-learn xgboost catboost tensorflow seaborn matplotlib
```

### 3. Run the Notebook
Open `Benchmark QRT.ipynb` using Jupyter Notebook or VS Code, and run all cells step-by-step.

### 4. Prepare the Dataset
Ensure the input data files (`x_train.csv`, `y_train.csv`, and `x_test.csv`) are located in the same directory as the notebook.

## Model Performance

After full training and validation, the ensemble model (combining Random Forest, XGBoost, CatBoost, and Neural Network with equal weights) achieved the following performance on the training set:

- **Ensemble Accuracy**: 56.92%
- **Ensemble ROC AUC**: 0.6020

**Classification Report:**
```
              precision    recall  f1-score   support

           0     0.5663    0.6014    0.5833    206807
           1     0.5725    0.5369    0.5541    205662

    accuracy                         0.5692    412469
   macro avg     0.5694    0.5691    0.5687    412469
weighted avg     0.5694    0.5692    0.5688    412469
```

## Conclusion
This project demonstrates the power of combining machine learning models for time-series classification in financial data. Through feature engineering, careful validation, and ensemble methods, we successfully improved prediction accuracy in a noisy and complex domain.
