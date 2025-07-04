# 🚀 Employee Promotion Prediction – ML Model (XGBoost)

This project aims to predict whether an employee is likely to be promoted, based on various personal and performance-related features. It uses **XGBoost** as the main machine learning algorithm, combined with preprocessing, visualization, and evaluation steps inside a Jupyter notebook.

---

## 📘 Notebook: `UNNIKRISHNAN C EXIT TEXT.ipynb`

This notebook includes:

- Data loading (from CSV)
- EDA (Exploratory Data Analysis) using `matplotlib` and `seaborn`
- Feature engineering and preprocessing
- Model training with `XGBClassifier`
- Evaluation using F1-score, Accuracy, and Classification Report
- Final predictions for submission

---

## 📁 Dataset

- `train_LZdllcl.csv`: Training data with features and target variable.
- `test_2umaH9m.csv`: Test data without the target column.

---

## ⚙️ Tools and Libraries

- Python 3
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- xgboost

---

## 📊 Model: XGBoostClassifier

Used model:
```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1,
    eval_metric='logloss',
    random_state=42
)
