# Credit Card Fraud Detection Using Machine Learning

## Overview
A supervised machine learning project for identifying potentially fraudulent credit card transactions. The notebook covers exploratory data analysis, preprocessing, model tuning, evaluation, and ensemble learning.

## Dataset
The notebook expects the original `creditcard.csv` dataset in the `data/` directory:

```text
credit-card-fraud-detection-ml/
├── data/
│   └── creditcard.csv   # not included in this repository
└── notebooks/
    └── Credit_Card_Fraud_Detection.ipynb
```

The dataset is not committed because it is not part of the uploaded project files. Download it from the original dataset source and place it at `data/creditcard.csv` before running the notebook.

## Workflow
1. Exploratory Data Analysis (EDA) and class distribution inspection.
2. Majority-class undersampling: 20,000 legitimate transactions are sampled and combined with the fraud transactions.
3. Duplicate and missing-value checks.
4. Stratified train/validation split.
5. Preprocessing with imputation, `log1p` transformation for `Amount`, and `StandardScaler`.
6. Hyperparameter tuning with `GridSearchCV` and 5-fold cross-validation.
7. Evaluation of Logistic Regression, Random Forest, and a soft Voting Classifier.

## Models
- Logistic Regression with `class_weight='balanced'`
- Random Forest with `class_weight='balanced'`
- Soft Voting Classifier combining the tuned Logistic Regression and Random Forest models

## Validation Results

### Soft Voting Classifier

| Metric | Score |
|---|---:|
| Precision | 91.21% |
| Recall | 87.37% |
| F1-score | 89.25% |
| PR-AUC (Average Precision) | 89.55% |
| ROC-AUC | 97.32% |

### Random Forest

| Metric | Score |
|---|---:|
| Precision | 96.34% |
| Recall | 83.16% |
| F1-score | 89.27% |
| PR-AUC (Average Precision) | 88.93% |
| ROC-AUC | 98.05% |

The Voting Classifier produced the stronger Precision/Recall balance by F1-score, while Random Forest achieved higher Precision and ROC-AUC in the reported validation results.

## Important Evaluation Note
The notebook evaluates on a validation split created **after** undersampling the majority class. Therefore, the validation class distribution differs substantially from the original real-world distribution. These metrics should be treated as results on the sampled validation set, not as a guarantee of production performance. A further evaluation on an untouched test set retaining the original class distribution is recommended before deployment.

## Run Locally
1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Put `creditcard.csv` in `data/`.
4. Open and run `notebooks/Credit_Card_Fraud_Detection.ipynb`.

## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter, Joblib.

## Author
Ali Mahmoud Ali
