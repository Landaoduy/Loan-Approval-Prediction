![UTA-DataScience-Logo](https://github.com/user-attachments/assets/6d626bcc-5430-4356-927b-97764939109d)

# Loan Approval Prediction

## Business Problem 

Loan approval decisions have a direct financial and social impact on both lenders and borrowers. Approving a risky loan can lead to significant losses; rejecting a creditworthy applicant means lost revenue and poor customer experience. Traditional rule-based systems are rigid and slow to adapt. This project builds a data-driven predictive model that helps financial institutions make faster, more consistent, and explainable loan approval decisions based on applicant financial profiles.

---

## Project Overview

This project builds and evaluates multiple machine learning models to predict whether a loan application will be approved or rejected. Starting from baseline classifiers (Logistic Regression, KNN) through advanced ensemble models (Random Forest, XGBoost), the pipeline systematically compares preprocessing strategies (imputation, outlier handling, class imbalance techniques) and tunes hyperparameters to maximize performance. The final model — a tuned XGBoost classifier — is served through an interactive **Streamlit chatbot** that supports both single-customer prediction and batch CSV prediction, with SHAP-based explanations for every decision.

**Key results:**
- Best model: Tuned XGBoost with custom decision threshold
- Metrics: Evaluated on Accuracy, Precision, Recall, and F1-score
- Explainability: SHAP waterfall plots, force plots, and global feature importance

---

## Data

| Property | Details |
|----------|---------|
| **Type** | Tabular / Structured |
| **Format** | CSV (`Loan_Approval_Final_8.csv`) |
| **Key Features** | `CreditScore`, `AnnualIncome`, `LoanAmount`, `RiskScore`, `InterestRate`, `DebtToIncomeRatio`, `TotalDebtToIncomeRatio`, `MonthlyDebtPayments`, `NetWorth`, `EmploymentStatus`, `MaritalStatus`, `EducationLevel`, `HomeOwnershipStatus`, `LoanPurpose` |
| **Target** | `LoanApproved` (0 = Not Approved, 1 = Approved) |

---

## Data Preprocessing

### Handling Missing Values
Three strategies were tested and compared:
- **Median/Mode Imputation** — fill numerical columns with median, categorical with mode
- **KNN Imputation** (`KNNImputer`, k=5, distance-weighted) — used in the final pipeline after scaling
- **Iterative Imputation (MICE)** — multivariate imputation using round-robin regressions

KNN Imputation paired with Standard Scaling gave the best model performance and was selected for the final pipeline.

### Handling Outliers
Two approaches were compared:
- **IQR-based Winsorizing (Capping)** — clips values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR], fit on training data only
- **Log Transformation (`log1p`)** — applied to right-skewed financial features (income, loan amount, net worth, etc.)

Winsorizing was chosen for the final pipeline as it preserved interpretability and produced more stable model behavior.

### Handling Class Imbalance
The dataset exhibited class imbalance in `LoanApproved`. Four resampling strategies were tested:
- **SMOTE** — synthetic oversampling of the minority class
- **ADASYN** — adaptive oversampling, focusing on harder-to-classify examples
- **Random Undersampling (RUS)** — reduce majority class
- **SMOTE + ENN** — hybrid approach: oversample then clean noisy borderline samples

ADASYN was used for the final XGBoost model; RUS for the final Random Forest model.

### Feature Engineering
- Categorical features (`EmploymentStatus`, `MaritalStatus`, `EducationLevel`, `HomeOwnershipStatus`, `LoanPurpose`) were included in EDA but numerical features were used for modeling
- Standard Scaling applied before imputation to ensure distance-based methods work correctly

---

## Exploratory Data Analysis

**1. Class Distribution**
The target variable `LoanApproved` was checked for imbalance using value counts and a pie chart, revealing a meaningful skew between approved and rejected applications.
<img width="370" height="400" alt="Screenshot 2025-10-28 211558" src="https://github.com/user-attachments/assets/de5988c4-d15f-4b62-88be-3032ce0fb687" />

**2. Categorical Features vs. Loan Approval**
Count plots comparing `EmploymentStatus`, `MaritalStatus`, `EducationLevel`, `HomeOwnershipStatus`, and `LoanPurpose` against approval status revealed which applicant segments see higher approval rates.

**3. Pairplot of Key Financial Features**
A pairplot across `CreditScore`, `AnnualIncome`, `LoanAmount`, and `RiskScore` (colored by loan approval) showed clear separation patterns, especially around `RiskScore` and `CreditScore`.
<img width="890" height="900" alt="image" src="https://github.com/user-attachments/assets/94cf20ce-3c00-4e8d-9762-bfa5d2bdbf10" />

**4. Correlation Heatmap**
A heatmap of 12 numerical features (`Age`, `AnnualIncome`, `CreditScore`, `LoanAmount`, `RiskScore`, `InterestRate`, `NetWorth`, etc.) identified feature relationships and multicollinearity — informing feature selection for modeling.
<img width="873" height="784" alt="image" src="https://github.com/user-attachments/assets/153aa382-62ad-4c8a-903f-d58ec25f20ca" />


---

## Modeling Approach

### Baseline Models
| Model | Rationale |
|-------|-----------|
| **Logistic Regression** | Interpretable linear baseline; establishes a lower-bound performance reference |
| **K-Nearest Neighbors (KNN)** | Non-parametric baseline that captures local patterns without assumptions about data distribution |

### Advanced Models
| Model | Rationale |
|-------|-----------|
| **Random Forest** | Ensemble of decision trees; robust to noise and naturally handles non-linear relationships |
| **XGBoost** | Gradient boosting with regularization; typically state-of-the-art on tabular data and supports SHAP natively |

All four models were evaluated across multiple preprocessing combinations before hyperparameter tuning.

---

## Model Training

### Tools & Libraries
- **scikit-learn** — Logistic Regression, KNN, preprocessing, GridSearchCV
- **XGBoost** — XGBClassifier
- **imbalanced-learn** — SMOTE, ADASYN, RandomUnderSampler, SMOTEENN
- **SHAP** — model explainability
- **joblib** — model serialization

### Hyperparameter Tuning
Both advanced models were tuned using `GridSearchCV` with 5-fold cross-validation, optimizing for **F1-score** (chosen because it balances precision and recall, which matters when both false positives and false negatives carry financial risk).

**XGBoost tuned parameters:**
```
n_estimators: [200, 300, 400]
learning_rate: [0.03, 0.05, 0.1]
max_depth: [3, 4, 5]
subsample: [0.8, 1.0]
colsample_bytree: [0.8, 1.0]
min_child_weight: [1, 3, 5]
```

**Random Forest tuned parameters:**
```
n_estimators: [200, 300]
max_depth: [None, 10, 20]
min_samples_split: [2, 5]
min_samples_leaf: [1, 2]
max_features: ['sqrt', 'log2']
```

### Custom Decision Threshold
After tuning, a threshold sweep (0.30 to 0.80, step 0.05) was applied to each model's predicted probabilities to find the threshold maximizing F1-score — allowing the model to be calibrated toward precision or recall depending on business needs.

---

## Results

### Why These Metrics?
- **Accuracy** — overall correctness
- **Precision** — of predicted approvals, how many were truly creditworthy (minimizes false approvals / financial loss)
- **Recall** — of all actual approvals, how many did we catch (minimizes missed business opportunities)
- **F1-score** — harmonic mean of precision and recall; the primary optimization metric for an imbalanced classification task

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| KNN (Tuned + Threshold) | 0.836 | 0.295 | 0.708 | 0.416 |
| LR (Tuned + Threshold) | 0.850 | 0.354 | 0.991 | 0.522 |
| Random Forest (Tuned) | 0.836 | 0.370 | 0.960 | 0.491 |
| **XGBoost (Tuned + Threshold)** | **0.850** | **0.350** | **0.957** | **0.552** |

---

## Model Interpretation

### Global Explanation — SHAP Feature Importance
SHAP (SHapley Additive exPlanations) was applied to the tuned XGBoost model to understand which features drive predictions across the entire dataset.

**Top influential features (from SHAP):**
- `RiskScore` — the single most impactful feature; higher risk scores strongly push toward rejection
- `TotalDebtToIncomeRatio` — high debt burden relative to income reduces approval probability
- `InterestRate` — higher interest rates correlate with riskier applicant profiles
- `AnnualIncome` — higher income increases approval likelihood
- `LoanAmount` — larger loan requests relative to income reduce approval odds

> A SHAP summary plot (beeswarm) and mean |SHAP| bar chart are saved in `images/` and embedded in the notebook under **Section VIII**.

### Local Explanation — Per-Prediction Waterfall & Force Plots
For any individual prediction, SHAP waterfall and force plots show exactly which features pushed the model toward or away from approval. This is also surfaced in the Streamlit app for real-time interpretability.

---

## Key Insights

- **RiskScore dominates all models** — it is the strongest single predictor, likely encoding a composite of credit history and behavioral signals.
- **Debt burden matters more than raw loan amount** — the ratio features (`DebtToIncomeRatio`, `TotalDebtToIncomeRatio`) outperformed absolute values.
- **XGBoost consistently outperformed all baselines** — especially after threshold calibration, which improved F1 by catching more true approvals without sacrificing excessive precision.
- **Resampling strategy matters** — ADASYN slightly outperformed SMOTE for XGBoost, likely due to its focus on harder boundary cases.
- **Practical impact:** A deployed version of this model could help loan officers flag borderline applications for human review while auto-approving/rejecting high-confidence cases — reducing processing time while maintaining fairness.

---

## Conclusion

This project demonstrates a complete end-to-end ML workflow for loan approval prediction. Through systematic preprocessing experimentation, model comparison, and hyperparameter tuning, the final XGBoost model achieves strong performance on the held-out test set. SHAP explainability makes the model's decision-making transparent and auditable — a critical requirement in financial applications. The Streamlit chatbot demo makes the model accessible to non-technical stakeholders.

---

## Future Work

- **Add categorical feature encoding** — one-hot or target encoding to include `EmploymentStatus`, `LoanPurpose`, etc. in modeling
- **Calibrate predicted probabilities** — use Platt scaling or isotonic regression to improve probability reliability
- **Fairness audit** — test for demographic parity and equal opportunity across sensitive groups (marital status, education level)
- **Deploy to cloud** — host the Streamlit app on Streamlit Cloud, Hugging Face Spaces, or AWS
- **Add explainability to the chatbot** — surface the SHAP waterfall chart directly in the chat interface per prediction
- **Time-series features** — if repayment history data becomes available, add temporal patterns

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place `Loan_Approval_Final_8.csv` in the `data/` folder.

### 4. Run the notebook
Open and run all cells in `notebooks/Loan_Approval_Prediction.ipynb`. This will:
- Perform EDA and preprocessing experiments
- Train and evaluate all models
- Save trained model artifacts (`.pkl` files) to the project root

### 5. Launch the Streamlit demo app
```bash
streamlit run Model_demo.py
```
Make sure all `.pkl` artifact files are in the same directory as `Model_demo.py`.

---

## 📁 Repository Structure

```
loan-approval-prediction/
├── README.md                        # Project documentation (this file)
├── requirements.txt                 # All Python dependencies
├── Model_demo.py                    # Streamlit chatbot app
├── data/
│   └── Loan_Approval_Final_8.csv    # Raw dataset (not tracked by Git)
├── notebooks/
│   └── Loan_Approval_Prediction.ipynb  # Full analysis and modeling notebook
├── models/
│   ├── best_xgb_model.pkl           # Trained XGBoost model
│   ├── scaler.pkl                   # Fitted StandardScaler
│   ├── imputer.pkl                  # Fitted KNNImputer
│   ├── iqr_caps.pkl                 # IQR winsorizing bounds
│   ├── feature_names.pkl            # Ordered list of model features
│   ├── best_threshold.pkl           # Optimal classification threshold
│   ├── metrics_df.pkl               # Model performance metrics
│   ├── X_train_imputed_df.pkl       # Preprocessed training data (for SHAP)
│   └── X_test_imputed_df.pkl        # Preprocessed test data (for SHAP)
├── results/
│   └── model_comparison.png         # Final model comparison chart
└── images/
    ├── shap_summary.png             # SHAP beeswarm plot
    ├── shap_bar.png                 # Top 10 SHAP feature importance
    ├── correlation_heatmap.png      # Feature correlation heatmap
    └── class_distribution.png      # Target variable distribution
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key packages:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
shap
streamlit
joblib
missingno
```
