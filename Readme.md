BAN 6800 Module 4 Assignment: Business Analytics Model
 
## Project Overview
This project focuses on evaluating marketing campaign effectiveness for *Unilever Nigeria* using predictive analytics. The goal is to identify key drivers of campaign success and optimize Return on Marketing Investment (ROMI).
 
## Dataset Description
- *Source*: Kaggle Bank Marketing Dataset
- *Records*: 41,000
- *Features*:
  - Demographics: age, job, marital, education
  - Financial: housing, loan
  - Campaign details: contact type, month, day_of_week, duration
  - Economic indicators: emp_var_rate, cons_price_idx, euribor3m
  - Target: response_binary (0 = No, 1 = Yes)
 
## Installation Instructions
1. Clone the repository:
bash
git clone <your-repo-url>
cd <your-repo-folder>

2. Install dependencies:
bash
pip install -r requirements.txt

 
## How to Run the Notebook
1. Open Jupyter Notebook:
bash
jupyter notebook BAN_6800_Module4.ipynb

2. Run all cells sequentially.
 
## Model Details
- *Models Used*:
  - Logistic Regression (baseline)
  - Random Forest (advanced)
- *Preprocessing*:
  - One-hot encoding for categorical variables
  - Standard scaling for numeric features
 
## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve & AUC
 
## Business Insights
- Customers with recent and longer interactions respond positively.
- Economic conditions influence campaign success.
- Middle-aged professionals show higher engagement.
 
## Recommendations
- Optimize contact strategy based on recency and duration.
- Align campaigns with favorable economic conditions.
- Target high-value segments (age/job categories).
 
## Repository Structure

├── BAN_6800_Module4.ipynb       # Main notebook for model development
├── bank_marketing_cleaned.csv   # Cleaned dataset
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── artifacts/                   # Saved models and schema
└── figures/                     # Saved plots
└── BAN 6800.ipynb/              # Previous milestone notebook



 
## Reproducibility & Deployment 
 
- **Artifacts in artifacts/**:
  - model_logreg.joblib – trained Logistic Regression model
  - model_rf.joblib – trained Random Forest model
  - scaler.joblib – fitted StandardScaler
  - schema.json – column names and preprocessing details
  - metrics.json – model performance summary
 
- **Figures in figures/**:
  - confusion_matrices.png
  - roc_curve.png
  - rf_feature_importance.png
  - logreg_coefficients.png
 
### Quick Inference Example
python
import pandas as pd, joblib, json
 
# Load new data
df_raw = pd.read_csv("data/new_customers.csv")
 
# Load artifacts
logreg = joblib.load("artifacts/model_logreg.joblib")
scaler = joblib.load("artifacts/scaler.joblib")
schema = json.load(open("artifacts/schema.json"))
 
# Preprocess new data
df_enc = pd.get_dummies(df_raw, columns=schema["categorical_cols"], drop_first=True)
df_enc = df_enc.reindex(columns=schema["train_columns"], fill_value=0)
df_enc[schema["numeric_cols"]] = scaler.transform(df_enc[schema["numeric_cols"]])
 
# Predict probabilities
proba = logreg.predict_proba(df_enc)[:, 1]
print(proba[:10])

  
## References
- Harvard Business Review (2022): How well does your company use analytics?
- Kaggle Bank Marketing Dataset
- scikit-learn: Model persistence best practices