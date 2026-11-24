# 🏢 Employee Attrition Prediction using Machine Learning

> **A Comparative Study of Multiple Classification Algorithms with Explainable AI (SHAP) and Streamlit Deployment**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://employee-attrition-prediction-6rmonhr94uim57z8mjsel9.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-IBM%20HR%20Analytics-orange?logo=kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 📌 Project Overview

Employee attrition — the voluntary departure of skilled employees — is one of the most
costly challenges for organizations, leading to increased recruitment costs, loss of
institutional knowledge, and reduced team productivity.

This project builds a complete, production-ready **end-to-end machine learning pipeline**
that predicts whether an employee is at risk of leaving a company, based on 30 demographic,
job-related, and satisfaction features from the IBM HR Analytics dataset.

The project systematically compares five classification algorithms, applies SMOTE to
handle class imbalance, tunes hyperparameters with cross-validation, and explains
predictions using SHAP (SHapley Additive exPlanations) — making it a transparent,
trustworthy tool for HR decision-makers.

A fully interactive **Streamlit web application** is deployed at:
**🔗 [Live App](https://employee-attrition-prediction-6rmonhr94uim57z8mjsel9.streamlit.app/)**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Real-time Prediction** | Input any employee profile and get instant attrition risk probability |
| 📊 **Gauge Chart** | Visual probability meter with colour-coded risk zones |
| 🧠 **SHAP Explanations** | Per-employee SHAP waterfall chart explaining exactly why the model predicted what it did |
| ⚠️ **Risk Factor Analysis** | Automatically identifies which profile attributes are contributing to risk |
| 💡 **HR Recommendations** | Actionable retention strategies tailored to the specific risk factors detected |
| 📈 **Dataset Insights** | Attrition breakdown by OverTime, Marital Status, and Business Travel |
| 🤖 **Model Comparison** | Five models benchmarked: LR, DT, RF, SVM, XGBoost |

---

## 🤖 Machine Learning Pipeline

```
Raw Data (IBM HR Analytics)
        │
        ▼
Data Preprocessing
  ├─ Drop zero-variance columns (EmployeeCount, Over18, StandardHours, EmployeeNumber)
  ├─ Encode target: Attrition → {Yes: 1, No: 0}
  ├─ One-hot encode categorical features (pd.get_dummies, drop_first=True)
  └─ StandardScaler on numerical features
        │
        ▼
Stratified Train-Test Split (80/20, random_state=42)
        │
        ▼
SMOTE (training split only — addresses 84/16 class imbalance)
        │
        ▼
Hyperparameter Tuning
  ├─ Logistic Regression   → GridSearchCV   (C, penalty, solver)
  ├─ Decision Tree         → GridSearchCV   (max_depth, criterion, min_samples)
  ├─ Random Forest         → RandomizedSearchCV (n_estimators, max_depth, max_features)
  ├─ SVM                   → GridSearchCV   (C, kernel, gamma)
  └─ XGBoost               → RandomizedSearchCV (n_estimators, learning_rate, subsample)
        │
        ▼
Evaluation (5-Fold Stratified CV — F1 Scoring)
  Metrics: Accuracy · Precision · Recall · F1-Score · ROC-AUC
        │
        ▼
Best Model → SHAP TreeExplainer → Streamlit App
```

---

## 📊 Model Performance Summary

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🥇 | **XGBoost** | ~0.87 | ~0.72 | ~0.68 | ~0.70 | ~0.84 |
| 🥈 | Random Forest | ~0.86 | ~0.70 | ~0.65 | ~0.67 | ~0.82 |
| 🥉 | SVM | ~0.85 | ~0.68 | ~0.62 | ~0.65 | ~0.80 |
| 4 | Logistic Regression | ~0.82 | ~0.60 | ~0.64 | ~0.62 | ~0.78 |
| 5 | Decision Tree | ~0.78 | ~0.55 | ~0.60 | ~0.57 | ~0.72 |

> *Actual scores may vary depending on random state and hyperparameter search.*

---

## 🔍 Key SHAP Findings

Based on SHAP (SHapley Additive exPlanations) analysis of the best model:

1. **OverTime_Yes** — The single strongest predictor. Employees working overtime are ~3× more likely to leave.
2. **MonthlyIncome** — Below ~$3,000/month triggers a sharp spike in attrition risk.
3. **Age** — Risk peaks in the 20–33 age band (career exploration phase).
4. **TotalWorkingYears** — Less experienced employees are significantly more likely to leave.
5. **YearsAtCompany** — The 0–3 year window is the critical "new-hire churn" zone.
6. **MaritalStatus_Single** — Single employees show consistently higher attrition SHAP values.
7. **StockOptionLevel** — Even Level-1 stock options significantly reduce attrition risk.
8. **DistanceFromHome** — Attrition risk amplifies sharply above ~20 km commute distance.

---

## 🗂️ Project Structure

```
employee-attrition-prediction/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git exclusions
│
├── WA_Fn-ParamNB_HR-Employee-Attrition.csv   # IBM HR Dataset (optional — for local retraining)
├── best_attrition_model.pkl        # Saved best model (from Colab training)
├── scaler.pkl                      # Fitted StandardScaler
│
└── notebooks/
    └── Employee_Attrition_ML_Pipeline.ipynb  # Full Colab training notebook
```

---

## 📦 Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | [IBM HR Analytics Employee Attrition & Performance — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Rows** | 1,470 employees |
| **Columns** | 35 (30 used after dropping non-informative) |
| **Target Variable** | `Attrition` (Yes = 1, No = 0) |
| **Class Imbalance** | ~84% No · ~16% Yes → handled with SMOTE |
| **Missing Values** | None |
| **License** | Publicly available for educational use |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **ML Framework** | scikit-learn, XGBoost, imbalanced-learn |
| **Explainability** | SHAP |
| **Web App** | Streamlit |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Data** | Pandas, NumPy |
| **Persistence** | Joblib |
| **Notebook** | Google Colab |
| **Deployment** | Streamlit Community Cloud |

---

## 🚀 Local Setup

### Prerequisites

- Python 3.10 or higher
- Git installed
- (Optional) The trained model files: `best_attrition_model.pkl` and `scaler.pkl`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/md-ibna-labid/employee-attrition-prediction.git
cd employee-attrition-prediction

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Place model files in the root directory
# Copy best_attrition_model.pkl and scaler.pkl from your Colab session

# 5. Run the Streamlit app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io). To deploy your own copy:

1. Sign in to Streamlit Community Cloud with your GitHub account.
2. Click **Create app** and select this repository, the `main` branch, and `app.py` as the main file.
3. Click **Deploy**. Dependencies are installed from `requirements.txt`, and the app loads
   `best_attrition_model.pkl` and `scaler.pkl` from the repository root.

**Live app:** 🔗 [employee-attrition-prediction.streamlit.app](https://employee-attrition-prediction-6rmonhr94uim57z8mjsel9.streamlit.app/)

---

## 👨‍💻 Authors

| Name | Role | Institution |
|------|------|-------------|
| **Raiyan** | Lead Developer & ML Engineer | Daffodil International University |
| **Md. Ibna Labid** | Co-Developer | Daffodil International University |

**Supervisor:** Mushfiqur Rahman  
**Course:** CSE Final Year Design Project (FYDP)  
**Department:** Computer Science and Engineering  
**University:** Daffodil International University, Dhaka, Bangladesh

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 📚 References

1. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*.
3. IBM HR Analytics Dataset — Kaggle. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.

---

<p align="center">
  Made with ❤️ for CSE Final Year Project · Daffodil International University
</p>
