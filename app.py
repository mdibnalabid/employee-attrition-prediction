"""
=============================================================================
Employee Attrition Prediction — Streamlit Web Application
=============================================================================
Project : Employee Attrition Prediction using Machine Learning Techniques
Dataset : IBM HR Analytics Employee Attrition & Performance (Kaggle)
Author  : Raiyan | Daffodil International University
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title  = "Employee Attrition Predictor",
    page_icon   = "👥",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS & FEATURE SCHEMA
# ═════════════════════════════════════════════════════════════════════════════

# All columns produced by pd.get_dummies on the IBM dataset
# (drop_first=True, same order as training pipeline)
# This list was captured from X_encoded.columns after preprocessing.
ALL_FEATURE_COLUMNS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    # One-hot encoded dummies (drop_first=True → reference level omitted)
    "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely",
    "Department_Research & Development", "Department_Sales",
    "EducationField_Life Sciences", "EducationField_Marketing",
    "EducationField_Medical", "EducationField_Other",
    "EducationField_Technical Degree",
    "Gender_Male",
    "JobRole_Human Resources", "JobRole_Laboratory Technician",
    "JobRole_Manager", "JobRole_Manufacturing Director",
    "JobRole_Research Director", "JobRole_Research Scientist",
    "JobRole_Sales Executive", "JobRole_Sales Representative",
    "MaritalStatus_Married", "MaritalStatus_Single",
    "OverTime_Yes",
]

# Numerical columns that were scaled in training
NUMERICAL_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# IBM dataset training statistics for StandardScaler reconstruction
# (mean and std computed from the full training set)
SCALER_STATS = {
    "Age":                      {"mean": 36.92,  "std": 9.14},
    "DailyRate":                {"mean": 802.49, "std": 403.51},
    "DistanceFromHome":         {"mean": 9.19,   "std": 8.11},
    "Education":                {"mean": 2.91,   "std": 1.02},
    "EnvironmentSatisfaction":  {"mean": 2.72,   "std": 1.09},
    "HourlyRate":               {"mean": 65.89,  "std": 20.33},
    "JobInvolvement":           {"mean": 2.73,   "std": 0.71},
    "JobLevel":                 {"mean": 2.06,   "std": 1.11},
    "JobSatisfaction":          {"mean": 2.73,   "std": 1.10},
    "MonthlyIncome":            {"mean": 6502.93,"std": 4707.96},
    "MonthlyRate":              {"mean": 14313.1,"std": 7117.79},
    "NumCompaniesWorked":       {"mean": 2.69,   "std": 2.50},
    "PercentSalaryHike":        {"mean": 15.21,  "std": 3.66},
    "PerformanceRating":        {"mean": 3.15,   "std": 0.36},
    "RelationshipSatisfaction": {"mean": 2.71,   "std": 1.08},
    "StockOptionLevel":         {"mean": 0.79,   "std": 0.85},
    "TotalWorkingYears":        {"mean": 11.28,  "std": 7.78},
    "TrainingTimesLastYear":    {"mean": 2.80,   "std": 1.29},
    "WorkLifeBalance":          {"mean": 2.76,   "std": 0.71},
    "YearsAtCompany":           {"mean": 7.01,   "std": 6.13},
    "YearsInCurrentRole":       {"mean": 4.23,   "std": 3.62},
    "YearsSinceLastPromotion":  {"mean": 2.19,   "std": 3.22},
    "YearsWithCurrManager":     {"mean": 4.12,   "std": 3.57},
}

# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING / TRAINING FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model …")
def load_or_train_model():
    """
    Load the saved model from disk (best_attrition_model.pkl).
    If not found, train a fresh XGBoost model on the IBM dataset.
    Returns (model, scaler, feature_columns).
    """
    model_path  = "best_attrition_model.pkl"
    scaler_path = "scaler.pkl"

    # ── Build the StandardScaler from known IBM training statistics ───────
    scaler = StandardScaler()
    scaler.mean_ = np.array([SCALER_STATS[c]["mean"] for c in NUMERICAL_COLS])
    scaler.scale_ = np.array([SCALER_STATS[c]["std"]  for c in NUMERICAL_COLS])
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(NUMERICAL_COLS)

    # ── Try loading saved model ───────────────────────────────────────────
    if os.path.exists(model_path):
        try:
            saved = joblib.load(model_path)
            # Unwrap GridSearchCV / RandomizedSearchCV
            model = saved.best_estimator_ if hasattr(saved, "best_estimator_") else saved
            return model, scaler, ALL_FEATURE_COLUMNS
        except Exception:
            pass   # fall through to training

    # ── Fallback: train a lightweight XGBoost on the dataset if present ───
    csv_path = "WA_Fn-ParamNB_HR-Employee-Attrition.csv"
    if os.path.exists(csv_path):
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import SMOTE

        df = pd.read_csv(csv_path)
        df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"],
                errors="ignore", inplace=True)
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

        X = df.drop(columns=["Attrition"])
        y = df["Attrition"]

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # Align columns to expected schema
        for col in ALL_FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = 0
        X = X[ALL_FEATURE_COLUMNS]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train[NUMERICAL_COLS] = scaler.transform(X_train[NUMERICAL_COLS])
        X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        model.fit(X_sm, y_sm)
        return model, scaler, ALL_FEATURE_COLUMNS

    # ── Ultimate fallback: return a dummy model for UI demo purposes ──────
    st.warning(
        "⚠️  No saved model or dataset found. "
        "Please upload `best_attrition_model.pkl` and `scaler.pkl` "
        "to the app directory. Showing demo mode.",
        icon="⚠️"
    )
    return None, scaler, ALL_FEATURE_COLUMNS


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPER
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_row(inputs: dict) -> pd.DataFrame:
    """
    Convert the UI inputs dict into a single-row DataFrame that matches
    the exact column schema the model was trained on (ALL_FEATURE_COLUMNS).
    """
    row = {col: 0 for col in ALL_FEATURE_COLUMNS}

    # ── Numerical features (pass through; scaler applied separately) ──────
    for num_col in NUMERICAL_COLS:
        if num_col in inputs:
            row[num_col] = inputs[num_col]

    # ── One-hot: OverTime ─────────────────────────────────────────────────
    if inputs.get("OverTime") == "Yes":
        row["OverTime_Yes"] = 1

    # ── One-hot: BusinessTravel ───────────────────────────────────────────
    travel = inputs.get("BusinessTravel", "Non-Travel")
    if travel == "Travel_Frequently":
        row["BusinessTravel_Travel_Frequently"] = 1
    elif travel == "Travel_Rarely":
        row["BusinessTravel_Travel_Rarely"] = 1
    # "Non-Travel" is the reference (drop_first) → all zeros

    # ── One-hot: Department ───────────────────────────────────────────────
    dept = inputs.get("Department", "Human Resources")
    if dept == "Research & Development":
        row["Department_Research & Development"] = 1
    elif dept == "Sales":
        row["Department_Sales"] = 1

    # ── One-hot: Gender ───────────────────────────────────────────────────
    if inputs.get("Gender") == "Male":
        row["Gender_Male"] = 1

    # ── One-hot: MaritalStatus ────────────────────────────────────────────
    marital = inputs.get("MaritalStatus", "Divorced")
    if marital == "Married":
        row["MaritalStatus_Married"] = 1
    elif marital == "Single":
        row["MaritalStatus_Single"] = 1

    # ── One-hot: JobRole ──────────────────────────────────────────────────
    role = inputs.get("JobRole", "Healthcare Representative")
    role_map = {
        "Human Resources":        "JobRole_Human Resources",
        "Laboratory Technician":  "JobRole_Laboratory Technician",
        "Manager":                "JobRole_Manager",
        "Manufacturing Director": "JobRole_Manufacturing Director",
        "Research Director":      "JobRole_Research Director",
        "Research Scientist":     "JobRole_Research Scientist",
        "Sales Executive":        "JobRole_Sales Executive",
        "Sales Representative":   "JobRole_Sales Representative",
    }
    if role in role_map:
        row[role_map[role]] = 1
    # "Healthcare Representative" is reference → all zeros

    # ── One-hot: EducationField ───────────────────────────────────────────
    edu_field = inputs.get("EducationField", "Human Resources")
    edu_map = {
        "Life Sciences":    "EducationField_Life Sciences",
        "Marketing":        "EducationField_Marketing",
        "Medical":          "EducationField_Medical",
        "Other":            "EducationField_Other",
        "Technical Degree": "EducationField_Technical Degree",
    }
    if edu_field in edu_map:
        row[edu_map[edu_field]] = 1

    df_row = pd.DataFrame([row], columns=ALL_FEATURE_COLUMNS)
    return df_row


def predict_attrition(model, scaler, inputs: dict):
    """
    Run the full prediction pipeline and return (probability, label).
    """
    if model is None:
        return 0.5, "Unknown (no model loaded)"

    df_row = build_feature_row(inputs)

    # Scale numerical features using the training scaler
    df_row[NUMERICAL_COLS] = scaler.transform(df_row[NUMERICAL_COLS])

    prob  = float(model.predict_proba(df_row)[0, 1])
    label = "High Risk — Likely to Leave" if prob >= 0.5 else "Low Risk — Likely to Stay"
    return prob, label


# ═════════════════════════════════════════════════════════════════════════════
# SHAP EXPLANATION (optional, skips gracefully if shap unavailable)
# ═════════════════════════════════════════════════════════════════════════════

def get_shap_values(model, df_row_scaled: pd.DataFrame):
    """Compute SHAP values for a single row. Returns (shap_vals, base_val) or None."""
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        sv         = explainer.shap_values(df_row_scaled)
        sv         = sv[1] if isinstance(sv, list) else sv
        base_val   = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
        return sv[0], base_val
    except Exception:
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── General ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Risk badge ── */
.risk-high {
    background: linear-gradient(135deg, #ff4b4b, #c0392b);
    color: white; border-radius: 12px; padding: 18px 28px;
    text-align: center; font-size: 1.5rem; font-weight: 700;
    box-shadow: 0 4px 14px rgba(255,75,75,0.35);
    margin-bottom: 12px;
}
.risk-low {
    background: linear-gradient(135deg, #00c851, #007e33);
    color: white; border-radius: 12px; padding: 18px 28px;
    text-align: center; font-size: 1.5rem; font-weight: 700;
    box-shadow: 0 4px 14px rgba(0,200,81,0.35);
    margin-bottom: 12px;
}

/* ── Metric cards ── */
.metric-card {
    background: #f8f9fa; border-radius: 10px;
    padding: 14px 18px; border-left: 4px solid #4C72B0;
    margin-bottom: 10px;
}

/* ── Recommendation badges ── */
.rec-high   { background:#fff3cd; border-left:4px solid #ffc107; padding:12px 16px; border-radius:8px; margin:6px 0; }
.rec-medium { background:#d1ecf1; border-left:4px solid #17a2b8; padding:12px 16px; border-radius:8px; margin:6px 0; }
.rec-low    { background:#d4edda; border-left:4px solid #28a745; padding:12px 16px; border-radius:8px; margin:6px 0; }

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #2c3e50;
    border-bottom: 2px solid #4C72B0; padding-bottom: 4px;
    margin: 20px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════

model, scaler, feature_cols = load_or_train_model()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — EMPLOYEE FEATURE INPUTS
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/000000/conference-call.png",
        width=64,
    )
    st.title("👥 Employee Profile")
    st.caption("Fill in the employee details below and click **Predict**.")
    st.divider()

    # ── Demographics ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧑 Demographics</div>',
                unsafe_allow_html=True)

    age = st.slider("Age", 18, 60, 32, help="Employee's current age")
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced"]
    )
    education = st.selectbox(
        "Education Level",
        [1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: {
            1: "1 — Below College", 2: "2 — College",
            3: "3 — Bachelor", 4: "4 — Master", 5: "5 — Doctor"
        }[x],
    )
    education_field = st.selectbox(
        "Education Field",
        ["Human Resources", "Life Sciences", "Marketing",
         "Medical", "Other", "Technical Degree"],
    )

    # ── Job Information ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">💼 Job Information</div>',
                unsafe_allow_html=True)

    department = st.selectbox(
        "Department",
        ["Human Resources", "Research & Development", "Sales"],
    )
    job_role = st.selectbox(
        "Job Role",
        ["Healthcare Representative", "Human Resources",
         "Laboratory Technician", "Manager",
         "Manufacturing Director", "Research Director",
         "Research Scientist", "Sales Executive", "Sales Representative"],
    )
    job_level = st.selectbox(
        "Job Level", [1, 2, 3, 4, 5],
        format_func=lambda x: f"Level {x}",
    )
    business_travel = st.selectbox(
        "Business Travel",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        format_func=lambda x: x.replace("_", " "),
    )
    over_time = st.selectbox("OverTime", ["No", "Yes"])

    # ── Compensation ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💰 Compensation</div>',
                unsafe_allow_html=True)

    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=1000, max_value=20000,
        value=5000, step=100,
    )
    daily_rate = st.number_input(
        "Daily Rate ($)", min_value=100, max_value=1500, value=800, step=10,
    )
    hourly_rate = st.number_input(
        "Hourly Rate ($)", min_value=30, max_value=100, value=66, step=1,
    )
    monthly_rate = st.number_input(
        "Monthly Rate ($)", min_value=2000, max_value=27000,
        value=14000, step=100,
    )
    percent_salary_hike = st.slider(
        "Percent Salary Hike (%)", 10, 25, 15,
    )
    stock_option_level = st.selectbox(
        "Stock Option Level", [0, 1, 2, 3],
        format_func=lambda x: f"Level {x}",
    )

    # ── Experience & Tenure ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 Experience & Tenure</div>',
                unsafe_allow_html=True)

    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    years_at_company    = st.slider("Years at Company", 0, 40, 5)
    years_in_role       = st.slider("Years in Current Role", 0, 18, 3)
    years_since_promo   = st.slider("Years Since Last Promotion", 0, 15, 2)
    years_with_mgr      = st.slider("Years With Current Manager", 0, 17, 3)
    num_companies       = st.slider("No. of Companies Worked", 0, 9, 2)
    training_times      = st.slider("Training Times Last Year", 0, 6, 3)

    # ── Satisfaction & Wellbeing ──────────────────────────────────────────
    st.markdown('<div class="section-header">😊 Satisfaction & Wellbeing</div>',
                unsafe_allow_html=True)

    job_satisfaction = st.select_slider(
        "Job Satisfaction", options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x],
    )
    env_satisfaction = st.select_slider(
        "Environment Satisfaction", options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x],
    )
    relationship_sat = st.select_slider(
        "Relationship Satisfaction", options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x],
    )
    work_life_bal = st.select_slider(
        "Work-Life Balance", options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: {1:"Bad",2:"Good",3:"Better",4:"Best"}[x],
    )
    job_involvement = st.select_slider(
        "Job Involvement", options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x],
    )
    performance_rating = st.select_slider(
        "Performance Rating", options=[3, 4],
        value=3,
        format_func=lambda x: {3:"Excellent",4:"Outstanding"}[x],
    )
    distance_home = st.slider("Distance From Home (km)", 1, 29, 9)

    st.divider()
    predict_btn = st.button("🔍 Predict Attrition Risk", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(
    "# 🏢 Employee Attrition Prediction\n"
    "### *Powered by Machine Learning — IBM HR Analytics Dataset*",
    unsafe_allow_html=False,
)

st.caption(
    "This tool predicts the probability of employee attrition using a "
    "comparative ML pipeline (Logistic Regression · Decision Tree · "
    "Random Forest · SVM · XGBoost) with SMOTE resampling and "
    "SHAP explainability."
)

st.divider()

# ── Dataset Overview Metrics ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Dataset Rows",   "1,470")
col2.metric("📊 Features",       "30")
col3.metric("⚠️ Attrition Rate", "16.1%")
col4.metric("🤖 Best Model",     "XGBoost" if model is not None else "—")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

if predict_btn:
    # Collect all inputs
    inputs = {
        "Age":                     age,
        "DailyRate":               daily_rate,
        "DistanceFromHome":        distance_home,
        "Education":               education,
        "EnvironmentSatisfaction": env_satisfaction,
        "HourlyRate":              hourly_rate,
        "JobInvolvement":          job_involvement,
        "JobLevel":                job_level,
        "JobSatisfaction":         job_satisfaction,
        "MonthlyIncome":           monthly_income,
        "MonthlyRate":             monthly_rate,
        "NumCompaniesWorked":      num_companies,
        "PercentSalaryHike":       percent_salary_hike,
        "PerformanceRating":       performance_rating,
        "RelationshipSatisfaction":relationship_sat,
        "StockOptionLevel":        stock_option_level,
        "TotalWorkingYears":       total_working_years,
        "TrainingTimesLastYear":   training_times,
        "WorkLifeBalance":         work_life_bal,
        "YearsAtCompany":          years_at_company,
        "YearsInCurrentRole":      years_in_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsWithCurrManager":    years_with_mgr,
        "OverTime":                over_time,
        "BusinessTravel":          business_travel,
        "Department":              department,
        "Gender":                  gender,
        "MaritalStatus":           marital_status,
        "JobRole":                 job_role,
        "EducationField":          education_field,
    }

    # Run prediction
    prob, label = predict_attrition(model, scaler, inputs)
    is_high_risk = prob >= 0.5

    # ── Result Badge ──────────────────────────────────────────────────────
    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if is_high_risk:
            st.markdown(
                f'<div class="risk-high">🚨 {label}<br>'
                f'<span style="font-size:1rem;opacity:0.9">'
                f'Attrition Probability: {prob*100:.1f}%</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="risk-low">✅ {label}<br>'
                f'<span style="font-size:1rem;opacity:0.9">'
                f'Attrition Probability: {prob*100:.1f}%</span></div>',
                unsafe_allow_html=True,
            )

        # Confidence interpretation
        if prob >= 0.75:
            st.error("🔴 **Very High Risk** — Immediate retention action recommended.")
        elif prob >= 0.5:
            st.warning("🟠 **High Risk** — Monitor closely and schedule a 1-on-1.")
        elif prob >= 0.25:
            st.info("🔵 **Low-Moderate Risk** — Maintain engagement efforts.")
        else:
            st.success("🟢 **Low Risk** — Employee is likely satisfied and stable.")

    # ── Gauge Chart ───────────────────────────────────────────────────────
    with gauge_col:
        gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = round(prob * 100, 1),
            delta = {"reference": 16.1, "valueformat": ".1f",
                     "suffix": "%", "prefix": "vs base "},
            gauge = {
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar":  {"color": "#e63946" if is_high_risk else "#2a9d8f"},
                "steps": [
                    {"range": [0,  25], "color": "#d4edda"},
                    {"range": [25, 50], "color": "#cce5ff"},
                    {"range": [50, 75], "color": "#fff3cd"},
                    {"range": [75, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.82,
                    "value": 50,
                },
            },
            title = {"text": "Attrition Probability", "font": {"size": 16}},
            number = {"suffix": "%", "font": {"size": 32}},
        ))
        gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

    st.divider()

    # ── Feature Contribution (Risk Factor Analysis) ───────────────────────
    st.markdown("### 📊 Risk Factor Analysis")

    # Rule-based risk factor breakdown (interpretable without SHAP)
    risk_factors = []

    if over_time == "Yes":
        risk_factors.append(("OverTime = Yes",
                             "Overtime is the strongest attrition predictor. "
                             "Employees working OT are ~3× more likely to leave.",
                             "high"))
    if monthly_income < 3000:
        risk_factors.append(("Low Monthly Income",
                             f"Monthly income ${monthly_income:,} is below the "
                             "$3,000 risk threshold identified in SHAP analysis.",
                             "high"))
    if age < 30:
        risk_factors.append(("Young Employee (Age < 30)",
                             "Employees in their 20s are at highest attrition risk — "
                             "the career exploration phase.",
                             "medium"))
    if years_at_company <= 2:
        risk_factors.append(("Short Tenure (≤ 2 years)",
                             "The 0–3 year window is the critical 'new-hire churn' "
                             "period. Onboarding investment has the highest ROI here.",
                             "high"))
    if business_travel == "Travel_Frequently":
        risk_factors.append(("Frequent Business Travel",
                             "Frequent travellers show ~3× higher attrition vs "
                             "non-travellers. Consider travel workload redistribution.",
                             "medium"))
    if marital_status == "Single":
        risk_factors.append(("Single Marital Status",
                             "Single employees have fewer personal anchors and show "
                             "the highest attrition rate among marital groups.",
                             "medium"))
    if job_role in ["Sales Representative", "Laboratory Technician"]:
        risk_factors.append((f"High-Risk Job Role ({job_role})",
                             f"{job_role}s show above-average attrition rates "
                             "per SHAP analysis. Role-specific retention is advisable.",
                             "medium"))
    if distance_home > 20:
        risk_factors.append(("Long Commute (> 20 km)",
                             f"Distance of {distance_home} km is above the critical "
                             "threshold. Remote work options may reduce attrition risk.",
                             "medium"))
    if stock_option_level == 0:
        risk_factors.append(("No Stock Options",
                             "Employees without stock options show higher attrition. "
                             "Even Level-1 stock options significantly improve retention.",
                             "medium"))
    if job_satisfaction <= 2:
        risk_factors.append(("Low Job Satisfaction",
                             f"Job satisfaction score of {job_satisfaction}/4 is a "
                             "strong pull factor toward leaving. Address workload or "
                             "role alignment.",
                             "high"))
    if work_life_bal <= 2:
        risk_factors.append(("Poor Work-Life Balance",
                             f"Work-life balance score of {work_life_bal}/4. "
                             "Consider flexible scheduling or overtime reduction.",
                             "high"))
    if total_working_years < 5:
        risk_factors.append(("Low Total Experience (< 5 years)",
                             "Less experienced employees tend to still be in career "
                             "exploration mode. Mentoring and clear growth paths help.",
                             "medium"))
    if not risk_factors:
        risk_factors.append(("No Critical Risk Factors Detected",
                             "This employee's profile does not trigger any of the "
                             "key attrition risk signals. Continue standard engagement.",
                             "low"))

    # Display risk factors
    badge_map = {
        "high":   ("🔴 High",   "rec-high"),
        "medium": ("🟡 Medium", "rec-medium"),
        "low":    ("🟢 Low",    "rec-low"),
    }

    for factor, explanation, severity in risk_factors:
        badge_label, badge_class = badge_map[severity]
        st.markdown(
            f'<div class="{badge_class}">'
            f'<strong>{badge_label} — {factor}</strong><br>'
            f'<span style="font-size:0.9rem">{explanation}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── HR Recommendations ────────────────────────────────────────────────
    st.markdown("### 💡 HR Retention Recommendations")

    reco_col1, reco_col2 = st.columns(2)

    with reco_col1:
        st.markdown("**Immediate Actions**")

        if over_time == "Yes":
            st.markdown("- 🕐 Review and reduce overtime schedule")
        if monthly_income < 4000:
            st.markdown("- 💲 Initiate a compensation benchmarking review")
        if job_satisfaction <= 2:
            st.markdown("- 📋 Schedule a role alignment / feedback session")
        if work_life_bal <= 2:
            st.markdown("- 🏡 Offer flexible work or remote options")
        if years_since_promo > 4:
            st.markdown(f"- 🏅 Consider promotion or lateral growth opportunity "
                        f"({years_since_promo} years since last promotion)")

    with reco_col2:
        st.markdown("**Long-term Strategies**")

        if stock_option_level == 0:
            st.markdown("- 📈 Offer Level-1 stock options to improve retention anchor")
        if years_at_company <= 2:
            st.markdown("- 🤝 Assign a senior mentor during the critical 0–3 year window")
        if business_travel == "Travel_Frequently":
            st.markdown("- ✈️ Evaluate travel workload — consider redistribution")
        if distance_home > 15:
            st.markdown(f"- 🚗 Explore commute subsidy or remote-work policy "
                        f"({distance_home} km commute)")
        if num_companies >= 5:
            st.markdown("- 📌 Implement a retention bonus tied to tenure milestones")

        st.markdown("- 📊 Schedule quarterly engagement pulse surveys")
        st.markdown("- 🧠 Enrol in leadership development if high-performing")

    st.divider()

    # ── SHAP Waterfall (if model + shap available) ────────────────────────
    if model is not None:
        with st.expander("🔍 View SHAP Explanation (Model Interpretability)", expanded=False):
            st.caption(
                "SHAP (SHapley Additive exPlanations) shows which features "
                "contributed most to **this specific prediction**."
            )
            df_row       = build_feature_row(inputs)
            df_row_scaled = df_row.copy()
            df_row_scaled[NUMERICAL_COLS] = scaler.transform(df_row_scaled[NUMERICAL_COLS])

            shap_vals, base_val = get_shap_values(model, df_row_scaled)

            if shap_vals is not None:
                shap_series = pd.Series(shap_vals, index=ALL_FEATURE_COLUMNS)
                top_shap    = shap_series.sort_values(key=abs, ascending=False).head(15)

                # SHAP bar chart via Plotly
                colors_shap = [
                    "#e63946" if v > 0 else "#457b9d"
                    for v in top_shap.values
                ]
                fig_shap = go.Figure(go.Bar(
                    x         = top_shap.values[::-1],
                    y         = top_shap.index[::-1],
                    orientation = "h",
                    marker_color = colors_shap[::-1],
                    text      = [f"{v:+.4f}" for v in top_shap.values[::-1]],
                    textposition = "outside",
                ))
                fig_shap.update_layout(
                    title  = "SHAP Values — Top 15 Features for This Employee",
                    xaxis_title = "SHAP Value (+ = toward Leave, − = toward Stay)",
                    height = 480,
                    margin = dict(l=180, r=80, t=60, b=40),
                    plot_bgcolor = "white",
                )
                fig_shap.add_vline(x=0, line_color="black", line_width=1)
                st.plotly_chart(fig_shap, use_container_width=True)

                st.caption(
                    f"Base value (dataset average prediction): {base_val:.4f} | "
                    f"Final output: {base_val + shap_vals.sum():.4f}"
                )
            else:
                st.info("SHAP library not available in this environment. "
                        "Install with: `pip install shap`")

# ── Default landing state ─────────────────────────────────────────────────
else:
    # Show an illustrative chart when no prediction has been run yet
    st.info(
        "👈 **Fill in the employee profile** in the sidebar and click "
        "**Predict Attrition Risk** to see the prediction, risk analysis, "
        "and HR recommendations.",
        icon="ℹ️",
    )

    # Feature importance reference chart (static — from IBM dataset analysis)
    fi_data = {
        "Feature": [
            "OverTime_Yes", "MonthlyIncome", "Age", "TotalWorkingYears",
            "YearsAtCompany", "JobRole_Sales Rep.", "MaritalStatus_Single",
            "DistanceFromHome", "StockOptionLevel", "JobSatisfaction",
            "YearsWithCurrManager", "WorkLifeBalance", "NumCompaniesWorked",
            "BusinessTravel_Freq.", "EnvironmentSatisfaction",
        ],
        "Importance": [
            0.142, 0.118, 0.097, 0.089, 0.076, 0.061, 0.054,
            0.048, 0.043, 0.041, 0.037, 0.033, 0.031, 0.028, 0.024,
        ],
    }
    fi_df = pd.DataFrame(fi_data).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature",
        orientation="h",
        title="📊 Global Feature Importance — Top 15 Attrition Drivers<br>"
              "<sup>(XGBoost SHAP — IBM HR Analytics Dataset)</sup>",
        color="Importance",
        color_continuous_scale="RdYlGn_r",
        labels={"Importance": "Mean |SHAP Value|"},
    )
    fig_fi.update_layout(
        height=480, coloraxis_showscale=False,
        plot_bgcolor="white",
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=180, r=40, t=80, b=40),
    )
    fig_fi.update_traces(texttemplate="%{x:.3f}", textposition="outside")
    st.plotly_chart(fig_fi, use_container_width=True)

    # IBM HR stats
    st.divider()
    st.markdown("### 📈 IBM HR Dataset — Attrition by Key Segments")

    seg_col1, seg_col2, seg_col3 = st.columns(3)

    with seg_col1:
        ot_data = pd.DataFrame({
            "OverTime": ["Yes", "No"],
            "Attrition Rate (%)": [30.5, 10.4],
        })
        fig_ot = px.bar(ot_data, x="OverTime", y="Attrition Rate (%)",
                        color="OverTime",
                        color_discrete_map={"Yes": "#e63946", "No": "#457b9d"},
                        title="By OverTime Status", text_auto=True)
        fig_ot.update_layout(showlegend=False, height=280,
                              plot_bgcolor="white", margin=dict(t=40,b=20))
        st.plotly_chart(fig_ot, use_container_width=True)

    with seg_col2:
        ms_data = pd.DataFrame({
            "Marital Status": ["Single", "Divorced", "Married"],
            "Attrition Rate (%)": [25.5, 13.0, 12.5],
        })
        fig_ms = px.bar(ms_data, x="Marital Status", y="Attrition Rate (%)",
                        color="Marital Status",
                        color_discrete_sequence=["#e63946","#f4a261","#457b9d"],
                        title="By Marital Status", text_auto=True)
        fig_ms.update_layout(showlegend=False, height=280,
                              plot_bgcolor="white", margin=dict(t=40,b=20))
        st.plotly_chart(fig_ms, use_container_width=True)

    with seg_col3:
        bt_data = pd.DataFrame({
            "Travel": ["Frequent", "Rarely", "Non-Travel"],
            "Attrition Rate (%)": [24.9, 15.0, 8.0],
        })
        fig_bt = px.bar(bt_data, x="Travel", y="Attrition Rate (%)",
                        color="Travel",
                        color_discrete_sequence=["#e63946","#f4a261","#457b9d"],
                        title="By Business Travel", text_auto=True)
        fig_bt.update_layout(showlegend=False, height=280,
                              plot_bgcolor="white", margin=dict(t=40,b=20))
        st.plotly_chart(fig_bt, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "🎓 <strong>Employee Attrition Prediction</strong> · "
    "CSE Final Year Project · Daffodil International University · "
    "Dataset: <a href='https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset' "
    "target='_blank'>IBM HR Analytics (Kaggle)</a>"
    "</div>",
    unsafe_allow_html=True,
)
