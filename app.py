import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Loan Approval System", layout="wide")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "rf":          None,
        "debiased":    None,
        "db_scaler":   None,
        "db_features": None,
        "X_train":     None
    }

    try:
        models["rf"] = joblib.load("random_forest.pkl")
    except Exception as e:
        st.error(f"Random Forest model not found: {e}")

    try:
        models["debiased"] = joblib.load("debiased_model.pkl")
    except Exception as e:
        st.warning(f"Debiased model not found: {e}")

    try:
        models["db_scaler"] = joblib.load("debiased_scaler.pkl")
    except Exception as e:
        st.warning(f"Debiased scaler not found: {e}")

    try:
        with open("Models/debiased_meta.json") as f:
            db_meta = json.load(f)
        models["db_features"] = db_meta["debiased_features"]
    except Exception as e:
        st.warning(f"debiased_meta.json not found — fairness check disabled: {e}")

    try:
        models["X_train"] = pd.read_csv("X_train.csv")
    except Exception as e:
        st.warning(f"Training data not found for LIME background: {e}")

    return models


models         = load_models()
rf_model       = models["rf"]
debiased_model = models["debiased"]
db_scaler      = models["db_scaler"]
db_features    = models["db_features"]
X_train        = models["X_train"]

# Feature names — must match columns the Random Forest was trained on
FEATURE_NAMES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
    "Total_Income", "LoanAmount_to_Income"
]

# -----------------------------
# HEADER
# -----------------------------
st.title("🏦 Loan Approval System")
st.caption("Random Forest model with fairness verification and LIME explanation")

st.divider()

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender       = st.selectbox("Gender",     ["Male", "Female"])
    married      = st.selectbox("Married",    ["Yes", "No"])
    dependents   = st.selectbox("Dependents", [0, 1, 2, 3])

with col2:
    education     = st.selectbox("Education",    ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col3:
    applicant_income = st.number_input("Applicant Income",    min_value=0, value=5000)
    co_income        = st.number_input("Co-applicant Income", min_value=0, value=0)
    loan_amount      = st.number_input("Loan Amount",         min_value=0, value=150)
    loan_term        = st.slider("Loan Term (months)", min_value=12, max_value=480, value=360)
    credit_history   = st.selectbox("Credit History", ["Good", "Bad"])

# -----------------------------
# PREPROCESS
# -----------------------------
def prepare_input():
    total_income          = applicant_income + co_income
    loan_to_income        = loan_amount / (total_income + 1)

    return pd.DataFrame([{
        "Gender":              1 if gender == "Male" else 0,
        "Married":             1 if married == "Yes" else 0,
        "Dependents":          dependents,
        "Education":           1 if education == "Graduate" else 0,
        "Self_Employed":       1 if self_employed == "Yes" else 0,
        "ApplicantIncome":     applicant_income,
        "CoapplicantIncome":   co_income,
        "LoanAmount":          loan_amount,
        "Loan_Amount_Term":    loan_term,
        "Credit_History":      1 if credit_history == "Good" else 0,
        "Property_Area":       {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area],
        "Total_Income":        total_income,
        "LoanAmount_to_Income": loan_to_income
    }])

# -----------------------------
# LIME HELPERS
# -----------------------------
def get_lime_explanation(model, input_df, X_train_df):
    if X_train_df is not None:
        cols = [f for f in FEATURE_NAMES if f in X_train_df.columns]
        background = X_train_df[cols].values
        feature_names_used = cols
    else:
        background = np.zeros((50, len(FEATURE_NAMES)))
        feature_names_used = FEATURE_NAMES

    explainer = LimeTabularExplainer(
        training_data         = background,
        feature_names         = feature_names_used,
        class_names           = ["Rejected", "Approved"],
        mode                  = "classification",
        discretize_continuous = True,
        random_state          = 42
    )

    instance = input_df[feature_names_used].values[0]

    explanation = explainer.explain_instance(
        data_row     = instance,
        predict_fn   = lambda x: model.predict_proba(
            pd.DataFrame(x, columns=feature_names_used)
        ),
        num_features = len(feature_names_used),
        num_samples  = 1000,
        top_labels   = 2
    )

    return explanation


def get_class_idx(explanation, pred_label):
    class_idx = 1 if pred_label == "Approved" else 0
    available = list(explanation.available_labels())
    if class_idx not in available:
        class_idx = available[0]
    return class_idx


def plot_lime(explanation, pred_label):
    class_idx = get_class_idx(explanation, pred_label)
    weights   = explanation.as_list(label=class_idx)

    weights_sorted = sorted(weights, key=lambda x: abs(x[1]))
    features = [w[0] for w in weights_sorted]
    values   = [w[1] for w in weights_sorted]
    colors   = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.45)))
    ax.barh(features, values, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Feature weight (contribution to prediction)")
    ax.set_title(
        f"LIME Explanation — Prediction: {pred_label}",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Run Prediction"):

    if rf_model is None:
        st.error("Model not loaded. Check that Models/random_forest.pkl exists.")
        st.stop()

    X = prepare_input()

    prob = rf_model.predict_proba(X)[0][1]
    pred = "Approved" if prob >= 0.5 else "Rejected"

    # ── Affordability override ────────────────────────────────────────────────
    # The Random Forest was trained on a small dataset where credit history
    # dominates. These hard financial thresholds catch cases the model misses.
    total_income   = applicant_income + co_income
    loan_to_income = loan_amount / (total_income + 1)

    affordability_failed = False
    affordability_reason = ""

    if total_income == 0:
        affordability_failed = True
        affordability_reason = "zero total income"
    elif loan_to_income > 10:
        affordability_failed = True
        affordability_reason = (
            f"loan-to-income ratio of {loan_to_income:.1f}x exceeds safe threshold (10x)"
        )

    if affordability_failed and pred == "Approved":
        pred = "Rejected"
        prob = 0.10
        st.warning(
            f"⚠️ Affordability check failed: application rejected due to {affordability_reason}. "
            "A loan cannot be approved without sufficient income to service repayments."
        )

    st.divider()

    # -------------------------
    # RESULT
    # -------------------------
    st.subheader("Prediction Result")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Decision", pred)
    with colB:
        st.metric("Approval Probability", f"{prob * 100:.2f}%")
    with colC:
        confidence = prob if pred == "Approved" else (1 - prob)
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    # -------------------------
    # LIME EXPLANATION
    # -------------------------
    st.subheader("Model Explanation (LIME)")

    explanation = None

    with st.spinner("Generating LIME explanation..."):
        try:
            explanation = get_lime_explanation(rf_model, X, X_train)
            fig = plot_lime(explanation, pred)
            st.pyplot(fig)
            plt.close(fig)

            class_idx = get_class_idx(explanation, pred)
            lime_df = pd.DataFrame(
                explanation.as_list(label=class_idx),
                columns=["Feature Condition", "Weight"]
            ).sort_values("Weight", key=abs, ascending=False)

            st.markdown("**Feature weights (LIME):**")
            st.dataframe(lime_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"LIME explanation failed: {e}")

    # -------------------------
    # EXPLANATION NARRATIVE
    # -------------------------
    st.subheader("Model Explanation")

    if pred == "Approved":
        st.write(
            "Based on the information provided, the model found several factors that "
            "increased the likelihood of loan approval."
        )
    else:
        st.write(
            "The model identified a few factors that reduced the likelihood of approval. "
            "These factors suggest potential risk from a repayment perspective."
        )

    # -------------------------
    # FAIRNESS CHECK
    # -------------------------
    if debiased_model is not None and db_scaler is not None and db_features is not None:

        st.subheader("Fairness Verification")

        try:
            db_cols     = [f for f in db_features if f in X.columns]
            X_db        = X[db_cols]
            X_db_scaled = db_scaler.transform(X_db)
            db_prob     = debiased_model.predict_proba(X_db_scaled)[0][1]
            db_pred     = "Approved" if db_prob >= 0.5 else "Rejected"

            # Apply same affordability override to debiased model
            if affordability_failed and db_pred == "Approved":
                db_pred = "Rejected"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Random Forest", pred)
            with col2:
                st.metric("Debiased Model", db_pred)

            if pred == db_pred:
                st.success("Models agree → decision is consistent")
            else:
                st.warning("Models disagree → possible bias detected")

        except Exception as e:
            st.warning(f"Fairness check failed: {e}")

    elif debiased_model is not None and db_features is None:
        st.warning("Fairness check skipped — debiased_meta.json not found.")

    # -------------------------
    # KEY INSIGHT
    # -------------------------
    st.subheader("Key Insight")

    if affordability_failed:
        st.info(
            "This application was rejected because the financial capacity to repay "
            f"could not be established ({affordability_reason}). "
            "Income relative to loan size is the primary factor in this decision."
        )
    elif explanation is not None:
        try:
            class_idx = get_class_idx(explanation, pred)
            top_weights = sorted(
                explanation.as_list(label=class_idx),
                key=lambda x: abs(x[1]), reverse=True
            )
            top_feature, top_impact = top_weights[0]

            if top_impact > 0:
                st.info(
                    f"The strongest factor supporting this decision was **{top_feature}**, "
                    "which significantly improved the approval probability."
                )
            else:
                st.info(
                    f"The most influential factor affecting this decision was **{top_feature}**, "
                    "which reduced the approval likelihood."
                )
        except Exception:
            pass
    else:
        st.info("Key insight unavailable — LIME explanation did not complete.")
