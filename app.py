"""Streamlit app: interactive bank customer churn risk scorer."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from config import METRICS_PATH, MODEL_PATH, FIGURES_DIR  # noqa: E402
from data_prep import add_features, encode  # noqa: E402

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


bundle = load_model()
metrics = load_metrics()

st.title("🏦 Bank Customer Churn Predictor")
st.caption("Random Forest model trained on 10,000 retail banking customers.")

if bundle is None:
    st.error("No trained model found. Run `python src/train_model.py` first.")
    st.stop()

model, columns = bundle["model"], bundle["columns"]

if metrics:
    rf = next(m for m in metrics["comparison"] if m["model"] == "Random Forest")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{rf['accuracy']:.1%}")
    c2.metric("ROC-AUC", f"{rf['roc_auc']:.3f}")
    c3.metric("Recall (churners)", f"{rf['recall']:.1%}")
    c4.metric("Records trained on", f"{metrics['n_records']:,}")

tab_predict, tab_insights = st.tabs(["Predict", "Model insights"])

with tab_predict:
    st.subheader("Customer profile")
    left, mid, right = st.columns(3)

    with left:
        credit_score = st.slider("Credit score", 350, 850, 650)
        age = st.slider("Age", 18, 92, 38)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
    with mid:
        balance = st.number_input("Account balance (€)", 0.0, 260000.0, 76000.0, step=1000.0)
        salary = st.number_input("Estimated salary (€)", 0.0, 200000.0, 100000.0, step=1000.0)
        products = st.selectbox("Number of products", [1, 2, 3, 4], index=0)
    with right:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        has_card = st.radio("Has credit card", ["Yes", "No"], horizontal=True)
        active = st.radio("Active member", ["Yes", "No"], horizontal=True)

    if st.button("Predict churn risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": 1 if has_card == "Yes" else 0,
            "IsActiveMember": 1 if active == "Yes" else 0,
            "EstimatedSalary": salary,
        }])
        features = encode(add_features(row)).reindex(columns=columns, fill_value=0)
        prob = float(model.predict_proba(features)[0, 1])

        st.progress(prob)
        if prob >= 0.6:
            st.error(f"### High risk — {prob:.1%} probability of churn")
            st.write("Recommended action: priority retention outreach, tailored offer.")
        elif prob >= 0.3:
            st.warning(f"### Medium risk — {prob:.1%} probability of churn")
            st.write("Recommended action: add to nurture campaign and monitor activity.")
        else:
            st.success(f"### Low risk — {prob:.1%} probability of churn")
            st.write("Recommended action: standard servicing, cross-sell opportunity.")

with tab_insights:
    st.subheader("What drives churn?")
    fig = FIGURES_DIR / "08_feature_importance.png"
    if fig.exists():
        st.image(str(fig))
    if metrics:
        st.subheader("Model comparison")
        st.dataframe(
            pd.DataFrame(metrics["comparison"]).set_index("model").style.format("{:.3f}"),
            use_container_width=True,
        )
    for name, caption in [("02_churn_by_category", "Churn rate by customer segment"),
                          ("03_numeric_distributions", "Numeric feature distributions"),
                          ("07_roc_curve", "ROC curve")]:
        p = FIGURES_DIR / f"{name}.png"
        if p.exists():
            st.image(str(p), caption=caption)