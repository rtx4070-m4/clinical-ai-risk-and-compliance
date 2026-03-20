
import streamlit as st
from deployment.utils import load_bundle, transform, predict

st.set_page_config(page_title="CKD Risk with Liability Warning", layout="centered")
st.title("CKD Risk Prediction (Liability-Aware)")

bundle = load_bundle()

with st.form("input"):
    age = st.number_input("Age", 18, 100, 50)
    creatinine = st.number_input("Creatinine", 0.2, 10.0, 1.2)
    bp = st.number_input("Blood Pressure", 60, 220, 120)
    hemoglobin = st.number_input("Hemoglobin", 4.0, 20.0, 12.0)
    rbc = st.selectbox("RBC", ["normal","abnormal"])
    hypertension = st.selectbox("Hypertension", ["no","yes"])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "age": age, "creatinine": creatinine, "bp": bp,
        "hemoglobin": hemoglobin, "rbc": rbc, "hypertension": hypertension
    }
    X = transform(bundle, row)
    proba, pred = predict(bundle, X)

    if pred == 1:
        st.error(f"High CKD Risk (p={proba:.2f})")
    else:
        st.warning(f"Low risk (p={proba:.2f}) — **Verify clinically (False Negative Risk!)**")

    if 0.4 < proba < 0.6:
        st.info("Low confidence region → heightened liability if relied upon without clinical validation.")
