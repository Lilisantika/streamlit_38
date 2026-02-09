import streamlit as st
import pandas as pd
import joblib

# =====================
# LOAD MODEL
# =====================
model = joblib.load("imdb_joblib.pkl")

st.set_page_config("Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction")

st.write("Masukkan data customer untuk prediksi risiko churn")

# =====================
# INPUT FORM
# =====================
df_sample = pd.read_csv("churn_clean.csv").drop("Churn", axis=1)

inputs = {}

for col in df_sample.columns:
    if df_sample[col].dtype == "object":
        inputs[col] = st.selectbox(col, df_sample[col].unique())
    else:
        inputs[col] = st.number_input(col, float(df_sample[col].min()), float(df_sample[col].max()))

input_df = pd.DataFrame([inputs])

# =====================
# PREDICT
# =====================
st.divider()

if st.button("🚀 Predict Churn Risk"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prob*100:.2f}%")

    with col2:
        if pred == 1:
            st.error("⚠ High Risk of Churn")
        else:
            st.success("✅ Low Risk of Churn")

    st.progress(int(prob*100))