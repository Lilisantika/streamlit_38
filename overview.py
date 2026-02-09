import streamlit as st
import pandas as pd

st.title("📊 Customer Churn Analysis Project")

st.markdown("""
Project ini menganalisis Customer Churn menggunakan dataset telecom.

### Tujuan:
- Memahami faktor penyebab churn
- Visualisasi interaktif
- Dasar predictive modeling
""")

df = pd.read_csv("churn_clean.csv")
df["Churn_num"] = df["Churn"].map({"Yes":1,"No":0})

st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

st.metric("Total Customer", len(df))
st.metric("Churn Rate (%)", round(df["Churn_num"].mean()*100,2))