import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv("churn_clean.csv")

st.title("📈 Exploratory Data Analysis - Churn")

# ===== BASIC COMPONENT 1 =====
show_raw = st.checkbox("Tampilkan raw data")

if show_raw:
    st.dataframe(df)

# ===== BASIC COMPONENT 2 =====
tenure_range = st.slider(
    "Filter Tenure (bulan):",
    int(df.Tenure.min()),
    int(df.Tenure.max()),
    (0, 72)
)

filtered = df[(df.Tenure >= tenure_range[0]) & (df.Tenure <= tenure_range[1])]

# ===== BASIC COMPONENT 3 =====
churn_filter = st.selectbox(
    "Filter Churn:",
    ["All", 0, 1]
)

if churn_filter != "All":
    filtered = filtered[filtered['Churn'] == churn_filter]

st.write(f"Jumlah data setelah filter: {len(filtered)}")

# =============================
# 📊 INTERACTIVE VISUALIZATION
# =============================

st.subheader("Churn Distribution")

fig1 = px.pie(
    filtered,
    names="Churn",
    title="Churn Proportion",
    hole=0.4
)
st.plotly_chart(fig1, use_container_width=True)

# =============================
st.subheader("Tenure vs Monthly Charges")

fig2 = px.scatter(
    filtered,
    x="Tenure",
    y="MonthlyCharges",
    color="Churn",
    title="Tenure vs Monthly Charges"
)

st.plotly_chart(fig2, use_container_width=True)

# =============================
st.subheader("Correlation Heatmap")

numeric_df = filtered.select_dtypes(include=['int','float'])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)