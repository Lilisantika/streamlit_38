import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix
)

# =======================
# PAGE
# =======================
st.set_page_config("Churn Model Evaluation", layout="wide")
st.title("📊 Customer Churn – Model Evaluation Dashboard")

# =======================
# LOAD DATA
# =======================
df = pd.read_csv("churn_clean.csv")

df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(include=["int64","float64"]).columns

# =======================
# PREPROCESSING
# =======================
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# =======================
# SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# =======================
# MODELS
# =======================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(max_depth=15, n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

# =======================
# METRIC FUNCTION
# =======================
def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true,y_pred)*100,2),
        "Recall": round(recall_score(y_true,y_pred),3),
        "Precision": round(precision_score(y_true,y_pred),3),
        "F1": round(f1_score(y_true,y_pred),3),
        "AUC": round(roc_auc_score(y_true,y_prob),3)
    }

results = {}
conf_matrices = {}

# =======================
# TRAIN + EVALUATE
# =======================
for name, model in models.items():

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:,1]

    results[name] = evaluate(y_test, pred, prob)
    conf_matrices[name] = confusion_matrix(y_test, pred)

# =======================
# METRIC TABLE
# =======================
st.subheader("📈 Model Performance Comparison")

metric_df = pd.DataFrame(results).T
st.dataframe(metric_df)

# =======================
# CONFUSION MATRIX
# =======================
st.subheader("📊 Confusion Matrix")

cols = st.columns(3)

for col, (name, cm) in zip(cols, conf_matrices.items()):
    with col:
        st.markdown(f"### {name}")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# =======================
# BEST MODEL (BY RECALL)
# =======================
best_model_name = metric_df["Recall"].idxmax()
best_cm = conf_matrices[best_model_name]

st.success(f"🏆 Best model for churn detection (highest Recall): {best_model_name}")

# =======================
# BUSINESS IMPACT
# =======================
st.subheader("💰 Potential Business Impact")

retention_cost = st.number_input(
    "Retention cost per customer (Rp)",
    value=500000
)

actual_churn = y_test.sum()
detected_churn = best_cm[1,1]

saving = detected_churn * retention_cost

col1, col2, col3 = st.columns(3)
col1.metric("Total Actual Churn", int(actual_churn))
col2.metric("Churn Detected by Model", int(detected_churn))
col3.metric("Potential Cost Efficiency (Rp)", f"{saving:,.0f}")

# =======================
# CONCLUSION
# =======================
st.markdown("### 📌 Conclusion")

st.write(f"""
Among all evaluated models, *{best_model_name}* achieved the highest recall, 
meaning it is the most effective in identifying customers at risk of churn.

Higher recall allows businesses to:
- Detect more churn-prone customers early  
- Deploy targeted retention campaigns  
- Reduce revenue loss efficiently  

With the current assumptions, the model could potentially save *Rp {saving:,.0f}*.
""")

# ==============================
# SAVE BEST MODEL & PREPROCESS
# ==============================