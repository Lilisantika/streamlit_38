import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("churn_clean.csv")

# target
y = df["Churn"].map({"Yes":1, "No":0})
X = df.drop("Churn", axis=1)

# column types
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# ======================
# PREPROCESSING
# ======================
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ======================
# MODEL PIPELINE
# ======================
model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])

# ======================
# TRAIN
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model.fit(X_train, y_train)

# ======================
# SAVE
# ======================
joblib.dump(model, "imdb_joblib.pkl")

print("✅ Model churn berhasil disimpan!")