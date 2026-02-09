import streamlit as st
import overview
import about_me
import EDA
import model_evaluation
import prediksi

st.set_page_config(page_title="Churn Portfolio", layout="wide")

st.sidebar.title("📂 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overview","About Me","EDA","Model Evaluation","Prediction"]
)

if page == "Overview":
    overview.app()

elif page == "About Me":
    about_me.app()

elif page == "EDA":
    EDA.app()

elif page == "Model Evaluation":
    model_evaluation.app()

elif page == "Prediction":
    prediksi.app()