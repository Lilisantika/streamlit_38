import streamlit as st

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "About Me", "EDA", "Model Evaluation", "Prediction"]
)

if page == "Overview":
    exec(open("overview.py").read())

elif page == "About Me":
    exec(open("about_me.py").read())

elif page == "EDA":
    exec(open("EDA.py").read())

elif page == "Model Evaluation":
    exec(open("model_evaluation.py").read())

elif page == "Prediction":
    exec(open("prediksi.py").read())