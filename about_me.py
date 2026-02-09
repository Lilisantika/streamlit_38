import streamlit as st

st.title("👩‍💻 About Me")

st.write("""
Saya Lilisantika, seorang data enthusiast yang sedang membangun portfolio 
di bidang Data Analyst & Data Science.
""")

st.divider()

st.header("🛠 Skills")

skills = {
    "Python": 80,
    "SQL": 70,
    "Power BI": 85,
    "Streamlit": 70
}

for skill, level in skills.items():
    st.write(skill)
    st.progress(level)

st.divider()

st.header("📬 Let's Connect")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/)")

with col2:
    st.markdown("💻 [GitHub](https://github.com/)")

with col3:
    st.markdown("📧 Email: lilisantika@email.com")

st.info("Klik link di atas untuk terhubung dengan saya!")