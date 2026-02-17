import streamlit as st
import pandas as pd
import joblib
import os
import xgboost as xgb
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CareerPath Pro",
    page_icon="🎃",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

@st.cache_resource
def load_models():
    clf = joblib.load(os.path.join(MODEL_DIR, 'placement_model.pkl'))
    reg = joblib.load(os.path.join(MODEL_DIR, 'salary_model.pkl'))
    return clf, reg

clf, reg = load_models()

st.title("🎃 CareerPath Pro: Placement & Salary Dashboard")
st.markdown("---")

st.sidebar.header("⚙️ Student Parameters")
with st.sidebar:
    iq = st.number_input("IQ Score", 50, 200, 110)
    cgpa = st.number_input("CGPA", 0.0, 10.0, 8.5, 0.1)
    prev_sem = st.slider("Previous Semester Result", 0.0, 10.0, 8.0)
    internships = st.selectbox("Internship Experience", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

col1, col2 = st.columns(2)
with col1:
    academic_perf = st.slider("Academic Performance Score", 1, 10, 8)
    comm_skills = st.slider("Communication Skills", 1, 10, 7)
with col2:
    extra_curr = st.slider("Extra Curricular Score", 1, 10, 6)
    projects = st.number_input("Projects Completed", 0, 15, 3)

if st.button("🔮 Predict My Future"):
    academic_power = cgpa * iq
    
    input_dict = {
        'IQ': iq,
        'Prev_Sem_Result': prev_sem,
        'CGPA': cgpa,
        'Academic_Performance': academic_perf,
        'Internship_Experience': internships,
        'Extra_Curricular_Score': extra_curr,
        'Communication_Skills': comm_skills,
        'Projects_Completed': projects,
        'Academic_Power': academic_power
    }
    
    df_features = pd.DataFrame([input_dict])
    
    is_placed = clf.predict(df_features)[0]
    prob = clf.predict_proba(df_features)[0][1]
    
    with st.container(border=True):
        res_col1, res_col2 = st.columns(2)
        
        if is_placed == 1:
            res_col1.success("### Status: PLACED!")
            res_col1.write(f"**Confidence:** {prob*100:.2f}%")
            
            salary = reg.predict(df_features)[0]
            res_col2.metric("Estimated Salary Package", f"₹{max(0, salary):,.2f}")
        else:
            res_col1.error("### Status: NOT PLACED")
            res_col1.write(f"**Confidence:** {(1-prob)*100:.2f}%")
            
            if internships == 0:
                tip = "💡 Tip: Focus on securing an **Internship** and building more **Technical Projects**."
            else:
                tip = "💡 Tip: You have experience! Focus on increasing your **CGPA** or **Communication Skills** to cross the final threshold."
            
            res_col2.info(tip)

st.divider()
st.subheader("📊 Global Model Insights")
tab1, tab2 = st.tabs(["Feature Importance", "About the Model"])

with tab1:
    importance_path = os.path.join(OUTPUT_DIR, 'salary_importance.png')
    if os.path.exists(importance_path):
        st.image(importance_path, caption="XGBoost Feature Importance: Getting a Job vs. Getting Paid")
    else:
        st.warning("Feature importance plot not found. Please run 'compare_features.py' first.")

with tab2:
    st.write("""
    **Model Architecture:** Dual XGBoost Pipeline (Classifier + Regressor)
    **Optimization:** Hyperparameter Tuning via Optuna
    **Key Drivers:** Academic metrics (CGPA/IQ) and professional experience (Internships).
    """)