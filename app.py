import streamlit as st
import requests
import json
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ML Projects Dashboard", page_icon="🧠", layout="wide")

# ── Sidebar ──────────────────────────────────────────
st.sidebar.title("🧠 ML Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select a Project", [
    "Credit Risk Prediction",
    "Fraud Detection",
    "Loan Prediction",
    "Sentiment Analysis",
])


# ── Credit Risk ──────────────────────────────────────
if page == "Credit Risk Prediction":
    st.title("💳 Credit Risk Prediction")
    st.markdown("Predict whether a borrower is likely to **default** on a loan.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        person_age = st.number_input("Age", 18, 100, 30)
        person_income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
        person_emp_length = st.number_input("Employment Length (years)", 0.0, 50.0, 5.0)
        loan_amnt = st.number_input("Loan Amount ($)", 0, 100000, 10000)
    with col2:
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    with col3:
        loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 10.0)
        loan_percent_income = st.number_input("Loan % of Income", 0.0, 1.0, 0.2, step=0.01)
        cb_person_default_on_file = st.selectbox("Historical Default", ["N", "Y"])
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", 0.0, 30.0, 5.0)

    if st.button("🔮 Predict Credit Risk", use_container_width=True):
        payload = {
            "person_age": person_age, "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length, "loan_intent": loan_intent,
            "loan_grade": loan_grade, "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }
        try:
            resp = requests.post(f"{API_URL}/predict/credit-risk", json=payload)
            result = resp.json()
            if resp.status_code == 200:
                col_a, col_b = st.columns(2)
                col_a.metric("Prediction", result['label'])
                col_b.metric("Default Probability", f"{result['default_probability']*100:.1f}%")
                if result['prediction'] == 1:
                    st.error("⚠️ High risk of default!")
                else:
                    st.success("✅ Low risk — likely to repay.")
            else:
                st.error(f"API Error: {result.get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running (`uvicorn api:app --reload`).")


# ── Fraud Detection ──────────────────────────────────
elif page == "Fraud Detection":
    st.title("🔒 Credit Card Fraud Detection")
    st.markdown("Detect fraudulent credit card transactions using PCA-transformed features.")
    st.markdown("---")

    st.info("This model uses 28 PCA components (V1–V28) plus Time and Amount. Enter values below or use defaults for a quick test.")

    with st.expander("📝 Transaction Features", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            Time = st.number_input("Time (seconds)", value=0.0)
            Amount = st.number_input("Amount ($)", value=100.0)
        features = {}
        for i in range(1, 29):
            col_idx = (i - 1) % 3
            cols = [col1, col2, col3]
            with cols[col_idx]:
                features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}")

    if st.button("🔮 Detect Fraud", use_container_width=True):
        payload = {"Time": Time, "Amount": Amount, **features}
        try:
            resp = requests.post(f"{API_URL}/predict/fraud-detection", json=payload)
            result = resp.json()
            if resp.status_code == 200:
                col_a, col_b = st.columns(2)
                col_a.metric("Prediction", result['label'])
                col_b.metric("Fraud Probability", f"{result['fraud_probability']*100:.1f}%")
                if result['prediction'] == 1:
                    st.error("🚨 This transaction looks FRAUDULENT!")
                else:
                    st.success("✅ This transaction appears legitimate.")
            else:
                st.error(f"API Error: {result.get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API server.")


# ── Loan Prediction ──────────────────────────────────
elif page == "Loan Prediction":
    st.title("💰 Loan Approval Prediction")
    st.markdown("Predict whether a loan application will be **approved** or **rejected**.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 18, 100, 35)
        Income = st.number_input("Income ($)", 0.0, 500000.0, 50000.0)
        LoanAmount = st.number_input("Loan Amount ($)", 0.0, 200000.0, 20000.0)
        CreditScore = st.number_input("Credit Score", 300.0, 850.0, 700.0)
        YearsExperience = st.number_input("Years of Experience", 0, 50, 10)
    with col2:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD", "Unknown"])
        City = st.selectbox("City", ["Houston", "San Francisco", "New York", "Chicago", "Los Angeles"])
        EmploymentType = st.selectbox("Employment Type", ["Employed", "Self-Employed", "Unemployed"])

    if st.button("🔮 Predict Loan Approval", use_container_width=True):
        payload = {
            "Age": Age, "Income": Income, "LoanAmount": LoanAmount,
            "CreditScore": CreditScore, "YearsExperience": YearsExperience,
            "Gender": Gender, "Education": Education,
            "City": City, "EmploymentType": EmploymentType,
        }
        try:
            resp = requests.post(f"{API_URL}/predict/loan-prediction", json=payload)
            result = resp.json()
            if resp.status_code == 200:
                col_a, col_b = st.columns(2)
                col_a.metric("Prediction", result['label'])
                col_b.metric("Approval Probability", f"{result['approval_probability']*100:.1f}%")
                if result['prediction'] == 1:
                    st.success("✅ Loan is likely to be APPROVED!")
                else:
                    st.warning("⚠️ Loan is likely to be REJECTED.")
            else:
                st.error(f"API Error: {result.get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API server.")


# ── Sentiment Analysis ───────────────────────────────
elif page == "Sentiment Analysis":
    st.title("📰 News Sentiment Analysis")
    st.markdown("Classify the sentiment of news text as **Positive**, **Neutral**, or **Negative**.")
    st.markdown("---")

    text = st.text_area("Enter news text:", height=150,
                        placeholder="e.g., The company reported record profits this quarter...")

    if st.button("🔮 Analyze Sentiment", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            payload = {"text": text}
            try:
                resp = requests.post(f"{API_URL}/predict/sentiment-analysis", json=payload)
                result = resp.json()
                if resp.status_code == 200:
                    st.metric("Sentiment", result['label'].upper())
                    st.markdown("**Probability breakdown:**")
                    for label, prob in result['probabilities'].items():
                        st.progress(prob, text=f"{label}: {prob*100:.1f}%")
                else:
                    st.error(f"API Error: {result.get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the API server.")
