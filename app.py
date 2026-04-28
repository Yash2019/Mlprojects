import streamlit as st
import requests
import os
import pandas as pd

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 30


def parse_api_response(resp):
    """Parse API JSON safely so Streamlit does not crash on HTML/error pages."""
    content_type = resp.headers.get("content-type", "")
    try:
        return resp.json()
    except requests.exceptions.JSONDecodeError:
        preview = resp.text.strip().replace("\n", " ")[:300] or "<empty response>"
        st.error(
            "The API did not return JSON. "
            f"Status: {resp.status_code}. Content-Type: {content_type or 'unknown'}."
        )
        st.code(preview)
        st.info(
            f"Current API_URL is {API_URL}. On Render, this must be the URL of "
            "your deployed FastAPI service, not the Streamlit service itself."
        )
        st.stop()


def show_probability_chart(title, probabilities):
    chart_data = pd.DataFrame(
        {"Probability (%)": {label: round(prob * 100, 1) for label, prob in probabilities.items()}}
    )
    st.markdown(f"**{title}**")
    st.bar_chart(chart_data)

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
            resp = requests.post(f"{API_URL}/predict/credit-risk", json=payload, timeout=REQUEST_TIMEOUT)
            result = parse_api_response(resp)
            if resp.status_code == 200:
                col_a, col_b = st.columns(2)
                col_a.metric("Prediction", result['label'])
                col_b.metric("Default Probability", f"{result['default_probability']*100:.1f}%")
                show_probability_chart("Prediction confidence", {
                    "No Default": 1 - result['default_probability'],
                    "Default": result['default_probability'],
                })
                if result['prediction'] == 1:
                    st.error("⚠️ High risk of default!")
                else:
                    st.success("✅ Low risk — likely to repay.")
            else:
                st.error(f"API Error: {result.get('detail', 'Unknown error')}")
        except requests.exceptions.Timeout:
            st.error(f"The API request timed out after {REQUEST_TIMEOUT} seconds.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running (`uvicorn api:app --reload`).")


# ── Fraud Detection ──────────────────────────────────
elif page == "Fraud Detection":
    st.title("🔒 Credit Card Fraud Detection")
    st.markdown("Detect fraudulent credit card transactions using PCA-transformed features.")
    st.markdown("---")

    st.info(
        "For a quick demo, choose a sample transaction. For accurate custom predictions, "
        "provide the advanced anonymized PCA fields from the original feature pipeline."
    )

    normal_sample = {
        "Time": 0.0,
        "Amount": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    }
    suspicious_sample = {
        "Time": 406.0,
        "Amount": 0.0,
        "V1": -2.3122265423263, "V2": 1.95199201064158, "V3": -1.60985073229769,
        "V4": 3.9979055875468, "V5": -0.522187864667764, "V6": -1.42654531920595,
        "V7": -2.53738730624579, "V8": 1.39165724829804, "V9": -2.77008927719433,
        "V10": -2.77227214465915, "V11": 3.20203320709635, "V12": -2.89990738849473,
        "V13": -0.595221881324605, "V14": -4.28925378244217, "V15": 0.389724120274487,
        "V16": -1.14074717980657, "V17": -2.83005567450437, "V18": -0.0168224681808257,
        "V19": 0.416955705037907, "V20": 0.126910559061474, "V21": 0.517232370861764,
        "V22": -0.0350493686052974, "V23": -0.465211076182388, "V24": 0.320198198514526,
        "V25": 0.0445191674731724, "V26": 0.177839798284401, "V27": 0.261145002567677,
        "V28": -0.143275874698919,
    }

    input_mode = st.radio("Transaction input", ["Normal sample", "Suspicious sample", "Custom"], horizontal=True)
    selected_sample = suspicious_sample if input_mode == "Suspicious sample" else normal_sample

    col1, col2 = st.columns(2)
    with col1:
        Time = st.number_input("Transaction time (seconds)", value=float(selected_sample["Time"]))
    with col2:
        Amount = st.number_input("Transaction amount ($)", value=float(selected_sample["Amount"]))

    features = {f"V{i}": float(selected_sample[f"V{i}"]) for i in range(1, 29)}
    advanced_open = input_mode == "Custom"
    with st.expander("Advanced anonymized model fields", expanded=advanced_open):
        st.caption("These V1-V28 values are PCA-transformed fields. Leave samples as-is for demo predictions.")
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        for i in range(1, 29):
            with cols[(i - 1) % 3]:
                features[f"V{i}"] = st.number_input(f"V{i}", value=features[f"V{i}"], key=f"v{i}")

    if st.button("🔮 Detect Fraud", use_container_width=True):
        payload = {"Time": Time, "Amount": Amount, **features}
        try:
            resp = requests.post(f"{API_URL}/predict/fraud-detection", json=payload, timeout=REQUEST_TIMEOUT)
            result = parse_api_response(resp)
            if resp.status_code == 200:
                col_a, col_b = st.columns(2)
                col_a.metric("Prediction", result['label'])
                col_b.metric("Fraud Probability", f"{result['fraud_probability']*100:.1f}%")
                show_probability_chart("Prediction confidence", {
                    "Legitimate": 1 - result['fraud_probability'],
                    "Fraud": result['fraud_probability'],
                })
                if result['prediction'] == 1:
                    st.error("🚨 This transaction looks FRAUDULENT!")
                else:
                    st.success("✅ This transaction appears legitimate.")
            else:
                st.error(f"API Error: {result.get('detail', 'Unknown error')}")
        except requests.exceptions.Timeout:
            st.error(f"The API request timed out after {REQUEST_TIMEOUT} seconds.")
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
            resp = requests.post(f"{API_URL}/predict/loan-prediction", json=payload, timeout=REQUEST_TIMEOUT)
            result = parse_api_response(resp)
            if resp.status_code == 200:
                col_a, col_b = st.columns(2)
                col_a.metric("Prediction", result['label'])
                col_b.metric("Approval Probability", f"{result['approval_probability']*100:.1f}%")
                show_probability_chart("Prediction confidence", {
                    "Rejected": 1 - result['approval_probability'],
                    "Approved": result['approval_probability'],
                })
                if result['prediction'] == 1:
                    st.success("✅ Loan is likely to be APPROVED!")
                else:
                    st.warning("⚠️ Loan is likely to be REJECTED.")
            else:
                st.error(f"API Error: {result.get('detail', 'Unknown error')}")
        except requests.exceptions.Timeout:
            st.error(f"The API request timed out after {REQUEST_TIMEOUT} seconds.")
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
                resp = requests.post(f"{API_URL}/predict/sentiment-analysis", json=payload, timeout=REQUEST_TIMEOUT)
                result = parse_api_response(resp)
                if resp.status_code == 200:
                    st.metric("Sentiment", result['label'].upper())
                    show_probability_chart("Sentiment confidence", result['probabilities'])
                else:
                    st.error(f"API Error: {result.get('detail', 'Unknown error')}")
            except requests.exceptions.Timeout:
                st.error(f"The API request timed out after {REQUEST_TIMEOUT} seconds.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the API server.")
