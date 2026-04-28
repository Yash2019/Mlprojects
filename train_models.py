
import os
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# 1. Credit Risk Prediction
# ─────────────────────────────────────────────────────
def train_credit_risk():
    print("\n" + "="*60)
    print("  Training Credit Risk Prediction Model")
    print("="*60)

    csv_path = os.path.join(BASE_DIR, "credit_risk_pred", "credit_risk_dataset.csv")
    df = pd.read_csv(csv_path)

    # Data cleaning (matching notebook)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Encode categorical columns
    categorical = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    label_encoders = {}
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost (best in notebook)
    from xgboost import XGBClassifier
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Save model and encoders
    artifact = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns),
    }
    joblib.dump(artifact, os.path.join(MODELS_DIR, "credit_risk_model.pkl"))
    print("  [OK] Saved credit_risk_model.pkl")


# ─────────────────────────────────────────────────────
# 2. Credit Card Fraud Detection
# ─────────────────────────────────────────────────────
def train_fraud_detection():
    print("\n" + "="*60)
    print("  Training Credit Card Fraud Detection Model")
    print("="*60)

    csv_path = os.path.join(BASE_DIR, "creditcard", "creditcard.csv")
    df = pd.read_csv(csv_path)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_sm, y_train_sm)

    y_pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:  {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")

    artifact = {
        'model': model,
        'feature_names': list(X.columns),
    }
    joblib.dump(artifact, os.path.join(MODELS_DIR, "fraud_detection_model.pkl"))
    print("  [OK] Saved fraud_detection_model.pkl")


# ─────────────────────────────────────────────────────
# 3. Loan Risk Prediction
# ─────────────────────────────────────────────────────
def train_loan_prediction():
    print("\n" + "="*60)
    print("  Training Loan Risk Prediction Model")
    print("="*60)

    csv_path = os.path.join(BASE_DIR, "loan prediction", "loan_risk_prediction_dataset.csv")
    df = pd.read_csv(csv_path)

    # Data cleaning (matching notebook)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median = df[col].median()
        df[col].fillna(median, inplace=True)

    # Clip outliers (matching notebook)
    for col in ['Income', 'LoanAmount', 'CreditScore']:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)

    # Fill Education NaN
    if 'Education' in df.columns:
        df['Education'].fillna("Unknown", inplace=True)

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    # Ensure all boolean columns become int
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop("LoanApproved", axis=1)
    y = df['LoanApproved']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.4f}")

    artifact = {
        'model': model,
        'feature_names': list(X.columns),
        'dummy_columns': list(X.columns),
    }
    joblib.dump(artifact, os.path.join(MODELS_DIR, "loan_prediction_model.pkl"))
    print("  [OK] Saved loan_prediction_model.pkl")


# ─────────────────────────────────────────────────────
# 4. Sentiment Analysis
# ─────────────────────────────────────────────────────
def train_sentiment_analysis():
    print("\n" + "="*60)
    print("  Training Sentiment Analysis Model")
    print("="*60)

    csv_path = os.path.join(BASE_DIR, "sentiment analysis", "news_sentiment_analysis.csv")
    df = pd.read_csv(csv_path)

    # Clean data
    df.dropna(subset=['Description', 'Sentiment'], inplace=True)
    df['Description'] = df['Description'].astype(str)

    # Encode labels
    le = LabelEncoder()
    df['Sentiment_encoded'] = le.fit_transform(df['Sentiment'])

    X = df['Description']
    y = df['Sentiment_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Random Forest (matching notebook approach)
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features='log2',
        max_depth=None,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    artifact = {
        'model': model,
        'tfidf': tfidf,
        'label_encoder': le,
    }
    joblib.dump(artifact, os.path.join(MODELS_DIR, "sentiment_analysis_model.pkl"))
    print("  [OK] Saved sentiment_analysis_model.pkl")


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting model training for all projects...\n")
    train_credit_risk()
    train_fraud_detection()
    train_loan_prediction()
    train_sentiment_analysis()
    print("\n" + "="*60)
    print("  All models trained and saved successfully!")
    print("="*60)
