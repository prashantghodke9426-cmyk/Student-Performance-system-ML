import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3
import smtplib
import jwt
import datetime
from email.mime.text import MIMEText

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Enterprise Student AI System", layout="wide")
st.title("🎓 Enterprise Student Performance AI System")

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/student_data.csv"
DB_PATH = "database.db"
JWT_SECRET = "supersecretkey"


# =========================================================
# DATABASE INITIALIZATION
# =========================================================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    prediction TEXT,
    confidence REAL
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS model_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    model_name TEXT,
    accuracy REAL,
    cv_score REAL
)
""")

conn.commit()


# =========================================================
# JWT AUTH SYSTEM (Login First Page)
# =========================================================
import jwt
import datetime

JWT_SECRET = "supersecretkey"

def generate_token(username):
    payload = {
        "user": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=5)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(token):
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return decoded
    except:
        return None

# Session state
if "token" not in st.session_state:
    st.session_state.token = None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# LOGIN SCREEN
if not st.session_state.authenticated:

    st.title("🔐 Secure Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            token = generate_token(username)
            st.session_state.token = token
            st.session_state.authenticated = True
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.stop()

# Verify token on reload
decoded = verify_token(st.session_state.token)
if not decoded:
    st.session_state.authenticated = False
    st.warning("Session expired. Please login again.")
    st.stop()

# Logout button
with st.sidebar:
    st.success(f"Logged in as {decoded['user']}")
    if st.button("Logout"):
        st.session_state.token = None
        st.session_state.authenticated = False
        st.rerun()


# =========================================================
# LOAD MODEL & DATA
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

encoded_df = df.copy()
for col in encoded_df.select_dtypes(include=['object']).columns:
    encoded_df[col] = encoded_df[col].astype('category').cat.codes

X = encoded_df.iloc[:, :-1]
y = encoded_df.iloc[:, -1]

y_pred = model.predict(X)

y_true_fixed = y.astype(str)
y_pred_fixed = pd.Series(y_pred).astype(str)

accuracy = accuracy_score(y_true_fixed, y_pred_fixed)
precision = precision_score(y_true_fixed, y_pred_fixed, average="weighted")
recall = recall_score(y_true_fixed, y_pred_fixed, average="weighted")
f1 = f1_score(y_true_fixed, y_pred_fixed, average="weighted")


# =========================================================
# KPI DASHBOARD
# =========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset Size", len(df))
col2.metric("Accuracy", f"{round(accuracy*100,2)}%")
col3.metric("Precision", f"{round(precision*100,2)}%")
col4.metric("F1 Score", f"{round(f1*100,2)}%")

st.divider()


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Prediction",
    "📊 Analytics Dashboard",
    "📈 Model Tracking",
    "📧 Email Alerts",
    "⚙️ Admin Control"
])


# =========================================================
# 🔮 PREDICTION + REAL TIME DB SAVE
# =========================================================
with tab1:
    st.header("Single Prediction")

    input_dict = {}
    for col in df.columns[:-1]:
        if df[col].dtype == 'object':
            val = st.selectbox(col, df[col].unique())
            input_dict[col] = df[col].astype('category').cat.codes[df[col] == val].values[0]
        else:
            input_dict[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        prediction = model.predict(input_df)

        if hasattr(model, "predict_proba"):
            proba = np.max(model.predict_proba(input_df))
            confidence = round(proba*100,2)
        else:
            confidence = None

        st.success(f"Prediction: {prediction[0]}")

        # Save to DB
        c.execute("INSERT INTO predictions (timestamp, prediction, confidence) VALUES (?, ?, ?)",
                  (str(datetime.datetime.now()), str(prediction[0]), confidence))
        conn.commit()

        if confidence and confidence < 60:
            st.error("⚠️ Early Warning: Low Confidence Student!")

        if confidence:
            st.progress(int(confidence))


# =========================================================
# 📊 ANALYTICS DASHBOARD (REAL TIME)
# =========================================================
with tab2:
    st.header("Prediction Trends")

    pred_df = pd.read_sql_query("SELECT * FROM predictions", conn)

    if not pred_df.empty:
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

        st.line_chart(pred_df["confidence"])

        st.subheader("Prediction Distribution")
        st.bar_chart(pred_df["prediction"].value_counts())
    else:
        st.info("No predictions logged yet.")


# =========================================================
# 📈 MODEL PERFORMANCE HISTORY
# =========================================================
with tab3:
    st.header("Model Performance History")

    history_df = pd.read_sql_query("SELECT * FROM model_history", conn)

    if not history_df.empty:
        st.dataframe(history_df)
        st.line_chart(history_df["accuracy"])
    else:
        st.info("No model history available.")


# =========================================================
# 📧 EMAIL ALERT SYSTEM
# =========================================================
with tab4:
    st.header("Send Risk Alert Email")

    receiver = st.text_input("Recipient Email")

    if st.button("Send Test Alert"):
        try:
            msg = MIMEText("⚠️ Student at risk detected.")
            msg["Subject"] = "Student Risk Alert"
            msg["From"] = "your_email@gmail.com"
            msg["To"] = receiver

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login("your_email@gmail.com", "your_app_password")
            server.send_message(msg)
            server.quit()

            st.success("Email Sent Successfully")
        except Exception as e:
            st.error(f"Email failed: {e}")


# =========================================================
# ⚙️ ADMIN CONTROL + MODEL RETRAINING
# =========================================================
with tab5:
    if not st.session_state.authenticated:
        st.warning("Admin login required.")
    else:
        st.header("Model Comparison & Retraining")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        models = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000)
        }

        if XGB_AVAILABLE:
            models["XGBoost"] = XGBClassifier(eval_metric="mlogloss")

        results = {}

        for name, m in models.items():
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            acc = accuracy_score(y_test, pred)
            cv = cross_val_score(m, X, y, cv=5).mean()
            results[name] = {"Accuracy": acc, "CV": cv}

            # Save history
            c.execute("INSERT INTO model_history (timestamp, model_name, accuracy, cv_score) VALUES (?, ?, ?, ?)",
                      (str(datetime.datetime.now()), name, acc, cv))
            conn.commit()

        result_df = pd.DataFrame(results).T
        st.dataframe(result_df)

        selected = st.selectbox("Select Model to Deploy", result_df.index)

        if st.button("Deploy Selected Model"):
            best_model = models[selected]
            best_model.fit(X, y)
            joblib.dump(best_model, MODEL_PATH)
            st.success(f"{selected} deployed successfully!")
