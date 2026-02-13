import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Prediction System")

# -----------------------------
# Load model and dataset
# -----------------------------
model = joblib.load("models/best_model.pkl")
df = pd.read_csv("data/student_data.csv")

# -----------------------------
# Sidebar: Student Input
# -----------------------------
st.sidebar.header("Enter Student Details")

# Create input dictionary for all columns except target
input_dict = {}
for col in df.columns[:-1]:
    if df[col].dtype == 'object':
        options = df[col].unique()
        value = st.sidebar.selectbox(col, options)
        # Convert categorical to numeric using pandas category codes
        input_dict[col] = df[col].astype('category').cat.codes[df[col] == value].values[0]
    else:
        value = st.sidebar.number_input(col, float(df[col].min()), float(df[col].max()))
        input_dict[col] = value

# Convert to DataFrame for model
input_df = pd.DataFrame([input_dict], columns=df.columns[:-1])

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Grade Category: {prediction[0]}")

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    st.subheader("🔍 Feature Impact on Prediction (SHAP)")
    try:
        # Create SHAP explainer
        explainer = shap.Explainer(model.predict, input_df)
        shap_values = explainer(input_df)
        # Waterfall plot for single prediction
        st.pyplot(shap.plots.waterfall(shap_values[0], show=False))
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")

# -----------------------------
# Data Insights
# -----------------------------
st.subheader("📊 Data Insights")

# Class distribution
if st.checkbox("Show Class Distribution"):
    st.bar_chart(df.iloc[:, -1].value_counts())

# Numerical correlation heatmap
if st.checkbox("Show Correlation Heatmap (Numerical Only)"):
    numerical_df = df.select_dtypes(include=[np.number])
    if not numerical_df.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap (Numerical Columns)", fontsize=16)
        st.pyplot(plt)
    else:
        st.write("No numerical columns available for heatmap.")

# Full correlation heatmap (numerical + categorical)
if st.checkbox("Show Full Correlation (Encoded)"):
    encoded_df = df.copy()
    for col in encoded_df.select_dtypes(include=['object']).columns:
        encoded_df[col] = encoded_df[col].astype('category').cat.codes
    plt.figure(figsize=(14, 10))
    sns.heatmap(encoded_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (All Columns Encoded)", fontsize=16)
    st.pyplot(plt)

# Global feature importance
if st.checkbox("Show Feature Importance"):
    try:
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "Feature": df.columns[:-1],
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            plt.figure(figsize=(12, 6))
            sns.barplot(x="Importance", y="Feature", data=fi)
            plt.title("Feature Importance", fontsize=16)
            st.pyplot(plt)
        else:
            st.write("Feature importance not available for this model type.")
    except Exception as e:
        st.warning(f"Feature importance error: {e}")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)
