import os
import joblib
import json
from datetime import datetime
from src.preprocess import load_and_preprocess
from src.train import train_models
from src.evaluate import evaluate_model
import sys

def main():
    # -----------------------------
    # Ensure folders exist
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    data_path = os.path.join("data", "student_data.csv")

    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        sys.exit(1)

    # -----------------------------
    # Load and preprocess
    # -----------------------------
    X_train, X_test, y_train, y_test, scaler, encoders, df = load_and_preprocess(data_path)
    print(f"✅ Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")

    # -----------------------------
    # Train models
    # -----------------------------
    best_model, best_name = train_models(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    evaluate_model(best_model, X_test, y_test)

    # -----------------------------
    # Save all artifacts
    # -----------------------------
    joblib.dump(best_model, os.path.join("models", "best_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    joblib.dump(encoders, os.path.join("models", "label_encoders.pkl"))
    joblib.dump(df.columns[:-1], os.path.join("models", "feature_columns.pkl"))

    metadata = {
        "model_name": best_name,
        "trained_on": str(datetime.now()),
        "dataset_size": len(df)
    }
    with open(os.path.join("models", "model_metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("✅ All 5 artifacts saved successfully in 'models/' folder!")

if __name__ == "__main__":
    main()
