import pandas as pd
import joblib
import json
import os
from fairlearn.metrics import demographic_parity_difference


def check_model_fairness():
    """
    Loads the trained fraud detection model and assesses its fairness based
    on the 'location' sensitive feature.
    """
    print("--- Checking Model Fairness ---")
    try:
        # Load model and data
        model = joblib.load("artifacts/model.pkl")
        df = pd.read_csv("data/transactions_2022.csv")
        if 'location' not in df.columns:
            print("❌ Error: 'location' column not found. Please run scripts/add_location.py first.")
            return
        # Get features the model was trained on
        expected_features = model.feature_names_in_
        X = df[expected_features]
        y_true = df['Class']
        sensitive_feature = df['location']
        print("✅ Model and data with 'location' feature loaded.")
    except (FileNotFoundError, AttributeError, KeyError) as e:
        print(f"❌ Error during data/model loading: {e}")
        return
    # Predict with the model
    y_pred = model.predict(X)
    # --- FIXED: Calculate Demographic Parity for the 'fraud' class (Class=1) ---
    print("\n📏 Calculating Demographic Parity Difference for the fraud class...")   
    # Manually create binary arrays for compatibility
    y_true_binary = (y_true == 1)
    y_pred_binary = (y_pred == 1)   
    dpd = demographic_parity_difference(
        y_true_binary,
        y_pred_binary,
        sensitive_features=sensitive_feature
    )   
    fairness_report = {
        "demographic_parity_difference_fraud": dpd
    }
    print("\n✅ Overall Fairness Report:")
    print(json.dumps(fairness_report, indent=2))
    # Save report to JSON file
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/fairness_report.json"
    with open(report_path, "w") as f:
        json.dump(fairness_report, f, indent=4)


    print(f"\n💾 Fairness report saved to: {report_path}")
    print("-----------------------------\n")


if __name__ == "__main__":
    check_model_fairness()
