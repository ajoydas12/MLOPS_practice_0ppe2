import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import sys


sys.stdout.reconfigure(line_buffering=True)


def generate_shap_explanations():
    """
    Loads the trained model, calculates SHAP values for a SAMPLE of the test set,
    and saves global and individual explanation plots.
    """
    print("--- Generating SHAP Explanations ---")
   
    try:
        model = joblib.load("artifacts/model.pkl")
        df = pd.read_csv("data/transactions.csv")
        expected_features = model.feature_names_in_
        X = df[expected_features]
        y = df['Class']
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return


    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   
    # --- Use a smaller sample of the test set for performance ---
    if len(X_test) > 2000:
        X_test_sample = X_test.sample(n=2000, random_state=42)
    else:
        X_test_sample = X_test
   
    print(f"Calculating SHAP values for a sample of {len(X_test_sample)} instances...")
   
    explainer = shap.TreeExplainer(model)
    shap_values_explanation = explainer(X_test_sample)
   
    # --- Global Summary Plot ---
    print("Generating global summary plot...")
    plt.figure()
    shap.summary_plot(shap_values_explanation, X_test_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/shap_summary.png")
    plt.close()
    print("Global summary plot saved.")


    # --- Force Plot for All Instances in the Sample ---
    print("Generating stacked force plot...")
    p_all = shap.force_plot(
        base_value=explainer.expected_value[1],
        shap_values=shap_values_explanation.values[:, :, 1],
        features=X_test_sample,
        matplotlib=False
    )
    shap.save_html("artifacts/shap_force_plot_all.html", p_all)
    print("Stacked force plot saved.")
   
    # --- Create and Save Textual Report Artifact ---
    try:
        # Determine the correct SHAP values to use for importance calculation
        if hasattr(model, "classes_"):  # Classification model
            if 1 in model.classes_:
                # Find index for the positive class '1'
                positive_class_index = np.where(model.classes_ == 1)[0][0]
            else:
                # Fallback for models where '1' isn't a class, use the second class
                positive_class_index = 1
            shap_values_for_importance = shap_values_explanation.values[:, :, positive_class_index]
        else:
            # For regression models, SHAP values array doesn't have a class dimension
            shap_values_for_importance = shap_values_explanation.values


        # Calculate feature importances
        importance_df = pd.DataFrame({
            'feature': X_test_sample.columns,
            'importance': np.abs(shap_values_for_importance).mean(axis=0)
        }).sort_values('importance', ascending=False)


        # Write the report
        report_path = 'artifacts/shap_report.txt'
        with open(report_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("🧠 MODEL EXPLAINABILITY RESULTS (SHAP)\n")
            f.write("="*60 + "\n")
            f.write("Top 10 Most Important Features:\n")
            f.write(importance_df.head(10).to_string(index=False))
            f.write("\n\n" + "="*60 + "\n")
            f.write("\n📋 Key Insights:\n")
            top_feature = importance_df.iloc[0]
            f.write(f"• Most predictive feature: {top_feature['feature']} (Avg SHAP value: {top_feature['importance']:.4f})\n")
            f.write("• Feature importance indicates which transaction patterns are most suspicious.\n")


        print(f"✅ Textual SHAP report saved to {report_path}")
    except Exception as e:
        print(f"❌ Error during textual report generation: {e}")


if __name__ == "__main__":
    generate_shap_explanations()
