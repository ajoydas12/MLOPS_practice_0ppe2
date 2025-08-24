from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import os
import pandas as pd
def check_data_drift(reference_path="data_orig/transactions_2022.csv",
                     current_path="data_orig/transactions_2023.csv",
                     output_dir="artifacts"):
    """
    Compares a reference (v0) and current (v1) dataset to generate a
    data drift report using Evidently AI.
    Args:
        reference_path (str): Path to the reference (e.g., v0) dataset.
        current_path (str): Path to the current (e.g., v1) dataset.
        output_dir (str): Directory to save the output report.
    """
    print("--- Checking for Data Drift ---")   
    # 1. Load the datasets
    try:
        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)
        print(f"Reference data loaded from: {reference_path}")
        print(f"Current data loaded from: {current_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data has been prepared.")
        return
    # 2. Run the drift report
    print("Running Evidently report...")
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    # The .run() method modifies the report in-place
    data_drift_report.run(reference_data=reference_df, current_data=current_df)


    my_eval=drift_report.run(reference_data=reference_df, current_data=current_df)
    my_eval


# Save the report
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    my_eval.save_html("artifacts/drift_report.html")
    print("Drift report saved to artifacts/drift_report.html")
    print("Drift analysis completed successfully. ✓")  


if __name__ == "__main__":
    check_data_drift()
