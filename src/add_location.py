import pandas as pd
import numpy as np
import os
def add_sensitive_feature(data_path="data/transactions.csv"):
    """
    Adds a synthetic 'location' column to the dataset for fairness analysis.
    This script modifies the file in-place.
    Args:
        data_path (str): Path to the transaction data CSV.
    """
    print(f"--- Adding sensitive feature to {data_path} ---")   
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    if 'location' in df.columns:
        print("'location' column already exists. Skipping.")
        return
    np.random.seed(42) # for reproducibility
    df['location'] = np.random.choice(['Location_A', 'Location_B'], size=len(df))
    df.to_csv(data_path, index=False)   
    print("Successfully added 'location' column.")
    print("---------------------------------------------\n")
if __name__ == "__main__":
    add_sensitive_feature()
