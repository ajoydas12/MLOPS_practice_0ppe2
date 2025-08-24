import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import argparse
import os
def find_suspicious_labels(data_path, k=5, threshold=0.5):
    """
    Analyzes a dataset to find rows with potentially flipped labels using KNN.


    Args:
        data_path (str): Path to the CSV data file.
        k (int): Number of neighbors to consider.
        threshold (float): Fraction of neighbors that must disagree to flag a point.
    """
    print(f"--- Checking for Suspicious Labels in: {data_path} ---")
   
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return []


    # Prepare data for KNN
    # We drop non-feature columns. 'errors="ignore"' makes it robust
    # in case a column like 'location' doesn't exist.
    X = df.drop(columns=['Class', 'Time', 'location'], errors='ignore')
    y = df['Class']


    # Use KNeighborsClassifier to find neighbors
    knn = KNeighborsClassifier(n_neighbors=k + 1)
    knn.fit(X, y)


    # Find the k+1 nearest neighbors for every point
    distances, indices = knn.kneighbors(X)


    suspicious_indices = []
    # Iterate through each data point to check its neighbors
    for i in range(len(df)):
        original_label = y.iloc[i]
        neighbor_indices = indices[i][1:] # Exclude the point itself
        neighbor_labels = y.iloc[neighbor_indices]
       
        # Count how many neighbors have a different label
        num_mismatched = np.sum(neighbor_labels != original_label)
       
        # If the mismatch ratio exceeds our threshold, flag the point
        if (num_mismatched / k) >= threshold:
            suspicious_indices.append(i)


    print(f"\nReport: Found {len(suspicious_indices)} suspicious labels out of {len(df)} total rows.")
    if suspicious_indices:
        print(f"Suspicious row indices (first 10): {suspicious_indices[:10]}")
    print("---------------------------------------------------\n")
   
    return suspicious_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for suspicious (potentially poisoned) labels in a dataset.")
    parser.add_argument("--data-path", type=str, default="data/transactions.csv", help="Path to the input CSV file.")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to check against.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fraction of neighbors that must disagree to flag a point.")
   
    args = parser.parse_args()


    find_suspicious_labels(data_path=args.data_path, k=args.k, threshold=args.threshold)
