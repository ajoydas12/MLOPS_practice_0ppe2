# for preparing the datasets
import pandas as pd
import os


def split_transactions(input_path: str, output_dir: str):
    # Read the CSV
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)


    # Sort by Time column
    df = df.sort_values("Time").reset_index(drop=True)


    # Split into two halves
    midpoint = len(df) // 2
    df_2022 = df.iloc[:midpoint].copy()
    df_2023 = df.iloc[midpoint:].copy()


    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)


    # Save the files
    path_2022 = os.path.join(output_dir, "transactions_2022.csv")
    path_2023 = os.path.join(output_dir, "transactions_2023.csv")
   
    df_2022.to_csv(path_2022, index=False)
    df_2023.to_csv(path_2023, index=False)


    print(f"Saved 2022 data to {path_2022}")
    print(f"Saved 2023 data to {path_2023}")
    print("✅ Done splitting and saving transactions.")


if __name__ == "__main__":
    input_csv_path = "orig_data/transactions.csv"         # Assumes this is in the project root
    output_directory = "data_orig"              # Output will be stored here
    split_transactions(input_csv_path, output_directory)
