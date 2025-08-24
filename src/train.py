
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn
import joblib
import os
from google.cloud import storage
# --- Helper Function for GCS Upload ---
def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """Uploads a file to the specified GCS bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        print(f"File {source_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
def train_model(data_path='data/transactions.csv'):
    """
    Trains a Decision Tree model for fraud detection, logs the experiment
    to MLflow, and saves the final model artifact locally and to GCS.
    """
    print("--- Training Model & Logging with MLflow ---")
    mlflow.set_experiment("Fraud_Detection_Training")   
    with mlflow.start_run() as run:
        # Load data
        df = pd.read_csv(data_path)       
        # --- FIXED: Define features (X) and target (y) for the fraud dataset ---
        X = df.drop(columns=['Class', 'Time'])
        y = df['Class']       
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)       
        # Train Decision Tree model
        params = {
            'class_weight': 'balanced',
            'max_depth': 5, # A reasonable depth to prevent overfitting
            'random_state': 42
        }
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)       
        # Evaluate model performance using F1-score, which is good for imbalanced data
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f"Validation F1-Score: {f1:.4f}")       
        # Log parameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)       
        # Log the model to MLflow for tracking and registration
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="fraud-detection-dt"
        )
        print(f"Model logged to MLflow. Run ID: {run.info.run_id}")
        # --- Save the final model locally for the API ---
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)
        print(f"Model artifact saved locally to: {model_path}")
        # --- Upload the final model directly to GCS ---
        bucket_name = "mlops_practice_bucket" # Your bucket name
        destination_path = "production_models/model.pkl"
        upload_to_gcs(bucket_name, model_path, destination_path)
if __name__ == "__main__":
    # Make sure your MLflow server is running and accessible
    # Replace with your server's IP address
    mlflow.set_tracking_uri("http://35.223.196.178/:5000") #<-- Example IP
    train_model()
