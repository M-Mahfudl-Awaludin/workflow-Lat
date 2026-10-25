
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load preprocessed data
df = pd.read_csv("preprocessing.csv")

# Select features for clustering
features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].copy()

# Pastikan fitur numerik dan tidak memiliki nilai kosong
X = X.apply(pd.to_numeric, errors="coerce")

if X.isnull().any().any():
    raise ValueError(
        "Fitur clustering mengandung nilai kosong atau nonnumerik."
    )

if len(X) < 6:
    raise ValueError(
        "Dataset harus memiliki minimal 6 baris untuk evaluasi silhouette."
    )

# Set MLflow Tracking URI lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Latihan Model Statis")

# Enable MLflow Autolog
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run() as run:

    # Define model without hyperparameter tuning
    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    # Train model
    model.fit(X)

    # Predict cluster
    y_pred = model.predict(X)

    # Evaluate clustering
    inertia = float(model.inertia_)
    silhouette = float(silhouette_score(X, y_pred))

    # Log additional metrics
    mlflow.log_metric("inertia", inertia)
    mlflow.log_metric("silhouette_score", silhouette)

    # Log model
    input_example = X.iloc[:5]

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    # Save clustering results
    df["Cluster"] = y_pred

    df.to_csv(
        "hasil_clustering.csv",
        index=False
    )

    print("Model training completed successfully.")
    print(f"Inertia: {inertia:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"MLflow Run ID: {run.info.run_id}")

print("Experiment completed.")
