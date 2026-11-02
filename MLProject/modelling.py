
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set project directory
BASE_DIR = Path(__file__).resolve().parent

# Load preprocessed data
df = pd.read_csv(BASE_DIR / "preprocessed_data.csv")

# Select features for clustering
features = [
    "Year_Built",
    "Length_meters",
    "Gross_Tonnage",
    "Top_Speed_knots",
    "Guests_Capacity"
]

X = df[features].copy()

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Handle missing values
X = X.fillna(X.median())

# Remove rows with remaining missing values
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows]
df = df.loc[valid_rows].copy()

if len(X) <= 5:
    raise ValueError("Jumlah data harus lebih dari 5.")

# Set MLflow Tracking URI
tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file:./mlruns"
)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Global Superyachts Clustering")

# Enable autologging
mlflow.sklearn.autolog()

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
inertia = model.inertia_
silhouette = silhouette_score(X, y_pred)

# Log additional metrics to the active MLflow Project run
mlflow.log_metric("inertia", float(inertia))
mlflow.log_metric("silhouette_score", float(silhouette))

# Save clustering results
df["Cluster"] = y_pred

df.to_csv(
    BASE_DIR / "hasil_clustering.csv",
    index=False
)

# Save cluster summary
df.groupby("Cluster")[features].mean().to_csv(
    BASE_DIR / "ringkasan_cluster.csv"
)

# Display results
active_run = mlflow.active_run()

print("Model training completed successfully.")
print(f"Inertia: {inertia:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

if active_run:
    print(f"MLflow Run ID: {active_run.info.run_id}")

print("Experiment completed.")
