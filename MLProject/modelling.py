
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
        "Dataset minimal harus memiliki 6 baris."
    )

# Jangan memanggil set_tracking_uri(), set_experiment(),
# atau start_run() di sini.
# MLflow Project mengelola run yang sedang aktif.

# Enable MLflow Autolog
mlflow.sklearn.autolog(log_models=False)

# Define model
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

# Log metrics to the active MLflow Project run
mlflow.log_metric("inertia", inertia)
mlflow.log_metric("silhouette_score", silhouette)

# Log model
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=X.head(5)
)

# Save clustering results
df["Cluster"] = y_pred
df.to_csv("hasil_clustering.csv", index=False)

print("Model training completed successfully.")
print(f"Inertia: {inertia:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"MLflow Run ID: {os.environ.get('MLFLOW_RUN_ID', 'managed by MLflow')}")

print("Experiment completed.")
