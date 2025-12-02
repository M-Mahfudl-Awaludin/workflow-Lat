# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import joblib

import mlflow
import mlflow.sklearn

# === FIX PENTING ===
mlflow.set_experiment("clustering-experiment")

with mlflow.start_run():

    # 1. Load Data
    df = pd.read_csv("preprocessed_data.csv")

    print("\n=== HEAD ===")
    print(df.head())

    print("\n=== INFO ===")
    print(df.info())

    print("\n=== DESCRIBE ===")
    print(df.describe())

    # 2. Elbow Method
    model = KMeans(random_state=42)
    visualizer = KElbowVisualizer(
        model,
        k=(2, 10),
        metric='silhouette',
        timings=False
    )
    visualizer.fit(df)

    best_k = visualizer.elbow_value_
    print("\nBest k from Elbow:", best_k)
    mlflow.log_param("elbow_best_k", best_k)

    # 3. Train Model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(df)

    mlflow.log_param("final_n_clusters", 3)

    score = silhouette_score(df, kmeans.labels_)
    print("Silhouette Score:", score)
    mlflow.log_metric("silhouette_score", score)

    # 4. Save model locally
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model_clustering.h5")
    joblib.dump(kmeans, model_path)
    print("\nModel saved to:", model_path)

    # 5. Log ke MLflow (INI SEKARANG BEKERJA)
    mlflow.log_artifacts("artifacts", artifact_path="model")
    # atau:
    # mlflow.sklearn.log_model(kmeans, artifact_path="model")

print("\n=== MLflow logging completed ===")
