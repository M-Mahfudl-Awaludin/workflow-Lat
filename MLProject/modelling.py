
import os
from contextlib import nullcontext
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "titanic_preprocessing_train.csv"
TEST_PATH = BASE_DIR / "titanic_preprocessing_test.csv"

RANDOM_STATE = 42

# Local run menggunakan tracking server lokal.
# MLflow Project akan mengelola run-nya sendiri.
if not os.environ.get("MLFLOW_RUN_ID"):
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment("Titanic Classification")
    run_context = mlflow.start_run(run_name="Random Forest Titanic")
else:
    run_context = nullcontext()

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError(
        "Dataset preprocessing tidak ditemukan. Pastikan file "
        "titanic_preprocessing_train.csv dan "
        "titanic_preprocessing_test.csv ada di folder MLProject."
    )

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target = "Survived"

if target not in train_df.columns or target not in test_df.columns:
    raise ValueError("Kolom target Survived tidak ditemukan.")

X_train = train_df.drop(columns=[target])
y_train = train_df[target].astype(int)

X_test = test_df.drop(columns=[target])
y_test = test_df[target].astype(int)

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Kolom fitur training dan testing tidak konsisten.")

if X_train.isnull().any().any() or X_test.isnull().any().any():
    raise ValueError("Fitur masih mengandung missing values.")

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Dataset training atau testing kosong.")

# Autolog parameter dan metrik dari scikit-learn.
mlflow.sklearn.autolog(log_models=False)

with run_context:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))
    mlflow.log_param("number_of_features", X_train.shape[1])

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, float(metric_value))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.head(5),
    )

    # Simpan hasil prediksi
    prediction_path = BASE_DIR / "hasil_prediksi_titanic.csv"
    pd.DataFrame({
        "actual_Survived": y_test.to_numpy(),
        "predicted_Survived": y_pred,
    }).to_csv(prediction_path, index=False)

    # Simpan classification report
    report_path = BASE_DIR / "classification_report.txt"
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Tidak Selamat", "Selamat"],
        zero_division=0,
    )
    report_path.write_text(report, encoding="utf-8")

    # Simpan confusion matrix
    cm_path = BASE_DIR / "confusion_matrix.csv"
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(cm_path)

    mlflow.log_artifact(str(prediction_path))
    mlflow.log_artifact(str(report_path))
    mlflow.log_artifact(str(cm_path))

    print("\nHasil evaluasi:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nClassification Report:")
    print(report)

    print("Hasil prediksi:", prediction_path)
    print("Classification report:", report_path)
    print("Confusion matrix:", cm_path)

print("\nEksperimen Titanic Classification selesai.")
