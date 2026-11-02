
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Konfigurasi direktori
BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "titanic_preprocessing_train.csv"
TEST_PATH = BASE_DIR / "titanic_preprocessing_test.csv"

# Memuat dataset
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target = "Survived"

if target not in train_df.columns or target not in test_df.columns:
    raise ValueError("Kolom Survived tidak ditemukan.")

X_train = train_df.drop(columns=[target])
y_train = train_df[target].astype(int)

X_test = test_df.drop(columns=[target])
y_test = test_df[target].astype(int)

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Kolom fitur training dan testing tidak konsisten.")

if X_train.isnull().any().any() or X_test.isnull().any().any():
    raise ValueError("Dataset masih mengandung missing values.")

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Dataset training atau testing kosong.")

# Konfigurasi MLflow
mlflow.set_experiment("Titanic Classification")

# Autolog parameter, metrik, dan model
mlflow.sklearn.autolog()

# Membuat dan melatih model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Simpan hasil prediksi
pd.DataFrame({
    "actual_Survived": y_test.to_numpy(),
    "predicted_Survived": y_pred,
}).to_csv(
    BASE_DIR / "hasil_prediksi_titanic.csv",
    index=False,
)

print("Training model selesai.")
print("Hasil prediksi berhasil disimpan.")
