import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from config import Config


def _feature_columns(cfg: Config) -> list[str]:
    return [f"f{i + 1}" for i in range(cfg.NUM_FEATURES)]


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load and validate all gesture CSV files from data folder."""
    if not os.path.isdir(cfg.DATA_PATH):
        raise ValueError(f"Folder data tidak ditemukan: {cfg.DATA_PATH}")

    if cfg.NUM_COORDINATES <= 0:
        raise ValueError("NUM_COORDINATES harus > 0")
    if cfg.NUM_FEATURES % cfg.NUM_COORDINATES != 0:
        raise ValueError(
            f"NUM_FEATURES ({cfg.NUM_FEATURES}) harus kelipatan NUM_COORDINATES ({cfg.NUM_COORDINATES})"
        )

    feature_size = cfg.NUM_FEATURES
    columns = _feature_columns(cfg)

    data_all = []
    print("\nMulai load data dari folder:", cfg.DATA_PATH)
    for fname in os.listdir(cfg.DATA_PATH):
        if fname.endswith(".csv") and fname.startswith("data_"):
            print(f"Memproses: {fname}")
            try:
                df = pd.read_csv(os.path.join(cfg.DATA_PATH, fname), header=None)
                print(f"  Jumlah kolom: {df.shape[1]}")
                if df.shape[1] != feature_size:
                    print(f"  Dilewati: Kolom tidak sesuai ({df.shape[1]} != {feature_size})")
                    continue

                label = fname.replace("data_", "").replace(".csv", "")
                df.columns = columns
                df["label"] = label
                data_all.append(df)
            except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
                print(f"  Gagal membaca {fname}: {e}")

    if not data_all:
        raise ValueError("Tidak ada file valid yang dimuat. Periksa format CSV dan kolom fitur.")

    return pd.concat(data_all, ignore_index=True)


def train_model(cfg: Config):
    """Train SVM model using the configured feature size."""
    dataset = load_dataset(cfg)
    print(f"\nTotal sampel: {len(dataset)}")

    columns = _feature_columns(cfg)
    X = dataset[columns].fillna(0).astype(float).values

    le = LabelEncoder()
    y = le.fit_transform(dataset["label"])

    class_counts = np.bincount(y)
    use_stratify = np.all(class_counts >= 2)
    if not use_stratify:
        print("Peringatan: Ada kelas dengan <2 sampel, stratify dinonaktifkan untuk menghindari error split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    model = SVC(kernel="rbf", C=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nHasil Evaluasi:")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    all_label_indices = np.arange(len(le.classes_))
    print(
        classification_report(
            y_test,
            y_pred,
            labels=all_label_indices,
            target_names=le.classes_,
            zero_division=0,
        )
    )

    print("\nAkurasi per huruf:")
    correct = y_pred == y_test
    for idx, label in enumerate(le.classes_):
        total = int(np.sum(y_test == idx))
        benar = int(np.sum((y_test == idx) & correct))
        akurasi = 100 * benar / total if total else 0
        print(f" - {label}: {akurasi:.1f}% ({benar}/{total})")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=le.classes_,
        cmap="Blues",
        xticks_rotation="vertical",
    ).plot()
    plt.title("Confusion Matrix - SIBI Gesture Model")
    plt.tight_layout()
    plt.show()

    return model, le


def save_artifacts(cfg: Config, model, le) -> None:
    """Save model and label encoder into model folder."""
    model_folder = os.path.dirname(cfg.MODEL_PATH) or "."
    os.makedirs(model_folder, exist_ok=True)

    with open(os.path.join(model_folder, "mymodel.sav"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(model_folder, "le.sav"), "wb") as f:
        pickle.dump(le, f)

    print("\nModel dan label encoder berhasil disimpan ke folder:", model_folder)


def main() -> None:
    cfg = Config()
    model, le = train_model(cfg)
    save_artifacts(cfg, model, le)


if __name__ == "__main__":
    main()
