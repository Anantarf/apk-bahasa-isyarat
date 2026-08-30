"""Evaluasi model dan validasi dataset SIBI.

Fitur utama:
1) Validasi dataset `data/data_*.csv` terhadap aturan `Config`.
2) In-sample evaluation untuk cek mismatch model/dataset.
3) Holdout evaluation (refit clone model).
4) Optional stratified cross-validation.

Run examples:
  python scripts/evaluasi.py --validate-only
  python scripts/evaluasi.py --test-size 0.2 --seed 42
  python scripts/evaluasi.py --cv 5 --seed 42
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

# Enable importing from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    labels_sorted: List[str]
    per_label_counts: Dict[str, int]
    n_features: int


def _discover_label_files(data_path: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(data_path):
        return []

    files = []
    try:
        names = sorted(os.listdir(data_path))
    except OSError:
        return []

    for name in names:
        if not (name.startswith("data_") and name.endswith(".csv")):
            continue
        label = name[len("data_") : -len(".csv")]
        files.append((label, os.path.join(data_path, name)))
    return files


def validate_dataset(config: Config) -> Tuple[bool, bool, str]:
    """Validate CSV shape and class distribution against config rules."""
    pairs = _discover_label_files(config.DATA_PATH)
    if not pairs:
        return False, False, f"Tidak ada data_*.csv di: {config.DATA_PATH}"

    expected_features = int(config.NUM_FEATURES)
    target_per_class = int(config.TARGET_PER_CLASS)

    invalid_shapes: List[str] = []
    counts: Dict[str, int] = {}

    for label, path in pairs:
        try:
            df = pd.read_csv(path, header=None)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
            return False, False, f"Gagal membaca {path}: {e}"

        cols = int(df.shape[1])
        rows = int(df.shape[0])
        counts[label] = rows

        if cols != expected_features:
            invalid_shapes.append(f"{os.path.basename(path)}: {cols} kolom (expected {expected_features})")

    lines = []
    lines.append("Dataset validation summary:")
    lines.append(f"  Path data: {config.DATA_PATH}")
    lines.append(f"  Expected features: {expected_features}")
    lines.append(f"  Target per class: {target_per_class}")
    lines.append("")

    lines.append("Per-class counts:")
    below_target: List[str] = []
    for label in sorted(counts):
        n = counts[label]
        status = "OK" if n >= target_per_class else "KURANG"
        if n < target_per_class:
            below_target.append(label)
        lines.append(f"  {label:>2}: {n} ({status})")

    if invalid_shapes:
        lines.append("")
        lines.append("Invalid file shapes:")
        lines.extend([f"  - {s}" for s in invalid_shapes])

    shape_ok = len(invalid_shapes) == 0
    target_ok = len(below_target) == 0

    if not target_ok:
        lines.append("")
        lines.append(
            "Kelas di bawah TARGET_PER_CLASS: " + ", ".join(sorted(below_target))
        )

    return shape_ok, target_ok, "\n".join(lines)


def load_dataset(config: Config) -> Dataset:
    pairs = _discover_label_files(config.DATA_PATH)
    if not pairs:
        raise RuntimeError(f"No data files found in: {config.DATA_PATH}")

    xs: List[np.ndarray] = []
    ys: List[str] = []
    counts: Dict[str, int] = {}
    n_features: int | None = None

    for label, path in pairs:
        df = pd.read_csv(path, header=None)
        arr = df.to_numpy(dtype=float)
        if arr.ndim != 2:
            raise RuntimeError(f"Invalid shape from {path}: {arr.shape}")

        if n_features is None:
            n_features = int(arr.shape[1])
        elif int(arr.shape[1]) != int(n_features):
            raise RuntimeError(
                f"Feature-size mismatch: {path} has {arr.shape[1]} columns, expected {n_features}"
            )

        xs.append(arr)
        ys.extend([label] * arr.shape[0])
        counts[label] = int(arr.shape[0])

    X = np.vstack(xs)
    y = np.array(ys, dtype=str)
    labels_sorted = [label for label, _ in pairs]

    return Dataset(
        X=X,
        y=y,
        labels_sorted=labels_sorted,
        per_label_counts=counts,
        n_features=int(n_features),
    )


def load_model(config: Config):
    with open(config.MODEL_PATH, "rb") as f:
        return pickle.load(f)


def coerce_pred_to_str_labels(y_pred: np.ndarray, labels_sorted: List[str]) -> np.ndarray:
    pred = np.asarray(y_pred)

    if pred.dtype.kind in {"U", "S"}:
        return pred.astype(str)

    out: List[str] = []
    for v in pred.tolist():
        try:
            idx = int(v)
        except (TypeError, ValueError, OverflowError):
            out.append(str(v))
            continue

        if 0 <= idx < len(labels_sorted):
            out.append(labels_sorted[idx])
        else:
            out.append(str(v))

    return np.asarray(out, dtype=str)


def summarize_scores(y_true: np.ndarray, y_pred: np.ndarray, labels_sorted: List[str]) -> str:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_sorted,
        average=None,
        zero_division=0,
    )

    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_sorted,
        average="macro",
        zero_division=0,
    )

    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_sorted,
        average="weighted",
        zero_division=0,
    )

    acc = accuracy_score(y_true, y_pred)

    lines = []
    lines.append(f"Accuracy: {acc:.4f}")
    lines.append(f"Macro avg  (P/R/F1): {macro[0]:.4f} / {macro[1]:.4f} / {macro[2]:.4f}")
    lines.append(f"Weighted   (P/R/F1): {weighted[0]:.4f} / {weighted[1]:.4f} / {weighted[2]:.4f}")
    lines.append("")
    lines.append("Per-class (Precision / Recall / F1 / Support):")
    for i, label in enumerate(labels_sorted):
        lines.append(f"  {label:>2}: {precision[i]:.4f} / {recall[i]:.4f} / {f1[i]:.4f} / {int(support[i])}")
    return "\n".join(lines)


def summarize_dataset_quality(dataset: Dataset, config: Config) -> str:
    """Build a practical dataset-quality report with data collection priorities."""
    target = int(config.TARGET_PER_CLASS)
    counts = dataset.per_label_counts
    min_count = min(counts.values()) if counts else 0
    max_count = max(counts.values()) if counts else 0
    imbalance_ratio = (max_count / min_count) if min_count else float("inf")

    lines = []
    lines.append("Dataset quality:")
    lines.append(f"  Target per class: {target}")
    lines.append(f"  Min/Max samples: {min_count} / {max_count}")
    lines.append(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    lines.append("")
    lines.append("Collection priority:")

    missing = []
    for label in dataset.labels_sorted:
        count = counts.get(label, 0)
        deficit = max(0, target - count)
        if deficit:
            missing.append((deficit, label, count))

    if not missing:
        lines.append("  Semua kelas sudah mencapai target.")
        return "\n".join(lines)

    for deficit, label, count in sorted(missing, reverse=True):
        lines.append(f"  {label:>2}: tambah {deficit:>3} sample (sekarang {count})")
    return "\n".join(lines)


def format_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray, labels_sorted: List[str], top_n: int = 12) -> str:
    """Return the most frequent off-diagonal confusion pairs."""
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    pairs = []
    for row_idx, true_label in enumerate(labels_sorted):
        for col_idx, pred_label in enumerate(labels_sorted):
            if row_idx == col_idx:
                continue
            count = int(cm[row_idx, col_idx])
            if count:
                pairs.append((count, true_label, pred_label))

    lines = ["Top confusion pairs (true -> predicted):"]
    if not pairs:
        lines.append("  Tidak ada confusion pada evaluasi ini.")
        return "\n".join(lines)

    for count, true_label, pred_label in sorted(pairs, reverse=True)[:top_n]:
        lines.append(f"  {true_label:>2} -> {pred_label:<2}: {count}")
    return "\n".join(lines)


def save_report(path: str, sections: List[str]) -> None:
    """Persist evaluation output to a Markdown-like text report."""
    report_path = Path(path)
    report_path.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    print(f"Report saved: {report_path}")


def evaluate_in_sample(model, dataset: Dataset) -> str:
    y_pred_raw = model.predict(dataset.X)
    y_pred = coerce_pred_to_str_labels(y_pred_raw, dataset.labels_sorted)
    section = "\n".join([
        "=" * 80,
        "(1) In-sample evaluation: saved model as-is on full dataset",
        "=" * 80,
        summarize_scores(dataset.y, y_pred, dataset.labels_sorted),
        format_confusion_pairs(dataset.y, y_pred, dataset.labels_sorted),
    ])
    print(section)
    print("")
    return section


def evaluate_holdout_refit(model, dataset: Dataset, test_size: float, seed: int) -> str:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=test_size,
            random_state=seed,
            stratify=dataset.y,
        )
    except ValueError as e:
        print(f"Peringatan holdout stratify dinonaktifkan: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    estimator = clone(model)
    estimator.fit(X_train, y_train)
    y_pred_raw = estimator.predict(X_test)
    y_pred = coerce_pred_to_str_labels(y_pred_raw, dataset.labels_sorted)
    section = "\n".join([
        "=" * 80,
        f"(2) Holdout evaluation: refit cloned model (test_size={test_size}, seed={seed})",
        "=" * 80,
        summarize_scores(y_test, y_pred, dataset.labels_sorted),
        format_confusion_pairs(y_test, y_pred, dataset.labels_sorted),
    ])
    print(section)
    print("")
    return section


def evaluate_crossval(model, dataset: Dataset, cv: int, seed: int) -> str:
    if cv <= 1:
        return ""

    min_class_count = min(dataset.per_label_counts.values()) if dataset.per_label_counts else 0
    if min_class_count < cv:
        section = f"Peringatan: CV dilewati karena n_splits={cv} > minimum sampel per kelas ({min_class_count})."
        print(section)
        print("")
        return section

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    estimator = clone(model)
    y_pred_raw = cross_val_predict(estimator, dataset.X, dataset.y, cv=splitter)
    y_pred = coerce_pred_to_str_labels(y_pred_raw, dataset.labels_sorted)
    cm = confusion_matrix(dataset.y, y_pred, labels=dataset.labels_sorted)
    section = "\n".join([
        "=" * 80,
        f"(3) Cross-validation (cv={cv}, seed={seed}) via cross_val_predict",
        "=" * 80,
        summarize_scores(dataset.y, y_pred, dataset.labels_sorted),
        format_confusion_pairs(dataset.y, y_pred, dataset.labels_sorted),
        "Confusion matrix (rows=true, cols=pred):",
        np.array2string(cm),
    ])
    print(section)
    print("")
    return section


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-in-sample", action="store_true")
    parser.add_argument("--cv", type=int, default=0, help="If >0, run stratified K-fold CV")
    parser.add_argument("--validate-only", action="store_true", help="Only validate dataset shape/counts")
    parser.add_argument("--report", default="", help="Optional path to save evaluation report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail jika ada kelas di bawah TARGET_PER_CLASS",
    )
    args = parser.parse_args()

    config = Config()

    shape_ok, target_ok, validation_report = validate_dataset(config)
    print(validation_report)
    print("")

    validation_ok = shape_ok and (target_ok if args.strict else True)

    if args.strict and not target_ok:
        print("STRICT mode: gagal karena masih ada kelas di bawah TARGET_PER_CLASS.")
        print("")

    if args.validate_only:
        return 0 if validation_ok else 2

    if not validation_ok:
        print("Dataset invalid untuk evaluasi. Perbaiki dulu data_*.csv.")
        return 2

    dataset = load_dataset(config)
    model = load_model(config)

    report_sections = [validation_report]
    quality_report = summarize_dataset_quality(dataset, config)
    print(quality_report)
    print("")
    report_sections.append(quality_report)

    print("Dataset summary:")
    print(f"  Samples: {dataset.X.shape[0]}")
    print(f"  Features per sample: {dataset.n_features}")
    print(f"  Classes: {len(dataset.labels_sorted)} -> {dataset.labels_sorted}")
    print("  Per-class counts:")
    for label in dataset.labels_sorted:
        print(f"    {label:>2}: {dataset.per_label_counts.get(label, 0)}")
    print("")

    if not args.no_in_sample:
        report_sections.append(evaluate_in_sample(model, dataset))

    report_sections.append(evaluate_holdout_refit(model, dataset, test_size=float(args.test_size), seed=int(args.seed)))

    if int(args.cv) > 1:
        cv_section = evaluate_crossval(model, dataset, cv=int(args.cv), seed=int(args.seed))
        if cv_section:
            report_sections.append(cv_section)

    if args.report:
        save_report(args.report, report_sections)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
