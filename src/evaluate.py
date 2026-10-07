from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true, y_pred, output_dir: Path) -> dict:
    """Compute binary classification metrics and persist evaluation artifacts."""
    # تمام معیارها فقط روی پیش‌بینی‌های داده test محاسبه می‌شوند.
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="spam", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    # پوشه خروجی در صورت نبودن ساخته می‌شود تا اجرای اول هم بدون خطا باشد.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ماتریس درهم‌ریختگی با ترتیب ثابت کلاس‌ها ذخیره می‌شود.
    matrix = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    pd.DataFrame(matrix, index=["ham", "spam"], columns=["ham", "spam"]).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    return metrics
