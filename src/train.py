from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (
    DATASET_NAME,
    MODEL_DIR,
    MODEL_PATH,
    METADATA_PATH,
    MODEL_VERSION,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from .data import load_raw_data
from .evaluate import evaluate_predictions
from .preprocessing import build_model, clean_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    # مسیر دیتاست از config گرفته می‌شود و برنامه هیچ فایل خارجی را به‌صورت خودکار دریافت نمی‌کند.
    # دیتاست فقط از فایل محلی داخل پوشه data خوانده می‌شود و هیچ دانلود خودکاری انجام نمی‌شود.
    raw = load_raw_data()
    # داده‌های ناقص، نامعتبر و تکراری قبل از تقسیم داده پاک‌سازی می‌شوند.
    df = clean_dataframe(raw)
    if len(df) < 5000:
        raise ValueError(f"Dataset has only {len(df)} valid rows; at least 5000 are required.")

    # ابتدا داده به train و test تقسیم می‌شود تا از data leakage جلوگیری شود.
    x_train, x_test, y_train_raw, y_test_raw = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    # انکودر برچسب فقط روی داده‌های آموزش fit می‌شود تا اطلاعات test وارد مرحله آموزش نشود.
    encoder = LabelEncoder().fit(y_train_raw)
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    label_names = list(encoder.classes_)

    # baseline با پرتکرارترین کلاس ساخته می‌شود و روی test دست‌نخورده ارزیابی می‌شود.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train.to_frame(), y_train)
    baseline_pred = baseline.predict(x_test.to_frame())
    spam_class = encoder.transform(["spam"])[0]
    baseline_metrics = {
        "accuracy": float(accuracy_score(y_test, baseline_pred)),
        "precision": float(precision_score(y_test, baseline_pred, pos_label=spam_class, zero_division=0)),
        "recall": float(recall_score(y_test, baseline_pred, pos_label=spam_class, zero_division=0)),
        "f1": float(f1_score(y_test, baseline_pred, pos_label=spam_class, zero_division=0)),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")

    # vocabulary مربوط به TF-IDF فقط روی x_train ساخته می‌شود تا data leakage رخ ندهد.
    model = build_model()
    model.fit(x_train, y_train)
    # انکودر fit شده همراه مدل ذخیره می‌شود تا inference بتواند برچسب را decode کند.
    model.label_encoder_ = encoder
    y_pred = encoder.inverse_transform(model.predict(x_test))
    metrics = evaluate_predictions(y_test_raw, y_pred, OUTPUT_DIR)

    # مدل آموزش‌دیده به‌صورت artifact روی دیسک ذخیره می‌شود تا API بعداً همان مدل را بارگذاری کند.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "TF-IDF + Logistic Regression",
        "dataset": DATASET_NAME,
        "dataset_rows_raw": int(len(raw)),
        "dataset_rows_valid": int(len(df)),
        "features_vectorizer": "TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, strip_accents=unicode)",
        "classifier": "LogisticRegression(max_iter=1000, class_weight=balanced)",
        "label_encoding": label_names,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "baseline": baseline_metrics,
        "metrics": metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Model saved to %s", MODEL_PATH)
    LOGGER.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
