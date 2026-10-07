from __future__ import annotations

from pathlib import Path

import joblib

from .config import MAX_TEXT_LENGTH, MODEL_PATH, MODEL_VERSION
from .preprocessing import normalize_text


def load_model(model_path: Path = MODEL_PATH):
    """Load the persisted model artifact without retraining."""
    # قبل از بارگذاری، وجود فایل مدل بررسی می‌شود.
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run `python -m src.train` first.")
    return joblib.load(model_path)


def predict(text: str, model=None) -> dict:
    """Run inference on one text without retraining the model."""
    # نوع ورودی باید رشته باشد.
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # همان نرمال‌سازی زمان آموزش برای ورودی inference اجرا می‌شود.
    normalized = normalize_text(text)
    # متن خالی نباید وارد مدل شود.
    if not normalized:
        raise ValueError("text must not be empty")
    # برای جلوگیری از ورودی‌های غیرعادی، طول متن محدود شده است.
    if len(normalized) > MAX_TEXT_LENGTH:
        raise ValueError(f"text exceeds maximum length of {MAX_TEXT_LENGTH} characters")
    # در API مدل از قبل بارگذاری شده است و در استفاده مستقیم، فایل ذخیره‌شده خوانده می‌شود.
    model = model or load_model()
    # احتمال هر کلاس برای محاسبه confidence از مدل دریافت می‌شود.
    probabilities = model.predict_proba([normalized])[0]
    classes = list(model.classes_)
    best_index = int(probabilities.argmax())
    predicted_class = classes[best_index]
    # در صورت استفاده از LabelEncoder، کلاس عددی به برچسب اصلی برگردانده می‌شود.
    if hasattr(model, "label_encoder_"):
        predicted_class = model.label_encoder_.inverse_transform([predicted_class])[0]
    return {
        "label": str(predicted_class),
        "confidence": round(float(probabilities[best_index]), 4),
        "model_version": MODEL_VERSION,
    }
