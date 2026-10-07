from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MAX_TEXT_LENGTH
from .transformer_utils import TRANSFORMER_DIR, TRANSFORMER_MAX_LENGTH, TRANSFORMER_VERSION


def load_transformer_model(model_dir: Path = TRANSFORMER_DIR):
    # مدل و tokenizer فقط از artifact ذخیره‌شده خوانده می‌شوند و هیچ آموزشی انجام نمی‌شود.
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Transformer model artifact not found: {model_dir}. Run `python -m src.transformer_train` first."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_transformer(text: str, model=None, tokenizer=None, device=None) -> dict:
    # ورودی باید رشته و در محدوده طول API باشد.
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"text exceeds maximum length of {MAX_TEXT_LENGTH} characters")

    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_transformer_model()

    # متن خام مستقیماً با tokenizer همان مدل به input_ids و attention_mask تبدیل می‌شود.
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=TRANSFORMER_MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_index = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_index].item())

    id2label = getattr(model.config, "id2label", {}) or {}
    label = id2label.get(predicted_index, {0: "ham", 1: "spam"}[predicted_index])
    return {
        "label": label.lower(),
        "confidence": round(confidence, 4),
        "model_version": TRANSFORMER_VERSION,
    }


def measure_transformer_latency(texts, model=None, tokenizer=None, device=None) -> float:
    # میانگین زمان inference برای چند متن ثابت و قابل تکرار اندازه‌گیری می‌شود.
    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_transformer_model()
    start = time.perf_counter()
    for text in texts:
        predict_transformer(text, model=model, tokenizer=tokenizer, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return float(elapsed / max(len(texts), 1))
