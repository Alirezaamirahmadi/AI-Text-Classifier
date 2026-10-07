from __future__ import annotations

from pathlib import Path

import torch

from .config import MAX_TEXT_LENGTH, MODEL_DIR
from .dl_dataset import Vocabulary, tokenize_text
from .dl_model import BiLSTMClassifier

CHECKPOINT_PATH = MODEL_DIR / "text_classifier_pytorch_v1.pt"


def load_pytorch_model(checkpoint_path: Path = CHECKPOINT_PATH):
    """It loads the saved PyTorch model without retraining."""
    # فایل checkpoint باید قبل از inference وجود داشته باشد.
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"PyTorch model artifact not found: {checkpoint_path}. Run `python -m src.dl_train` first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BiLSTMClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    vocabulary = Vocabulary(token_to_idx=checkpoint["vocabulary"])
    return model, vocabulary, checkpoint


def predict_dl(text: str, model=None, vocabulary: Vocabulary | None = None, checkpoint=None) -> dict:
    """برای یک متن بدون اجرای آموزش پیش‌بینی PyTorch انجام می‌دهد."""
    # نوع ورودی باید رشته باشد.
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # همان محدودیت API برای مدل PyTorch نیز اعمال می‌شود.
    if not text.strip():
        raise ValueError("text must not be empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"text exceeds maximum length of {MAX_TEXT_LENGTH} characters")

    if model is None or vocabulary is None or checkpoint is None:
        model, vocabulary, checkpoint = load_pytorch_model()

    # متن با همان normalize و tokenizer استفاده‌شده در آموزش به sequence تبدیل می‌شود.
    token_ids = vocabulary.encode(tokenize_text(text))
    if not token_ids:
        token_ids = [1]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, lengths)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_index].item())

    index_to_label = {int(index): label for label, index in checkpoint["label_mapping"].items()}
    return {
        "label": index_to_label[predicted_index],
        "confidence": round(confidence, 4),
        "model_version": "pytorch_v1",
    }