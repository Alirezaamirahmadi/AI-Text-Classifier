from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .config import DATASET_NAME, MODEL_DIR, OUTPUT_DIR, RAW_DATA_PATH
from .data import load_raw_data
from .dl_predict import load_pytorch_model, predict_dl
from .preprocessing import build_model, clean_dataframe
from .transformer_predict import load_transformer_model, predict_transformer
from .transformer_utils import (
    LABEL_MAPPING,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_DIR,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_GRADIENT_ACCUMULATION_STEPS,
    TRANSFORMER_LEARNING_RATE,
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_PATIENCE,
    TRANSFORMER_VERSION,
    TRANSFORMER_WEIGHT_DECAY,
    classification_metrics,
    set_transformer_seed,
    split_transformer_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def build_hf_dataset(df, tokenizer) -> HFDataset:
    # Hugging Face Dataset از متن و label ساخته می‌شود و tokenization روی همین Dataset انجام می‌گیرد.
    dataset = HFDataset.from_dict(
        {
            "text": df["text"].tolist(),
            "labels": [LABEL_MAPPING[label] for label in df["label"].tolist()],
        }
    )

    def tokenize_batch(batch):
        # tokenizer مربوط به مدل، متن را به input_ids و attention_mask تبدیل می‌کند.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=TRANSFORMER_MAX_LENGTH,
            padding=False,
        )

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])


def evaluate_epoch(model, loader, device):
    # در validation و test فقط inference انجام می‌شود و gradient ساخته نمی‌شود.
    model.eval()
    labels = []
    predictions = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.item()))
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    # حلقه آموزش شامل forward، loss، backward، optimizer step و zero_grad است.
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / TRANSFORMER_GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        if (step + 1) % TRANSFORMER_GRADIENT_ACCUMULATION_STEPS == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(outputs.loss.item())
    return total_loss / max(len(loader), 1)


def save_best_checkpoint(model, tokenizer, optimizer, epoch, validation_metrics, history):
    # بهترین مدل بر اساس validation F1 روی دیسک ذخیره می‌شود و برای inference آماده است.
    TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(TRANSFORMER_DIR, safe_serialization=True)
    tokenizer.save_pretrained(TRANSFORMER_DIR)
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_f1": validation_metrics["f1"],
            "history": history,
            "training_config": {
                "model_name": TRANSFORMER_MODEL_NAME,
                "learning_rate": TRANSFORMER_LEARNING_RATE,
                "batch_size": TRANSFORMER_BATCH_SIZE,
                "num_train_epochs": TRANSFORMER_EPOCHS,
                "weight_decay": TRANSFORMER_WEIGHT_DECAY,
                "max_length": TRANSFORMER_MAX_LENGTH,
                "patience": TRANSFORMER_PATIENCE,
                "random_state": 42,
                "gradient_accumulation_steps": TRANSFORMER_GRADIENT_ACCUMULATION_STEPS,
            },
            "label_mapping": LABEL_MAPPING,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        TRANSFORMER_DIR / "training_state.pt",
    )


def directory_size(path: Path) -> int:
    # artifact پوشه‌ای به مجموع اندازه فایل‌های ذخیره‌شده تبدیل می‌شود.
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def measure_classic_latency(texts):
    # زمان inference مدل کلاسیک روی همان متن‌های ثابت اندازه‌گیری می‌شود.
    from .predict import load_model, predict
    model = load_model()
    start = time.perf_counter()
    for text in texts:
        predict(text, model=model)
    return float((time.perf_counter() - start) / len(texts))


def measure_bilstm_latency(texts):
    # زمان inference مدل BiLSTM روی همان متن‌های ثابت اندازه‌گیری می‌شود.
    model, vocabulary, checkpoint = load_pytorch_model()
    start = time.perf_counter()
    for text in texts:
        predict_dl(text, model=model, vocabulary=vocabulary, checkpoint=checkpoint)
    return float((time.perf_counter() - start) / len(texts))


def main() -> None:
    set_transformer_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Transformer device: %s", device)
    LOGGER.info("Transformer memory settings: batch_size=%d max_length=%d", TRANSFORMER_BATCH_SIZE, TRANSFORMER_MAX_LENGTH)

    raw = load_raw_data(RAW_DATA_PATH)
    df = clean_dataframe(raw)
    LOGGER.info("Dataset size: raw=%d valid=%d", len(raw), len(df))
    train_df, val_df, test_df = split_transformer_data(df)
    LOGGER.info("Split sizes: train=%d validation=%d test=%d", len(train_df), len(val_df), len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME,
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1},
    )
    model.to(device)

    train_dataset = build_hf_dataset(train_df, tokenizer)
    val_dataset = build_hf_dataset(val_df, tokenizer)
    test_dataset = build_hf_dataset(test_df, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=TRANSFORMER_LEARNING_RATE, weight_decay=TRANSFORMER_WEIGHT_DECAY)
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, TRANSFORMER_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(val_metrics["loss"]),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
        }
        history.append(record)
        LOGGER.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f", epoch, TRANSFORMER_EPOCHS, train_loss, val_metrics["loss"], val_metrics["f1"])
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            save_best_checkpoint(model, tokenizer, optimizer, epoch, val_metrics, history)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= TRANSFORMER_PATIENCE:
                LOGGER.info("Early stopping triggered after %d epochs without validation F1 improvement", TRANSFORMER_PATIENCE)
                break

    # بهترین checkpoint بر اساس validation F1 بارگذاری می‌شود؛ test فقط پس از پایان آموزش یک بار دیده می‌شود.
    best_model, best_tokenizer, best_device = load_transformer_model()
    test_metrics = evaluate_epoch(best_model, test_loader, best_device)
    test_predictions = []
    test_labels = []
    best_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(best_device) for key, value in batch.items()}
            outputs = best_model(**batch)
            test_predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            test_labels.extend(batch["labels"].cpu().tolist())

    report = classification_report(test_labels, test_predictions, labels=[0, 1], target_names=["ham", "spam"], output_dict=True, zero_division=0)
    errors = []
    for text, actual, predicted in zip(test_df["text"].tolist(), test_labels, test_predictions):
        if actual != predicted:
            result = predict_transformer(text, model=best_model, tokenizer=best_tokenizer, device=best_device)
            errors.append({
                "text": text,
                "actual_label": "spam" if actual == 1 else "ham",
                "predicted_label": "spam" if predicted == 1 else "ham",
                "confidence": result["confidence"],
            })
    if len(errors) < 10:
        LOGGER.warning(
    "Transformer produced only %s test errors; fewer than 10 real misclassifications are available for error analysis.",
    len(errors),
)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transformer_metrics = {
        "model_version": TRANSFORMER_VERSION,
        "model_type": "DistilBERT fine-tuned for sequence classification",
        "pretrained_model": TRANSFORMER_MODEL_NAME,
        "dataset": DATASET_NAME,
        "dataset_rows_raw": len(raw),
        "dataset_rows_valid": len(df),
        "split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "device": str(best_device),
        "hyperparameters": {
            "learning_rate": TRANSFORMER_LEARNING_RATE,
            "batch_size": TRANSFORMER_BATCH_SIZE,
            "num_train_epochs": TRANSFORMER_EPOCHS,
            "weight_decay": TRANSFORMER_WEIGHT_DECAY,
            "max_length": TRANSFORMER_MAX_LENGTH,
            "early_stopping_patience": TRANSFORMER_PATIENCE,
            "gradient_accumulation_steps": TRANSFORMER_GRADIENT_ACCUMULATION_STEPS,
        },
        "best_epoch": int(torch.load(TRANSFORMER_DIR / "training_state.pt", map_location="cpu", weights_only=False)["epoch"]),
        "best_validation_f1": float(best_val_f1),
        "history": history,
        "test": test_metrics,
        "spam_classification": report["spam"],
        "model_size_bytes": directory_size(TRANSFORMER_DIR),
    }
    (OUTPUT_DIR / "transformer_metrics.json").write_text(json.dumps(transformer_metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "transformer_errors.json").write_text(json.dumps(errors[:10], indent=2, ensure_ascii=False), encoding="utf-8")

    sample_texts = test_df["text"].head(50).tolist()
    classic_latency = measure_classic_latency(sample_texts)
    bilstm_latency = measure_bilstm_latency(sample_texts)
    # زمان Transformer با perf_counter جداگانه اندازه‌گیری می‌شود تا فقط inference محاسبه شود.
    start = time.perf_counter()
    for text in sample_texts:
        predict_transformer(text, model=best_model, tokenizer=best_tokenizer, device=best_device)
    if best_device.type == "cuda":
        torch.cuda.synchronize()
    transformer_latency = float((time.perf_counter() - start) / len(sample_texts))

    comparison = {
        "evaluation_split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "models": {
            "tfidf_logistic_regression": {
                "accuracy": 0.979381443298969,
                "precision": 0.9270833333333334,
                "recall": 0.9081632653061225,
                "f1": 0.9175257731958762,
                "inference_time_seconds_per_sample": classic_latency,
                "model_size_bytes": (MODEL_DIR / "text_classifier_v1.joblib").stat().st_size,
            },
            "pytorch_bilstm": {
                "accuracy": 0.9755154639175257,
                "precision": 0.8910891089108911,
                "recall": 0.9183673469387755,
                "f1": 0.9045226130653267,
                "inference_time_seconds_per_sample": bilstm_latency,
                "model_size_bytes": (MODEL_DIR / "text_classifier_pytorch_v1.pt").stat().st_size,
            },
            "transformer": {
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "inference_time_seconds_per_sample": transformer_latency,
                "model_size_bytes": directory_size(TRANSFORMER_DIR),
            },
        },
        "note": "The Transformer is evaluated once on the held-out test set after training and validation-based model selection.",
    }
    (OUTPUT_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    LOGGER.info("Transformer test metrics: %s", test_metrics)
    LOGGER.info("Transformer checkpoint saved to %s", TRANSFORMER_DIR)


if __name__ == "__main__":
    main()