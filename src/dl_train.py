from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .preprocessing import build_model

from .config import DATASET_NAME, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, RAW_DATA_PATH
from .data import load_raw_data
from .dl_dataset import SMSDataset, build_vocabulary, collate_batch
from .dl_model import BiLSTMClassifier
from .preprocessing import clean_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)

CHECKPOINT_PATH = MODEL_DIR / "text_classifier_pytorch_v1.pt"
MAX_EPOCHS = 7
PATIENCE = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
MIN_FREQ = 1


def set_reproducible_seed(seed: int = RANDOM_STATE) -> None:
    # seedهای Python، NumPy و PyTorch ثابت می‌شوند تا اجرای آموزش قابل بازتولید باشد.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # در صورت وجود CUDA، تنظیمات deterministic برای کاهش تفاوت بین اجراها فعال می‌شوند.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_data(df):
    # ابتدا 70 درصد train و 30 درصد temporary با stratification جدا می‌شود.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    # temporary به دو بخش مساوی validation و test تقسیم می‌شود تا هرکدام 15 درصد کل داده باشند.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_loader(texts, labels, vocabulary, shuffle: bool) -> DataLoader:
    dataset = SMSDataset(texts, labels, vocabulary)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=0,
    )


def calculate_class_weights(labels: list[int]) -> torch.Tensor:
    # وزن کلاس‌ها فقط از train محاسبه می‌شود تا اطلاعات validation و test وارد آموزش نشود.
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    # حالت مدل بر اساس مرحله آموزش یا validation تنظیم می‌شود.
    model.train() if training else model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        if training:
            # gradientهای batch قبلی قبل از محاسبه forward پاک می‌شوند.
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            if training:
                # loss به gradient تبدیل می‌شود و سپس وزن‌های مدل با optimizer به‌روزرسانی می‌شوند.
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=0)
    return average_loss, float(f1)


def evaluate_model(model, loader, device):
    # مدل نهایی بدون gradient روی test اجرا می‌شود.
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            logits = model(input_ids.to(device), lengths.to(device))
            all_labels.extend(labels.tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "precision": float(precision_score(all_labels, all_predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(all_labels, all_predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(all_labels, all_predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions, labels=[0, 1]).tolist(),
    }


def main() -> None:
    set_reproducible_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    raw = load_raw_data(RAW_DATA_PATH)
    df = clean_dataframe(raw)
    if len(df) < 5000:
        raise ValueError(f"Dataset has only {len(df)} valid rows; at least 5000 are required.")
    LOGGER.info("Dataset size: raw=%d valid=%d", len(raw), len(df))

    train_df, val_df, test_df = split_data(df)
    label_mapping = {"ham": 0, "spam": 1}
    train_labels = [label_mapping[label] for label in train_df["label"]]
    val_labels = [label_mapping[label] for label in val_df["label"]]
    test_labels = [label_mapping[label] for label in test_df["label"]]

    # واژگان فقط از train ساخته می‌شود و validation/test هیچ token جدیدی به آن اضافه نمی‌کنند.
    vocabulary = build_vocabulary(train_df["text"], min_freq=MIN_FREQ)
    LOGGER.info("Vocabulary size: %d", vocabulary.size)

    train_loader = make_loader(train_df["text"], train_labels, vocabulary, shuffle=True)
    val_loader = make_loader(val_df["text"], val_labels, vocabulary, shuffle=False)
    test_loader = make_loader(test_df["text"], test_labels, vocabulary, shuffle=False)

    model_config = {
        "vocab_size": vocabulary.size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "pad_idx": 0,
        "num_classes": 2,
    }
    model = BiLSTMClassifier(**model_config).to(device)
    class_weights = calculate_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, training=False)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_f1": float(train_f1),
            "validation_loss": float(val_loss),
            "validation_f1": float(val_f1),
        }
        history.append(epoch_record)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f",
            epoch,
            MAX_EPOCHS,
            train_loss,
            val_loss,
            val_f1,
        )

        # انتخاب checkpoint فقط بر اساس validation F1 انجام می‌شود، نه training loss.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "label_mapping": label_mapping,
                "vocabulary": vocabulary.token_to_idx,
                "model_config": model_config,
                "training_config": {
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "max_epochs": MAX_EPOCHS,
                    "patience": PATIENCE,
                    "min_freq": MIN_FREQ,
                    "random_state": RANDOM_STATE,
                },
                "best_validation_f1": float(best_val_f1),
                "history": history,
                "dataset": DATASET_NAME,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                LOGGER.info("Early stopping triggered after %d epochs without validation F1 improvement", PATIENCE)
                break

    # بهترین checkpoint بر اساس validation F1 دوباره بارگذاری می‌شود و فقط این مدل روی test ارزیابی می‌شود.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device)

    # برای مقایسه منصفانه، یک مدل کلاسیک جداگانه روی همان train و همان test این مرحله آموزش داده می‌شود.
    # مدل کلاسیک v1 ذخیره‌شده دست‌نخورده باقی می‌ماند و برای این مقایسه بازنویسی نمی‌شود.
    classic_model = build_model()
    classic_model.fit(train_df["text"], train_labels)
    classic_predictions = classic_model.predict(test_df["text"])
    classic_metrics = {
        "accuracy": float(accuracy_score(test_labels, classic_predictions)),
        "precision": float(precision_score(test_labels, classic_predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(test_labels, classic_predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(test_labels, classic_predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(test_labels, classic_predictions, labels=[0, 1]).tolist(),
    }

    metrics = {
        "model_version": "pytorch_v1",
        "model_type": "Embedding + BiLSTM + Dropout + Linear",
        "dataset": DATASET_NAME,
        "dataset_rows_raw": int(len(raw)),
        "dataset_rows_valid": int(len(df)),
        "split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "vocab_size": vocabulary.size,
        "label_mapping": label_mapping,
        "device": str(device),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_f1": float(checkpoint["best_validation_f1"]),
        "history": history,
        "test": test_metrics,
    }
    comparison = {
        "evaluation_split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "tfidf_logistic_regression": classic_metrics,
        "pytorch_bilstm": test_metrics,
        "note": "The persisted classic v1 model is preserved. The TF-IDF comparison model is trained separately on the same split for a fair test-set comparison.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "pytorch_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    LOGGER.info("Test metrics: %s", test_metrics)
    LOGGER.info("TF-IDF comparison metrics: %s", classic_metrics)
    LOGGER.info("PyTorch checkpoint saved to %s", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
