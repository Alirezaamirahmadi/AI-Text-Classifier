from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE

TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_VERSION = "transformer_v1"
TRANSFORMER_DIR = Path(__file__).resolve().parent.parent / "models" / TRANSFORMER_VERSION
TRANSFORMER_MAX_LENGTH = 64
TRANSFORMER_BATCH_SIZE = 4
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_EPOCHS = 2
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_PATIENCE = 1
TRANSFORMER_GRADIENT_ACCUMULATION_STEPS = 1
LABEL_MAPPING = {"ham": 0, "spam": 1}


def set_transformer_seed(seed: int = RANDOM_STATE) -> None:
    # seedهای Python، NumPy و PyTorch ثابت می‌شوند تا اجرای آموزش تا حد ممکن قابل بازتولید باشد.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # در صورت وجود CUDA، تنظیمات deterministic برای کاهش تفاوت بین اجراها فعال می‌شوند.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_transformer_data(df):
    # همان split ثابت مرحله PyTorch با stratification استفاده می‌شود تا مقایسه مدل‌ها منصفانه باشد.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_STATE
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def classification_metrics(labels, predictions) -> dict:
    # معیارهای اصلی با spam به‌عنوان کلاس مثبت محاسبه می‌شوند.
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
