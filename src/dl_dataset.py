from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .preprocessing import normalize_text

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def tokenize_text(text: str) -> list[str]:
    """It converts the normalized text into tokens using a simple and understandable tokenizer."""
    # همان نرمال‌سازی مرحله قبلی حفظ می‌شود تا ورودی آموزش و inference یکسان باشد.
    normalized = normalize_text(text)
    # حروف، عدد و نشانه‌های غیر فاصله به‌صورت token جدا می‌شوند.
    return re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE)


@dataclass
class Vocabulary:
    token_to_idx: dict[str, int]

    @property
    def idx_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_idx.items()}

    @property
    def size(self) -> int:
        return len(self.token_to_idx)

    def encode(self, tokens: list[str]) -> list[int]:
        # tokenهای خارج از واژگان با UNK جایگزین می‌شوند.
        return [self.token_to_idx.get(token, UNK_IDX) for token in tokens]


def build_vocabulary(texts, min_freq: int = 1) -> Vocabulary:
    """It builds the vocabulary solely from the training texts."""
    # شمارش tokenها فقط روی داده‌ای انجام می‌شود که برای آموزش در نظر گرفته شده است.
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))

    # PAD و UNK باید index ثابت داشته باشند تا padding و token ناشناخته قابل مدیریت باشند.
    token_to_idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    # ترتیب واژه‌های واقعی قطعی می‌شود تا artifact آموزش قابل بازتولید باشد.
    for token, frequency in sorted(counter.items()):
        if frequency >= min_freq and token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)
    return Vocabulary(token_to_idx=token_to_idx)


class SMSDataset(Dataset):
    """A real PyTorch dataset for converting text into sequence and label tensors."""

    def __init__(self, texts, labels, vocabulary: Vocabulary):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        # هر متن هنگام دریافت به token و سپس به indexهای واژگان تبدیل می‌شود.
        token_ids = self.vocabulary.encode(tokenize_text(self.texts[index]))
        # حتی متن‌های بسیار کوتاه باید حداقل یک token قابل استفاده داشته باشند.
        if not token_ids:
            token_ids = [UNK_IDX]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


def collate_batch(batch):
    """It converts a variable-length batch into a fixed-size tensor using padding."""
    sequences, labels = zip(*batch)
    # طول واقعی هر sequence برای pack_padded_sequence نگه داشته می‌شود.
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    # padding فقط درون batch انجام می‌شود و مقدار آن index مربوط به PAD است.
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths, torch.stack(labels)