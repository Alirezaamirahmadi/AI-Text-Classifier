from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    """Embedding, BiLSTM, Dropout, and Linear for binary classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_classes: int = 2,
    ):
        super().__init__()
        # embedding برای تبدیل indexهای واژگان به بردارهای قابل یادگیری استفاده می‌شود.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # LSTM دوطرفه هم context قبل و هم context بعد از هر token را یاد می‌گیرد.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # dropout برای کاهش overfitting قبل از لایه خروجی اعمال می‌شود.
        self.dropout = nn.Dropout(dropout)
        # دو جهت LSTM در خروجی به یک بردار با اندازه دو برابر hidden_dim تبدیل می‌شوند.
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # token indexها ابتدا به embedding تبدیل می‌شوند.
        embedded = self.embedding(input_ids)
        # sequenceهای paddingشده با طول واقعی به LSTM داده می‌شوند تا PAD در محاسبه توالی اثر نگذارد.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        # آخرین hidden state هر دو جهت به یک representation مشترک متصل می‌شود.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        features = torch.cat((forward_hidden, backward_hidden), dim=1)
        # قبل از classifier dropout اعمال می‌شود.
        features = self.dropout(features)
        return self.classifier(features)