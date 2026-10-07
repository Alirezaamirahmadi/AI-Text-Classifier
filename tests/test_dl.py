from pathlib import Path

import torch
from fastapi.testclient import TestClient

from src.api import create_app
from src.dl_dataset import PAD_IDX, UNK_IDX, SMSDataset, build_vocabulary, collate_batch, tokenize_text
from src.dl_model import BiLSTMClassifier
from src.dl_predict import CHECKPOINT_PATH, load_pytorch_model, predict_dl


def test_tokenization_and_vocabulary_are_train_only():
    # واژگان فقط باید tokenهای متن‌های train را داشته باشند.
    train_texts = ["hello world", "hello friend"]
    vocabulary = build_vocabulary(train_texts)
    assert vocabulary.token_to_idx["<PAD>"] == PAD_IDX
    assert vocabulary.token_to_idx["<UNK>"] == UNK_IDX
    assert "hello" in vocabulary.token_to_idx
    assert "secret" not in vocabulary.token_to_idx
    assert tokenize_text("Hello, world!") == ["hello", ",", "world", "!"]


def test_pytorch_dataset_and_collate():
    # Dataset باید sequence و label tensor واقعی PyTorch تولید کند.
    vocabulary = build_vocabulary(["hello world", "free prize"])
    dataset = SMSDataset(["hello world", "free prize"], [0, 1], vocabulary)
    sequence, label = dataset[0]
    assert isinstance(sequence, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    padded, lengths, labels = collate_batch([dataset[0], dataset[1]])
    assert padded.shape[0] == 2
    assert lengths.tolist() == [2, 2]
    assert labels.tolist() == [0, 1]


def test_bilstm_forward_shape():
    # خروجی مدل باید برای دو کلاس logits با شکل batch_size در دو کلاس داشته باشد.
    model = BiLSTMClassifier(vocab_size=20, embedding_dim=8, hidden_dim=8, dropout=0.1)
    input_ids = torch.tensor([[2, 3, 0], [4, 5, 6]], dtype=torch.long)
    lengths = torch.tensor([2, 3], dtype=torch.long)
    logits = model(input_ids, lengths)
    assert logits.shape == (2, 2)


def test_pytorch_checkpoint_exists_and_has_required_fields():
    # checkpoint باید artifact دائمی مدل را نگه دارد و فقط به RAM وابسته نباشد.
    assert Path(CHECKPOINT_PATH).exists()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "epoch" in checkpoint
    assert "label_mapping" in checkpoint
    assert "vocabulary" in checkpoint


def test_pytorch_inference():
    # مدل ذخیره‌شده باید بدون آموزش مجدد inference انجام دهد.
    model, vocabulary, checkpoint = load_pytorch_model()
    result = predict_dl("Congratulations! You won a prize.", model, vocabulary, checkpoint)
    assert result["label"] in {"ham", "spam"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_version"] == "pytorch_v1"


def test_pytorch_api_endpoint():
    # endpoint جدید باید از مدل PyTorch ذخیره‌شده پاسخ معتبر بگیرد.
    app = create_app(dl_bundle=load_pytorch_model())
    response = TestClient(app).post("/predict/dl", json={"text": "Congratulations! You won a prize."})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"ham", "spam"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "pytorch_v1"