from pathlib import Path

from fastapi.testclient import TestClient

from src.api import create_app
from src.transformer_predict import load_transformer_model, predict_transformer
from src.transformer_utils import TRANSFORMER_DIR, TRANSFORMER_VERSION


def test_transformer_model_loads():
    # مدل Fine-tuned باید از artifact ذخیره‌شده بدون آموزش دوباره بارگذاری شود.
    model, tokenizer, device = load_transformer_model()
    assert model.config.num_labels == 2
    assert tokenizer is not None
    assert device.type in {"cpu", "cuda"}


def test_transformer_tokenizer_loads():
    # tokenizer باید به‌صورت artifact محلی همراه مدل ذخیره شده باشد.
    tokenizer = load_transformer_model()[1]
    encoded = tokenizer("hello world", truncation=True, max_length=64)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded


def test_transformer_prediction():
    # inference باید از متن خام تا label و confidence را بدون training انجام دهد.
    model, tokenizer, device = load_transformer_model()
    result = predict_transformer("Congratulations! You won a free prize.", model, tokenizer, device)
    assert result["label"] in {"ham", "spam"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_version"] == TRANSFORMER_VERSION


def test_transformer_prediction_confidence():
    # confidence باید یک مقدار احتمال معتبر باشد.
    model, tokenizer, device = load_transformer_model()
    result = predict_transformer("Please call me when you arrive.", model, tokenizer, device)
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_transformer_api_endpoint():
    # endpoint جدید باید از artifact موجود پاسخ معتبر دریافت کند.
    bundle = load_transformer_model()
    app = create_app(transformer_bundle=bundle)
    response = TestClient(app).post("/predict/transformer", json={"text": "Congratulations! You won a prize."})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"ham", "spam"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == TRANSFORMER_VERSION


def test_transformer_invalid_input():
    # ورودی خالی و متن بیش از حد مجاز باید قبل از inference رد شوند.
    app = create_app(transformer_bundle=load_transformer_model())
    client = TestClient(app)
    assert client.post("/predict/transformer", json={"text": "   "}).status_code == 422
    assert client.post("/predict/transformer", json={"text": "x" * 5001}).status_code == 422


def test_health_all_models():
    # health باید وضعیت هر سه مدل را به‌صورت جداگانه گزارش کند.
    app = create_app(
        model=object(),
        dl_bundle=object(),
        transformer_bundle=object(),
    )
    body = TestClient(app).get("/health").json()
    assert body["classic_model_loaded"] is True
    assert body["pytorch_model_loaded"] is True
    assert body["transformer_model_loaded"] is True


def test_transformer_artifact_contains_model_and_tokenizer_files():
    # artifact باید config، وزن مدل و فایل‌های tokenizer را روی دیسک داشته باشد.
    assert (TRANSFORMER_DIR / "config.json").exists()
    assert any(path.name.startswith("model") and path.suffix in {".safetensors", ".bin"} for path in TRANSFORMER_DIR.iterdir())
    assert (TRANSFORMER_DIR / "tokenizer_config.json").exists()
