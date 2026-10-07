from fastapi.testclient import TestClient


def test_health(test_app):
    # endpoint سلامت باید بدون خطا پاسخ دهد و وجود مدل را تأیید کند.
    response = TestClient(test_app).get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_valid_prediction(test_app):
    # یک درخواست معتبر باید برچسب، confidence و نسخه مدل را برگرداند.
    response = TestClient(test_app).post("/predict", json={"text": "Congratulations! You won a prize."})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"ham", "spam"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "v1"


def test_empty_text(test_app):
    # متن خالی یا فقط شامل فاصله نباید وارد مرحله inference شود.
    response = TestClient(test_app).post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_invalid_input(test_app):
    # نوع اشتباه برای text باید توسط validation رد شود.
    response = TestClient(test_app).post("/predict", json={"text": 123})
    assert response.status_code == 422

    # وجود فیلد ناشناخته به جای text نیز باید درخواست را نامعتبر کند.
    response = TestClient(test_app).post("/predict", json={"wrong": "field"})
    assert response.status_code == 422

    # نبودن فیلد اجباری text باید خطای validation ایجاد کند.
    response = TestClient(test_app).post("/predict", json={})
    assert response.status_code == 422


def test_malformed_request(test_app):
    # JSON خراب نباید باعث نمایش traceback داخلی برنامه به کاربر شود.
    response = TestClient(test_app).post(
        "/predict",
        content="{not-valid-json}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "traceback" not in response.text.lower()


def test_model_loaded(test_app):
    # endpoint سلامت باید مشخص کند که مدل برای inference در دسترس است.
    response = TestClient(test_app).get("/health")
    assert response.json()["model_loaded"] is True


def test_extremely_long_text(test_app):
    # متن بسیار طولانی باید قبل از رسیدن به مدل محدود و رد شود.
    response = TestClient(test_app).post("/predict", json={"text": "x" * 5001})
    assert response.status_code == 422
