from src.predict import predict


def test_inference_function_returns_prediction(test_app):
    # مدل داخل برنامه آزمایشی از route مربوط به predict استخراج می‌شود.
    route = next(r for r in test_app.routes if getattr(r, "path", None) == "/predict")
    model = route.endpoint.__closure__[0].cell_contents

    # تابع inference باید بدون اجرای دوباره آموزش، یک پیش‌بینی معتبر برگرداند.
    result = predict("free prize claim now", model=model)
    assert result["label"] in {"ham", "spam"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_version"] == "v1"
