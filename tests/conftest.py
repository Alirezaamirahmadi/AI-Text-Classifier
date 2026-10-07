from __future__ import annotations

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.api import create_app


@pytest.fixture
def test_app():
    # چند نمونه کوچک برای تست سریع API ساخته می‌شوند تا تست‌ها به دیتاست اصلی وابسته نباشند.
    texts = [
        "free prize claim now", "win cash prize", "call now for reward", "urgent free offer",
        "are we meeting today", "please call me later", "see you at dinner", "happy birthday",
    ]
    labels = ["spam", "spam", "spam", "spam", "ham", "ham", "ham", "ham"]

    # برای تست API یک مدل سبک در حافظه آموزش داده می‌شود و مدل اصلی پروژه دوباره آموزش داده نمی‌شود.
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    model.fit(texts, labels)

    # برنامه FastAPI با همین مدل آزمایشی ساخته می‌شود تا هر تست محیط مستقل داشته باشد.
    return create_app(model=model)
