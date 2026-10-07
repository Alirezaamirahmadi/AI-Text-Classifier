from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def normalize_text(text: str) -> str:
    """Deterministic normalization; it learns nothing from the dataset."""
    # فاصله‌های ابتدا و انتها حذف و متن به حروف کوچک تبدیل می‌شود.
    text = text.strip().lower()
    # چند فاصله متوالی به یک فاصله تبدیل می‌شود.
    text = re.sub(r"\s+", " ", text)
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove missing/invalid records without learning corpus statistics."""
    # پاک‌سازی روی یک کپی انجام می‌شود تا فایل خام برای بررسی و ردیابی دست‌نخورده بماند.
    result = df.copy()
    # مقادیر خالی متن به رشته خالی تبدیل می‌شوند تا قابل بررسی باشند.
    result["text"] = result["text"].fillna("").astype(str)
    # برچسب‌ها یکدست می‌شوند تا فقط ham و spam معتبر باقی بمانند.
    result["label"] = result["label"].astype(str).str.strip().str.lower()
    # نمونه‌های بدون متن حذف می‌شوند.
    result = result[result["text"].str.strip().str.len() > 0]
    # فقط دو کلاس مورد انتظار نگه داشته می‌شوند.
    result = result[result["label"].isin({"ham", "spam"})]
    # رکوردهای کاملاً تکراری بر اساس متن و برچسب حذف می‌شوند تا نمونه تکراری روی ارزیابی اثر نگذارد.
    result = result.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return result


def build_model() -> Pipeline:
    """Build the complete train-time pipeline. TF-IDF is fit only on training data."""
    # Pipeline ترتیب ثابت پیش‌پردازش و طبقه‌بندی را حفظ می‌کند و از fit شدن TF-IDF روی test جلوگیری می‌کند.
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    preprocessor=normalize_text,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )
