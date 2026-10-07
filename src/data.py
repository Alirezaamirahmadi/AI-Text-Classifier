from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from .config import RAW_DATA_PATH

LOGGER = logging.getLogger(__name__)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the local UCI SMS Spam Collection file."""
    # مسیر ورودی می‌تواند مسیر پیش‌فرض پروژه یا یک مسیر مشخص برای تست باشد.
    # ابتدا بررسی می‌کنیم که دیتاست محلی در مسیر مورد انتظار وجود داشته باشد.
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Place SMSSpamCollection in the data directory."
        )

    # فایل خام UCI با Tab جدا شده و header ندارد.
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )

    # تعداد رکوردهای خوانده‌شده برای بررسی روند اجرای آموزش ثبت می‌شود.
    LOGGER.info("Loaded %d rows from %s", len(df), path)
    return df
