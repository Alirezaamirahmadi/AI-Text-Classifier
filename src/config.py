from pathlib import Path

# ریشه پروژه از محل همین فایل پیدا می‌شود تا مسیرها به سیستم شخصی وابسته نباشند.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# مسیرهای اصلی پروژه بر اساس ریشه پروژه ساخته می‌شوند.
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# مسیر دیتاست محلی؛ فایل باید از قبل داخل پوشه data قرار گرفته باشد.
RAW_DATA_PATH = DATA_DIR / "SMSSpamCollection"

# مسیر فایل مدل و متادیتای نسخه اول.
MODEL_PATH = MODEL_DIR / "text_classifier_v1.joblib"
METADATA_PATH = MODEL_DIR / "metadata_v1.json"

MODEL_VERSION = "v1"
DATASET_NAME = "UCI SMS Spam Collection"
MAX_TEXT_LENGTH = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.2
