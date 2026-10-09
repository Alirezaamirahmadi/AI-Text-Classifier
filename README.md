# AI Text Classifier

سرویس طبقه‌بندی پیام‌های متنی (Spam / Ham) به‌صورت یک AI Text Classification Service کامل: از دیتاست خام تا یک API واقعی و قابل اجرا.

## 1. Problem Definition

- **Problem:** تشخیص این‌که یک پیام متنی ورودی هرزنامه (Spam) است یا پیام عادی (Ham).
- **Task:** Binary Text Classification.
- **Input:** یک رشته‌ی متنی (پیام کوتاه).
- **Output:** برچسب کلاس (`ham` یا `spam`) به همراه میزان اطمینان مدل (`confidence`).
- این مسئله از طریق یک API واقعی (`POST /predict`) در دسترس قرار می‌گیرد، نه فقط یک اسکریپت یا Notebook.

## 2. Dataset

- **نام:** SMS Spam Collection
- **منبع:** [UCI Machine Learning Repository – SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **حجم:** 5,574 پیام متنی خام (بعد از پاک‌سازی و حذف تکراری‌ها: 5,171 نمونه‌ی معتبر — همچنان بالای آستانه‌ی 5,000 نمونه)
- **ستون‌ها:** `label` (`ham` / `spam`) و `text` (متن پیام)
- **Labelها:**
  - `ham`: پیام عادی و مجاز
  - `spam`: پیام تبلیغاتی یا ناخواسته
- **چرا این دیتاست؟** یک دیتاست واقعی، عمومی و به‌خوبی شناخته‌شده در پژوهش‌های Spam Filtering است؛ اندازه‌ی مناسبی دارد (نه خیلی کوچک، نه نیازمند منابع سنگین)، Label‌های تمیز و باینری دارد، و به دلیل Class Imbalance طبیعی‌اش (حدود 87٪ ham در برابر 13٪ spam) محک خوبی برای ارزیابی درست‌تر از Accuracy به‌تنهایی است.
- فایل دیتاست به‌صورت محلی در `data/SMSSpamCollection` نگه‌داری می‌شود و هیچ دانلود خودکاری در زمان اجرا انجام نمی‌شود.

## 3. Architecture

```
Raw Data (data/SMSSpamCollection)
        ↓
   load_raw_data()           [src/data.py]
        ↓
   clean_dataframe()         [src/preprocessing.py]
        ↓
   Train / Test Split        (قبل از هر Fit، برای جلوگیری از Leakage)
        ↓
   Baseline: DummyClassifier
        ↓
   Pipeline(TF-IDF → Logistic Regression)   [src/preprocessing.py, src/train.py]
        ↓
   Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)   [src/evaluate.py]
        ↓
   Model Artifact (.joblib) + Metadata (.json)   [models/]
        ↓
   predict(text)              [src/predict.py]
        ↓
   FastAPI: POST /predict ,  GET /health      [src/api.py]
```

هر جعبه دقیقاً در یک فایل مستقل در `src/` پیاده‌سازی شده؛ هیچ منطق اصلی داخل Notebook نیست.

## 4. Preprocessing

در `src/preprocessing.py` و `src/data.py`:

- **Handling Missing Text:** مقدار خالی متن با رشته‌ی خالی جایگزین می‌شود تا قابل فیلتر باشد (`fillna("")`).
- **Invalid Record Removal:** نمونه‌هایی که متن‌شان بعد از حذف فاصله خالی است، یا برچسب‌شان خارج از `{ham, spam}` است، حذف می‌شوند.
- **Deduplication:** رکوردهای کاملاً تکراری (متن + برچسب یکسان) حذف می‌شوند تا روی ارزیابی اثر مصنوعی نگذارند.
- **Normalization:** `normalize_text()` حروف را به کوچک تبدیل و فاصله‌های اضافه را یکدست می‌کند؛ این تابع هیچ آماری از کل دیتاست یاد نمی‌گیرد (Deterministic)، و دقیقاً همین تابع هم در Training و هم در Inference استفاده می‌شود.
- **Tokenization/Vectorization:** `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` — یعنی unigram و bigram، با حذف کلماتی که در کمتر از ۲ سند دیده شده‌اند.
- **Label Encoding:** `LabelEncoder` برای تبدیل `ham`/`spam` به مقدار عددی برای آموزش.

### جلوگیری از Data Leakage

- Split به Train/Test **قبل از** هرگونه Fit انجام می‌شود (`train_test_split` در `src/train.py`).
- `TfidfVectorizer` و `LogisticRegression` در یک `Pipeline` واحد قرار دارند، بنابراین vocabulary TF-IDF فقط از `x_train` ساخته می‌شود؛ روی `x_test` فقط `transform` اجرا می‌شود.
- `LabelEncoder` هم فقط روی `y_train` فیت می‌شود.
- Normalization متن (`normalize_text`) یک تبدیل Deterministic per-sample است و هیچ آماری را از کل مجموعه یاد نمی‌گیرد؛ بنابراین اجرای آن پیش از Split هیچ نشتی ایجاد نمی‌کند.

## 5. Baseline

- **مدل:** `DummyClassifier(strategy="most_frequent")` — همیشه پرتکرارترین کلاس (`ham`) را پیش‌بینی می‌کند.
- **چرا این Baseline؟** کف مطلق عملکرد را نشان می‌دهد؛ هر مدل واقعی باید به‌وضوح از آن بهتر عمل کند.
- **نتیجه‌ی Baseline** (ذخیره‌شده در `outputs/baseline_metrics.json`):

| Metric | Value |
|---|---|
| Accuracy | 0.8734 |
| Precision (spam) | 0.0000 |
| Recall (spam) | 0.0000 |
| F1 (spam) | 0.0000 |

Accuracy بالای Baseline (87.3٪) دقیقاً نشان می‌دهد چرا Accuracy به‌تنهایی برای این مسئله کافی نیست: یک مدل کاملاً بی‌فایده هم بدون شناسایی حتی یک پیام Spam، Accuracy بالایی می‌گیرد.

## 6. Model

- **معماری:** `TF-IDF → Logistic Regression`، هر دو در یک `sklearn.pipeline.Pipeline`.
- **TfidfVectorizer:** `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`, `strip_accents="unicode"`, با `preprocessor=normalize_text`.
- **LogisticRegression:** `max_iter=1000`, `class_weight="balanced"` (برای جبران نامتوازن بودن کلاس‌ها)، `random_state=42`.
- چرا این مدل؟ TF-IDF + Logistic Regression یک baseline صنعتی استاندارد و قوی برای Text Classification است؛ سریع، تفسیرپذیر و بدون نیاز به منابع سنگین (طبق خواسته‌ی این مرحله، از Transformer استفاده نشده).

## 7. Evaluation

Metricها روی Test Set (ذخیره‌شده در `outputs/metrics.json`):

| Metric | Value |
|---|---|
| Accuracy | 0.9797 |
| Precision (spam) | 0.9231 |
| Recall (spam) | 0.9160 |
| F1 (spam) | 0.9195 |
| F1 (macro) | 0.9540 |
| F1 (weighted) | 0.9797 |

**Confusion Matrix** (`outputs/confusion_matrix.csv`):

| | Predicted: ham | Predicted: spam |
|---|---|---|
| **Actual: ham** | 894 | 10 |
| **Actual: spam** | 11 | 120 |

### Averaging Strategy

چون این مسئله دودویی (`ham`/`spam`) است، Precision/Recall/F1 اصلی با `pos_label="spam"` گزارش می‌شوند — یعنی کلاس Spam (کلاس کم‌تعدادتر و مهم‌تر از نظر کسب‌وکار) معیار اصلی است. علاوه بر آن، برای کامل بودن گزارش و آماده بودن برای توسعه‌ی احتمالی به چندکلاسه در آینده، دو معیار اضافه هم ذخیره شده‌اند:
- **Macro F1:** میانگین ساده‌ی F1 بین `ham` و `spam`، بدون توجه به تعداد نمونه‌ی هر کلاس — به هر دو کلاس وزن یکسان می‌دهد.
- **Weighted F1:** میانگین F1 با وزن‌دهی بر اساس تعداد نمونه‌ی هر کلاس — چون `ham` اکثریت دارد، این عدد به Accuracy نزدیک‌تر است.

فاصله‌ی زیاد بین Macro F1 (0.9540) و Weighted F1 (0.9797) بازتاب طبیعی Class Imbalance دیتاست است، نه ضعف مدل.

## 8. Inference

تابع اصلی در `src/predict.py`:

```python
from src.predict import predict

result = predict("Congratulations! You won a prize.")
print(result)
# {"label": "spam", "confidence": 0.941, "model_version": "v1"}
```

- مدل در این مرحله **دوباره Train نمی‌شود**؛ فقط Model Artifact از `models/text_classifier_v1.joblib` بارگذاری می‌شود.
- پیش از Prediction: نوع ورودی (باید `str` باشد)، خالی نبودن متن (بعد از Normalization) و رعایت حداکثر طول مجاز (`MAX_TEXT_LENGTH = 5000`) بررسی می‌شود.
- همان تابع `normalize_text` که در Training استفاده شده، روی ورودی Inference هم اعمال می‌شود تا رفتار مدل سازگار بماند.

## 9. API

ساخته‌شده با **FastAPI**. اجرای Local:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### `GET /health`
وضعیت سرویس و بارگذاری مدل‌های کلاسیک، PyTorch و Transformer را، بدون انجام هیچ Training‌ای، گزارش می‌دهد.

### `POST /predict`
متن را می‌گیرد و برچسب + میزان اطمینان را با مدل TF-IDF + Logistic Regression برمی‌گرداند.

### `POST /predict/dl`
متن را با مدل PyTorch BiLSTM طبقه‌بندی می‌کند و `model_version` آن `pytorch_v1` است.

### `POST /predict/transformer`
متن را با مدل DistilBERT ذخیره‌شده طبقه‌بندی می‌کند و `model_version` آن `transformer_v1` است.

تمام خطاهای پیش‌بینی‌نشده با یک Exception Handler سراسری گرفته می‌شوند و فقط پیام عمومی `"Internal server error"` (بدون Traceback داخلی پایتون) به کاربر بازگردانده می‌شود.

## 10. Example Request / Response

```bash
curl.exe -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a prize."}'
```

پاسخ:

```json
{
  "label": "spam",
  "confidence": 0.941,
  "model_version": "v1"
}
```

نمونه برای `/health`:

```bash
curl.exe http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "pytorch_model_loaded": true,
  "transformer_model_loaded": true,
  "model_version": "v1"
}
```

### Input Validation

درخواست‌های زیر رد می‌شوند (کد وضعیت `422`)، بدون این‌که هیچ Traceback داخلی به کاربر نمایش داده شود:

| حالت ورودی | رفتار |
|---|---|
| متن خالی یا فقط فاصله | رد می‌شود |
| نبودن فیلد `text` | رد می‌شود |
| نوع اشتباه (مثلاً عدد به‌جای رشته) | رد می‌شود |
| فیلد اضافه و ناشناخته در Body | رد می‌شود (`extra="forbid"`) |
| متن طولانی‌تر از `MAX_TEXT_LENGTH` (5000 کاراکتر) | رد می‌شود |
| JSON خراب (Malformed) | رد می‌شود، بدون نمایش جزئیات داخلی |
| مدل هنوز Train نشده | پاسخ `503` با پیام راهنما برای اجرای `python -m src.train` |

## 11. Testing

```bash
python -m pytest -v
```

| فایل | شامل |
|---|---|
| `tests/test_api.py` | `test_health`, `test_valid_prediction`, `test_empty_text`, `test_invalid_input`, `test_malformed_request`, `test_model_loaded`, `test_extremely_long_text` |
| `tests/test_prediction.py` | تست مستقیم تابع `predict()` روی یک مدل آزمایشیِ درون‌حافظه‌ای |
| `tests/conftest.py` | یک مدل کوچک و سریع (روی چند نمونه‌ی دستی) برای تست‌ها می‌سازد، تا تست‌ها به دیتاست اصلی یا اجرای قبلی `train.py` وابسته نباشند |

## 12. Model Artifact & Metadata

- **Artifact:** `models/text_classifier_v1.joblib` — کل `Pipeline` (TF-IDF + Logistic Regression)، نه فقط وزن‌های مدل.
- **Metadata:** `models/metadata_v1.json`، شامل:

```json
{
  "model_version": "v1",
  "model_type": "TF-IDF + Logistic Regression",
  "dataset": "UCI SMS Spam Collection",
  "dataset_rows_raw": 5574,
  "dataset_rows_valid": 5171,
  "features_vectorizer": "...",
  "classifier": "...",
  "label_encoding": ["ham", "spam"],
  "training_date": "...",
  "test_size": 0.2,
  "random_state": 42,
  "baseline": { ... },
  "metrics": { ... }
}
```

## 13. Docker

```bash
docker build -t ai-text-classifier .
docker run -p 8000:8000 ai-text-classifier
```

در نسخه‌ی نهایی، Model Artifactهای آموزش‌دیده همراه پروژه داخل Image قرار می‌گیرند و `Dockerfile` هیچ Training یا Download مدل در زمان Build انجام نمی‌دهد. Container هنگام Startup فقط Artifactهای موجود را Load می‌کند و API هیچ‌وقت هنگام اجرا مدل را Train نمی‌کند.

برای Build موفق نسخه‌ی نهایی، فایل‌های `models/` شامل Artifactهای سه مدل باید در Context ساخت Docker موجود باشند.

## 14. Project Structure

```
AI-Text-Classifier/
├── data/
│   └── SMSSpamCollection
├── models/
│   ├── text_classifier_v1.joblib
│   ├── metadata_v1.json
│   ├── text_classifier_pytorch_v1.pt
│   └── transformer_v1/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_state.pt
├── outputs/
│   ├── baseline_metrics.json
│   ├── metrics.json
│   ├── confusion_matrix.csv
│   ├── transformer_metrics.json
│   ├── transformer_errors.json
│   └── model_comparison.json
├── notebooks/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── dl_dataset.py
│   ├── dl_model.py
│   ├── dl_train.py
│   ├── dl_predict.py
│   ├── transformer_utils.py
│   ├── transformer_train.py
│   ├── transformer_predict.py
│   └── api.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_prediction.py
│   ├── test_dl.py
│   └── test_transformer.py
├── Dockerfile
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

مسیرها در `src/config.py` با `pathlib.Path(__file__)` نسبت به ریشه‌ی پروژه ساخته می‌شوند؛ کد به هیچ مسیر مخصوص یک کامپیوتر وابسته نیست.

## 15. From Clone to First Prediction

```bash
git clone <repository-url>
cd AI-Text-Classifier

python -m venv venv
source venv/bin/activate        # ویندوز: venv\Scripts\activate

pip install -r requirements.txt

python -m src.train              # آموزش مدل + ذخیره Artifact و Metadata
python -m pytest -v                        # اجرای تست‌ها

uvicorn src.api:app --host 0.0.0.0 --port 8000   # بالا آوردن API

# در یک ترمینال دیگر:
curl.exe -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a prize."}'
```

## 16. Limitations

- دیتاست فقط شامل پیام‌های کوتاه انگلیسی (SMS) از یک دوره‌ی زمانی خاص است؛ تعمیم مستقیم به پیام‌های ایمیل، زبان‌های دیگر یا سبک‌های نوشتاری جدیدتر (مثل اسپم‌های شبکه‌های اجتماعی امروزی) بدون آموزش مجدد توصیه نمی‌شود.
- مدل از TF-IDF (مبتنی بر فرکانس کلمات) استفاده می‌کند، نه یک نمایش معنایی/Context-aware؛ بنابراین در برابر جمله‌هایی با ساختار یا کلمات کاملاً جدید (خارج از Vocabulary آموزش‌دیده) ضعیف‌تر عمل می‌کند.
- Recall کلاس Spam حدود 91.6٪ است؛ یعنی حدود 8٪ از پیام‌های واقعاً Spam هنوز به‌اشتباه Ham شناسایی می‌شوند.
- سرویس فعلی تک‌مدلی و تک‌نسخه‌ای (`v1`) است؛ مدیریت چند نسخه‌ی مدل هم‌زمان یا A/B Testing در این مرحله پیاده‌سازی نشده.
- API به‌صورت Local/Single-instance طراحی شده و مواردی مثل Authentication، Rate Limiting یا مقیاس‌پذیری افقی (Horizontal Scaling) در این مرحله جزو اهداف نبوده‌اند.
- این مرحله عمداً از مدل‌های Transformer/Deep Learning استفاده نکرده (طبق خواسته‌ی این مرحله از پروژه)؛ در صورت نیاز به دقت بالاتر روی متن‌های پیچیده‌تر، قدم بعدی می‌تواند استفاده از چنین مدل‌هایی باشد.

## 17. Deep Learning Architecture

در این مرحله، همان مسئله‌ی طبقه‌بندی Spam/Ham با یک مدل Deep Learning مبتنی بر PyTorch (به‌جای TF-IDF) حل شده، بدون این‌که مسیر و مدل مرحله‌ی قبل (`src/train.py`, `models/text_classifier_v1.joblib`) حذف یا خراب شود.

```
Raw Data (data/SMSSpamCollection)
        ↓
   load_raw_data() + clean_dataframe()      [همان پاک‌سازی مرحله قبل]
        ↓
   Train (70%) / Validation (15%) / Test (15%)   [Stratified Split]
        ↓
   build_vocabulary(train_only)              [src/dl_dataset.py]
        ↓
   SMSDataset + DataLoader (collate + padding)   [src/dl_dataset.py]
        ↓
   BiLSTMClassifier: Embedding → BiLSTM → Dropout → Linear   [src/dl_model.py]
        ↓
   Training Loop دستی + Early Stopping         [src/dl_train.py]
        ↓
   Checkpoint (.pt) با model/optimizer/vocabulary/history   [models/text_classifier_pytorch_v1.pt]
        ↓
   predict_dl(text)                            [src/dl_predict.py]
        ↓
   FastAPI: POST /predict/dl   (در کنار /predict قبلی، بدون تغییر آن)   [src/api.py]
```

این پایپ‌لاین به‌طور کامل مستقل از پایپ‌لاین TF-IDF اجراست؛ هر دو مدل، آرتیفکت و Endpoint خودشان را دارند و در کنار هم روی سرویس فعال‌اند.

## 18. Tokenization

تابع `tokenize_text` در `src/dl_dataset.py`:

1. ابتدا همان `normalize_text` مرحله‌ی قبل (حروف کوچک + یکدست‌سازی فاصله‌ها) روی متن اعمال می‌شود تا رفتار Normalization بین دو مدل یکسان بماند.
2. سپس با یک عبارت منظم (`\w+|[^\w\s]`) متن به کلمات و نشانه‌های نگارشی جدا تقسیم می‌شود؛ یعنی هم کلمات (`congratulations`) و هم علائم (`!`) به‌صورت Token مستقل در نظر گرفته می‌شوند.

این Tokenizer ساده و کاملاً قطعی (Deterministic) است — چیزی از داده یاد نمی‌گیرد، فقط متن را می‌شکند؛ یادگیری واقعی در مرحله‌ی ساخت Vocabulary اتفاق می‌افتد.

## 19. Vocabulary

تابع `build_vocabulary` در `src/dl_dataset.py`:

- Vocabulary **فقط از Token‌های متن‌های Train** ساخته می‌شود؛ Validation و Test هیچ تأثیری روی آن ندارند (دقیقاً برای جلوگیری از Leakage در لایه‌ی بازنمایی متن، مشابه چیزی که در مرحله‌ی قبل برای TF-IDF رعایت شده بود).
- دو Token ویژه با Index ثابت رزرو شده‌اند:
  - `<PAD> = 0`
  - `<UNK> = 1`
- **چرا PAD لازم است؟** چون جملات طول متفاوتی دارند، اما یک Batch در PyTorch باید Tensor مستطیلی (طول ثابت) باشد؛ `<PAD>` جای خالی جملات کوتاه‌تر را در هر Batch پر می‌کند تا بشود همه را در یک Tensor کنار هم قرار داد. (در مدل، `padding_idx=0` در Embedding و `pack_padded_sequence` باعث می‌شوند PAD روی محاسبه‌ی واقعی LSTM اثر نگذارد.)
- **چرا UNK لازم است؟** چون در زمان Inference یا حتی روی Validation/Test ممکن است کلمه‌ای ظاهر شود که در Train دیده نشده؛ به‌جای خطا دادن، چنین Tokenهایی به `<UNK>` نگاشت می‌شوند.
- در دیتاست فعلی (5,171 نمونه‌ی معتبر، Train=3,619 نمونه)، اندازه‌ی Vocabulary ساخته‌شده **7,147 Token** (شامل PAD و UNK) است.

## 20. PyTorch Dataset / DataLoader

- **`SMSDataset`** (در `src/dl_dataset.py`) یک `torch.utils.data.Dataset` واقعی است: هر متن را Tokenize و با Vocabulary به دنباله‌ای از Index تبدیل می‌کند و برچسب متناظر را هم به‌صورت Tensor برمی‌گرداند.
- **`collate_batch`** تابع `collate_fn` سفارشی برای `DataLoader` است: طول واقعی هر جمله را قبل از Padding نگه می‌دارد (برای استفاده در `pack_padded_sequence`) و سپس با `pad_sequence` همه‌ی جمله‌های یک Batch را به یک طول مشترک (طول بیشینه‌ی همان Batch) می‌رساند.
- سه `DataLoader` مستقل برای Train (با `shuffle=True`)، Validation و Test (بدون Shuffle) ساخته می‌شوند؛ `batch_size=128`.

## 21. Model Architecture

کلاس `BiLSTMClassifier` در `src/dl_model.py`، دقیقاً مطابق معماری خواسته‌شده:

```
Input (token indices)
    ↓
Embedding (vocab_size → embedding_dim=64, padding_idx=0)
    ↓
BiLSTM (hidden_dim=64, bidirectional=True)
    ↓
Dropout (p=0.3)
    ↓
Linear (hidden_dim*2 → 2 کلاس)
```

- از `pack_padded_sequence` استفاده شده تا بخش `<PAD>` در محاسبات LSTM بی‌اثر بماند (LSTM هیچ‌وقت واقعاً "Padding" را به‌عنوان یک Token معنادار پردازش نمی‌کند).
- چون LSTM دوطرفه (BiLSTM) است، هم جهت رفت (`hidden[-2]`) و هم جهت برگشت (`hidden[-1]`) در انتها به هم متصل (`concat`) می‌شوند تا یک بازنمایی واحد از کل جمله به دست آید.
- خروجی نهایی، Logits خام برای دو کلاس (`ham`, `spam`) است؛ تبدیل به Probability در مرحله‌ی Inference با `softmax` انجام می‌شود (بخش 10 و 25).

## 22. Training Loop

در `src/dl_train.py`، تابع `run_epoch` حلقه‌ی آموزش را به‌صورت کاملاً دستی پیاده‌سازی می‌کند (بدون `Trainer` آماده‌ی هیچ کتابخانه‌ای):

```
for هر batch:
    optimizer.zero_grad()      # پاک‌سازی gradient مرحله قبل
    logits = model(x)          # forward
    loss = criterion(logits, y)
    loss.backward()            # backward
    optimizer.step()           # به‌روزرسانی وزن‌ها
```

- همین تابع با پارامتر `training=False` و داخل `torch.no_grad()`/`torch.set_grad_enabled(False)` برای Validation هم استفاده می‌شود (بدون `backward`/`step`)، تا کد Training و Evaluation تکراری نشود.
- هر Epoch هم روی Train و هم روی Validation اجرا می‌شود و `train_loss`, `train_f1`, `validation_loss`, `validation_f1` با `logging` (نه `print`) ثبت می‌گردد.

## 23. Loss & Optimizer

- **Loss:** `nn.CrossEntropyLoss` با `weight` بر اساس نسبت معکوس فراوانی کلاس‌ها در **Train** (`calculate_class_weights`) — چون دیتاست نامتوازن است (حدود 87٪ ham)، این وزن‌دهی باعث می‌شود مدل کلاس Spam را نادیده نگیرد.
- **Optimizer:** `AdamW` با `learning_rate=1e-3`.
- **اثر Learning Rate:** نرخ یادگیری اندازه‌ی گام به‌روزرسانی وزن‌ها در هر Step را تعیین می‌کند. مقدار خیلی بزرگ باعث نوسان یا واگرایی Loss می‌شود (مدل هیچ‌وقت به کمینه نمی‌رسد یا رد می‌شود)؛ مقدار خیلی کوچک یادگیری را به‌شدت کند می‌کند و ممکن است مدل در تعداد Epoch محدود (اینجا حداکثر 7) به‌خوبی Converge نشود. مقدار `1e-3` یک نقطه‌ی شروع استاندارد و معمول برای AdamW/Adam روی این نوع معماری‌هاست.

## 24. Validation & Early Stopping

- بعد از هر Epoch، مدل روی Validation (نه Test) ارزیابی و F1 آن ثبت می‌شود.
- اگر F1 روی Validation نسبت به بهترین مقدار تا آن لحظه بهبود یابد، Checkpoint جدید ذخیره و شمارنده‌ی «بدون بهبود» صفر می‌شود.
- اگر طی **3 Epoch متوالی** (`PATIENCE=3`) هیچ بهبودی در Validation F1 دیده نشود، Training زودتر از `MAX_EPOCHS=7` متوقف می‌شود.
- **چرا بر اساس F1 و نه Loss یا Accuracy؟** چون دیتاست نامتوازن است؛ Accuracy یا حتی Loss می‌توانند با وجود نادیده گرفتن کلاس Spam هم عدد خوبی نشان دهند، در حالی که F1 (به‌خصوص برای کلاس Spam) مستقیماً کیفیت شناسایی کلاس اقلیت را می‌سنجد — همان منطقی که در انتخاب Metric مرحله‌ی قبل هم استفاده شده بود.
- در اجرای فعلی، بهترین Validation F1 (0.9137) در **Epoch 7** (آخرین Epoch مجاز) به دست آمده؛ یعنی تا پایان 7 Epoch روند بهبود هنوز ادامه داشت و Early Stopping زودتر فعال نشد.

## 25. Checkpoint & Reproducibility

Checkpoint در `models/text_classifier_pytorch_v1.pt` (با `torch.save`) ذخیره می‌شود و شامل این موارد است:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "label_mapping": {"ham": 0, "spam": 1},
    "vocabulary": {...},          # کل Vocabulary برای بازسازی دقیق Index‌ها
    "model_config": {...},        # ابعاد لازم برای بازسازی خود کلاس مدل
    "training_config": {...},
    "best_validation_f1": ...,
    "history": [...],             # تاریخچه کامل هر Epoch
    "dataset": "UCI SMS Spam Collection",
    "saved_at": "...",
}
```

به همین دلیل مدل فقط در RAM باقی نمی‌ماند و `src/dl_predict.py` می‌تواند بدون هیچ Training‌ای، دقیقاً همان مدل (همراه Vocabulary و تنظیماتش) را از روی دیسک بازسازی کند.

**چرا Reproducibility در Deep Learning سخت‌تر از مدل قبلی (TF-IDF + Logistic Regression) است؟**

- مدل کلاسیک مرحله‌ی قبل یک راه‌حل بهینه‌سازی محدب (Convex) دارد؛ با همان داده و همان `random_state`، همیشه به همان نتیجه می‌رسد.
- مدل PyTorch چندین منبع تصادفی‌بودن دارد که همه باید هم‌زمان کنترل شوند: مقداردهی اولیه‌ی وزن‌های Embedding/LSTM/Linear، ترتیب Shuffle در DataLoader، و (در صورت اجرا روی GPU) رفتار غیرقطعی برخی عملیات CUDA.
- به همین دلیل در `dl_train.py` تابع `set_reproducible_seed` هم‌زمان Seed پایتون، NumPy و PyTorch (و در صورت وجود CUDA، تنظیمات Deterministic آن) را ثابت می‌کند — چیزی که برای مدل کلاسیک اصلاً لازم نبود (فقط یک `random_state` کافی بود).
- حتی با Seed ثابت، نسخه‌ی کتابخانه‌ها (PyTorch, CUDA) یا سخت‌افزار اجرا (CPU در برابر GPU) می‌تواند روی چند رقم اعشار نتیجه اثر بگذارد؛ به همین دلیل Checkpoint واقعی (نه فقط کد) تنها راه تضمین‌شده برای بازتولید دقیقاً همان مدل است.

## 26. Evaluation & Comparison with TF-IDF + Logistic Regression

برای مقایسه‌ی منصفانه، هر دو مدل روی **همان تقسیم‌بندی** (Train=3,619 / Validation=776 / Test=776، از همان 5,171 نمونه‌ی پاک‌سازی‌شده) ارزیابی شده‌اند. مدل TF-IDF مرحله‌ی قبل (`models/text_classifier_v1.joblib`) دست‌نخورده باقی مانده؛ برای این مقایسه یک نسخه‌ی جداگانه با همان معماری روی همین Split آموزش دیده (`outputs/model_comparison.json`).

| Metric | TF-IDF + Logistic Regression | PyTorch (Embedding+BiLSTM) |
|---|---|---|
| Accuracy | **0.9794** | 0.9755 |
| Precision (spam) | **0.9271** | 0.8911 |
| Recall (spam) | 0.9082 | **0.9184** |
| F1 (spam) | **0.9175** | 0.9045 |

Confusion Matrix (Test, 776 نمونه):

| | TF-IDF: Pred ham / spam | PyTorch: Pred ham / spam |
|---|---|---|
| **Actual ham** | 671 / 7 | 667 / 11 |
| **Actual spam** | 9 / 89 | 8 / 90 |

**چرا در این Dataset ممکن است TF-IDF + Logistic Regression از LSTM بهتر باشد؟**

- حجم داده‌ی Train تنها **3,619 نمونه** است؛ یک شبکه‌ی Embedding+LSTM که کاملاً از صفر (بدون Embedding از‌پیش‌آموزش‌دیده) یاد می‌گیرد، معمولاً به مراتب بیش از این برای یادگیری بازنمایی‌های معنایی قابل‌اعتماد نیاز دارد.
- TF-IDF یک بازنمایی ساده و پراکنده (Sparse) مبتنی بر فراوانی n-gramهاست؛ برای مسئله‌ای مثل تشخیص Spam که اغلب با وجود/عدم وجود چند کلمه و عبارت کلیدی (`free`, `win`, `prize`, `call now`, ...) به‌خوبی قابل تفکیک است، این بازنمایی ساده و یک مدل خطی (Logistic Regression) کفایت می‌کند و کمتر مستعد Overfitting روی داده‌ی کم است.
- مدل LSTM درجه آزادی (پارامتر) بسیار بیشتری دارد (Embedding 64 بعدی برای ۷ هزار Token + وزن‌های BiLSTM)، که با Validation محدود (776 نمونه) و تنها 7 Epoch، فرصت کافی برای رسیدن به بهترین نقطه‌ی ممکن خودش را پیدا نمی‌کند.
- نتیجه در عمل هم همین را نشان می‌دهد: فاصله‌ی دو مدل کم است (F1: 0.9175 در برابر 0.9045)، یعنی روی این دیتاست کوچک، مدل کلاسیک حداقل هم‌تراز یا کمی بهتر از مدل یادگیری عمیق از‌صفر‌آموزش‌دیده عمل می‌کند — نتیجه‌ای که در ادبیات NLP روی دیتاست‌های کوچک کاملاً شناخته‌شده است.

## 27. New API Endpoint & Testing

- **Endpoint جدید:** `POST /predict/dl` — دقیقاً مشابه `POST /predict` قبلی (همان ساختار Request/Response)، با این تفاوت که از مدل PyTorch استفاده می‌کند و `model_version` آن `"pytorch_v1"` است.
- **Endpoint جدید:** `POST /predict/transformer` — متن را دریافت می‌کند و با Transformer ذخیره‌شده، `label`, `confidence` و `model_version` را برمی‌گرداند.
- `GET /health` وضعیت هر سه مدل (`TF-IDF`, `PyTorch`, `Transformer`) را گزارش می‌کند.
- Endpointها و تست‌های قبلی بدون حذف یا جایگزینی باقی مانده‌اند و تست‌های جدید به‌صورت افزایشی اضافه شده‌اند.

تست‌های PyTorch در `tests/test_dl.py`: 

| تست | بررسی می‌کند |
|---|---|
| `test_tokenization_and_vocabulary_are_train_only` | Tokenizer و این‌که Vocabulary فقط از متن‌های Train ساخته می‌شود، نه داده‌ی دیده‌نشده |
| `test_pytorch_dataset_and_collate` | خروجی `SMSDataset` و `collate_batch` از نوع Tensor واقعی PyTorch با شکل صحیح است |
| `test_bilstm_forward_shape` | خروجی مدل برای یک Batch، شکل `(batch_size, num_classes)` دارد |
| `test_pytorch_checkpoint_exists_and_has_required_fields` | فایل Checkpoint روی دیسک وجود دارد و فیلدهای ضروری را دارد |
| `test_pytorch_inference` | تابع `predict_dl` بدون Training مجدد، یک پیش‌بینی معتبر برمی‌گرداند |
| `test_pytorch_api_endpoint` | Endpoint `POST /predict/dl` پاسخ معتبر HTTP 200 با ساختار درست برمی‌گرداند |

تست‌های Transformer در `tests/test_transformer.py`: 

| تست | بررسی می‌کند |
|---|---|
| `test_transformer_model_loads` | مدل Transformer ذخیره‌شده بدون Training مجدد Load می‌شود |
| `test_transformer_tokenizer_loads` | Tokenizer ذخیره‌شده Load می‌شود و برای Inference قابل استفاده است |
| `test_transformer_prediction` | تابع Prediction خروجی معتبر برای متن ورودی تولید می‌کند |
| `test_transformer_prediction_confidence` | مقدار `confidence` معتبر و در بازه‌ی مورد انتظار است |
| `test_transformer_api_endpoint` | Endpoint `POST /predict/transformer` پاسخ معتبر HTTP 200 برمی‌گرداند |
| `test_transformer_invalid_input` | ورودی نامعتبر با کد وضعیت مناسب رد می‌شود |
| `test_health_all_models` | Health endpoint بارگذاری هر سه مدل را بررسی می‌کند |

تست‌های API همچنین با `Invoke-WebRequest` روی Windows برای Endpointهای جدید و ورودی‌های معتبر و نامعتبر بررسی شدند؛ پاسخ‌های موفق `200` و ورودی‌های نامعتبر `422` دریافت کردند.

اجرای کامل همه‌ی تست‌ها (قدیم و جدید):

```bash
python -m pytest -v
```

**نتیجه:** `22 passed`

## 28. Limitations

- دیتاست کوچک (5,171 نمونه‌ی معتبر، فقط 3,619 برای Train) باعث می‌شود مدل PyTorch از صفر، مزیت واقعی نسبت به مدل کلاسیک نشان ندهد (بخش 26)؛ برای برتری واقعی LSTM/GRU نسبت به TF-IDF معمولاً به داده‌ی قابل‌توجه بیشتری یا استفاده از Embeddingهای از‌پیش‌آموزش‌دیده نیاز است.
- Vocabulary با `min_freq=1` ساخته شده؛ یعنی حتی کلماتی که فقط یک‌بار در کل Train دیده شده‌اند هم وارد Vocabulary می‌شوند (اندازه‌ی Vocabulary: 7,147). این می‌تواند باعث Overfitting خفیف روی کلمات کم‌تکرار شود؛ افزایش `min_freq` یک تنظیم ساده برای آزمایش بعدی است.
- معماری فعلی از Embedding تصادفی (نه از‌پیش‌آموزش‌دیده مثل GloVe/Word2Vec) استفاده می‌کند؛ بنابراین تمام دانش زبانی باید فقط از همین 3,619 نمونه یاد گرفته شود.
- هر دو Endpoint (`/predict` و `/predict/dl`) هم‌زمان روی یک Process اجرا می‌شوند؛ مدیریت منابع (حافظه/CPU) بین دو مدل بهینه‌سازی نشده و برای استقرار واقعی با ترافیک بالا نیاز به بررسی بیشتر دارد.
- طبق خواسته‌ی این مرحله، از Transformer یا مدل‌های از‌پیش‌آموزش‌دیده استفاده نشده؛ گام منطقی بعدی (طبق نقشه‌ی راه) استفاده از Fine-tuning با Hugging Face است.

## 29. Transformer Architecture

در این مرحله، همان مسئله‌ی Spam/Ham با یک مدل Transformer ازپیش‌آموزش‌دیده‌ی سبک‌تر، `distilbert-base-uncased`، حل شده است. مدل Transformer به‌صورت افزایشی در کنار دو مدل قبلی اضافه شده و Artifactهای قبلی حذف یا جایگزین نشده‌اند.

```
Raw Data (data/SMSSpamCollection)
        ↓
   همان پاک‌سازی و Stratified Split قبلی
        ↓
   Train (70%) / Validation (15%) / Test (15%)
        ↓
   DistilBERT Tokenizer
        ↓
   input_ids + attention_mask + labels
        ↓
   DistilBERT + Sequence Classification Head
        ↓
   Fine-tuning روی Train + Validation Evaluation
        ↓
   Best Checkpoint بر اساس Validation F1
        ↓
   Test Evaluation یک‌باره بعد از انتخاب مدل
        ↓
   مدل ذخیره‌شده + Tokenizer      [models/transformer_v1/]
        ↓
   predict_transformer(text)
        ↓
   FastAPI: POST /predict/transformer
```

### Model Selection

`distilbert-base-uncased` به‌دلیل اندازه‌ی کوچک‌تر و هزینه‌ی محاسباتی پایین‌تر نسبت به بسیاری از مدل‌های Transformer انتخاب شد. این مدل برای CPU نیز قابل اجراست و در عین حال از یک نمایش Context-aware از متن استفاده می‌کند. مدل‌های بزرگ‌تر برای محدودیت سخت‌افزاری این پروژه انتخاب نشده‌اند.

## 30. Tokenization

برای Transformer از Tokenizer رسمی همان مدل (`DistilBertTokenizerFast`) استفاده شده است. برخلاف Tokenizer ساده‌ی بخش PyTorch، این Tokenizer از Subword Tokenization استفاده می‌کند؛ بنابراین یک کلمه‌ی ناشناخته لزوماً به یک Token ناشناخته تبدیل نمی‌شود و می‌تواند به چند Subword شکسته شود.

خروجی اصلی Tokenizer شامل موارد زیر است:

- `input_ids`: شناسه‌ی عددی Tokenها در Vocabulary مدل.
- `attention_mask`: مشخص می‌کند کدام موقعیت‌ها Token واقعی هستند و کدام موقعیت‌ها Padding هستند.
- `labels`: برچسب عددی کلاس برای محاسبه‌ی Loss در زمان Training.

برای این پروژه `max_length=64` استفاده شده است تا طول ورودی و مصرف حافظه‌ی CPU کنترل شود. Padding و Truncation بر اساس همین مقدار انجام می‌شوند.

## 31. Transformer Training

مدل با `AutoModelForSequenceClassification.from_pretrained` و دو کلاس (`ham`, `spam`) ساخته شده و به‌صورت واقعی روی داده‌ی Train Fine-tune شده است. Validation در طول Training استفاده شده و Test فقط بعد از پایان انتخاب مدل و برای ارزیابی نهایی استفاده شده است.

Hyperparameterهای اجرای نهایی:

| Parameter | Value |
|---|---|
| Pretrained model | `distilbert-base-uncased` |
| Learning rate | `2e-5` |
| Batch size | `4` |
| Epochs | `2` |
| Weight decay | `0.01` |
| Max length | `64` |
| Early stopping patience | `1` |
| Gradient accumulation steps | `1` |
| Device | `CPU` |

Learning rate `2e-5` برای Fine-tuning مدل Pretrained انتخاب شده است تا وزن‌های ازپیش‌آموزش‌دیده با گام‌های کوچک و کنترل‌شده روی مسئله‌ی Spam/Ham تطبیق داده شوند. Batch کوچک نیز با توجه به محدودیت حافظه‌ی سیستم انتخاب شده است.

در اجرای نهایی، بهترین Validation F1 برابر **0.9583** در Epoch 1 به دست آمد. در Epoch 2، Validation F1 به `0.9366` کاهش یافت؛ بنابراین Early Stopping فعال شد و Checkpoint مربوط به Epoch 1 به‌عنوان مدل نهایی انتخاب شد.

## 32. Transformer Evaluation

مدل نهایی Transformer فقط یک‌بار روی Test Set جداشده ارزیابی شد. نتیجه‌ی نهایی در `outputs/transformer_metrics.json` ذخیره شده است.

| Metric | Value |
|---|---|
| Accuracy | **0.99098** |
| Precision (spam) | **1.00000** |
| Recall (spam) | **0.92857** |
| F1 (spam) | **0.96296** |

Confusion Matrix (Test, 776 نمونه):

| | Predicted: ham | Predicted: spam |
|---|---:|---:|
| **Actual: ham** | 678 | 0 |
| **Actual: spam** | 7 | 91 |

مدل هیچ False Positive نداشت و 7 پیام Spam را به‌اشتباه Ham تشخیص داد. معیارهای مربوط به کلاس Spam به‌صورت جداگانه در خروجی Evaluation ذخیره شده‌اند.

## 33. Transformer Error Analysis

خطاهای واقعی مدل در `outputs/transformer_errors.json` ذخیره شده‌اند و شامل متن، برچسب واقعی، برچسب پیش‌بینی‌شده و Confidence هستند. در Test Set فقط **7 misclassification واقعی** وجود داشت؛ این تعداد بدون ساختن نمونه‌ی مصنوعی گزارش شده است.

الگوی اصلی خطاها نشان می‌دهد که بیشتر نمونه‌های اشتباه، پیام‌های Spam با یکی از این ویژگی‌ها هستند:

- زبان بسیار محاوره‌ای و کوتاه‌شده (`U`, `u`, `THNQ`, ...).
- پیام‌های تبلیغاتی یا تجاری که در قالب یک جمله‌ی نسبتاً عادی نوشته شده‌اند.
- محتوای Spam با ساختار غیرمعمول، URL یا رشته‌های نویزی.
- پیام‌هایی که با وجود محتوای Spam، از نظر سبک نوشتاری شبیه پیام‌های عادی هستند.

نکته‌ی مهم این است که Confidence مدل در چند خطای موجود بسیار بالا بوده است؛ بنابراین Confidence بالا لزوماً به معنی درستی Prediction نیست و برای کاربردهای حساس، Threshold و Human Review می‌تواند اهمیت داشته باشد.

## 34. Model Comparison

برای مقایسه‌ی نهایی، سه مدل روی همان Test Split شامل 776 نمونه ارزیابی شده‌اند.

| Model | Accuracy | Precision | Recall | F1 | Avg. Inference / Sample | Model Size |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9794 | 0.9271 | 0.9082 | 0.9175 | 1.00 ms | ~304 KB |
| PyTorch BiLSTM | 0.9755 | 0.8911 | **0.9184** | 0.9045 | 1.93 ms | ~6.44 MB |
| **DistilBERT Transformer** | **0.9910** | **1.0000** | 0.9286 | **0.9630** | 28.95 ms | ~268 MB weights |

اندازه‌ی حدود 268 MB مربوط به `model.safetensors` است. `training_state.pt` با اندازه‌ی حدود 536 MB برای ادامه‌ی Training نگه‌داری شده و بخشی از مدل موردنیاز برای Inference نیست. مقدار `model_size_bytes` موجود در `outputs/model_comparison.json` اندازه‌ی Artifactهای Transformer را شامل می‌شود و برای مقایسه‌ی Runtime بهتر است اندازه‌ی `model.safetensors` ملاک قرار گیرد.

در این مقایسه، Transformer بهترین Accuracy و F1 را دارد، در حالی که TF-IDF + Logistic Regression با اختلاف عملکردی کم، به‌مراتب سبک‌تر و سریع‌تر است. بنابراین انتخاب مدل به محدودیت‌های Deployment بستگی دارد: Transformer برای دقت بالاتر و مدل کلاسیک برای سرعت و مصرف منابع کمتر مناسب‌تر است.

## 35. Transformer Inference

تابع `predict_transformer(text)` متن خام را دریافت می‌کند، Tokenizer را اجرا می‌کند، `input_ids` و `attention_mask` را به مدل می‌دهد و از روی Logits با `softmax` برچسب و Confidence را تولید می‌کند.

مدل و Tokenizer فقط یک‌بار از `models/transformer_v1/` بارگذاری می‌شوند و در زمان هر Request هیچ Download یا Training انجام نمی‌شود.

Endpoint مربوط به این مدل:

```text
POST /predict/transformer
```

ساختار Response با Endpointهای قبلی سازگار است و شامل `label`, `confidence` و `model_version` با مقدار `transformer_v1` است.

## 36. Limitations

- مدل Transformer نسبت به TF-IDF و BiLSTM منابع محاسباتی و حافظه‌ی بیشتری مصرف می‌کند و میانگین زمان Inference آن روی CPU حدود 28.95 میلی‌ثانیه برای هر نمونه است؛ بنابراین برای سرویس‌های با محدودیت شدید منابع، مدل کلاسیک همچنان گزینه‌ی سبک‌تری است.
- مدل Transformer روی همین دیتاست کوچک آموزش داده شده و عملکرد آن الزاماً به دیتاست‌های بزرگ‌تر، زبان‌های دیگر یا انواع جدیدتر Spam تعمیم پیدا نمی‌کند؛ برای محیط واقعی، ارزیابی روی داده‌ی جدید و پایش افت عملکرد لازم است.
- در Test Set مدل Transformer فقط 7 خطای واقعی داشته است؛ بنابراین Error Analysis شامل همین 7 misclassification واقعی است.
- اندازه‌ی فایل `model.safetensors` مدل Transformer حدود 268 MB است؛ فایل `training_state.pt` برای ادامه‌ی Training نگه‌داری شده و برای Inference لازم نیست.
- اجرای فعلی Transformer روی CPU انجام شده و برای ترافیک بالا یا Latency پایین، استفاده از سخت‌افزار مناسب‌تر یا بهینه‌سازی Serving می‌تواند لازم باشد.
- `distilbert-base-uncased` یک مدل انگلیسی است؛ بنابراین استفاده از آن برای پیام‌های غیرانگلیسی بدون بررسی و ارزیابی جداگانه توصیه نمی‌شود.

---

# RAG Documentation Retrieval

## 37. RAG Overview

این بخش یک سامانه‌ی بازیابی اسناد (Retrieval-Augmented Generation pipeline بدون مرحله‌ی تولید پاسخ توسط LLM) را به پروژه اضافه می‌کند. هدف این پیاده‌سازی، یافتن قطعه‌های مرتبط از اسناد واقعی و برگرداندن متن، امتیاز شباهت و فراداده‌ی منبع است. هیچ سرویس خارجی برای تولید پاسخ فراخوانی نمی‌شود.

اجزای اصلی:

- خواندن و پاک‌سازی فایل‌های Markdown از `data/rag_documents/`.
- قطعه‌بندی متن با اندازه و هم‌پوشانی قابل تنظیم.
- تولید embedding با `sentence-transformers/all-MiniLM-L6-v2`؛ بردارها 384 بعدی و نرمال‌شده هستند.
- جست‌وجوی برداری با FAISS CPU و معیار inner product روی بردارهای نرمال‌شده.
- ذخیره‌سازی ایندکس و فراداده روی دیسک برای استفاده‌ی مجدد.
- ارزیابی retrieval با Recall@1، Recall@3، Recall@5 و MRR.

## 38. RAG Architecture

```text
Markdown files (data/rag_documents/)
        |
        v
load_documents() -> cleaning + title/source metadata
        |
        v
chunk_documents(size, overlap)
        |
        v
all-MiniLM-L6-v2 embeddings (384 dimensions)
        |
        v
FAISS IndexFlatIP + metadata.json + config.json
        |
        v
query embedding -> top-k similarity search
        |
        v
chunk text + score + source metadata
```

## 39. Source Documents and Reproducibility

مجموعه‌ی اسناد شامل فایل‌های مستندات FastAPI به فرمت Markdown است که در `data/rag_documents/` نگه‌داری می‌شوند. فایل‌های موجود در مخزن برای اجرای تکرارپذیر قابل استفاده‌اند. برای دانلود مجدد نسخه‌ی فعلی منابع تعریف‌شده در اسکریپت:

```powershell
python scripts/download_rag_documents.py
```

اسکریپت فایل‌ها را در `data/rag_documents/` ذخیره می‌کند. چون منبع دانلود از شاخه‌ی `master` استفاده می‌کند، محتوای دانلودشده ممکن است در طول زمان تغییر کند؛ برای تکرارپذیری دقیق‌تر، نسخه‌ی فایل‌های موجود در مخزن یا یک commit مشخص از منبع را نگه دارید.

## 40. Ingestion and Chunking

بارگذاری اسناد در `src/rag/ingestion.py` انجام می‌شود. هر سند شناسه، نام فایل منبع، عنوان و متن پاک‌سازی‌شده دارد. پاک‌سازی فاصله‌های اضافی را یکدست می‌کند و عنوان از اولین سطر Markdown که با `# ` شروع می‌شود استخراج می‌شود.

قطعه‌بندی در `src/rag/chunking.py` بر اساس تعداد کاراکتر انجام می‌شود؛ این روش ساده و قابل بازتولید است، اما مرز قطعه‌ها الزاماً با مرز معنایی جمله یا پاراگراف یکی نیست. هر قطعه اطلاعاتی مانند `chunk_id`، `document_id`، `source`، `title` و محدوده‌ی کاراکتر را حفظ می‌کند.

دو پیکربندی ارزیابی‌شده:

| نام راهبرد | اندازه‌ی قطعه | هم‌پوشانی |
|---|---:|---:|
| `strategy_500_100` | 500 کاراکتر | 100 کاراکتر |
| `strategy_800_150` | 800 کاراکتر | 150 کاراکتر |

## 41. Embeddings and Vector Store

مدل embedding از `sentence-transformers/all-MiniLM-L6-v2` استفاده می‌کند. ابعاد خروجی 384 است و embeddingها با `normalize_embeddings=True` نرمال می‌شوند. `src/rag/vector_store.py` بردارها را به نوع `float32` تبدیل و در `faiss.IndexFlatIP` ذخیره می‌کند. با توجه به نرمال بودن بردارها، inner product به‌عنوان امتیاز شباهت کسینوسی استفاده می‌شود.

فایل‌های ذخیره‌شده در هر پوشه‌ی ایندکس:

- `index.faiss`: ایندکس برداری FAISS.
- `metadata.json`: متن قطعه‌ها و فراداده‌ی مربوط به آن‌ها.
- `config.json`: پیکربندی ایندکس، از جمله ابعاد embedding.

ایندکس پس از ساخت روی دیسک ذخیره می‌شود و در درخواست‌های بعدی از روی فایل‌ها بارگذاری می‌شود؛ ایندکس‌سازی کامل با هر درخواست retrieval تکرار نمی‌شود.

## 42. Build the RAG Index

از ریشه‌ی پروژه، محیط مجازی فعال و وابستگی‌ها نصب شده باشند. برای راهبرد 500/100:

```powershell
python scripts/build_rag_index.py --chunk-size 500 --chunk-overlap 100 --output-dir data/rag_index/strategy_500_100
```

برای راهبرد 800/150:

```powershell
python scripts/build_rag_index.py --chunk-size 800 --chunk-overlap 150 --output-dir data/rag_index/strategy_800_150
```

بار اول ممکن است مدل embedding از مخزن Hugging Face دریافت شود؛ پس از دانلود، مدل در cache محلی کتابخانه نگه‌داری می‌شود. این مرحله برای ساخت embedding است، نه برای فراخوانی یک API خارجی تولید پاسخ.

## 43. RAG API Endpoints

APIهای RAG از طریق router در `src/rag/api.py` به برنامه‌ی FastAPI اضافه می‌شوند.

### `POST /ingest`

بدنه‌ی نمونه:

```json
{
  "input_dir": "data/rag_documents",
  "chunk_size": 800,
  "chunk_overlap": 150,
  "output_dir": "data/rag_index/strategy_800_150"
}
```

این endpoint اسناد را می‌خواند، قطعه‌بندی و embedding تولید می‌کند، سپس ایندکس و فراداده را ذخیره می‌کند. این عملیات را هنگام تغییر اسناد یا پیکربندی اجرا کنید، نه برای هر پرسش.

### `POST /retrieve`

بدنه‌ی نمونه:

```json
{
  "query": "How do I define a path parameter in FastAPI?",
  "top_k": 5,
  "index_dir": "data/rag_index/strategy_800_150"
}
```

پاسخ شامل پرسش، تعداد نتایج و فهرست نتایج است. هر نتیجه دارای `chunk_id`، `score`، `text` و `metadata` است. فراداده شامل شناسه‌ی سند، نام منبع، عنوان و محدوده‌ی کاراکترهاست. `top_k` باید بین 1 و 20 باشد.

### `GET /health`

endpoint موجود وضعیت مدل‌های طبقه‌بندی را گزارش می‌کند. برای این‌که retrieval کار کند، باید پوشه‌ی ایندکس انتخاب‌شده ساخته شده و در دسترس باشد.

## 44. Evaluate Retrieval

مجموعه‌ی پرسش‌های ارزیابی و منبع مورد انتظار در `data/rag_evaluation/queries.json` قرار دارد. اجرای ارزیابی:

```powershell
python scripts/evaluate_rag.py
```

اسکریپت برای هر دو مسیر `data/rag_index/strategy_500_100` و `data/rag_index/strategy_800_150` معیارهای زیر را محاسبه می‌کند:

- **Recall@1:** نسبت پرسش‌هایی که منبع مورد انتظار در رتبه‌ی اول دیده می‌شود.
- **Recall@3:** نسبت پرسش‌هایی که منبع مورد انتظار در سه نتیجه‌ی اول دیده می‌شود.
- **Recall@5:** نسبت پرسش‌هایی که منبع مورد انتظار در پنج نتیجه‌ی اول دیده می‌شود.
- **MRR:** میانگین معکوس رتبه‌ی اولین نتیجه‌ی دارای منبع مورد انتظار؛ اگر منبع در نتایج نباشد، سهم آن پرسش صفر است.

## 45. Evaluation Results and Interpretation

در اجرای ثبت‌شده روی 20 پرسش، خروجی‌ها به این شکل بودند:

| راهبرد | تعداد پرسش | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| `500/100` | 20 | 0.8000 | 0.9000 | 0.9500 | 0.8517 |
| `800/150` | 20 | 0.9000 | 0.9500 | 1.0000 | 0.9350 |

در همین مجموعه‌ی محدود، راهبرد `800/150` در هر چهار معیار بهتر بوده است. این نتایج تنها به فایل پرسش‌های فعلی و نسخه‌ی اسناد مورد استفاده مربوط‌اند؛ نباید آن‌ها را تضمین عملکرد روی پرسش‌های خارج از مجموعه تلقی کرد. برای مقایسه‌ی قابل اتکاتر، مجموعه‌ی بزرگ‌تر و متنوع‌تری از پرسش‌ها لازم است.

## 46. Retrieval Error Analysis

در خروجی ارزیابی فعلی، پرسش‌هایی که منبع مورد انتظار آن‌ها در سه رتبه‌ی اول ظاهر نمی‌شود، به همراه منابع بازیابی‌شده و امتیازهای شباهت چاپ می‌شوند. این گزارش کمک می‌کند خطاهای واقعی بررسی شوند، نه این‌که نتیجه‌ی مطلوب از پیش در کد ثابت شود.

نمونه‌های ثبت‌شده:

- در راهبرد `500/100`، پرسش درباره‌ی ساخت و ثبت middleware سفارشی، سند مورد انتظار را در رتبه‌ی پنجم پیدا کرد. نتیجه‌های اول شامل مستندات CORS و middleware پیشرفته بودند؛ موضوع‌های نزدیک می‌توانند باعث رقابت بین قطعه‌ها شوند.
- در راهبرد `500/100`، پرسش درباره‌ی رسیدگی FastAPI به اولین درخواست و برگرداندن پاسخ، در پنج نتیجه‌ی اول سند مورد انتظار را پیدا نکرد. قطعه‌های response model امتیاز بالاتری داشتند؛ احتمالاً عبارت «response» به جای مفهوم چرخه‌ی اولین درخواست، روی قطعه‌های پاسخ‌دهی اثر گذاشته است.
- در راهبرد `800/150`، پرسش درباره‌ی middleware برای پردازش request و response، سند مورد انتظار را در رتبه‌ی پنجم پیدا کرد. قطعه‌های عمومی middleware و CORS در رتبه‌های بالاتر قرار گرفتند.

این سه مورد نمونه‌های ثبت‌شده‌اند، نه تحلیل ده خطای مستقل. برای گزارش ده مورد ضعیف مستقل، باید مجموعه‌ی پرسش‌ها گسترش یابد یا پرسش‌های بیشتری به‌صورت جداگانه ارزیابی شوند؛ نباید نمونه‌های تکراری را به‌عنوان خطاهای مستقل شمرد.

## 47. Context Builder and Scope

تابع `build_context` در `src/rag/context_builder.py` متن قطعه‌های بازیابی‌شده را با اطلاعات منبع، عنوان، شناسه‌ها و امتیاز شباهت ترکیب می‌کند. پارامتر `max_chars` امکان محدود کردن طول متن زمینه را می‌دهد.

این پروژه در حال حاضر یک سیستم retrieval است؛ پاسخ نهایی توسط LLM تولید نمی‌شود. بنابراین کیفیت بازیابی، امتیازها و متن اسناد مستقیماً قابل مشاهده‌اند و خطاهای retrieval با تولید متن توسط یک مدل زبانی پنهان نمی‌شوند.

## 48. Docker Usage

ساخت image از ریشه‌ی پروژه:

```powershell
docker build -t ai-text-classifier .
```

اجرای سرویس با نگه‌داری داده‌ها و ایندکس روی میزبان:

```powershell
docker run --rm -p 8000:8000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/models:/app/models" `
  ai-text-classifier
```

mount کردن `data` باعث می‌شود اسناد، ایندکس‌های RAG و خروجی‌های ذخیره‌شده خارج از لایه‌ی موقت کانتینر باقی بمانند. پوشه‌ی `models` نیز برای بارگذاری artifactهای طبقه‌بندی روی میزبان نگه‌داری می‌شود. در Windows PowerShell از backtick برای ادامه‌ی خط استفاده شده است؛ دستور را می‌توان در یک خط هم نوشت.

برای retrieval، پیش از درخواست `/retrieve` باید ایندکس در مسیر داخل کانتینر موجود باشد یا با `POST /ingest` ساخته شود. دانلود مدل embedding ممکن است در اولین اجرا به اتصال اینترنت نیاز داشته باشد؛ برای محیط بدون اینترنت باید cache مدل را از قبل آماده و در دسترس قرار دهید.

## 49. Dependencies and Runtime Notes

وابستگی‌های اصلی RAG عبارت‌اند از `sentence-transformers` برای embedding و `faiss-cpu` برای جست‌وجوی برداری. نسخه‌ی Python پایه‌ی Docker در `Dockerfile` برابر 3.11 است. برای محیط قابل بازتولید در استقرار نهایی، نسخه‌های کتابخانه‌ها و نسخه‌ی منبع اسناد را pin کنید و تست‌ها را در همان محیط اجرا کنید.

## 50. Limitations

- قطعه‌بندی فعلی بر اساس تعداد کاراکتر است و ساختار معنایی عنوان‌ها و پاراگراف‌ها را به‌صورت ویژه مدل نمی‌کند.
- مجموعه‌ی ارزیابی فقط 20 پرسش دارد؛ بنابراین نتایج اولیه‌اند و برای ادعای کیفیت عمومی کافی نیستند.
- منبع مورد انتظار در ارزیابی در سطح فایل تعریف شده است؛ این معیار لزوماً درست بودن دقیق قطعه‌ی بازیابی‌شده را نمی‌سنجد.
- شباهت embedding معادل درستی factual نیست؛ امتیاز بالا تضمین نمی‌کند قطعه دقیقاً پاسخ پرسش را پوشش دهد.
- محتوای منابع از مستندات FastAPI است؛ عملکرد روی حوزه‌های دیگر باید جداگانه ارزیابی شود.
- بهبودهای بعدی می‌توانند شامل افزایش تنوع پرسش‌های ارزیابی، pin کردن commit منبع اسناد، قطعه‌بندی آگاه از Markdown و بررسی خطاهای رتبه‌بندی باشند.

---

## References

- Dataset: [SMS Spam Collection – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- کتابخانه‌ها: [scikit-learn](https://scikit-learn.org/stable/documentation.html) (`TfidfVectorizer`, `LogisticRegression`, `Pipeline`)، [FastAPI](https://fastapi.tiangolo.com/)، [Pydantic](https://docs.pydantic.dev/)، [joblib](https://joblib.readthedocs.io/)، [pytest](https://docs.pytest.org/)
- Transformer: [Hugging Face Transformers](https://huggingface.co/docs/transformers/)، [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)
- PyTorch: [PyTorch Documentation](https://pytorch.org/docs/stable/)