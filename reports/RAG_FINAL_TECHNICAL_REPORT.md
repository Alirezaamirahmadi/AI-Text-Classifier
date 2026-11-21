# گزارش فنی نهایی RAG (مرحله‌ی ۱۰)

- وضعیت نهایی: **PARTIALLY_READY**
- Commit مرجع: `314f479` (قبل از تغییرات این مرحله)
- مدل تولید: Qwen2.5-3B-Instruct (Q4_K_M) محلی با `llama-server`؛ مدل دوم: Qwen2.5-0.5B
- Dataset: ۷۸ سؤال (۴۰ پاسخ‌دار، ۳۸ بی‌پاسخ)؛ بدون تغییر

## ۱. معماری

| بخش | فایل | وضعیت |
|---|---|---|
| Document Loading / Chunking / Ingestion | `ingestion.py`، `chunking.py` | ورودی نامعتبر با خطای 400/422 رد می‌شود |
| Embedding | `embeddings.py` (all-MiniLM-L6-v2، محلی) | خطای بازیابی حالا کنترل‌شده است (`retrieval_unavailable`) |
| Vector Index | `vector_store.py` (FAISS) | خطای فایل ناموجود و ناسازگاری بُعد بررسی می‌شود |
| Retrieval | `retrieval.py` | ورودی خالی و top_k نامعتبر رد می‌شود |
| Context Construction | `context_builder.py` | سقف طول رعایت می‌شود؛ قطعه‌های تکراری حذف می‌شوند |
| Generation | `generation.py` | OpenRouter یا `LLM_BASE_URL`؛ Timeout، 429/401/5xx، پاسخ Malformed به خطای کنترل‌شده تبدیل می‌شوند؛ مقدار `usage` ذخیره می‌شود |
| Citation Validation | `citation_validation.py` | شماره‌ی منبع و حضور در Context؛ **ارتباط معنایی ادعا با منبع بررسی نمی‌شود** |
| Abstention | `answering.py` | جمله‌ی استاندارد (حتی با متن اضافه) → `insufficient_context` |
| Evaluation | `evaluation_dataset.py`، `run_eval_simple.py` | Dataset ثابت ۷۸ سؤال |

## ۲. تغییرات این مرحله (کد)
1. **خطای بازیابی/Embedding کنترل شد:** اگر `retrieve` خطا بدهد، وضعیت `retrieval_unavailable` برمی‌گردد و برنامه کرش نمی‌کند.
2. **حذف قطعه‌های تکراری:** قطعه‌هایی با متن یکسان فقط یک بار وارد Context می‌شوند.
3. **ذخیره‌ی توکن:** مقدار `usage` پاسخ سرویس در `generation.LAST_USAGE` نگه داشته می‌شود.
4. **تست‌های جدید:** ۷ تست پایداری در `tests/test_rag_stability.py` و ۴ تست خطای واقعی سرویس در `tests/test_rag_live_failures.py`.

## ۳. نتایج تست‌ها
- `python -m pytest -q` از محیط تمیز با وابستگی‌های پین‌شده: **۱۴۹ قبول، ۰ شکست**.
- وابستگی‌ها: `requirements.txt` با نسخه‌های دقیق (از محیط واقعی).

## ۴. تست‌های پایداری (رفتار واقعی)
| سناریو | نتیجه |
|---|---|
| ورودی خالی | رد می‌شود (ValueError) |
| Context یا سند خالی | `insufficient_context`، بدون فراخوانی مدل |
| Context بیش از حد مجاز | سقف طول رعایت می‌شود |
| Prompt Injection در سند | به‌عنوان متن وارد Context می‌شود |
| ارجاع به منبع ناموجود | `invalid_citations` |
| سؤال چندبخشی | هر دو منبع ارجاع داده می‌شوند |
| قطع Retrieval/Embedding (شبیه‌سازی در مسیر کد) | `retrieval_unavailable`، بدون کرش |
| سند تکراری | یک بار در Context |
| Timeout واقعی سرویس (سرور HTTP محلی) | `generation_unavailable` |
| پاسخ Malformed واقعی | `generation_unavailable` |
| HTTP 500 واقعی | `generation_unavailable` |
| پاسخ معتبر واقعی | پذیرفته می‌شود |
| قطع سرویس مدل (آدرس اشتباه، واقعی) | `generation_unavailable`، بدون کرش |
| هم‌زمانی سه درخواست (واقعی، `-np 3`) | **ناموفق**: هر سه `generation_unavailable` |

## ۵. کیفیت Retrieval (اندازه‌گیری‌شده)
- Recall@1 = 0.7458، Recall@3 = 0.8458، Recall@5 = 0.8708، MRR = 0.9125 (۴۰ سؤال پاسخ‌دار).
- Precision@1 = 0.875، Precision@3 = 0.3333، Precision@5 = 0.21 (همان ۴۰ سؤال).
- بهینه‌سازی Retrieval (Hybrid، Reranking) انجام نشد؛ هیچ تغییری نسبت به Benchmark قبلی وجود ندارد.

## ۶. کیفیت Generation

| وضعیت (۷۸ سؤال) | Qwen 0.5B | Qwen 3B |
|---|---:|---:|
| answered | 3 | 6 |
| invalid_generation | 12 | 2 |
| insufficient_context | 36 | 42 |
| uncited | 26 | 28 |
| generation_unavailable | 1 | 0 |

- بررسی دستی: از ۴ پاسخ `answered` در اجرای قبلی مدل ۳B، ۱ درست، ۲ بخشی درست، ۱ غلط. از ۲۷ پاسخ `uncited` بررسی‌شده، تقریباً ۹ جواب درست بدون ارجاع معتبر بودند.
- پاسخ کامل درست: تقریباً صفر. مدل ۳B تعداد پاسخ‌های غیرمعتبر را کم کرده ولی بیشتر خودداری می‌کند.
- پذیرش کاذب باقی‌مانده: RAG-055 و RAG-071.
- ارجاع به سند اشتباه: مثلاً RAG-013 و RAG-016 (ارزیابی فقط وجود شماره‌ی منبع را می‌بیند).

## ۷. Performance و توکن (اجرای نهایی، مدل ۳B)
- میانگین زمان: **3.06 s**، P95: **5.68 s**
- توکن: prompt = **110٬679**، completion = **2٬373**
- وضعیت‌ها: answered 6، insufficient_context 42، uncited 28، invalid_generation 2، خطای سرویس صفر
- یک اجرا. این زمان بسیار کمتر از اجراهای قبلی (حدود ۲۰ تا ۸۰ ثانیه) است و احتمالاً به کش پرامپت `llama-server` مربوط است؛ با اجراهای قبلی قابل مقایسه نیست.
- شواهد: `outputs/rag_performance_final.json`

## ۸. معیارهای معنایی (وضعیت)
| معیار | وضعیت |
|---|---|
| Answer Correctness | `not_measured` — داور معنایی مستقل وجود ندارد |
| Faithfulness | `not_measured` — همان دلیل |
| Citation Support (معنایی) | `not_measured` — همان دلیل |
| Answer Relevancy | `not_measured` — همان دلیل |
| Abstention Quality (معنایی) | `not_measured` — همان دلیل |

مدل محلی به‌عنوان داور قابل‌اعتماد نیست، پس استفاده نشد.

## ۹. BLOCKED و موارد باز (با دلیل)
- **داور معنایی مستقل:** BLOCKED — هیچ داور مستقل و قابل‌اعتماد در دسترس نیست.
- **هم‌زمانی چند درخواست:** ناموفق با تنظیمات فعلی (CPU، ۸ گیگ RAM، timeout=30s). برای حل: افزایش timeout یا سخت‌افزار قوی‌تر؛ انجام نشد.
- **Prompt Injection در سطح زنده:** فقط در سطح کد بررسی شد (سند مخرب به‌عنوان متن)؛ آزمون زنده با مدل انجام نشد.
- **قطع Embedding Provider زنده:** Embedding محلی است؛ شبیه‌سازی در مسیر کد انجام شد، آزمون زنده انجام نشد.
- **بنچمارک مدل بزرگ‌تر یا ابری:** انجام نشد (RAM محدود).
- **برچسب RAG-029:** نادرست است و عمداً تغییر نکرده (ثبت‌شده).

## ۱۰. محدودیت‌های اصلی
- کیفیت پاسخ کم است؛ تقریباً هیچ پاسخ کاملاً درستی وجود ندارد.
- معیارهای معنایی `not_measured` هستند.
- Performance یک اجرا است.
- ارزیابی ارجاع فقط وجود شماره‌ی منبع را می‌سنجد، نه درست بودن سند یا معنا.
- هیچ ادعایی درباره‌ی حذف کامل Hallucination وجود ندارد.

## ۱۱. وضعیت نهایی
**PARTIALLY_READY.** اجزای اصلی پایدار و تست‌ها قبول‌اند؛ خطاهای سرویس کنترل شده‌اند؛ اندازه‌گیری Retrieval و توکن انجام شده. کیفیت Generation، معیارهای معنایی و همزمانی هنوز به تکمیل نیاز دارند.