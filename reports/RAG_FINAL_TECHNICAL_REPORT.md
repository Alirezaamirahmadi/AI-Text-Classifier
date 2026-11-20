# گزارش فنی نهایی RAG

- وضعیت کلی: **PARTIALLY_READY**
- Commit مرجع: `8c8404a` (آخرین commit مرحله‌ی ۹)

## ۱. معماری فعلی (از کد)
| بخش | فایل | وضعیت |
|---|---|---|
| Document Loading / Chunking / Ingestion | `ingestion.py`, `chunking.py` | موجود؛ ورودی نامعتبر با خطای 400/422 رد می‌شود |
| Embedding | `embeddings.py` (all-MiniLM-L6-v2) | موجود؛ قطع ارتباط با Embedding Provider تست نشده |
| Vector Index | `vector_store.py` (FAISS) | خطای فایل ناموجود و ناسازگاری بُعد بررسی می‌شود |
| Retrieval | `retrieval.py` | ورودی خالی و top_k نامعتبر رد می‌شود |
| Context Construction | `context_builder.py` | سقف طول رعایت می‌شود (تست پایداری) |
| Generation | `generation.py` | OpenRouter یا `LLM_BASE_URL`؛ Timeout و 429/401/5xx و پاسخ Malformed به خطای قابل‌کنترل تبدیل می‌شوند |
| Citation Validation | `citation_validation.py` | شماره‌ی منبع و حضور در Context؛ **ارتباط معنایی ادعا با منبع بررسی نمی‌شود** |
| Abstention | `answering.py` | جمله‌ی استاندارد (حتی با متن اضافه) → insufficient_context |
| Evaluation | `evaluation_dataset.py`، `run_eval_simple.py`، `build_stage9_outputs.py` | Dataset ثابت ۷۸ سؤال |

## ۲. تغییرات این مرحله
- افزودن `tests/test_rag_stability.py` (۷ تست): ورودی خالی، Context خالی، چانک خالی، سقف طول Context، Prompt Injection به‌عنوان متن، ارجاع به منبع ناموجود، سؤال چندبخشی.
- هیچ تغییر معماری یا قابلیت جدید انجام نشد.

## ۳. نتایج تست‌ها
- تست‌های خالص‌پایتونی: ۷۷ قبول، ۰ شکست، ۲ تست پارامتری فقط در pytest واقعی اجرا می‌شوند (در لپ‌تاپ قبلاً قبول شدند).
- تست‌های API و یکپارچگی: در این محیط اجرا نشدند.
- گزارش pytest واقعی لپ‌تاپ (مرحله‌ی ۹): ۱۳۵ قبول.

## ۴. کیفیت Retrieval (اندازه‌گیری‌شده)
Recall@1 = 0.7458، Recall@3 = 0.8458، Recall@5 = 0.8708، MRR = 0.9125 (۴۰ سؤال پاسخ‌دار). Precision@K: not_measured.

## ۵. کیفیت Generation
- وضعیت‌ها برای دو مدل محلی در `outputs/rag_final_evaluation.json` آمده است.
- Answer Correctness، Faithfulness، Citation Support، Answer Relevancy، Abstention Quality (معنایی): **not_measured**. داور مستقل وجود ندارد؛ مدل محلی به‌عنوان داور قابل‌اعتماد نیست.
- پاسخ کامل درست در دو مدل تقریباً صفر است؛ مدل ۳ میلیاردی تعداد پاسخ‌های غیرمعتبر را کم کرده ولی بیشتر خودداری کرده است.

## ۶. Performance (اجرای واقعی روی لپ‌تاپ، مدل Qwen2.5-3B محلی)
- میانگین زمان پاسخ: **۲۳٫۲۴ ثانیه** برای هر سؤال
- P95: **۲۷٫۱۷ ثانیه**
- توکن: `not_measured` (تابع تولید تعداد توکن برنمی‌گرداند)
- خطای Provider در این اجرا: ۴ مورد (`generation_unavailable`)
- یک اجرا فقط؛ تکرار و ثبات زمان‌ها اندازه‌گیری نشده است.
- شواهد: `outputs/rag_final_performance.json`

## ۷. خطاهای باقی‌مانده و محدودیت‌ها
- پذیرش کاذب: RAG-055، RAG-071 (هر دو مدل).
- پاسخ بخشی/غلط با شکل ارجاع معتبر (مدل ۳ میلیاردی).
- برچسب RAG-029 در دیتاست نادرست است (ثبت شده در گزارش مرحله‌ی ۹).
- ارجاع به سند اشتباه (مثلاً RAG-013، RAG-016): ارزیابی فقط وجود شماره‌ی منبع را می‌بیند.
- مدل محلی کوچک؛ مدل بزرگ‌تر یا ابری بنچمارک نشده است.

## ۸. وضعیت نهایی
**PARTIALLY_READY**. زیرساخت پایدار و تست‌ها موجودند؛ کیفیت Generation پایین است و معیارهای معنایی و Performance اندازه‌گیری نشده‌اند.
