# گزارش نهایی Stage 12 — Semantic Validation, Citation Reliability & End-to-End Verification

**وضعیت نهایی: PARTIALLY_READY**

**SHA آخرین Commit:** `c0694ac430749c0acf9bd638796f16e92801d83d` (Stage 11 base؛ تغییرات Stage 12 فعلاً commit نشده‌اند)

> این گزارش فقط شامل نتایجی است که واقعاً اجرا شده‌اند. هر چیزی که اجرا نشده، صریحاً `not_measured` یا `not_run` ثبت شده است.

## ۱. خلاصه

- ساختار اعتبارسنجی Citation اصلاح شد و رفتار هر Claim جداگانه بررسی می‌شود.
- ۲۲ تست Regression جدید در فایل فعلی دیده می‌شود. گزارش اجرای قبلی **۱۷۱ از ۱۷۱ passed** ثبت کرده است؛ اجرای مجدد در این محیط ممکن نشد و SHA گزارش‌شده همان Stage 11 است، پس این نتیجه به کد فعلی منتسب نمی‌شود.
- ۷۸ سؤال با مدل واقعی اجرا شد. پاسخ‌های بدون Citation معتبر دیگر «تأییدشده» ارائه نمی‌شوند.
- پنج معیار معنایی روی هر ۷۸ مورد امتیازدهی شدند؛ جزئیات، روش و نمرهٔ تک‌تک سؤال‌ها در دو فایل `outputs/rag_stage12_semantic_evaluation.json` و `outputs/rag_stage12_semantic_evaluation_assistant_review.json` است.
- کیفیت مدل (۳۰ از ۴۰ سؤال پاسخ‌پذیر بدون `[Source N]`) هنوز مشکل اصلی است.

### ارزیابی معنایی (همهٔ ۷۸ پاسخ)

برای بررسی متنی و فهرست‌بندی خطاها، هر ۷۸ خروجی E2E با پاسخ مرجع، required facts، متن خام مدل و Context/Citation همان اجرا مرور شد. جزئیات و امتیاز هر سؤال در `outputs/rag_stage12_semantic_evaluation_assistant_review.json` است.

روش امتیازدهی در JSON ثبت شده است: پاسخ با مرجع و required facts مقایسه شد؛ ادعاها با Context نهایی سنجیده شدند؛ برای Citation، متن دقیق منبع شماره‌خورده بررسی شد؛ ارتباط پاسخ با سؤال از ۰ تا ۲ امتیاز گرفت.

| معیار | نتیجه |
|---|---:|
| Answer Correctness | ۳۴ correct، ۳۳ partial، ۱۱ wrong |
| Faithfulness | ۶۶ yes، ۱۲ no |
| Citation Support | ۳۵ yes، ۴۳ no |
| Answer Relevancy | میانگین ۰٫۹۷۴ از ۲ (۳۵ نمرهٔ ۰، ۶ نمرهٔ ۱، ۳۵ نمرهٔ ۲) |

در میان ۷ پاسخ `answered`، سه سؤال غیرقابل‌پاسخ به‌اشتباه پاسخ داده شدند: RAG-046 (ادعای ساختگی دربارهٔ گذرواژه)، RAG-047 (ارجاع Cloud به‌عنوان منطقه بدون تعیین منطقه)، و RAG-071 (پاسخ FastAPI Cloud به سؤال Kubernetes/Helm). جزئیات reasoning هر مورد در فایل review است.

## ۲. مشکلات حل‌شده (هرکدام با تست Regression)

| مشکل | اصلاح | تست |
|---|---|---|
| `split_claims` وقتی Citation بعد از نقطه می‌آمد جمله‌ها را جدا نمی‌کرد و پاسخ چندادعایی یک Claim حساب می‌شد | الگوی جداسازی اصلاح شد | `test_multi_claim_*` |
| خروجی `validate_citations` در تصمیم نهایی استفاده نمی‌شد | ارجاع نامعتبر، Claim بدون Citation و پشتیبانی تأییدنشده → پاسخ withheld | `test_citation_to_*`, `test_multi_claim_*` |
| `context_results` با شمردن `[Source N]` در متن Context ساخته می‌شد و متن سند (Prompt Injection) می‌توانست آن را جعل کند | `build_context_with_sources` فهرست دقیق منابع واردشده را برمی‌گرداند | `test_prompt_injection_*`, `test_citation_to_source_cut_*` |
| بررسی تکراری پاسخ خالی در `answering.py` | حذف شد | `test_empty_or_incomplete_*` |
| Token Usage سراسری بود و در اجرای هم‌زمان قاطی می‌شد | مقدار per-thread (`get_last_usage`) | `test_usage_is_kept_per_thread` |
| ثبت Context، Raw Generation و Token Usage هر سؤال ممکن نبود | پارامتر اختیاری `trace` در `answer_question` (خروجی تابع تغییر نمی‌کند) | `test_trace_receives_*` |
| CI در ZIP قابل مشاهده نیست | Workflow در آرشیو حاضر نشده است | تعریف و اجرا نشده |

## ۳. سناریوهای پایداری و امنیت (تست مستقل برای هر کدام، Mock)

Citation به منبع ناموجود، Citation به منبع نامرتبط، پاسخ چندادعایی (یکی بدون Citation / هرکدام جداگانه)، پاسخ بدون Citation، سؤال بی‌پاسخ، Context Dump، پاسخ خالی/ناقص، Timeout و قطع Provider، Prompt Injection در متن سند، Context بیش از حد مجاز، درخواست‌های هم‌زمان، تطبیق شماره Source بعد از حذف تکراری‌ها. همه passed شدند.
محدودیت: Timeout و هم‌زمانی فقط با Mock تست شده‌اند، نه با Provider واقعی.

## ۴. اجرای واقعی ۷۸ سؤال (E2E)

- مدل پاسخ‌دهنده: `qwen2.5-3b-instruct` (llama-server محلی)، `top_k=5`، `max_context_chars=10000`، `temperature=0`.
- ایندکس: `data/rag_index/strategy_800_150`.
- فایل نتیجه: `outputs/rag_stage12_e2e_results.json` (برای هر سؤال: Expected/Retrieved Sources، Final Context، Raw Generation، Final Answer، Citation Validation، Status، زمان، Token Usage، خطا).

| وضعیت نهایی | تعداد |
|---|---|
| insufficient_context | 36 |
| uncited | 34 |
| answered | 7 |
| invalid_generation | 1 |
| generation_unavailable | 0 |

### مقایسه با اجراهای قبلی (ادغام نشده)

| وضعیت | Stage 11 (فایل‌های تاریخی) | Stage 12 اجرای ذخیره‌شده |
|---|---|---|
| answered | 6 | 7 |
| uncited | 28 | 34 |
| insufficient_context | 42 | 36 |
| invalid_generation | 2 | 1 |
| generation_unavailable | 0 | 0 |

مقایسه عددی با Stage 9 در این گزارش انجام نشد، چون تنظیمات و مدل آن اجرا متفاوت است. اعداد آن مرحله در فایل‌های `outputs` تاریخی خودش باقی می‌مانند و با این نتایج ادغام نشده‌اند.

### نوسان بین اجراها (مهم)

فقط اجرای ذخیره‌شده در `outputs/rag_stage12_e2e_results.json` قابل ممیزی است. اشارهٔ قبلی به اجرای اول بدون فایل پشتیبان، شاهد قابل بازبینی محسوب نمی‌شود و از مقایسه حذف شده است.
نکته: Run metadata و regression report تاریخ 2026-11-27 به SHA پایه Stage 11 اشاره می‌کنند؛ بنابراین E2E/تست‌های آن اجرا اثبات اجرای کد تغییریافته Stage 12 نیستند.

## ۵. ارزیابی معنایی (Semantic Evaluation)

فایل خلاصه: `outputs/rag_stage12_semantic_evaluation.json`. امتیاز هر سؤال و توضیح داوری: `outputs/rag_stage12_semantic_evaluation_assistant_review.json`.

روش: Answer Correctness با پاسخ مرجع و required facts سنجیده شد (`correct / partial / wrong`). Faithfulness بر اساس پشتیبانی ادعاهای پاسخ خام از Context نهایی سنجیده شد (`yes / no`). Citation Support با خواندن متن دقیق Source ارجاع‌شده ارزیابی شد (`yes / no`). Answer Relevancy از ۰ (نامرتبط/بی‌پاسخ) تا ۲ (مستقیم و مرتبط) امتیاز گرفت. برای هر ۷۸ سؤال همهٔ این برچسب‌ها ثبت شده‌اند.

| معیار | نتیجه |
|---|---|
| Answer Correctness | ۳۴ correct، ۳۳ partial، ۱۱ wrong |
| Faithfulness | ۶۶ yes، ۱۲ no |
| Citation Support | ۳۵ yes، ۴۳ no |
| Answer Relevancy | میانگین ۰٫۹۷۴ از ۲؛ ۳۵ امتیاز ۰، ۶ امتیاز ۱، ۳۵ امتیاز ۲ |
| Abstention Quality | ۳۲/۳۸ خودداری درست (۸۴٫۲٪)، ۴/۴۰ خودداری اشتباه (۱۰٪) |

**Abstention Quality** — روش: برچسب `answerable` در Dataset، و «رد کردن» یعنی وضعیت نهایی `insufficient_context`. این معیار رفتار سیستم را می‌سنجد، نه کیفیت متن پاسخ.

- سؤال بی‌پاسخ: ۳۲ از ۳۸ با وضعیت `insufficient_context` رد شد → **۸۴٫۲٪**. سه مورد `uncited`، سه مورد `answered`.
- سؤال پاسخ‌پذیر: ۴ از ۴۰ اشتباهاً `insufficient_context` شد → **۱۰٪**.

## ۶. عملکرد (Performance)

فایل: `outputs/rag_stage12_performance.json` — اجرای پشت‌سرهم، یک درخواست در هر لحظه.

| شاخص | Stage 12 | Stage 11 (تاریخی) |
|---|---|---|
| میانگین Latency | 19.887 s | 3.056 s |
| P95 | 24.122 s | 5.676 s |
| حداکثر | 27.349 s | — |
| مجموع Token (۷۸ سؤال) | 115,152 | 110,679 |

تفاوت‌ها در محدوده نوسان معمول اجرا روی CPU است و ادعای بهبود نیست. Token Usage برای هر ۷۸ سؤال ثبت شد.

## ۷. Clean Install و تست‌ها (اجرای محلی)

- لاگ قبلی ادعای محیط جدید (`venv-clean`)، Python **3.14.10** و نصب کامل از `requirements.txt` را ثبت می‌کند؛ محیط و نصب در آرشیو حاضر نیست و اینجا بازتولید نشد.
- در لاگ ذخیره‌شده **۱۷۱ از ۱۷۱ passed** (`pytest_exit_code: 0`)، شامل ۲۲ تست Stage 12 در فایل فعلی. این اجرا روی SHA پایه Stage 11 ثبت شده، نه محتوای فعلی working tree.
- Smoke Test (`scripts/smoke_test_rag_citation.py`): ۷ بررسی passed.
- در محیط جاری فقط Python 3.15 نصب است و `pytest` در دسترس نیست؛ `py -0p` هم Python 3.14 دیگری پیدا نکرد.
- Secrets: جست‌وجوی متنی ساده انجام شد؛ کلید واقعی پیدا نشد (فقط Placeholder در README) و `.env` در Git ردیابی نمی‌شود. ابزار تخصصی اسکن Secrets اجرا نشد.

## ۸. اجرا نشده / محدودیت‌ها

- **GitHub Actions:** workflow ساده در `.github/workflows/ci.yml` اضافه شده است؛ هنوز روی GitHub اجرا نشده و باید پس از push نتیجه‌اش بررسی شود.
- **تطبیق README با کد:** به‌صورت کامل بررسی نشد. README هنوز Stage 12 و نسخه Python 3.14 را توضیح نمی‌دهد.
- هم‌پوشانی واژگانی اثبات معنایی نیست. پاسخ غلطی که واژه‌های منبع را تکرار کند (مثلاً نفی) عبور می‌کند؛ تست `test_known_limit_*` این را مستند می‌کند. `semantic_entailment_verified` همیشه `False` است.
- `split_claims` یک Heuristic است و ممکن است مخفف‌ها را اشتباه جدا کند.
- آستانه هم‌پوشانی روی داده واقعی کالیبره نشده است.
- درخواست هم‌زمان و Timeout با Provider واقعی تست نشد.
- امتیازهای معنایی هر ۷۸ مورد در خروجی ثبت شده‌اند؛ روش و نتایج در بخش ۵ آمده است.

## ۹. مشکلات باقی‌مانده

1. **۳۴ از ۴۰ سؤال پاسخ‌پذیر `uncited` شدند** و پاسخ نهایی‌شان withheld بود.
2. ۴ سؤال پاسخ‌پذیر اشتباهاً با `insufficient_context` رد شدند.
3. اجرای تازهٔ E2E روی تغییرات فعلی لازم است؛ خروجی ثبت‌شده به SHA Stage 11 منتسب است.

## ۱۰. وضعیت پذیرش: PARTIALLY_READY

دلایل: CI هنوز از GitHub اجرا نشده، اجرای تست/E2E فعلی به SHA تغییرات منتسب نیست، و ۳۴/۴۰ سؤال پاسخ‌پذیر به پاسخ نهاییِ قابل‌قبول نرسیدند. نتیجهٔ ارزیابی معنایی و روش آن در فایل‌های خروجی ثبت شده‌اند. برای READY_FOR_ACCEPTANCE لازم است اجرای تست/E2E روی commit نهایی و CI سبز بررسی شود.
