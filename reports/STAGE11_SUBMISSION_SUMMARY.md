# خلاصه تحویل مرحله ۱۱ — زیرسیستم RAG

**وضعیت: PARTIALLY_READY**

## شواهد موفق
- اجرای محلی گزارش‌شده: `python -m pytest -q` → 149 passed، صفر failed، یک warning، در 8.37 ثانیه.
- اجرای محلی گزارش‌شده: `python scripts/smoke_test_rag_citation.py` → هر 7 بررسی موفق.
- `python scripts/build_rag_stage11_reports.py` اجرا شده و خروجی‌های تاریخی را بازسازی/همگام کرده است.
- در محیط بازبینی این بسته، Smoke Test هفت‌گانه مجدداً اجرا شد و موفق بود؛ فایل‌های Python نیز از نظر Syntax کامپایل شدند.

## نتایج تاریخی RAG
- Retrieval روی نتایج ذخیره‌شده: Recall@1=0.875، Recall@3=0.95، Recall@5=0.95، MRR=0.9125.
- Generation روی 78 سؤال: 4 answered، 29 uncited، 42 insufficient_context، 2 invalid_generation و 1 generation_unavailable.
- از 40 سؤال پاسخ‌پذیر، فقط 4 مورد answered و 27 مورد uncited بوده‌اند؛ این گلوگاه اصلی قابل پیگیری است.
- Performance از اجرای تاریخی جداگانه: میانگین 3.056s و P95 برابر 5.676s.

## موارد باز
- اجرای تازه و یکپارچه Retrieval و Generation روی هر 78 سؤال.
- ارزیابی معنایی Correctness/Faithfulness/Citation Support/Relevancy/Abstention.
- تست با Provider واقعی و مدل زنده، و تست مستقل هم‌زمانی.
- اجرای clean-install و بررسی امنیتی نهایی Secrets.

## تفسیر
نتایج تست واحد و Smoke Test مثبت‌اند، اما به‌تنهایی اثبات‌کننده کیفیت معنایی Generation یا تکمیل کامل پذیرش عملیاتی RAG نیستند. وضعیت نهایی عمداً `PARTIALLY_READY` باقی مانده است.

**منشأ شواهد تست محلی:** خروجی ترمینالی ارائه‌شده توسط مسئول پروژه؛ این خروجی در محیط گزارش‌سازی مستقلاً بازاجرا نشده است. جزئیات در `outputs/rag_external_validation.json` و `outputs/rag_final_regression.json` ثبت شده‌اند.
