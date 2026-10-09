# این اسکریپت اسناد مستندات FastAPI را از منبع رسمی دریافت می‌کند.

from pathlib import Path
from urllib.request import urlopen


# نشانی ریشه اسناد رسمی FastAPI که به‌صورت بازتولیدپذیر دریافت می‌شوند.
BASE_URL = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs"

# نگاشت نام محلی هر فایل به مسیر آن در مخزن منبع.
DOCUMENTS = {
    "index.md": "index.md",
    "tutorial_first_steps.md": "tutorial/first-steps.md",
    "tutorial_path_params.md": "tutorial/path-params.md",
    "tutorial_query_params.md": "tutorial/query-params.md",
    "tutorial_body.md": "tutorial/body.md",
    "tutorial_response_model.md": "tutorial/response-model.md",
    "tutorial_request_files.md": "tutorial/request-files.md",
    "tutorial_handling_errors.md": "tutorial/handling-errors.md",
    "tutorial_dependencies.md": "tutorial/dependencies/index.md",
    "tutorial_security.md": "tutorial/security/first-steps.md",
    "tutorial_background_tasks.md": "tutorial/background-tasks.md",
    "tutorial_cors.md": "tutorial/cors.md",
    "tutorial_sql_databases.md": "tutorial/sql-databases.md",
    "tutorial_testing.md": "tutorial/testing.md",
    "tutorial_middleware.md": "tutorial/middleware.md",
    "tutorial_websockets.md": "advanced/websockets.md",
    "tutorial_events.md": "advanced/events.md",
    "tutorial_metadata.md": "tutorial/metadata.md",
    "tutorial_static_files.md": "tutorial/static-files.md",
    "advanced_middleware.md": "advanced/middleware.md",
}


# دانلود اسناد و ذخیره آن‌ها در پوشه داده‌های RAG.
def download_documents() -> None:
    output_dir = Path("data/rag_documents")
    output_dir.mkdir(parents=True, exist_ok=True)

    for output_name, source_path in DOCUMENTS.items():
        url = f"{BASE_URL}/{source_path}"
        output_path = output_dir / output_name

        with urlopen(url, timeout=30) as response:
            content = response.read().decode("utf-8")

        output_path.write_text(content, encoding="utf-8")
        print(f"Downloaded: {output_name}")


if __name__ == "__main__":
    download_documents()