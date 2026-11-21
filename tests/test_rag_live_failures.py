"""تست خطاهای واقعی سرویس مدل با یک سرور HTTP محلی (بدون Mock)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.rag.generation import GenerationError, generate_answer


class _Handler(BaseHTTPRequestHandler):
    mode = "ok"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.mode == "slow":
            time.sleep(3)
        if self.mode == "http500":
            self.send_response(500)
            self.end_headers()
            return
        if self.mode == "malformed":
            body = b"this is not json"
        else:
            body = json.dumps({"choices": [{"message": {"content": "ok [Source 1]"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()


def _url(srv):
    return f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"


def test_real_timeout_is_generation_error(server, monkeypatch):
    _Handler.mode = "slow"
    monkeypatch.setenv("LLM_BASE_URL", _url(server))
    with pytest.raises(GenerationError):
        generate_answer("q", "[Source 1] text", timeout=0.5)


def test_real_malformed_reply_is_generation_error(server, monkeypatch):
    _Handler.mode = "malformed"
    monkeypatch.setenv("LLM_BASE_URL", _url(server))
    with pytest.raises(GenerationError):
        generate_answer("q", "[Source 1] text")


def test_real_http_500_is_generation_error(server, monkeypatch):
    _Handler.mode = "http500"
    monkeypatch.setenv("LLM_BASE_URL", _url(server))
    with pytest.raises(GenerationError):
        generate_answer("q", "[Source 1] text")


def test_real_ok_reply_is_returned(server, monkeypatch):
    _Handler.mode = "ok"
    monkeypatch.setenv("LLM_BASE_URL", _url(server))
    assert generate_answer("q", "[Source 1] text") == "ok [Source 1]"