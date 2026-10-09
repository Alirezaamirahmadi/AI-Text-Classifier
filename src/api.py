from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.responses import JSONResponse

from .config import MAX_TEXT_LENGTH, MODEL_PATH, MODEL_VERSION
from .predict import load_model, predict
from .dl_predict import load_pytorch_model, predict_dl
from src.rag.api import router as rag_router

try:
    from .transformer_predict import load_transformer_model, predict_transformer
except ImportError:
    load_transformer_model = None
    predict_transformer = None

LOGGER = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    # مدل ورودی API فقط فیلد text را می‌پذیرد.
    model_config = ConfigDict(extra="forbid")
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        if len(value) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"text exceeds maximum length of {MAX_TEXT_LENGTH} characters"
            )
        return value


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


def create_app(
    model_path=MODEL_PATH,
    model: Any | None = None,
    dl_bundle: Any | None = None,
    transformer_bundle: Any | None = None,
) -> FastAPI:
    # در محیط واقعی مدل ذخیره‌شده خوانده می‌شود و در تست‌ها مدل آزمایشی تزریق می‌شود.
    loaded_model = model if model is not None else load_model(model_path)

    # مدل PyTorch فقط در صورت ارسال bundle آماده استفاده می‌شود.
    loaded_dl_bundle = dl_bundle
    loaded_transformer_bundle = transformer_bundle

    app = FastAPI(
    title="AI Text Classifier",
    version=MODEL_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
    app.include_router(rag_router)

    # خطاهای پیش‌بینی‌نشده نباید traceback داخلی را در پاسخ API نمایش دهند.
    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):
        LOGGER.exception("Unhandled API error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    @app.get("/health")
    def health():
        # وضعیت هر سه مدل بدون انجام آموزش گزارش می‌شود.
        return {
            "status": "ok",
            "model_loaded": loaded_model is not None,
            "classic_model_loaded": loaded_model is not None,
            "pytorch_model_loaded": loaded_dl_bundle is not None,
            "transformer_model_loaded": loaded_transformer_bundle is not None,
            "model_version": MODEL_VERSION,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def prediction(request: PredictionRequest):
        # درخواست معتبر مستقیماً به مدل کلاسیک ارسال می‌شود.
        try:
            return predict(
                request.text,
                model=loaded_model,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            raise HTTPException(
                status_code=500,
                detail="Prediction failed",
            ) from exc

    @app.post("/predict/transformer")
    def prediction_transformer(request: PredictionRequest):
        # مدل Transformer فقط برای inference استفاده می‌شود و در زمان API آموزش داده نمی‌شود.
        if loaded_transformer_bundle is None:
            raise HTTPException(
                status_code=503,
                detail="Transformer model is not loaded. Run python -m src.transformer_train first.",
            )

        try:
            model, tokenizer, device = loaded_transformer_bundle
            return predict_transformer(
                request.text,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Transformer prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from exc

    @app.post("/predict/dl")
    def prediction_dl(request: PredictionRequest):
        # مدل PyTorch فقط برای inference استفاده می‌شود و در زمان API آموزش داده نمی‌شود.
        if loaded_dl_bundle is None:
            raise HTTPException(
                status_code=503,
                detail="PyTorch model is not loaded. Run python -m src.dl_train first.",
            )

        try:
            model, vocabulary, checkpoint = loaded_dl_bundle
            return predict_dl(
                request.text,
                model=model,
                vocabulary=vocabulary,
                checkpoint=checkpoint,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            LOGGER.exception("PyTorch prediction failed")
            raise HTTPException(
                status_code=500,
                detail="Prediction failed",
            ) from exc

    return app


# هنگام import شدن برنامه، مدل‌های ذخیره‌شده بدون آموزش بارگذاری می‌شوند.
if MODEL_PATH.exists():
    loaded_dl_bundle = None
    loaded_transformer_bundle = None

    try:
        loaded_dl_bundle = load_pytorch_model()
    except FileNotFoundError:
        LOGGER.warning(
            "PyTorch model artifact not found. Run python -m src.dl_train first."
        )

    try:
        if load_transformer_model is not None:
            loaded_transformer_bundle = load_transformer_model()
    except FileNotFoundError:
        LOGGER.warning(
            "Transformer model artifact not found. Run python -m src.transformer_train first."
        )

    app = create_app(
        model_path=MODEL_PATH,
        dl_bundle=loaded_dl_bundle,
        transformer_bundle=loaded_transformer_bundle,
    )

else:
    app = FastAPI(
        title="AI Text Classifier",
        version=MODEL_VERSION,
    )

    @app.get("/health")
    def health_without_model():
        return {
            "status": "ok",
            "model_loaded": False,
            "classic_model_loaded": False,
            "pytorch_model_loaded": False,
            "transformer_model_loaded": False,
            "model_version": MODEL_VERSION,
        }

    @app.post("/predict")
    def predict_without_model():
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run python -m src.train first.",
        )

    @app.post("/predict/dl")
    def predict_dl_without_model():
        raise HTTPException(
            status_code=503,
            detail="PyTorch model is not loaded. Run python -m src.dl_train first.",
        )
    @app.post("/predict/transformer")
    def predict_transformer_without_model():
        raise HTTPException(status_code=503, detail="Transformer model is not loaded. Run python -m src.transformer_train first.")