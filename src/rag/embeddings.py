# این ماژول مدل تعبیه‌سازی را بارگذاری و بردارهای متن را تولید می‌کند.

from sentence_transformers import SentenceTransformer


# مدل تعبیه‌سازی جمله‌ها که بردارهای ۳۸۴بعدی تولید می‌کند.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# پوشش ساده برای بارگذاری مدل و تولید بردارهای نرمال‌شده.
class EmbeddingModel:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        if not texts:
            return []

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


# نگه‌داری یک نمونه مشترک برای جلوگیری از بارگذاری تکراری مدل.
_model = None


# بازگرداندن نمونه مشترک مدل تعبیه‌سازی.
def get_embedding_model() -> EmbeddingModel:
    global _model

    if _model is None:
        _model = EmbeddingModel()

    return _model