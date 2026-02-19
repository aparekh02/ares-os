class Embedder:
    """Shared embedding provider. Lazy-loads the model on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._get_model()
        return model.encode(text, convert_to_numpy=True).tolist()

    @property
    def dimensions(self) -> int:
        return 384
