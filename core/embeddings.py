from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL


class _STWrapper(Embeddings):
    """Wrapper to use SentenceTransformer directly with LangChain."""
    def __init__(self, model_name: str, normalize: bool = False):
        self.model = SentenceTransformer(model_name, cache_folder=r'D:\Arjun\Env_set_files')
        self.normalize = normalize

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        result = self.model.encode(
            [text],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return result[0].tolist()


def get_embeddings(model_name: str = EMBEDDING_MODEL, normalize: bool = True):
    """Create an Embeddings instance."""
    return _STWrapper(model_name, normalize=normalize)
