# Makes 'core' a Python package
# I am also exposing main functions here for easier imports
from .pubmed_fetcher import fetch_pubmed_articles
from .chunker import chunk_documents
from .vector_store import ingest_to_faiss, load_vector_store
from .query_engine import query_articles

__all__ = [
    'fetch_pubmed_articles',
    'chunk_documents',
    'ingest_to_faiss',
    'load_vector_store',
    'query_articles',
]
