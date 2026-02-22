from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings
from .config import EMBEDDING_MODEL, VECTOR_DIR


def ingest_to_faiss(chunks):
    """Store chunks in FAISS with embeddings."""
    print(f"Creating FAISS index...")

    embeddings = get_embeddings(EMBEDDING_MODEL, normalize=True)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    vectorstore.save_local(VECTOR_DIR)
    print(f"Successfully saved FAISS index to {VECTOR_DIR}")
    return vectorstore


def load_vector_store():
    """Load existing FAISS index."""
    embeddings = get_embeddings(EMBEDDING_MODEL, normalize=True)
    return FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
