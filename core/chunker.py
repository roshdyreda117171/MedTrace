from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks."""
    print("Chunking documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = i
        chunks.extend(doc_chunks)

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
