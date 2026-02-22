from core.pubmed_fetcher import fetch_pubmed_articles
from core.chunker import chunk_documents
from core.vector_store import ingest_to_faiss
from core.query_engine import query_articles


def run_ingestion():
    """Run the ingestion pipeline."""
    print("=== Starting Ingestion Pipeline ===\n")

    # Step 1: Fetch articles
    articles = fetch_pubmed_articles("Intermittent Fasting", max_results=90)
    if not articles:
        print("No articles found!")
        return False

    # Step 2: Chunk documents
    chunks = chunk_documents(articles)

    # Step 3: Store in FAISS
    ingest_to_faiss(chunks)

    print("\n=== Ingestion Complete! ===\n")
    return True


def run_query_loop():
    """Run the interactive query loop."""
    print("Enter your questions (type 'quit' to exit):")
    while True:
        user_query = input("\nQuestion: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        if not user_query:
            continue

        answer = query_articles(user_query)
        print(f"\nAnswer: {answer}")


def main():
    """Main entry point."""
    # Run ingestion first
    success = run_ingestion()
    if not success:
        return

    # Then run query loop
    run_query_loop()


if __name__ == "__main__":
    main()
