from langchain_groq import ChatGroq
from .vector_store import load_vector_store
from .config import GROQ_API_KEY, GROQ_MODEL


def query_articles(query, k=4):
    """Query the vector store and get LLM response."""
    print(f"\nQuery: {query}")

    vectorstore = load_vector_store()
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    print(f"Retrieved {len(retrieved_docs)} relevant chunks")

    # Format context
    context = "\n\n---\n\n".join([
        f"Source (PMID: {doc.metadata['pmid']}):\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3
    )

    # Build prompt
    prompt = f"""Answer the following question based ONLY on the provided research articles.
If the answer is not in the provided context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


def query_with_sources(query, k=4):
    """Query and return both answer and source documents."""
    vectorstore = load_vector_store()
    retrieved_docs = vectorstore.similarity_search(query, k=k)

    # Format context
    context = "\n\n---\n\n".join([
        f"Source (PMID: {doc.metadata['pmid']}):\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3
    )

    # Build prompt
    prompt = f"""Answer the following question based ONLY on the provided research articles.
If the answer is not in the provided context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "sources": retrieved_docs
    }


def query_with_analysis(query, k=4):
    """
    Query with contradiction detection and stance analysis.
    Returns answer, sources, and contradiction analysis.
    """
    from .contradiction_detector import detect_contradictions, format_stance_badges

    vectorstore = load_vector_store()
    retrieved_docs = vectorstore.similarity_search(query, k=k)

    # Detect contradictions
    contradiction_analysis = detect_contradictions(query, retrieved_docs)

    # Format context (prioritize supporting, then neutral, then opposing for main answer)
    pmid_to_stance = contradiction_analysis.get("pmid_to_stance", {})

    # Sort documents by stance priority
    def get_stance_priority(doc):
        pmid = doc.metadata.get("pmid", "")
        stance = pmid_to_stance.get(pmid, "NEUTRAL")
        if stance == "SUPPORTS":
            return 0
        elif stance == "NEUTRAL":
            return 1
        else:
            return 2

    context_order = sorted(retrieved_docs, key=get_stance_priority)

    context = "\n\n---\n\n".join([
        f"Source (PMID: {doc.metadata['pmid']}):\n{doc.page_content}"
        for doc in context_order[:k]
    ])

    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3
    )

    # Enhanced prompt mentioning contradiction awareness
    contradiction_note = ""
    if contradiction_analysis["has_contradiction"]:
        contradiction_note = """
IMPORTANT: The retrieved studies show conflicting evidence. 
Acknowledge this contradiction in your answer and present the majority view 
while noting the existence of opposing studies."""

    prompt = f"""Answer the following question based on the provided research articles.
{contradiction_note}

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": retrieved_docs,
        "contradiction_analysis": contradiction_analysis,
        "stance_badges": format_stance_badges(contradiction_analysis["summary"])
    }
