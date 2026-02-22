import time
from typing import Dict
from langchain.schema import Document
from langchain_groq import ChatGroq
from .config import GROQ_API_KEY, GROQ_MODEL


def evaluate_response(query: str, context: str, answer: str) -> Dict:
    """
    Simple RAG evaluation using LLM as judge.
    Returns metrics: relevance, groundedness, confidence.
    """
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0)

    # Metric 1: Context Relevance
    rel_prompt = f"""Rate how relevant the retrieved context is to answering the query.

Query: {query}
Context (truncated): {context[:500]}...

Rate 1-10 where 10 = highly relevant. Reply with ONLY a number."""

    relevance = llm.invoke(rel_prompt).content.strip()
    try:
        relevance_score = float(relevance) / 10
    except:
        relevance_score = 0.5

    # Metric 2: Answer Groundedness
    ground_prompt = f"""Check if the answer is fully supported by the context.

Context: {context[:500]}...
Answer: {answer}

Is the answer grounded in the context? Reply ONLY: YES, PARTIAL, or NO."""

    grounded = llm.invoke(ground_prompt).content.strip().upper()
    if "YES" in grounded:
        grounded_score = 1.0
    elif "PARTIAL" in grounded:
        grounded_score = 0.5
    else:
        grounded_score = 0.0

    # Metric 3: Confidence (based on citations presence)
    has_citations = any(char.isdigit() for char in answer) or "study" in answer.lower()
    confidence = 0.8 if has_citations else 0.5

    return {
        "context_relevance": round(relevance_score, 2),
        "groundedness": round(grounded_score, 2),
        "confidence": round(confidence, 2),
        "average": round((relevance_score + grounded_score + confidence) / 3, 2)
    }


def get_quality_badge(score: float) -> str:
    if score >= 0.8:
        return "🟢 High"
    elif score >= 0.6:
        return "🟡 Medium"
    else:
        return "🔴 Low"
