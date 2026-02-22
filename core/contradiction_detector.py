from typing import List, Dict
from langchain.schema import Document
from langchain_groq import ChatGroq
from .config import GROQ_API_KEY, GROQ_MODEL


def analyze_stance(query: str, context: str, llm: ChatGroq) -> str:
    """Determine if the context supports, opposes, or is neutral to the query."""
    prompt = f"""Analyze the following research context and determine the stance regarding the question.

Question: {query}

Context: {context}

Determine the stance as one of:
- SUPPORTS: Evidence clearly supports/affirms the premise of the question
- OPPOSES: Evidence contradicts or shows no effect for the premise
- NEUTRAL/Mixed: Evidence is inconclusive, preliminary, or mixed results

Respond with ONLY one word: SUPPORTS, OPPOSES, or NEUTRAL"""

    response = llm.invoke(prompt)
    stance = response.content.strip().upper()

    if "SUPPORT" in stance:
        return "SUPPORTS"
    elif "OPPOSE" in stance or "CONTRADICT" in stance:
        return "OPPOSES"
    else:
        return "NEUTRAL"


def detect_contradictions(query: str, documents: List[Document]) -> Dict:
    """
    Detect contradictions among retrieved documents.
    Returns analysis with stance distribution and synthesis.
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.1
    )

    # Analyze stance for each document and track by PMID
    stances = {
        "SUPPORTS": [],
        "OPPOSES": [],
        "NEUTRAL": []
    }

    # Map PMID to stance for easy lookup
    pmid_to_stance = {}

    for doc in documents:
        pmid = doc.metadata.get("pmid", "unknown")
        stance = analyze_stance(query, doc.page_content, llm)
        stances[stance].append(pmid)
        pmid_to_stance[pmid] = stance

    # Determine if contradictions exist
    has_contradiction = len(stances["SUPPORTS"]) > 0 and len(stances["OPPOSES"]) > 0

    # Generate synthesis if contradictions exist
    synthesis = None
    if has_contradiction:
        supporting_docs = [d for d in documents if d.metadata.get("pmid") in stances["SUPPORTS"]]
        opposing_docs = [d for d in documents if d.metadata.get("pmid") in stances["OPPOSES"]]

        supporting_evidence = "\n".join([f"- {d.metadata.get('title', 'Untitled')}" for d in supporting_docs[:2]])
        opposing_evidence = "\n".join([f"- {d.metadata.get('title', 'Untitled')}" for d in opposing_docs[:2]])

        synthesis_prompt = f"""Synthesize the conflicting evidence on this medical question:

Question: {query}

Supporting Studies:
{supporting_evidence}

Opposing Studies:
{opposing_evidence}

Provide a balanced 2-3 sentence synthesis acknowledging the contradiction and explaining possible reasons (study design differences, populations, outcome measures, publication dates)."""

        synthesis = llm.invoke(synthesis_prompt).content

    return {
        "has_contradiction": has_contradiction,
        "pmid_to_stance": pmid_to_stance,
        "stances": stances,
        "synthesis": synthesis,
        "summary": {
            "supporting": len(stances["SUPPORTS"]),
            "opposing": len(stances["OPPOSES"]),
            "neutral": len(stances["NEUTRAL"])
        }
    }


def format_stance_badges(summary: Dict) -> str:
    """Format stance summary as emoji badges."""
    badges = []
    if summary["supporting"] > 0:
        badges.append(f"✅ {summary['supporting']} Supporting")
    if summary["opposing"] > 0:
        badges.append(f"❌ {summary['opposing']} Opposing")
    if summary["neutral"] > 0:
        badges.append(f"⚖️ {summary['neutral']} Neutral/Mixed")
    return " | ".join(badges) if badges else "⚖️ Analyzing..."
