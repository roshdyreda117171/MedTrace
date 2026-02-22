import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set environment variables from .env
os.environ['TMP'] = os.getenv('TMP', r'./temp')
os.environ['TEMP'] = os.getenv('TEMP', r'./temp')
os.environ['TMPDIR'] = os.getenv('TMPDIR', r'./temp')
os.environ['CHROMA_DIR'] = os.getenv('CHROMA_DIR', r'./vectorstore')
os.environ['HF_HOME'] = os.getenv('HF_HOME', r'./model_cache')
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', r'./model_cache')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', r'./model_cache')

# Create directories if they don't exist
os.makedirs(os.environ['TMP'], exist_ok=True)
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['CHROMA_DIR'], exist_ok=True)


import streamlit as st

from core import fetch_pubmed_articles, chunk_documents, ingest_to_faiss
from core.query_engine import query_with_sources
from core.config import VECTOR_DIR

# Page configuration
st.set_page_config(
    page_title="PubMed Research Assistant",
    page_icon="🔬",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .disclaimer {
        font-size: 0.75em;
        font-style: italic;
        color: #666;
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    .source-link {
        font-size: 0.9em;
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Header with Logo
col1, col2 = st.columns([1, 5])
with col1:
    # Use placeholder if logo not found
    logo_path = os.path.join(current_dir, "assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
    else:
        st.markdown("### 🔬")
with col2:
    st.title("MedTrace: PubMed Research Assistant")
    st.caption("AI-powered biomedical literature search")

st.markdown("---")

# Session state initialization
if 'ingestion_complete' not in st.session_state:
    st.session_state.ingestion_complete = False
if 'index_available' not in st.session_state:
    st.session_state.index_available = False

# Check if index already exists
if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
    st.session_state.index_available = True

# INGESTION SECTION
st.subheader("📚 Literature Search")

# Topic input (compact)
topic = st.text_input(
    "Research Topic",
    placeholder="e.g., Intermittent Fasting, Diabetes Treatment...",
    help="Enter a biomedical topic to search PubMed",
    key="topic_input"
)

# Number of articles slider
num_articles = st.slider(
    "Number of Articles to Ingest",
    min_value=10,
    max_value=299,
    value=50,
    step=5,
    help="Select how many recent articles to fetch from PubMed"
)

# Ingest button
if st.button("🔍 Ingest Articles", type="primary", use_container_width=True):
    if not topic.strip():
        st.error("Please enter a research topic.")
    else:
        with st.status(f"Processing {num_articles} articles on '{topic}'...", expanded=True) as status:
            try:
                # Step 1: Fetch
                st.write("📡 Fetching articles from PubMed...")
                articles = fetch_pubmed_articles(topic, max_results=num_articles)

                if not articles:
                    st.error("No articles found for this topic.")
                    status.update(label="No articles found", state="error")
                else:
                    # Step 2: Chunk
                    st.write("✂️ Chunking documents...")
                    chunks = chunk_documents(articles)

                    # Step 3: Embeddings
                    st.write("🧠 Creating embeddings...")

                    # Step 4: Store
                    st.write("💾 Saving to vector store...")
                    ingest_to_faiss(chunks)

                    status.update(label=f"✅ Successfully ingested {len(articles)} articles!", state="complete")
                    st.session_state.ingestion_complete = True
                    st.session_state.index_available = True
                    st.rerun()

            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")
                status.update(label="Ingestion failed", state="error")

# Show success message if already ingested or previously completed
if st.session_state.index_available:
    st.success("✅ Vector database ready for queries!")
    st.session_state.ingestion_complete = True

st.markdown("---")

# QUERY SECTION (Only show if index exists)
if st.session_state.ingestion_complete or st.session_state.index_available:
    st.subheader("❓ Ask a Question")

    # Larger query input
    query = st.text_area(
        "Your Question",
        placeholder="Ask anything about the ingested research articles...\n\nExamples:\n- What are the main findings regarding cardiovascular benefits?\n- Does intermittent fasting affect insulin sensitivity?\n- What are the potential side effects mentioned?",
        height=120,
        help="Enter your question about the ingested articles",
        key="query_input"
    )

    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Analyzing evidence and detecting contradictions..."):
                try:
                    # Use the new analysis function
                    from core.query_engine import query_with_analysis

                    result = query_with_analysis(query, k=5)

                    # Display stance badges
                    st.markdown(f"**Evidence Distribution:** {result['stance_badges']}")

                    # Show contradiction warning if detected
                    if result['contradiction_analysis']['has_contradiction']:
                        st.warning("⚠️ **Contradictory Evidence Detected**")
                        with st.expander("View Conflict Analysis", expanded=True):
                            st.markdown("**Evidence Synthesis:**")
                            st.markdown(result['contradiction_analysis']['synthesis'])

                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown("**✅ Supporting Studies:**")
                                for doc in result['contradiction_analysis']['stances']['SUPPORTS'][:2]:
                                    st.markdown(f"- {doc.metadata['title'][:60]}...")

                            with cols[1]:
                                st.markdown("**❌ Opposing Studies:**")
                                for doc in result['contradiction_analysis']['stances']['OPPOSES'][:2]:
                                    st.markdown(f"- {doc.metadata['title'][:60]}...")

                    else:
                        st.success("✅ Evidence is consistent across retrieved studies")

                    st.markdown("---")

                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(result["answer"])

                    # Evaluation Metrics Section
                    with st.expander("📊 View Response Quality Metrics"):
                        from core.evaluation import evaluate_response, get_quality_badge

                        # Combine context for evaluation
                        eval_context = "\n".join([d.page_content[:300] for d in result["sources"]])

                        metrics = evaluate_response(query, eval_context, result["answer"])

                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Relevance", f"{metrics['context_relevance']:.0%}")
                            st.caption(get_quality_badge(metrics['context_relevance']))
                        with cols[1]:
                            st.metric("Groundedness", f"{metrics['groundedness']:.0%}")
                            st.caption(get_quality_badge(metrics['groundedness']))
                        with cols[2]:
                            st.metric("Confidence", f"{metrics['confidence']:.0%}")
                            st.caption(get_quality_badge(metrics['confidence']))
                        with cols[3]:
                            st.metric("Overall", f"{metrics['average']:.0%}")
                            st.caption(get_quality_badge(metrics['average']))

                        st.info("""
                        **Metrics Explained:**
                        - **Relevance:** How well retrieved articles match your query
                        - **Groundedness:** Whether answer is supported by cited sources
                        - **Confidence:** Based on citation density and specificity
                        """)

                    # Sources section (collapsible)
                    st.markdown("---")
                    with st.expander("📎 View All Sources"):
                        for i, doc in enumerate(result["sources"], 1):
                            pmid = doc.metadata["pmid"]
                            title = doc.metadata.get("title", "Untitled")
                            journal = doc.metadata.get("journal", "Unknown Journal")
                            year = doc.metadata.get("publication_date", "Unknown Year")  # FIXED: was 'Year'

                            # Show stance for each source using PMID lookup
                            pmid_to_stance = result['contradiction_analysis'].get('pmid_to_stance', {})
                            stance = pmid_to_stance.get(pmid, "NEUTRAL")

                            if stance == "SUPPORTS":
                                stance_icon = "✅ Supports"
                            elif stance == "OPPOSES":
                                stance_icon = "❌ Opposes"
                            else:
                                stance_icon = "⚖️ Neutral"

                            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                            st.markdown(f"**{i}.** {title}")
                            st.markdown(f"*Journal:* {journal} ({year}) | *Stance:* {stance_icon}")
                            st.markdown(f"[🔗 PubMed Link]({pubmed_url})")
                            st.markdown("---")

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")



st.markdown("---")

# Disclaimer at bottom
st.markdown(
    '<p class="disclaimer">'
    'All content is sourced from PubMed/NCBI and remains the property of the National Library of Medicine. '
    'This AI-generated response is for informational purposes only and should not be used for diagnosis, '
    'treatment decisions, or as a substitute for professional medical advice. '
    'Always consult qualified doctors for medical concerns.'
    '</p>',
    unsafe_allow_html=True
)
