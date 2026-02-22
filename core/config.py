import os
from dotenv import load_dotenv

# Set cache location BEFORE any imports
os.environ['HF_HOME'] = os.getenv('HF_HOME', './model_cache')
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', './model_cache')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', './model_cache')

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
VECTOR_DIR = os.getenv("CHROMA_DIR", "vector_resources/vectorstore")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# PubMed API URLs
SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
