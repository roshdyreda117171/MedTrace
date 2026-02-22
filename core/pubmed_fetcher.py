import requests
from xml.etree import ElementTree
from langchain.schema import Document
from .config import SEARCH_URL, FETCH_URL


def fetch_pubmed_articles(search_term, max_results=90):
    """Fetch articles from PubMed."""
    print(f"Fetching {max_results} articles on '{search_term}'...")

    # Get PMIDs
    params = {
        'db': 'pubmed',
        'term': search_term,
        'retmax': max_results,
        'retmode': 'xml',
        'sort': 'date'
    }

    response = requests.get(SEARCH_URL, params=params)
    root = ElementTree.fromstring(response.content)
    pmids = [elem.text for elem in root.findall(".//Id")]
    print(f"Found {len(pmids)} article IDs")

    # Fetch article details
    articles = []
    fetch_params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'xml'
    }
    fetch_response = requests.get(FETCH_URL, params=fetch_params)
    fetch_root = ElementTree.fromstring(fetch_response.content)

    for article in fetch_root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text
        title = article.find(".//ArticleTitle").text or "No Title"

        abstract_sections = article.findall(".//AbstractText")
        abstract_text = " ".join([
            section.text for section in abstract_sections
            if section.text is not None
        ]) if abstract_sections else "No Abstract"

        journal = article.find(".//Journal/Title").text or "Unknown Journal"
        pub_date = article.find(".//PubDate/Year").text or "Unknown Year"

        full_text = f"Title: {title}\n\nAbstract: {abstract_text}"

        doc = Document(
            page_content=full_text,
            metadata={
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "publication_date": pub_date
            }
        )
        articles.append(doc)

    print(f"Successfully fetched {len(articles)} articles")
    return articles
