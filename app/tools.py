from langchain.tools import tool
from langchain_community.retrievers import WikipediaRetriever, ArxivRetriever
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from app.vector_store import get_vector_stores

# Get initialized vector stores
wiki_vector_store, arxiv_vector_store = get_vector_stores()

# --- Tool Configuration ---
MAX_DOCS_RETRIEVED = 5

# --- Tool Definitions ---

@tool
def wikipedia_tool(query: str) -> list[Document]:
    """
    Use this tool to find information on general knowledge, concepts, and topics.
    It queries Wikipedia and returns the most relevant articles.
    Example: wikipedia_tool(query="The history of the internet")
    """
    # 1. Check cache first
    cached_results = wiki_vector_store.similarity_search(query, k=MAX_DOCS_RETRIEVED)
    if len(cached_results) >= MAX_DOCS_RETRIEVED:
        print(f"[INFO] Found {len(cached_results)} results in Wikipedia cache.")
        return cached_results

    # 2. If cache is insufficient, query live source
    print(f"[INFO] Cache miss for query: '{query}'. Querying live Wikipedia.")
    retriever = WikipediaRetriever(top_k_results=MAX_DOCS_RETRIEVED, doc_content_chars_max=2000)
    live_results = retriever.invoke(query)

    # 3. Add new results to cache
    if live_results:
        # Filter complex metadata to prevent database errors
        filtered_results = filter_complex_metadata(live_results)
        wiki_vector_store.add_documents(filtered_results)
        print(f"[INFO] Added {len(filtered_results)} new documents to Wikipedia cache.")

    return live_results

@tool
def arxiv_tool(query: str) -> list[Document]:
    """
    Use this tool to find scientific papers and research articles on ArXiv.
    It is best for technical, scientific, or academic queries.
    It can also fetch a paper directly if you provide an ArXiv paper ID.
    Example: arxiv_tool(query="Quantum computing algorithms 2023")
    Example: arxiv_tool(query="1706.03762")
    """
    # Clean the query to better handle direct ID searches
    cleaned_query = query.lower().replace("arxiv:", "").strip()

    # 1. Check cache first
    cached_results = arxiv_vector_store.similarity_search(cleaned_query, k=MAX_DOCS_RETRIEVED)
    # If we are searching for a specific ID, one result is enough
    if cached_results and cleaned_query in cached_results[0].metadata.get('entry_id', ''):
        print(f"[INFO] Found paper {cleaned_query} in ArXiv cache.")
        return cached_results[:1]

    # 2. If cache is insufficient, query live source
    print(f"[INFO] Cache miss for query: '{cleaned_query}'. Querying live ArXiv.")
    retriever = ArxivRetriever(top_k_results=MAX_DOCS_RETRIEVED, load_max_docs=MAX_DOCS_RETRIEVED)
    live_results = retriever.invoke(cleaned_query)

    # 3. Add new results to cache
    if live_results:
        # Filter complex metadata to prevent database errors
        filtered_results = filter_complex_metadata(live_results)
        arxiv_vector_store.add_documents(filtered_results)
        print(f"[INFO] Added {len(filtered_results)} new documents to ArXiv cache.")

    return live_results