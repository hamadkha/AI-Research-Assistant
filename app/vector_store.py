import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configure Embedding Model
# Uses a high-performance, open-source model that runs locally.
embedding_function = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Initialize ChromaDB Client
# Creates a persistent client that saves data to the specified directory.
client = chromadb.PersistentClient(path="./chroma_db")

# 3. Get or Create Vector Stores
# Each vector store acts as a cache for a specific data source.
wiki_vector_store = Chroma(
    client=client,
    collection_name="wiki_cache",
    embedding_function=embedding_function,
)

arxiv_vector_store = Chroma(
    client=client,
    collection_name="arxiv_cache",
    embedding_function=embedding_function,
)

def get_vector_stores():
    """Returns the initialized vector stores for Wikipedia and ArXiv."""
    return wiki_vector_store, arxiv_vector_store
