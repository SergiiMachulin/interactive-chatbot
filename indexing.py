import pinecone

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone

# Loading documents from a directory with LangChain
DIRECTORY = "./content/data/"


def load_docs(directory: str):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


documents = load_docs(DIRECTORY)


# Splitting documents
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


docs = split_docs(documents)

# Creating embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


pinecone.init(
    api_key="b325823f-1bb4-4dea-a0fc-868f6b20d80b",
    environment="us-west4-gcp-free",
)
index_name = "interactive-chatbot"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs
