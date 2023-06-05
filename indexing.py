import os
import pinecone

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone


DIRECTORY = "./content/data"


# Loading documents from a directory with LangChain
def load_docs(directory: str) -> list:
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


current_documents = load_docs(DIRECTORY)


# Splitting documents
def split_docs(documents: list, chunk_size=500, chunk_overlap=20) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


split_current_docs = split_docs(current_documents)

# Creating embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone with environment variables
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = pinecone_index_name

index = Pinecone.from_documents(
    split_current_docs, embeddings, index_name=index_name
)


# To delete all data from Pinecone - uncomment 2 lines below & comment line above

# index = pinecone.Index(index_name=index_name)
# index.delete(deleteAll="true", namespace="")


# Tracking the Conversation
def get_similar_docs(query: str, k=1, score=False) -> list:
    if not score:
        similar_docs = index.similarity_search(query, k=k)
    similar_docs = index.similarity_search_with_score(query, k=k)
    return similar_docs
