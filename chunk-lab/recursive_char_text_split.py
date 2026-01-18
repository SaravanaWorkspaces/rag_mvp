from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# Load document
loader = TextLoader("../data/company_policy.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store in FAISS vector DB
db = FAISS.from_documents(chunks, embeddings)

# Save index locally
db.save_local("../vector_db")

print("Vector DB created and saved.")