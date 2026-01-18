from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# Load document
loader = TextLoader("../data/company_policy.txt")
documents = loader.load()

def fixed_chunk_documents(documents, chunk_size):
    chunks = []
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
            )
    return chunks

chunks = fixed_chunk_documents(documents, chunk_size=500)

print(f"Chunks created: {len(chunks)}")

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store in FAISS vector DB
db = FAISS.from_documents(chunks, embeddings)

# Save index locally
db.save_local("../vector_db")

print("Vector DB created and saved.")