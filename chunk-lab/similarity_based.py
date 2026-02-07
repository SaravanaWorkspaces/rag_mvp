from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------
# 1. Raw text (same as basic_chunking)
# ---------------------------------
text = """
Employees are entitled to 20 days of paid leave per year.
Unused leave can be carried forward up to 5 days.
Sick leave requires a medical certificate if more than 2 consecutive days.

Work from home is allowed up to 3 days a week with manager approval.
Employees must be available during core hours from 10 AM to 4 PM.
VPN access is mandatory while working remotely.

All employees must follow the company code of conduct.
Harassment of any kind will result in immediate termination.
Confidential data must not be shared outside the organization.
"""

print(f"Loaded text with {len(text)} characters")

# ---------------------------------
# 2. Semantic chunking (LangChain does all the math)
# ---------------------------------
embeddings = OpenAIEmbeddings()

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # same as our np.percentile approach
    breakpoint_threshold_amount=30           # bottom 30% = breakpoint
)

semantic_chunks = semantic_splitter.create_documents([text])

print(f"Created {len(semantic_chunks)} semantic chunks:\n")
for i, chunk in enumerate(semantic_chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content.strip())
    print()

db = FAISS.from_documents(semantic_chunks, embeddings)
db.save_local("../vector_db")

print("Vector DB created and saved.")