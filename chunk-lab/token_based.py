from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import TokenTextSplitter

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
#  Token-based chunking
# ---------------------------------
# Why tokens instead of characters?
#   "Employees" = 1 token but 9 characters
#   LLMs think in tokens, not characters
#   chunk_size=100 chars ≠ 100 tokens
#
# TokenTextSplitter uses tiktoken (OpenAI's tokenizer)
# so chunks align with how the model actually reads text

token_splitter = TokenTextSplitter(
    chunk_size=50,       # max 50 tokens per chunk (not characters!)
    chunk_overlap=10     # last 10 tokens repeat in next chunk
)

token_chunks = token_splitter.create_documents([text])

print(f"Created {len(token_chunks)} token-based chunks:\n")
for i, chunk in enumerate(token_chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content.strip())
    print()

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(token_chunks, embeddings)
db.save_local("../vector_db")

print("Vector DB created and saved.")
