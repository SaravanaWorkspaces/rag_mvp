from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import numpy as np
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
# 2. Split into sentence-level chunks
# ---------------------------------
sentence_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", "? ", "! "]
)

sentences = sentence_splitter.create_documents([text])
print(f"Created {len(sentences)} sentence-level chunks")

for i, s in enumerate(sentences):
    print(f"  [{i}] {s.page_content.strip()}")

# ---------------------------------
# 3. Embed each sentence
# ---------------------------------
embeddings = OpenAIEmbeddings()

sentence_texts = [s.page_content for s in sentences]
sentence_vectors = embeddings.embed_documents(sentence_texts)
print(f"\nGenerated {len(sentence_vectors)} embeddings")

# ---------------------------------
# 4. Cosine similarity between consecutive sentences
# ---------------------------------
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = []
for i in range(len(sentence_vectors) - 1):
    sim = cosine_similarity(sentence_vectors[i], sentence_vectors[i + 1])
    similarities.append(sim)

print("\nConsecutive similarities:")
for i, sim in enumerate(similarities):
    print(f"  Sentence {i} → {i+1}: {sim:.4f}")

# ---------------------------------
# 5. Find breakpoints (low similarity = topic change)
# ---------------------------------
threshold = np.percentile(similarities, 30)
print(f"\nThreshold (30th percentile): {threshold:.4f}")

breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
print(f"Breakpoints at: {breakpoints}")

# ---------------------------------
# 6. Group sentences into semantic chunks
# ---------------------------------
semantic_chunks = []
start = 0

for bp in breakpoints:
    chunk_text = " ".join(sentence_texts[start:bp])
    semantic_chunks.append(Document(page_content=chunk_text))
    start = bp

# Last chunk
chunk_text = " ".join(sentence_texts[start:])
semantic_chunks.append(Document(page_content=chunk_text))

print(f"\nCreated {len(semantic_chunks)} semantic chunks:\n")
for i, chunk in enumerate(semantic_chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content.strip())
    print()

db = FAISS.from_documents(
    semantic_chunks,
    embeddings
)

db.save_local("../vector_db")
print("Vector DB created and saved.")