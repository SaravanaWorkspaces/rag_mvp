from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------
# 1. Load PDF (structure-aware)
# ---------------------------------
loader = UnstructuredPDFLoader(
    "../data/sample_policy.pdf",
    mode="elements"   # structure-aware
)

elements = loader.load()
print(f"Loaded {len(elements)} structured elements")

# ---------------------------------
# 2. Group elements by section (PDF-aware chunking)
# ---------------------------------
section_docs = []
current_section = "Introduction"
buffer = []

for el in elements:
    category = el.metadata.get("category")

    # Treat Title / Header as section boundaries
    if category in ["Title", "Header", "Section-header", "Headline"]:
        if buffer:
            section_docs.append(
                Document(
                    page_content="\n".join(buffer),
                    metadata={"section": current_section}
                )
            )
            buffer = []

        current_section = el.page_content.strip()

    else:
        buffer.append(el.page_content)

# Add last section
if buffer:
    section_docs.append(
        Document(
            page_content="\n".join(buffer),
            metadata={"section": current_section}
        )
    )

print(f"Created {len(section_docs)} PDF-aware sections")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(section_docs)
print(f"Final chunks created: {len(chunks)}")

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save index locally
db.save_local("../vector_db")

print("Vector DB created and saved.")