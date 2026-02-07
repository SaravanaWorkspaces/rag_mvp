from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader("/Users/saravana/workspace/ai/rag_mvp/data/sample_markdown.md")
documents = loader.load()

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

chunks = markdown_splitter.split_text(documents[0].page_content)

print(f"Header-based chunks created: {len(chunks)}")

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(chunks, embeddings)

db.save_local("../vector_db")

print("Markdown-based Vector DB created.")

