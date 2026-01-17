from langchain_community.document_loaders import TextLoader

loader = TextLoader("/Users/saravana/workspace/ai/rag_mvp/data/company_policy.txt")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
#print(documents[0].page_content)

