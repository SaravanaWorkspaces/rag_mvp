from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load vector database
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("../vector_db", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.7})


# Create LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("\nAsk a question (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
            You are a helpful assistant.
            Answer strictly from the provided context.
            If the answer is not in the context, say: "I don't know from the given documents."


Context:
{context}

Question: {query}
Answer:"""

    answer = llm.invoke(prompt)
    print("\nAI:", answer.content, "\n")

if "I don't know" in answer.content:
    print("⚠️  Answer not found in knowledge base.")