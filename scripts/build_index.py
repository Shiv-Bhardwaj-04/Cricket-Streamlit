import os
import sys

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.ingestion import load_documents, split_documents
from src.retrieval import get_vector_store, get_qa_chain


def main():
    print("--- Starting Data Ingestion ---")
    docs = load_documents()
    if not docs:
        print("No documents found!")
        return

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("--- Creating Vector Store ---")
    vector_store = get_vector_store(chunks)
    print("Vector Store created and saved to 'faiss_index'.")

    print("--- Verifying with Test Queries ---")
    qa_chain = get_qa_chain(vector_store)

    query1 = "Who has the most wickets in Test cricket?"
    print(f"\nQuery: {query1}")
    res1 = qa_chain.invoke({"query": query1})
    print(f"Answer: {res1['result']}")

    query2 = "Who has the most runs in ODI cricket?"
    print(f"\nQuery: {query2}")
    res2 = qa_chain.invoke({"query": query2})
    print(f"Answer: {res2['result']}")


if __name__ == "__main__":
    main()
