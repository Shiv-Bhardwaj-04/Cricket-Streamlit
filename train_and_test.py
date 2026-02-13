import sys
import os

# Add src to path
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
    
    print("--- Creating Vector Store (Training) ---")
    vector_store = get_vector_store(chunks)
    print("Vector Store created and saved to 'faiss_index'.")
    
    print("--- Verifying with Test Query ---")
    qa_chain = get_qa_chain(vector_store)
    
    # Test Query 1: Cricket
    query1 = "Who did Virat Kohli score his 1st century against?"
    print(f"\nQuery: {query1}")
    res1 = qa_chain.invoke({"query": query1})
    print(f"Answer: {res1['result']}")
    
    # Test Query 2: Football
    query2 = "Which tournament matches were played in Norway on 2022-10-16?"
    print(f"\nQuery: {query2}")
    res2 = qa_chain.invoke({"query": query2})
    print(f"Answer: {res2['result']}")

if __name__ == "__main__":
    main()
