import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Embeddings (Free local embeddings)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create or Load Vector Store
def get_vector_store(chunks=None):
    if chunks:
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    else:
        embeddings = get_embeddings()
        if os.path.exists("faiss_index"):
            return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            return None

# Simple retrieval without LLM - just return relevant documents
def get_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    class SimpleQAWrapper:
        def __init__(self, retriever):
            self.retriever = retriever
        
        def invoke(self, inputs):
            query = inputs.get("query", "")
            source_docs = self.retriever.invoke(query)
            
            # Create answer from retrieved documents
            if source_docs:
                answer = "Based on the data, here's what I found:\n\n"
                for i, doc in enumerate(source_docs[:3], 1):
                    content = doc.page_content[:300].strip()
                    answer += f"{i}. {content}...\n\n"
                answer += "\n(Note: Install Ollama or add OpenAI credits for AI-generated answers)"
            else:
                answer = "I couldn't find relevant information in the database."
            
            return {"result": answer, "source_documents": source_docs}
    
    return SimpleQAWrapper(retriever)
