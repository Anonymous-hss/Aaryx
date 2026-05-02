import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

KNOWLEDGE_FILE = "knowledge/portfolio.md"
FAISS_INDEX_DIR = "faiss_index"

# Initialize embeddings globally to reuse
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_vector_db():
    if not os.path.exists(FAISS_INDEX_DIR):
        print("Initializing FAISS index...")
        loader = TextLoader(KNOWLEDGE_FILE)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(FAISS_INDEX_DIR)
        print("FAISS index created and saved.")
        return vectorstore
    else:
        print("Loading existing FAISS index...")
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# Load vector store once
vectorstore = initialize_vector_db()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

@tool
def portfolio_search(query: str) -> str:
    """
    Search Harshal Sawatkar's portfolio knowledge base.
    Use this tool FIRST for any questions about Harshal, his projects, experience, skills, contact info, resume, or qualities.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the portfolio."
    
    # Format the results into a readable string
    formatted_results = "\n\n".join([doc.page_content for doc in results])
    return f"Portfolio Knowledge Base Results:\n{formatted_results}"
