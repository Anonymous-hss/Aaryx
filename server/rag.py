import os
import pickle
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool

KNOWLEDGE_FILE = "knowledge/portfolio.md"
BM25_INDEX_FILE = "faiss_index/bm25.pkl"

def get_retriever():
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
        
    if os.path.exists(BM25_INDEX_FILE):
        with open(BM25_INDEX_FILE, "rb") as f:
            return pickle.load(f)
            
    print("Initializing BM25 index...")
    loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    retriever = BM25Retriever.from_documents(splits)
    retriever.k = 4
    
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(retriever, f)
        
    print("BM25 index created and saved.")
    return retriever

retriever = get_retriever()

@tool
def portfolio_search(query: str) -> str:
    """
    Search Harshal Sawatkar's portfolio knowledge base.
    Use this tool FIRST for any questions about Harshal, his projects, experience, skills, contact info, resume, or qualities.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the portfolio."
    
    formatted_results = "\n\n".join([doc.page_content for doc in results])
    return f"Portfolio Knowledge Base Results:\n{formatted_results}"
