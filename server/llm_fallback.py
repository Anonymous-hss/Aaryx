from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def get_llm_with_fallback(tools=None):
    """
    Returns an LLM instance with fallback configured.
    Primary: Groq (Llama 3.1 8B)
    Fallback: Gemini 2.0 Flash
    """
    
    # Initialize Groq (Primary)
    llm_primary = ChatGroq(model="llama-3.1-8b-instant")
    
    # Initialize Gemini (Fallback)
    llm_fallback = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    if tools:
        llm_primary_with_tools = llm_primary.bind_tools(tools=tools)
        llm_fallback_with_tools = llm_fallback.bind_tools(tools=tools)
        
        # In Langchain we can create fallbacks with with_fallbacks()
        llm_with_fallback = llm_primary_with_tools.with_fallbacks([llm_fallback_with_tools])
        return llm_with_fallback
        
    return llm_primary.with_fallbacks([llm_fallback])
