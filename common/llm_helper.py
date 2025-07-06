"""
Helper for initializing LLMs and embedding models from OpenAI.
"""
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings

def get_llm(temperature=0):
    """Returns an LLM instance."""
    return OpenAI(temperature=temperature)

def get_chat_llm(temperature=0):
    """Returns a ChatLLM instance."""
    return ChatOpenAI(temperature=temperature)

def get_embeddings():
    """Returns an Embeddings instance."""
    return OpenAIEmbeddings()
