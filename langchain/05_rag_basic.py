"""
Demonstrates a basic RAG system using a local text file.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from common.llm_helper import get_llm, get_embeddings, get_chat_llm
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

def run():
    """
    Builds and queries a RAG system.
    """
    # Load the document
    loader = TextLoader("data/sample_docs/rag_text.txt")
    documents = loader.load()

    # Chunk the document
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = get_embeddings()

    # Store in Chroma vector store
    db = Chroma.from_documents(texts, embeddings)

    # Define the prompt template for RAG
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Create the document chain (combines documents with the LLM)
    document_chain = create_stuff_documents_chain(get_chat_llm(), prompt)

    # Create the retrieval chain (retrieves documents and passes them to the document chain)
    retrieval_chain = create_retrieval_chain(db.as_retriever(), document_chain)

    # Query the system
    query = "What is LangChain?"
    response = retrieval_chain.invoke({"input": query, "chat_history": []})
    print(f"Query: {query}")
    print(f"Response: {response['answer']}")

if __name__ == "__main__":
    run()
