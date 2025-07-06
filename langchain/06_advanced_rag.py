"""
Demonstrates an advanced RAG system using a PDF document.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from common.llm_helper import get_chat_llm, get_embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from colorama import Fore, Style, init

# Initialize colorama
init()

# Load environment variables
load_dotenv()

def run():
    """
    Builds and queries an advanced RAG system with a PDF.
    """
    pdf_path = "data/sample_docs/how-to-build-ai-career.pdf"

    # Load the PDF document
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # Chunk the document
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # Create embeddings and store in Chroma vector store
    print("Creating embeddings and storing in vector store...")
    embeddings = get_embeddings()
    db = Chroma.from_documents(texts, embeddings)
    print("Vector store created.")

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

    print("RAG system ready. Type your questions (type 'exit' to quit):")

    while True:
        user_query = input(f"{Fore.CYAN}You: {Style.RESET_ALL}")
        if user_query.lower() == 'exit':
            break

        print(f"{Fore.YELLOW}\nQuerying vector DB for: '{user_query}'{Style.RESET_ALL}")
        response = retrieval_chain.invoke({"input": user_query, "chat_history": []})
        
        print(f"{Fore.MAGENTA}\nRetrieved Context:{Style.RESET_ALL}")
        for doc in response['context']:
            print(f"{Fore.MAGENTA}---\n{doc.page_content}\n---{Style.RESET_ALL}")

        print(f"{Fore.GREEN}\nAI: {response['answer']}{Style.RESET_ALL}")

if __name__ == "__main__":
    run()
