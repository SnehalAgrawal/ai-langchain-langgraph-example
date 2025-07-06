"""
Demonstrates a chat system with memory using RunnableWithMessageHistory.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import uuid
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from common.llm_helper import get_chat_llm

# Load environment variables
load_dotenv()

# Store for session histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def run():
    """
    Runs a conversational chain with memory.
    """
    # Initialize the language model
    llm = get_chat_llm(temperature=0)

    # Define the prompt template with a messages placeholder for history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Create the chain
    chain = prompt | llm

    # Wrap the chain with RunnableWithMessageHistory
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Start chatting
    session_id = str(uuid.uuid4())
    print(f"Starting new conversation with session ID: {session_id}")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = with_message_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        ).content
        print("AI:", response)

    print("\n--- Conversation History ---")
    for message in store[session_id].messages:
        print(f"{message.type.capitalize()}: {message.content}")

if __name__ == "__main__":
    run()
