"""
Demonstrates using ChatOpenAI with a custom system and user prompt.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from common.llm_helper import get_chat_llm

# Load environment variables
load_dotenv()

def run():
    """
    Uses ChatOpenAI with a custom prompt structure and allows for continuous conversation.
    """
    # Initialize the chat model
    chat = get_chat_llm(temperature=0)

    # Define the initial messages
    messages = [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        AIMessage(content="Hello! I am TranslateBot, your English to French translation assistant. I can help you translate any English text into French. Type 'exit' to end our conversation.")
    ]

    print("Chatbot initialized. Type 'exit' to end the conversation.")
    print("AI:", messages[1].content)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Add user message to the list
        messages.append(HumanMessage(content=user_input))

        # Get the response from the chat model
        response = chat.invoke(messages)

        # Add AI response to the list
        messages.append(response)

        print("AI:", response.content)

if __name__ == "__main__":
    run()
