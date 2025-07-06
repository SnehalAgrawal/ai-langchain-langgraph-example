"""
Demonstrates a simple LLMChain using OpenAI or Azure OpenAI.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from common.llm_helper import get_llm

# Load environment variables from .env
load_dotenv()

def run():
    """
    Runs a simple LLMChain.
    """
    # Initialize the language model using the helper
    llm = get_llm(temperature=0.7)

    # Define the prompt template
    template = "Tell me a joke about {topic}."
    prompt = PromptTemplate(template=template, input_variables=["topic"])

    # Create the chain using the | syntax
    chain = prompt | llm

    # Run the chain with a topic
    response = chain.invoke({"topic": "bears"})
    print("Response:", response)

if __name__ == "__main__":
    run()
