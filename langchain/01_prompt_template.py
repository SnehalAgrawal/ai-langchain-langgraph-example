"""
Demonstrates the use of LangChain's PromptTemplate to create structured prompts.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.prompts import PromptTemplate

def run():
    """
    Initializes and uses a PromptTemplate.
    """
    # Define a simple prompt template with one variable
    template_single = "What is the capital of {country}?"
    prompt_template_single = PromptTemplate(template=template_single, input_variables=["country"])

    # Format the prompt with a specific country
    formatted_prompt_single = prompt_template_single.format(country="France")
    print("Formatted Prompt (Single Variable):", formatted_prompt_single)

    # Define a prompt template with multiple variables
    template_multiple = "Tell me a {adjective} joke about {subject}."
    prompt_template_multiple = PromptTemplate(template=template_multiple, input_variables=["adjective", "subject"])

    # Format the prompt with multiple variables
    formatted_prompt_multiple = prompt_template_multiple.format(adjective="funny", subject="programmers")
    print("Formatted Prompt (Multiple Variables):", formatted_prompt_multiple)

if __name__ == "__main__":
    run()
