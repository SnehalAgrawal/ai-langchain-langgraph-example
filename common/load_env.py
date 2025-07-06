"""
Shared utility for loading environment variables.
"""
import os
from dotenv import load_dotenv

def load_environment():
    """
    Loads environment variables from a .env file.
    """
    load_dotenv()
    # You can add more complex logic here, like checking for required variables
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY not found in .env file.")
