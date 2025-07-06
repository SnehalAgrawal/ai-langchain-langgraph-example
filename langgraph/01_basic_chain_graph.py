"""
Demonstrates a basic LangGraph with conditional routing.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv
from common.llm_helper import get_llm

# Load environment variables
load_dotenv()

# Define the state for our graph
class BasicGraphState(TypedDict):
    text: str
    summary: str

# Define the nodes
def summarize_node(state: BasicGraphState) -> BasicGraphState:
    llm = get_llm(temperature=0)
    summary = llm.invoke(f"Summarize this text: {state['text']}")
    state['summary'] = summary
    return state

def echo_node(state: BasicGraphState) -> BasicGraphState:
    state['summary'] = state['text']  # Just echo the input
    return state

# Define the conditional edge
def should_summarize(state: BasicGraphState) -> str:
    if len(state['text'].split()) > 20:
        return "summarize"
    return "echo"

def run():
    """
    Builds and runs the basic conditional graph.
    """
    # Build the graph
    graph = StateGraph(BasicGraphState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("echo", echo_node)

    # Define the entry point and conditional edges
    graph.add_conditional_edges(
        "__start__",
        should_summarize,
        {
            "summarize": "summarize",
            "echo": "echo",
        },
    )
    graph.add_edge("summarize", END)
    graph.add_edge("echo", END)

    # Compile and run
    app = graph.compile()

    # Test with long text
    long_text = "LangChain is a powerful framework for building applications with language models. It offers various components for chaining, agents, and memory."
    result_long = app.invoke({"text": long_text}) # Pass dictionary for TypedDict
    print("Long text result:", result_long['summary'])

    # Test with short text
    short_text = "Hello, world!"
    result_short = app.invoke({"text": short_text}) # Pass dictionary for TypedDict
    print("Short text result:", result_short['summary'])

if __name__ == "__main__":
    run()
