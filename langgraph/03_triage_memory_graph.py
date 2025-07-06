"""
Demonstrates a stateful graph for email triage with memory.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv
from common.llm_helper import get_llm

# Load environment variables
load_dotenv()

# Define the state
class TriageState(TypedDict):
    emails: List[str]
    triage_results: List[str]

# Define the nodes
def triage_node(state: TriageState):
    llm = get_llm(temperature=0)
    results = []
    for email in state["emails"]:
        prompt = f"Triage this email: '{email}'. Is it 'important' or 'spam'?"
        response = llm.invoke(prompt).strip().lower()
        results.append(response)
    state["triage_results"] = results
    return state

def run():
    """
    Builds and runs the triage graph.
    """
    # Build the graph
    graph = StateGraph(TriageState)
    graph.add_node("triage", triage_node)
    graph.add_edge("__start__", "triage")
    graph.add_edge("triage", END)

    # Compile and run
    app = graph.compile()

    # Sample emails
    emails = [
        "Your monthly invoice is ready.",
        "Win a free vacation now!",
        "Project update meeting at 2 PM."
    ]

    # Run the triage
    result = app.invoke({"emails": emails})
    print("Triage Results:", result["triage_results"])

if __name__ == "__main__":
    run()
