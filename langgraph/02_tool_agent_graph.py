"""
Demonstrates a LangGraph wrapping a tool-using ReAct agent.
"""

# Code is having some issue, unable to run it

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent, AgentExecutor
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# A simple tool
def get_word_length(word: str) -> int:
    return len(word)


# State to pass in graph
class AgentState(TypedDict):
    input: str
    output: str


# Node logic
def run_agent(state: AgentState) -> AgentState:
    tools = [
        Tool(
            name="get_word_length",
            func=get_word_length,
            description="Returns the number of characters in a word"
        )
    ]

    # Define prompt with required placeholders
    prompt = ChatPromptTemplate.from_messages(
        """You are a helpful agent. Use tools when helpful. You have access to the following tools:

        {tools}

        Use the following format in your response:

        Question: the input question you must answer
        Thought: your thought process
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the answer to the original question

        Begin!

        Question: {input}
        {agent_scratchpad}"""
        )

    # LLM
    llm = ChatOpenAI(temperature=0)

    # React agent setup
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Important: include scratchpad and tool info
    result = executor.invoke({
        "input": state["input"],
        "agent_scratchpad": [],
        "tools": tools,
        "tool_names": ", ".join([tool.name for tool in tools])
    })

    return {
        "input": state["input"],
        "output": result["output"]
    }


# Build LangGraph
def run():
    graph = StateGraph(AgentState)
    graph.add_node("agent", run_agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    app = graph.compile()

    query = "What is the length of the word 'onomatopoeia'?"
    result = app.invoke({"input": query})
    print("\n--- Result ---")
    print("Query:", query)
    print("Output:", result["output"])


if __name__ == "__main__":
    run()
