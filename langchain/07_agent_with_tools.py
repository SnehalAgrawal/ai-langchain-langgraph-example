"""
Demonstrates a LangChain agent with access to tools like a calculator.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from dotenv import load_dotenv
from common.llm_helper import get_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

# Load environment variables
load_dotenv()

def run():
    """
    Initializes and runs an agent with tools.
    """
    # Initialize the language model
    llm = get_llm(temperature=0)

    # Load some tools
    tools = load_tools(["llm-math"], llm=llm)

    # Define the prompt for the agent
    prompt = PromptTemplate.from_template(
        """You are a helpful AI agent. You have access to the following tools:

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

    # Create the agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run the agent with a query
    # query = "What is 25 percent of 500?"
    # response = agent_executor.invoke({"input": query, "chat_history": []})
    # print(f"Query: {query}")
    # print(f"Response: {response['output']}")

    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break

        response = agent_executor.invoke({"input": query, "chat_history": chat_history})
        print(f"AI: {response['output']}")
        chat_history.append(("human", query))
        chat_history.append(("ai", response['output']))

if __name__ == "__main__":
    run()
