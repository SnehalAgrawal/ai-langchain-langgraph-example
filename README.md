# LangChain and LangGraph Starter Kit

This project provides a practical, hands-on introduction to the core features of LangChain and LangGraph. It's designed as a beginner-friendly starter kit with modular, well-commented examples that you can run individually.

The project is configured to work with the standard **OpenAI** API.

## Folder Structure

```
/
|-- .env
|-- requirements.txt
|-- README.md
|-- data/
|   |-- sample_docs/
|       |-- rag_text.txt
|-- common/
|   |-- llm_helper.py
|   |-- load_env.py
|-- langchain/
|   |-- 01_prompt_template.py
|   |-- 02_simple_chain.py
|   |-- 03_chat_model_custom_prompt.py
|   |-- 04_chain_with_memory.py
|   |-- 05_rag_basic.py
|   |-- 06_advanced_rag.py
|   |-- 07_agent_with_tools.py
|-- langgraph/
|   |-- 01_basic_chain_graph.py
|   |-- 02_tool_agent_graph.py
|   |-- 03_triage_memory_graph.py
```

## Setup Instructions

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository (if applicable)

If you don't have the project files, clone the repository:
```bash
git clone <repository_url>
cd ai-langchain-langgraph-langmem
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key

All model configuration is handled in the `.env` file. Add your OpenAI API key to the `OPENAI_API_KEY` variable.

```env
# .env
OPENAI_API_KEY="sk-..."
```

## How to Run the Examples

Each Python script in the `langchain/` and `langgraph/` directories is a self-contained example. You can run them directly from your terminal.

Make sure you are in the root directory of the project and your virtual environment is activated.

### LangChain Examples

*   **01_prompt_template.py**: Demonstrates how to use `PromptTemplate` to create structured prompts with variables.
*   **02_simple_chain.py**: Shows how to create a simple LLM chain using the `|` syntax.
*   **03_chat_model_custom_prompt.py**: Illustrates how to use a chat model with a custom system and user prompt for a continuous conversation.
*   **04_chain_with_memory.py**: Implements a chat system with memory using `RunnableWithMessageHistory`.
*   **05_rag_basic.py**: A basic example of a Retrieval-Augmented Generation (RAG) system using a local text file.
*   **06_advanced_rag.py**: An advanced RAG system that processes a PDF document.
*   **07_agent_with_tools.py**: Demonstrates how to create a LangChain agent that has access to tools like a calculator.

```bash
# Example: Run the simple chain script
python langchain/02_simple_chain.py
```

### LangGraph Examples

*   **01_basic_chain_graph.py**: A basic LangGraph with conditional routing.
*   **02_tool_agent_graph.py**: A LangGraph that wraps a tool-using ReAct agent.
*   **03_triage_memory_graph.py**: A stateful graph for email triage with memory.

```bash
# Example: Run the basic graph
python langgraph/01_basic_chain_graph.py
```
