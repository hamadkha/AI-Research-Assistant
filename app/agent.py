import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_groq import ChatGroq

from app.tools import wikipedia_tool, arxiv_tool

# 1. Load Environment Variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# 2. Initialize the LLM
# Using Groq for high-speed inference. The model can be swapped out easily.
llm = ChatGroq(model=llm_model, temperature=0, groq_api_key=groq_api_key)

# 3. Define the Tools for the Agent
tools = [wikipedia_tool, arxiv_tool]

# 4. Create the Agent
# Pulls a pre-defined ReAct prompt that is optimized for chat models
prompt = hub.pull("hwchase17/react-chat")

# Create the agent by binding the tools and LLM to the prompt
agent = create_react_agent(llm, tools, prompt)

# 5. Create the Agent Executor
# This is the runtime for the agent, which orchestrates the tool calls and responses.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Set to True to see the agent's thought process
    handle_parsing_errors=True, # Gracefully handle errors if the LLM output is not formatted correctly
    max_iterations=5, # Prevents the agent from getting stuck in a loop
)

def get_agent_executor():
    """Returns the initialized agent executor."""
    return agent_executor