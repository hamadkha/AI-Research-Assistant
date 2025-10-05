import sys
import os
import streamlit as st

# Add the project root directory to the Python path
# This is necessary to ensure that modules from the 'app' package can be imported correctly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import AIMessage, HumanMessage
from app.agent import get_agent_executor

# --- App Configuration ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬")

# --- State Management ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Agent Initialization ---
agent_executor = get_agent_executor()

# --- UI Rendering ---
st.title("🔬 AI Research Assistant")
st.caption("Your intelligent partner for Wikipedia and ArXiv queries.")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Handle user input
user_query = st.chat_input("Ask your research question...")

if user_query:
    # Add user message to history and display it
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    # Get AI response
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            ai_response_content = response["output"]
            st.write(ai_response_content)

    # Add AI response to history
    st.session_state.chat_history.append(AIMessage(content=ai_response_content))