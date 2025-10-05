# 🔬 AI Research Assistant

An intelligent, conversational AI agent designed to assist with research queries by leveraging information from Wikipedia and ArXiv. This application uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, cited answers in a real-time chat interface.

---
## ✨ Features

- **Conversational Interface:** Interact with the assistant in a natural, chat-based UI built with Streamlit.
- **Multi-Turn Memory:** The agent remembers the context of your conversation for follow-up questions.
- **Dynamic Tool Use:** Intelligently decides whether to query **Wikipedia** for general knowledge or **ArXiv** for scientific papers based on your question.
- **Persistent Caching:** Automatically caches retrieved documents in a local **ChromaDB** vector store, making subsequent queries on the same topic significantly faster.
- **Open-Source & Local-First:** Built entirely with open-source components, with a focus on running locally for privacy and control.
- **Powered by Modern LLMs:** Utilizes fast and powerful language models (like Llama 3.1) served via the Groq API.

## ⚙️ How It Works

This project is built on the **LangChain** framework and uses a **ReAct (Reasoning and Acting) Agent** to orchestrate the entire process:

1.  **User Query:** The user asks a question in the Streamlit frontend.
2.  **Agent Reasoning:** The agent analyzes the query and decides which tool is best suited to answer it (Wikipedia or ArXiv).
3.  **Tool Execution:** The agent executes the chosen tool.
4.  **Caching Layer:** The tool first checks the local ChromaDB cache for relevant documents. If found, it uses them directly. If not, it queries the live API (Wikipedia/ArXiv).
5.  **Document Caching:** Any new documents retrieved from the live APIs are embedded and stored in the ChromaDB cache for future use.
6.  **Response Generation:** The retrieved documents are passed along with the user query and chat history to the LLM, which generates a comprehensive, cited answer.
7.  **UI Update:** The final answer is displayed in the chat interface.

## 🛠️ Tech Stack

- **Backend Framework:** [LangChain](https://www.langchain.com/)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM Serving:** [Groq](https://groq.com/) (for high-speed Llama 3.1 inference)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Data Sources:** [Wikipedia API](https://pypi.org/project/wikipedia/), [ArXiv API](https://pypi.org/project/arxiv/)

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.9 or newer
- A free API key from [Groq](https://console.groq.com/keys)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv .venv
.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the root of the project directory and add your Groq API key:

```
GROQ_API_KEY="YOUR_API_KEY_HERE"
LLM_MODEL="llama-3.1-8b-instant"
```

### 5. Run the Application

Launch the Streamlit web server.

```bash
streamlit run app/main.py
```

Your web browser should automatically open a new tab with the application running. If not, the terminal will provide a local URL (usually `http://localhost:8501`).

## 📁 Project Structure

```
/ai-research-assistant
|-- /app
|   |-- main.py             # Streamlit frontend code
|   |-- agent.py            # Core agent logic and executor
|   |-- tools.py            # Definitions for Wikipedia & ArXiv tools
|   |-- vector_store.py     # ChromaDB setup and caching logic
|-- /chroma_db/             # (Generated) Persistent vector store data
|-- .gitignore              # Files to be ignored by Git
|-- README.md               # This file
|-- requirements.txt        # Project dependencies
|-- .env                    # (Local) Environment variables with API keys
```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
