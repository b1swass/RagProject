# RagProject

A collection of practical **LLM, RAG, AI Agent, and LangChain experiments** built while learning and implementing modern AI application patterns with Python.

The repository focuses on understanding how LLM applications actually work—from basic question answering and retrieval to context-aware RAG systems and tool-using AI agents.

---

## 🚀 What This Project Contains

### 1. Fast Q&A Chatbot

A lightweight question-answering chatbot focused on getting fast responses from an LLM.

The main goal was to understand:

* LLM interaction
* Prompting
* Response generation
* Streaming responses
* Building a simple chatbot workflow
* Improving response speed

---

### 2. Context-Aware RAG

A Retrieval-Augmented Generation system that allows an LLM to answer questions using information retrieved from a knowledge base while maintaining relevant conversational context.

The general flow is:

```text
User Query
    ↓
Query Processing
    ↓
Retriever
    ↓
Relevant Context
    ↓
LLM
    ↓
Context-Aware Answer
```

This project helped explore the relationship between:

* Documents
* Chunking
* Embeddings
* Vector databases
* Retrievers
* Context
* LLMs

---

### 3. SQL To-Do AI Agent

An AI agent capable of interacting with a SQL database to perform tasks based on natural-language instructions.

Instead of manually writing SQL queries, the user can communicate with the agent using natural language.

Example:

```text
User:
Show me all unfinished tasks.

        ↓

AI Agent

        ↓

SQL Database

        ↓

Query Result

        ↓

Natural Language Response
```

This project explores:

* AI agents
* Tool calling
* SQL databases
* LLM reasoning
* Database interaction
* LangChain agents
* LangGraph concepts

---

## 🧠 Technologies

| Technology         | Purpose                                      |
| ------------------ | -------------------------------------------- |
| Python             | Core programming language                    |
| LangChain          | LLM application framework                    |
| LangGraph          | Agent and workflow orchestration             |
| Mistral AI         | LLM provider                                 |
| Hugging Face       | Models and embeddings                        |
| Chroma / Vector DB | Vector storage and retrieval                 |
| SQL                | Database interaction                         |
| Streamlit          | User interfaces                              |
| uv                 | Python dependency and environment management |
| Git & GitHub       | Version control                              |

---

## 📂 Project Structure

```text
RagProject/
│
├── src/
│   └── MainProject/
│       ├── FastQ&A.py
│       ├── UIfastQ&A.py
│       └── ...
│
├── pyproject.toml
├── uv.lock
└── README.md
```

The project structure may evolve as new AI experiments and implementations are added.

---

## 🔄 Learning Path

This repository follows a progression from simple LLM applications toward more advanced AI systems:

```text
LLM
 │
 ▼
Prompting
 │
 ▼
Q&A Chatbot
 │
 ▼
Tool Calling
 │
 ▼
AI Agents
 │
 ▼
Embeddings
 │
 ▼
Vector Database
 │
 ▼
RAG
 │
 ▼
Context-Aware RAG
 │
 ▼
Advanced Agentic RAG
```

The goal is not just to use frameworks, but to understand the **architecture and flow behind each system**.

---

## ⚙️ Setup

### Clone the repository

```bash
git clone <your-repository-url>
cd RagProject
```

### Install dependencies

This project uses `uv` for Python package and environment management.

```bash
uv sync
```

### Activate the environment

```bash
source .venv/bin/activate
```

Or run commands directly with:

```bash
uv run python <file>
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root and add the API keys required by the specific project.

Example:

```env
MISTRAL_API_KEY=your_api_key
```

Never commit your `.env` file or expose API keys publicly.

---

## ▶️ Running the Projects

For example:

```bash
uv run python src/MainProject/FastQ&A.py
```

For the UI version:

```bash
uv run streamlit run src/MainProject/UIfastQ&A.py
```

Other projects may have their own required commands.

---

## 🌱 Git Branches

The repository uses Git branches to separate stable code from ongoing experiments.

```text
main
 │
 └── project
      ├── Fast Q&A
      ├── SQL AI Agent
      └── Context-Aware RAG
```

`main` is intended to represent the stable version, while `project` is used for active development and experimentation.

---

## 🎯 Current Focus

The current focus of the project is understanding and implementing:

* LLM applications
* RAG pipelines
* Embeddings
* Vector databases
* Retrieval strategies
* Conversational context
* Tool calling
* AI agents
* SQL agents
* LangChain
* LangGraph
* Performance optimization

---

## 🛣️ Future Direction

Planned areas of exploration include:

* Advanced RAG architectures
* Query transformation
* Query routing
* Query decomposition
* Reranking
* Hybrid search
* Agentic RAG
* Long-term memory
* Better evaluation
* Production-ready AI application architecture

---

## 📌 Purpose

This repository is primarily a **hands-on AI engineering learning project**.

Instead of only following tutorials, each implementation is used to understand a specific concept and how different components fit together into a complete AI application.

> Learn the architecture. Build the system. Understand why it works.

---

## 👨‍💻 Author

**Biswash Bista**

Learning and building with Python, LLMs, RAG, AI Agents, and modern AI engineering tools.

---

⭐ If you find this repository useful, consider giving it a star.
