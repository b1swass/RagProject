from dotenv import load_dotenv
load_dotenv()

import uuid
import streamlit as st

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SQL AI Agent",
    page_icon="🗄️",
    layout="centered",
)


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# ============================================================
# DATABASE
# ============================================================

db = SQLDatabase.from_uri("sqlite:///my_test.db")

db.run("""
    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK (
            status IN ('pending', 'in_progress', 'completed')
        ),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")


# ============================================================
# SQL TOOLS
# ============================================================

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

tools = toolkit.get_tools()


# ============================================================
# MEMORY
# ============================================================

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()


memory = st.session_state.memory


# ============================================================
# SESSION / THREAD ID
# ============================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


thread_id = st.session_state.thread_id


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = """
Your Name is SQL to do Manager remember it 
example:
question:who are you ?
answer:I am SQL to do Manager. How can i help you?
You are a task management assistant that interacts with a SQL database
containing a 'tasks' table.

TASK RULES:

1. Limit SELECT queries to a maximum of 10 results.
2. For task lists, order results by created_at DESC.
3. After CREATE, UPDATE, or DELETE operations, verify the change with a SELECT query.
4. Never invent tables, columns, or data.
5. If the user's request is unclear, ask for clarification.
6. When listing tasks, present the results in a clean structured format.

CRUD OPERATIONS:

CREATE:
INSERT INTO tasks(title, description, status)

READ:
SELECT * FROM tasks WHERE ... LIMIT 10

UPDATE:
UPDATE tasks SET status=? WHERE id=? OR title=?

DELETE:
DELETE FROM tasks WHERE id=? OR title=?

TABLE SCHEMA:

tasks(
    id,
    title,
    description,
    status,
    created_at
)

Valid status values:
- pending
- in_progress
- completed
"""


# ============================================================
# AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    system_prompt=system_prompt,
)


# ============================================================
# UI
# ============================================================

st.title("🗄️ SQL AI Agent")
st.caption("Manage your tasks using natural language.")


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

query = st.chat_input("Ask me about your tasks...")


if query:

    # -------------------------
    # Display user message
    # -------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)


    # -------------------------
    # Agent response
    # -------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                },
                {
                    "configurable": {
                        "thread_id": thread_id
                    }
                }
            )

            result = response["messages"][-1].content

        st.markdown(result)


    # -------------------------
    # Save assistant response
    # -------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })