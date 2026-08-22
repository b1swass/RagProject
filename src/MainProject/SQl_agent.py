from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit


# -------------------------
# LLM
# -------------------------
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit


# -------------------------
# LLM
# -------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# -------------------------
# Database
# -------------------------

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


# -------------------------
# SQL Tools
# -------------------------

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

tools = toolkit.get_tools()


# -------------------------
# Memory
# -------------------------

memory = InMemorySaver()


# -------------------------
# System Prompt
# -------------------------

system_prompt = """
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


# -------------------------
# Agent
# -------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    system_prompt=system_prompt
)


# -------------------------
# Chat Loop
# -------------------------

while True:

    query = input("\nUSER: ")

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
                "thread_id": "1"
            }
        }
    )

    result = response["messages"][-1].content

    print("\nAI:", result)
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# -------------------------
# Database
# -------------------------

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


# -------------------------
# SQL Tools
# -------------------------

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

tools = toolkit.get_tools()


# -------------------------
# Memory
# -------------------------

memory = InMemorySaver()


# -------------------------
# System Prompt
# -------------------------

system_prompt = """
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


# -------------------------
# Agent
# -------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    system_prompt=system_prompt
)


# -------------------------
# Chat Loop
# -------------------------

while True:

    query = input("\nUSER: ")

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
                "thread_id": "1"
            }
        }
    )

    result = response["messages"][-1].content

    print("\nAI:", result)