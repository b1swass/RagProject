from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


# -----------------------------
# Model & Tools
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    streaming=True
)

search = GoogleSerperAPIWrapper()

tools = [search.run]


# -----------------------------
# Session Memory
# -----------------------------

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Agent
# -----------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are an amazing AI agent and can search on Google as well"
)


# -----------------------------
# Streamlit UI
# -----------------------------

st.subheader("QuickAnswer - Answers at the speed of thought")


# Display previous messages
for message in st.session_state.history:
    st.chat_message(message["role"]).markdown(message["content"])


# User input
query = st.chat_input("Ask Anything?")


if query:

    # Display user message
    st.chat_message("user").markdown(query)

    st.session_state.history.append({
        "role": "user",
        "content": query
    })


    # Stream agent response
    response = agent.stream(
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
        },
        stream_mode="messages"
    )


    # Display streaming response
    with st.chat_message("assistant"):
        space = st.empty()

        message = ""

        for chunk in response:
            message += chunk[0].content
            space.markdown(message)


    # Save assistant response
    st.session_state.history.append({
        "role": "assistant",
        "content": message
    })