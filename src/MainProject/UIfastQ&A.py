from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# -----------------------------
# Page Configuration
# -----------------------------

st.title("🤖 AI Assistant")


# -----------------------------
# Agent Setup
# -----------------------------

memory = InMemorySaver()

search = TavilySearch()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
)

agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt="You are an amazing AI assistant and can search on google as well",
    checkpointer=memory
)


# -----------------------------
# Session Memory
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session-1"


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

query = st.chat_input("Ask me anything...")


if query:

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)

    # Save user message in Streamlit session
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


    # -----------------------------
    # Invoke Agent
    # -----------------------------

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
                "thread_id": st.session_state.thread_id
            }
        }
    )


    answer = response["messages"][-1].content


    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)


    # Save assistant response in Streamlit session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })