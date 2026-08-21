from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
import streamlit as st
from langchain.agents import create_agent

memory=InMemorySaver()
if "memory" not in  st.session_state
st.session_state.memory=InMemorySaver()
search=TavilySearch()

llm=init_chat_model(
    model='openai/gpt-oss-20b'
)
agent=create_agent(
    model=llm,
    tools=[search],
    system_prompt="You are an amazing AI assistant and can search on google as well",
    checkpointer=memory
)

while True:
    query=input("\n Ask me:-")
    if query.lower()=="quit":
        print("GoodBye")
        break
    response=agent.invoke(
        {'messages':[{"role":'user','content':query}]},
        {'configurable':{"thread_id":"1"}},
        )
    print(response['messages'][-1].content)