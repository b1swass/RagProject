from dotenv import  load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langchain_classic.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import InMemorySaver

llm=init_chat_model(
    model="mistral-small-latest",

)
agent=create_agent(
    model=llm,
    tools=,
    checkpointer=InMemorySaver(),
)
agent.invoke(
    {"messages":[{"role":"user","con"}]}
)