from dotenv import load_dotenv 
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI


@tool
def google_search(query:str)->str:
    """Search google for up to date information on the web
    Agrs:What is to search on google
    Returns:Top search results as plain text"""

    search=TavilySearch(
        max_result=5
    )
    return search.invoke({"query":query})

llm=ChatGroq(model="openai/gpt-oss-120b")
# #llm=ChatMistralAI(
#     model="mistral-small-latest"
# )
SYSTEM_PROMPT="""
    You are a helpful research assistant with access to google and search the result to get top 5 results in plain text and summarise it to use """


agent=create_agent(
    model=llm,
    tools=[google_search],
    system_prompt=SYSTEM_PROMPT,
    #checkpointer=InMemorySaver()
)

# result=agent.invoke({"messages":[{'role':'user','content':"how is the flood that hit Nepal? recently the rasuwa , trishuli nuwakot flood ?"}]},
#                     config={"configurable":{"thread_id":"1"}})

# print(result["messages"][-1].content)
