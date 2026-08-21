from dotenv import load_dotenv
load_dotenv()
from langchain_classic.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch

memory=MemorySaver()
search=TavilySearch()

llm=init_chat_model(
    model="mistral-small-latest",
    model_provider="mistralai"
)

agent=create_agent(
    model=llm,
    tools=[search],
    system_prompt="You are an agent who can search from google",
    checkpointer=memory,
)


while True:
    query=input('\n\nAsk your queries:')
    if query.lower()=="quit":
        print("\n Good Bye")
        break
    response=agent.invoke({"messages":[{'role':'user',"content":query}]},
                          {"configurable":{'thread_id':"0"}})
    print("\n\nqAI:",response["messages"][-1].content)