from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

from langchain.agents import create_agent

llm_model=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
 )
agent = create_agent(
    model=llm_model,
    tools=[web_search],
    system_prompt=system_prompt
)

while True:
    quey=input('\n what you wanna make ?')
    res=agent.invoke({'messages':[{'role':'user','content':quey}]},
                     {'configurable':{'thread_id':'1'}})
    print(res['messages'][-1].content)

    
    