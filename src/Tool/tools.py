from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient



#search_tool=TavilyClient()
search_tool=TavilySearchResults(max_result=5)

model=init_chat_model(
    model="mistral-small-latest",
    model_provider="mistralai"
)

prompt=ChatPromptTemplate.from_template(
    """
    You are a helpful  AI assistant 
    summarise the following news into clear bullet points
    and provide a clean output
    {news}
"""
)
parser=StrOutputParser()

chain=prompt | model | parser

news_result=search_tool.run("im a email marketing agency and i wanna automate the designing part so, which AI tool can design the email marketing poster and image generation im more into designing poster, image etc not the marketing tools? which ai is better in image creation? ")
result=chain.invoke({'news':news_result})
print(result)