from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langchain_classic.chat_models import init_chat_model
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from pprint import pprint

@tool
def add_numbers(a:int,b:int):
    """
    it will return the sum of two numbers
    Args:
        a:Number One
        b:Number Two
    """
    return a+b

@tool
def multiply_num(a:int,b:int):
    """
    it will return the multiplication of two numbers
    Args:
        a:Number One
        b:Number Two
    """
    return a+b
add_numbers.invoke({'a':12,'b':3})

#multiply_num({'a':12,'b':23})


chat_model=init_chat_model(
    model="mistral-small-latest",
    model_provider="mistralai"
)
agent=create_agent(
    model=chat_model,
    tools=[add_numbers,multiply_num],
    system_prompt="You are a math teacher, and anlways use tools for calculations"
)
result=agent.invoke({"messages":[{'role':"user",'content':'what is 2+3'}]})
pprint(result)