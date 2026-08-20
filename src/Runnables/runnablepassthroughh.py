from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough


model=init_chat_model(
    model="mistral-small-latest",
)
parser=StrOutputParser()
chat_prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a code generator"),
        ('human',"{topic}")
    ]
)
explain_prompt=ChatPromptTemplate.from_messages([
    ("system","you are an AI assistant who helps in explaining the code"),
    ("human","Explain the following codes in simpler language:\n{code}")
])
seq=chat_prompt |model|parser


seq2=RunnableParallel(
    code=RunnablePassthrough(),
     explanation=RunnableLambda(lambda x:{'code':x})|explain_prompt |model |parser
)

chain = seq | seq2
result=chain.invoke({"topic":"write a code of palindrome in Python"})
print(result['code'])
print(result['explanation'])
  

#chat gpt ans
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)


model = init_chat_model(
    model="mistral-small-latest",
    model_provider="mistralai"
)

parser = StrOutputParser()


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])


explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant who helps in explaining code"),
    ("human", "Explain the following code in simpler language:\n{code}")
])


seq = chat_prompt | model | parser


seq2 = RunnableParallel(
    code=RunnablePassthrough(),

    explanation=
        RunnableLambda(lambda x: {"code": x})
        | explain_prompt
        | model
        | parser
)


chain = seq | seq2


result = chain.invoke({
    "topic": "Write a code of palindrome in Python"
})


print(result["code"])

print("-+*" * 50)

print(result["explanation"])