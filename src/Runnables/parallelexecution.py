from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda


model=init_chat_model(
    model="mistral-small-latest",
)
parser=StrOutputParser()

short_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)
detailed_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in details"
)
short_topic=input("\n Ask please: ")
detailed_topic=input("\n Ask please: ")

chain = RunnableParallel(
    short=RunnableLambda(lambda x:x["short"]) | short_prompt | model | parser,
    detailed=RunnableLambda(lambda x:x["detailed"]) | detailed_prompt | model | parser
)

result=chain.invoke({
    "short":{"topic":short_topic},
    "detailed":{"topic":detailed_topic}
                })
print(result['short'])
print("-+*" *50)
print(result['detailed'])