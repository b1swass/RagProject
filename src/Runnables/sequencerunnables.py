from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in simple word"
)
model=init_chat_model(
    model="mistral-small-latest",

)
parser=StrOutputParser()

chain = prompt | model | parser   # pipeline
result=chain.invoke({"topic":"Dhanusha"})
print(result)


# formatted_prompt=prompt.format_messages(topic="Jose Maurinho")
# response=model.invoke(formatted_prompt)
# final_output=parser.parse(response.content)
# print(final_output)