from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


embedding_model=MistralAIEmbeddings()
vector_store=Chroma(
    persist_directory="Chroma_DB",
    embedding_function=embedding_model
)
retriver=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":4,
                   "fetch_k":10,
                   "lambda_mult":0.5}
)

llm=ChatMistralAI(model="mistral-small-latest")

#prompt_template
prompt=ChatPromptTemplate.from_messages([
    ("system",
     """You are an AI assistant.
     use only the provided context to answer the question.
     if the asnwer is not present in the context,
     say:" I could not find the answer in the document".

"""),
("human",
 """
Context:{context}
Question:{question}

""")
])

print("Rag System Created")
print("press 0 to exit")

while True:
    query=input("You:")
    if query=="0":
        break
    docs=retriver.invoke(query)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt=prompt.invoke({
        "context": context,
        "question": query
    })

    response=llm.invoke(final_prompt)
    print(f"\nAI:{response.content}")