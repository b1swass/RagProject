from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader=PyPDFLoader("/home/b1swas/RAG/RagProject/src/Agentic_RAG/medical_report.pdf")
docs=loader.load()
len(docs)


splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splitted_docs=splitter.split_documents(docs)
len(splitted_docs)

embedding_model=MistralAIEmbeddings()
vector_store=InMemoryVectorStore.from_documents(
    documents=splitted_docs,
    embedding=embedding_model,
)

@tool
def retriever_tool(query:str):
    """This tool can help you to retrieve the relevant data of the PDF Documents, and then answer according to the details on medical report"""
    print("Tool called",query)
    docs=vector_store.similarity_search(query=query,k=4)
    context=""
    for doc in docs:
        context=doc.page_content+"\n\n"
    return context

#retriever_tool("Patient Name")

llm_model=init_chat_model(
    model="mistral-small-latest"
)

System_Prompt="""
you are a helpfull assistant that answers questions using retrived context.
Always use the "retriver_tool" for questions requiring external knowledge
"""
agent=create_agent(
    model=llm_model,
    tools=[retriever_tool],
    system_prompt=System_Prompt,
)

query="what is doctor name ? wwrite a pyhton code to add two sum"

response=agent.invoke({"messages":[{'role':'user',"content":query}]})

result=response['messages'][-1].content
print(result)