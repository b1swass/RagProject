from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st


#load the document
loader=PyPDFLoader("/home/b1swas/RAG/RagProject/src/Agentic_RAG/medical_report.pdf")
docs=loader.load()

#split and chunk
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splitted_docs=splitter.split_documents(documents=docs)

#embed the chunk
embedding=MistralAIEmbeddings()
#store the embed
vector_store=InMemoryVectorStore.from_documents(
    documents=splitted_docs,
    embedding=embedding,
    
)
#initialise the model/LLM
llm=init_chat_model(
    model="mistral-small-latest"
)

#initialize the retriever tool
@tool
def retrieve_context(query:str):
    """Retrive Documents Relevant to a query from knowledge Base"""
    context=""
    docs=vector_store.similarity_search(query=query,k=4)
    for doc in docs:
        context += doc.page_content+"\n\n"
    return context

system_prompt = """
You are a document-based AI assistant.

For every user question, use the retrieve_context tool to find relevant information
from the uploaded documents.

Answer using only the retrieved document context.
If the answer is not present in the retrieved context, say:
"I couldn't find that information in the provided documents."

Do not use outside knowledge or make assumptions.
"""
memory=InMemorySaver()

#create the agent with the model, tools, system prompt, and memory
agent=create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=system_prompt,
    checkpointer=memory
)
#create a simple  interface for user interaction
while True:
    query=input("\n User:")
    if query.lower()=="quit":
        break
    response=agent.invoke({"messages":[{'role':'user','content':query}]},
                          {"configurable":{'thread_id':1}})
    result=response["messages"][-1].content
    print("\nAI:",result)

