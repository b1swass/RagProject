from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings


document=["""
Runnable Passthrough is used when we want to keep the original input or some intermediate data while passing it
through the pipeline. Normally, in a sequence, each step replaces the previous output, so earlier data gets lost. But in
many real-world scenarios, we need to carry multiple pieces of information together. RunnablePassthrough allows us to
forward the input as it is, without modifying it, so that it can be used along with other outputs in later steps. In simple
terms, it helps us preserve data while still continuing the flow of the pipeline.
"""
]
embeddings_model=MistralAIEmbeddings()
vector=embeddings_model.embed_documents(document)
print(len(vector[0]))

vector_store=Chroma.from_texts(
    texts=document,
    embedding=embeddings_model,
    persist_directory="./vector_chroma"
)


query="what is runnable"
result=vector_store.similarity_search(query=query,k=2)
print(result)