from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pprint import pprint
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

loader=PyPDFLoader(file_path="/home/b1swas/RAG/RagProject/GenAIpart2.pdf")
data=loader.load()
text_split=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_docs=text_split.split_documents(documents=data)
embeddings=MistralAIEmbeddings(
    model="mistral-embed"
)
# vector_store=QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="Chai_genai",
#     embedding=embeddings
# )
#vector_store.add_documents(split_docs)
print("Injection Done")

retriver=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="Chai_genai",
    embedding=embeddings
)
relevant_chunks=retriver.similarity_search(
    query="what is Text Splitter"
)
SYSTEM_PROMPT="""
 you are an helful ai assistant who responds base of the avcailabe contents
{relevant_chunks}
"""
print("Relevant Chunks",relevant_chunks)
#print("Docs",len(data))
#print("Split",len(split_docs))