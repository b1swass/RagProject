from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_classic.retrievers import MultiQueryRetriever

docs=[
    Document(page_content="kajshfjhksfjhksdfhjaf",metadata={"source":"AI model"}),
     Document(
        page_content="Machine learning allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every task.",
        metadata={"source": "Machine Learning"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation combines information retrieval with large language models to provide answers based on external documents or knowledge bases.",
        metadata={"source": "RAG"}
    ),
    Document(
        page_content="Embeddings represent text as numerical vectors. Similar pieces of text have vectors that are close to each other in the embedding space.",
        metadata={"source": "Embeddings"}
    ),
    Document(
        page_content="Vector databases store embeddings and allow applications to efficiently search for documents that are semantically similar to a user's query.",
        metadata={"source": "Vector Database"}
    ),
]
embeddings=HuggingFaceBgeEmbeddings()
vector_store=Chroma.from_documents(docs,embeddings)
retriever=vector_store.as_retriever()
llm=ChatMistralAI(model="mistral-small-latest")
multi_query_retriver=MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query="who won the worldcup most?"
docs=multi_query_retriver.invoke(query)

print("\nRetrived Documents")
for doc in docs:
    print(doc.page_content)