from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


from langchain_core.documents import Document

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
embedding_model=MistralAIEmbeddings()
vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)
response=vectorstore.similarity_search("What is use for storing ?")
for i in response:
    print(i.page_content)
    print(i.metadata)

retriver = vectorstore.as_retriever()
docs=retriver.invoke("what is ML?")
for d in docs:
    print(d.page_content)