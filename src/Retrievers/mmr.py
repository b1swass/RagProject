from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
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
embeddings=HuggingFaceBgeEmbeddings()
vector_store=Chroma.from_documents(docs,embeddings)
print("_______________SIMILARRITY SEARCH_______________")
similarity_retriever=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

similarity_docs=similarity_retriever.invoke("what is ml?")
for i in similarity_docs:
    print(i.page_content)


print("-#"*50)
print("___________________MMR SEARCH _________________")

similarity_retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)
similarity_docs=similarity_retriever.invoke("what is ml?")
for j in similarity_docs:
    print(j.page_content)