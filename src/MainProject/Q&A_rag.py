from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


# -------------------------
# 1. Load PDF
# -------------------------

loader = PyPDFLoader(
    "/home/b1swas/RAG/RagProject/GenAIpart2.pdf"
)

docs = loader.load()


# -------------------------
# 2. Split documents
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splitted_data = splitter.split_documents(docs)


# -------------------------
# 3. Create embeddings
# -------------------------

embedding_model = MistralAIEmbeddings()


# -------------------------
# 4. Create vector database
# -------------------------

vector_db = Chroma.from_documents(
    documents=splitted_data,
    embedding=embedding_model,
)


# -------------------------
# 5. Create retriever
# -------------------------

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)


# -------------------------
# 6. Create LLM
# -------------------------

llm = init_chat_model(
    model="mistral-small-latest"
)


# -------------------------
# 7. Create prompt
# -------------------------

prompt = PromptTemplate.from_template(
    """
You are a helpful assistant that answers questions based only on
the provided context.

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)


# -------------------------
# 8. Format retrieved docs
# -------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# -------------------------
# 9. Build RAG chain
# -------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# -------------------------
# 10. Ask question
# -------------------------
while True:
    query = input("\nAsk a question:")

    response = rag_chain.invoke(query)
    print('\nAnswer:')
    print(response.content)