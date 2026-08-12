import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("🤖 RAG Assistant")
st.caption("Ask questions about your documents")


# -----------------------------
# Initialize RAG components
# -----------------------------
@st.cache_resource
def initialize_rag():

    embedding_model = MistralAIEmbeddings()

    vector_store = Chroma(
        persist_directory="Chroma_DB",
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-latest"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an AI assistant.

Use only the provided context to answer the question.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Do not use outside knowledge.

Context:
{context}
"""
        ),
        (
            "human",
            "Question: {question}"
        )
    ])

    return retriever, llm, prompt


retriever, llm, prompt = initialize_rag()


# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User input
# -----------------------------
query = st.chat_input("Ask something about your documents...")


if query:

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching your documents..."):

            # Retrieve relevant documents
            docs = retriever.invoke(query)

            # Create context
            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            # Create final prompt
            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            # Ask LLM
            response = llm.invoke(final_prompt)

            answer = response.content

            st.markdown(answer)


    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })