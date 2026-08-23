from dotenv import load_dotenv
load_dotenv()

import tempfile

import streamlit as st

from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain.agents import create_agent
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.caption("Upload your documents and ask questions about them.")


# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# --------------------------------------------------
# Sidebar - PDF upload
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "Process Documents",
        use_container_width=True
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} PDF(s) selected**")

        for file in uploaded_files:
            st.write(f"• {file.name}")


# --------------------------------------------------
# Process uploaded PDFs
# --------------------------------------------------

if process_button:

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

    else:

        with st.spinner("Processing your documents..."):

            all_docs = []

            # Load every uploaded PDF
            for uploaded_file in uploaded_files:

                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                # Load PDF
                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                # Add filename metadata
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)

            # --------------------------------------------------
            # Split documents
            # --------------------------------------------------

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            split_docs = splitter.split_documents(all_docs)

            # --------------------------------------------------
            # Create embeddings
            # --------------------------------------------------

            embedding = MistralAIEmbeddings()

            # --------------------------------------------------
            # Create vector store
            # --------------------------------------------------

            vector_store = InMemoryVectorStore.from_documents(
                documents=split_docs,
                embedding=embedding
            )

            st.session_state.vector_store = vector_store

            # --------------------------------------------------
            # Create retrieval tool
            # --------------------------------------------------

            @tool
            def retrieve_context(query: str):
                """
                Retrieve relevant information from the uploaded
                documents.
                """

                docs = vector_store.similarity_search(
                    query=query,
                    k=4
                )

                context = ""

                for doc in docs:

                    source = doc.metadata.get(
                        "source",
                        "Unknown document"
                    )

                    context += (
                        f"\nSource: {source}\n"
                        f"{doc.page_content}\n"
                    )

                return context

            # --------------------------------------------------
            # LLM
            # --------------------------------------------------

            llm = init_chat_model(
                model="mistral-small-latest"
            )

            # --------------------------------------------------
            # System prompt
            # --------------------------------------------------

            system_prompt = """
You are a document-based AI assistant.

For every user question, use the retrieve_context tool
to search the uploaded documents.

Answer using ONLY the information retrieved from the
uploaded documents.

If the answer cannot be found in the retrieved documents,
say:

"I couldn't find that information in the provided documents."

Do not use outside knowledge.
Do not make assumptions.
Do not invent information.

When possible, mention the source PDF from which the
information was retrieved.
"""

            # --------------------------------------------------
            # Memory
            # --------------------------------------------------

            memory = InMemorySaver()

            # --------------------------------------------------
            # Create agent
            # --------------------------------------------------

            agent = create_agent(
                model=llm,
                tools=[retrieve_context],
                system_prompt=system_prompt,
                checkpointer=memory
            )

            st.session_state.agent = agent

            # Store uploaded filenames
            st.session_state.uploaded_files = [
                file.name for file in uploaded_files
            ]

            # Clear previous chat
            st.session_state.messages = []

        st.success(
            f"Processed {len(uploaded_files)} document(s) "
            f"with {len(split_docs)} chunks."
        )


# --------------------------------------------------
# Display uploaded documents
# --------------------------------------------------

if st.session_state.uploaded_files:

    st.info(
        "Loaded documents: "
        + ", ".join(st.session_state.uploaded_files)
    )


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

query = st.chat_input(
    "Ask something about your documents..."
)


if query:

    # Check whether documents have been processed

    if st.session_state.agent is None:

        st.warning(
            "Please upload and process your PDF documents first."
        )

        st.stop()

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Generate response

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = st.session_state.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                },
                {
                    "configurable": {
                        "thread_id": "pdf-chat"
                    }
                }
            )

            result = response["messages"][-1].content

            st.markdown(result)

    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result
        }
    )