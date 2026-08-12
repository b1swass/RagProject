from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#initializing the text loader
#data=TextLoader("/home/b1swas/RAG/RagProject/src/Document_loaders/notes.txt")
#docs=data.load() #it will always be in list

data=PyPDFLoader("/home/b1swas/RAG/RagProject/src/Document_loaders/indiividual.pdf")
docs=data.load()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunk=text_splitter.split_documents(docs)

template=ChatPromptTemplate([
    ('system',
    """
You are AI that summarise the text
"""),
('human',"""{data}""")
])

#invoking the LLM intializing it 
llm=ChatMistralAI(model="mistral-small-2603")
prompt=template.format_messages(data=docs[0].page_content)
response=llm.invoke( 
prompt)
print(response.content)