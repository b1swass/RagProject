from langchain_text_splitters import RecursiveCharacterTextSplitter  #its best we can chunk anything so its best till now 
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

pdf_chunk=PyPDFLoader("/home/b1swas/RAG/RagProject/src/Document_loaders/indiividual.pdf")
data1=pdf_chunk.load() #its loading the data from pdf

llm=TextLoader("/home/b1swas/RAG/RagProject/src/Document_loaders/notes.txt")
data=llm.load() # loading the data from txt file 


text_splitter=RecursiveCharacterTextSplitter( #using the text splitte for chunking
    chunk_size=100,
    chunk_overlap=0
)
chunk=text_splitter.split_documents(data1) #passing the data into the splitter 
print(len(chunk))

for i in chunk:
    print(i.page_content) #to check the content of chunk  