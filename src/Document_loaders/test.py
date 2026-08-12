from langchain_community.document_loaders import TextLoader

data=TextLoader("/home/b1swas/RAG/RagProject/src/Document_loaders/notes.txt")
docs=data.load() #it will always be in list
print(docs[0]) 