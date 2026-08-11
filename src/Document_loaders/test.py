from langchain_community.document_loaders import TextLoader

data=TextLoader("/home/b1swas/Desktop/learnGenai/RagProject/src/Document_loaders/notes.txt")
docs=data.load() #it will always be in list
print(docs[0]) 