from langchain_community.document_loaders import WebBaseLoader
url="https://www.apple.com/macbook-pro/"
data=WebBaseLoader(url)
doc=data.load()
print(doc)
