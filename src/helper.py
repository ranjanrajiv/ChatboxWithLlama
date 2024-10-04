from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings


#splits the data into chunks
def returnChuknks():
    loader = PyPDFDirectoryLoader("data") # name of the local folder to read the pdf
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks


# using the sentence-transformers/all-MiniLM-L6-v2 from HuggingFace for the embeddings
def loadEmbeddings():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding