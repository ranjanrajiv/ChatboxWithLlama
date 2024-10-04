from src.helper import returnChuknks, loadEmbeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') # Put the pinecone api key in .env 

text_chunks = returnChuknks()
embeddings = loadEmbeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "test" #name of the index in pinecone

# finds the index from the Pinecone DB
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,          # change this to the length of each vector embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

uids = [str(uuid4()) for _ in range(len(text_chunks))]
vector_store.add_documents(documents=text_chunks, ids=uids) # stroes the embeddings in PineCone Database