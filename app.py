from flask import Flask, render_template, jsonify, request
from src.helper import loadEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.prompt import *
import time
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "test" #name of the index in pinecone



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


llm = CTransformers(model="<Insert mode name>" # eg. any chat model from llama on HuggingFace
                    model_type="llama",
                    config={'max_new_tokens':512,
                            'temperature':0.1})
embeddings = loadEmbeddings()

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "questions"])
chain_type_kwargs={"prompt": prompt}


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)