# ChatboxWithLlama

This is a generalized chatbot. It can be used for any topic of interest with any chat llm.

## Instruction - Steps to run

### Clone the repository
https://github.com/ranjanrajiv/ChatboxWithLlama


### Create the directories
    1. model - Download any language model for chat in this
    2. data - Download the pdf, for the subject of interest. Used   to populate the vector database

### Create a python virtual environment using conda or virtualvenv

### Install all the requirements with
    pip install -r requirements.txt

### Perform the steps to access the cloud pinecone database. Then get the Pinecone api key (refer pinecone.io)

### Run vectordb_handler.py to populate the pinecone vectordb

### run python app.py - Go to 127.0.0.1:8080