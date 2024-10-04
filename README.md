# ChatboxWithLlama

This is a generalized chatbot. It can be used for any topic of interest with any chat llm.

It uses pinecone vector DB to store the embeddings.

Huggingface embedding is used.

Two similarity vector for any query is generated. These two are then given to the LLMs to get the final result.

## Instruction - Steps to run

### Clone the repository
https://github.com/ranjanrajiv/ChatboxWithLlama


### Create the directories
    # 1. model - Download any language model for chat in this
    # 2. data - Download the pdf, for the subject of interest. Used   to populate the vector database

### Create a python virtual environment using conda or virtualvenv
    conda create -n venv python=3.12 -y
    conda activate venv

### Install all the requirements with
    pip install -r requirements.txt


### Run vectordb_handler.py
    # Pouplate the Pinecone vectordb
    python vectordb_handler.py

### run the app
    python app.py
