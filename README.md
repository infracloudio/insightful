# InSightful-rerank

Enhancing the original InSightful with a reranker.

## Features

Everything InSightful can do but better. By utilizing and exploiting the methods of Advanced RAG using a reranker, we significantly improve the quality of retrieved context from the vector store.

## Overview of workflow

![RAG-FC-Rerank](https://github.com/user-attachments/assets/f56de040-05e8-4307-be70-16929a72bafb)

## Prerequisites

- Python 3.10.13
- TGI Docker Image
- TEI Docker Image
- ChromaDB Docker Image

## Usage

Insightful uses HuggingFace's [TGI](https://huggingface.co/docs/text-generation-inference/index) server for compatible models. Clone this repository and update the environment variables in the `.envrc` file with your server's address.

For the Retrieval Augmented Generation (RAG) aspect, Insightful uses HuggingFace's [TEI](https://huggingface.co/docs/text-embeddings-inference/en/index) server for compatible Embedding models. Update the environment variables in the `.envrc` file with your server's address.

To store these embeddings, Insightful uses a hosted instance of a ChromaDB server. Update the environment variables in the `.envrc` file with your vectorstore server's address.

```bash
git clone https://github.com/infracloudio/insightful.git
cd insightful
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

Set the environment variables.

```bash
source .envrc
```

Ensure that all environment variables are correctly set before running the application.

The Docker containers for each service (except Python) must be running before starting the application. Refer to their Docker installation guides for more information.

When running the TGI docker container, replace $model with $TGI_MODEL.

Similarly, replace $model with $TEI_MODEL when running the TEI container.

For the ChromaDB container, set the `--port` flag to $VECTORDB_PORT:$VECTORDB_PORT.

These flags ensure there are no conflicting ports between the services and allow access to the servers on the same host without issues.

Finally, run the application.

```bash
streamlit run app.py
```
