# insightful

The AI assistant for tech communities.

## Features

- [x] **Conversation Analysis**: insightful can analyze the conversation in a tech community and provide insights on the topics being discussed.

- [x] **Community Health Analysis**: insightful can analyze the health of a tech community and provide insights on the community's engagement, sentiment, and more.

- [x] **Search Stack Overflow**: insightful can search Stack Overflow for relevant questions and answers from the community discussions.

- [x] **Browse The Web**: insightful can browse the web for relevant information on the topics being discussed in the community.

## Prerequisites

- Python 3.10.13
- TGI Docker Image
- TEI Docker Image
- ChromaDB Docker Image

## Usage

Insightful uses HuggingFace's [TGI] (<https://huggingface.co/docs/text-generation-inference/index>) server to serve compatible models. To use insightful, simply clone this repository and change the environment variables in the `.envrc` file according to your server's address.

For the Retrieval Augmented Generation (RAG) aspect, Insightful uses HuggingFace's [TEI] (<https://huggingface.co/docs/text-embeddings-inference/en/index>) server to serve compatible Embedding models. Change the environment variables in the `.envrc` file according to your server's address.

To store these embeddings, Insightful uses a hosted instance of a ChromaDB server. Change the environment variables in the `.envrc` file according to your vectorstore server's address.

Please make sure that the environment variables are all set correctly before running the application.

```bash
git clone
cd insightful
```

Then, install the dependencies.

```bash
pip install -r requirements.txt
```

Set the environment variables.

```bash
source .envrc
```

Make sure the Docker containers for each of the above services (except Python)are running before starting the application.

Refer to each of their Docker installation guides for more information.

Replace the $model with $TGI_MODEL when running the TGI docker container.

Similarly with the TEI container. replace $model with $TEI_MODEL.

For the ChromaDB container, set the ```--port``` flag to $VECTORDB_PORT:$VECTORDB_PORT

Setting these flags ensures there are no conflicting ports between the services and to access the servers on the same host without any issues.

Finally, run the application.

```bash
streamlit run InSightful.py
```