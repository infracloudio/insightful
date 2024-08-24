import os
import uuid
import datasets
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.chroma import Chroma
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer

from langchain.retrievers import ContextualCompressionRetriever
from tei_rerank import TEIRerank
from transformers import AutoTokenizer
from tools import get_tools

import streamlit as st
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader

from langchain.globals import set_verbose, set_debug

set_verbose(True)
set_debug(True)

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from urllib3.exceptions import ProtocolError

st.set_page_config(layout="wide", page_title="InSightful")


def authenticate():
    with open(".streamlit/config.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["pre-authorized"],
    )

    name, authentication_status, username = authenticator.login()
    st.session_state["authentication_status"] = authentication_status
    st.session_state["username"] = username
    return authenticator


# Set up Chroma DB client
@st.cache_resource
def setup_chroma_client():
    client = chromadb.HttpClient(
        host="http://{host}:{port}".format(
            host=os.getenv("VECTORDB_HOST", "localhost"),
            port=os.getenv("VECTORDB_PORT", "8000"),
        ),
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )
    return client


# Set up Chroma embedding function
@st.cache_resource
def hf_embedding_server():
    _embedding_function = HuggingFaceEmbeddingServer(
        url="http://{host}:{port}/embed".format(
            host=os.getenv("TEI_HOST", "localhost"), port=os.getenv("TEI_PORT", "8081")
        )
    )
    return _embedding_function


# Set up HuggingFaceEndpoint model
@st.cache_resource
def setup_chat_endpoint():
    model = ChatOpenAI(
        base_url="http://{host}:{port}/v1".format(
            host=os.getenv("TGI_HOST", "localhost"), port=os.getenv("TGI_PORT", "8080")
        ),
        max_tokens=os.getenv("MAX_TOKENS", 1024),
        temperature=0.7,
        api_key="dummy",
    )
    return model


# Set up Portkey integrated model
@st.cache_resource
def setup_portkey_integrated_model():
    from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
    from langchain_openai import ChatOpenAI

    portkey_headers = createHeaders(
        api_key=os.getenv("PORTKEY_API_KEY"),
        custom_host=os.getenv("PORTKEY_CUSTOM_HOST"),
        provider=os.getenv("PORTKEY_PROVIDER"),
    )

    model = ChatOpenAI(
        api_key="None",
        base_url=PORTKEY_GATEWAY_URL,
        model="qwen2",  # Verify the exact model name
        default_headers=portkey_headers,
    )

    return model


# Set up HuggingFaceEndpointEmbeddings embedder
@st.cache_resource
def setup_huggingface_embeddings():
    embedder = HuggingFaceEndpointEmbeddings(
        model="http://{host}:{port}".format(
            host=os.getenv("TEI_HOST", "localhost"), port=os.getenv("TEI_PORT", "8081")
        ),
        task="feature-extraction",
    )
    return embedder


@st.cache_resource
def load_prompt_and_system_ins(
    template_file_path="templates/prompt_template.tmpl", template=None
):
    # prompt = hub.pull("hwchase17/react-chat")
    prompt = PromptTemplate.from_file(template_file_path)

    # Set up prompt template
    template = """
    Based on the retrieved context, respond with an accurate answer. Use the provided tools to support your response.

    Be concise and always provide accurate, specific, and relevant information.
    """

    system_instructions = SystemMessage(
        content=template,
        metadata={"role": "system"},
    )

    return prompt, system_instructions


class RAG:
    def __init__(self, collection_name, db_client):
        self.collection_name = collection_name
        self.db_client = db_client

    @retry(
        retry=retry_if_exception_type(ProtocolError),
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
    )
    def load_documents(self, doc, num_docs=250):
        documents = []
        for data in datasets.load_dataset(
            doc, split=f"train[:{num_docs}]", num_proc=10
        ).to_list():
            documents.append(
                Document(
                    page_content=data["text"],
                    metadata=dict(user=data["user"], workspace=data["workspace"]),
                )
            )
        print("Document loaded")
        return documents

    def chunk_doc(self, pages, chunk_size=512, chunk_overlap=30):
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(pages)
        print("Document chunked")
        return chunks

    def insert_embeddings(self, chunks, chroma_embedding_function, batch_size=32):
        print(
            "Inserting embeddings into collection: {collection_name}".format(
                collection_name=self.collection_name
            )
        )
        collection = self.db_client.get_or_create_collection(
            self.collection_name, embedding_function=chroma_embedding_function
        )
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            chunk_ids = [str(uuid.uuid1()) for _ in batch]
            metadatas = [chunk.metadata for chunk in batch]
            documents = [chunk.page_content for chunk in batch]

            collection.add(ids=chunk_ids, metadatas=metadatas, documents=documents)
        print("Embeddings inserted\n")

    def query_docs(
        self, model, question, vector_store, prompt, chat_history, use_reranker=False
    ):
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        )
        if use_reranker:
            compressor = TEIRerank(
                url="http://{host}:{port}".format(
                    host=os.getenv("RERANKER_HOST", "localhost"),
                    port=os.getenv("RERANKER_PORT", "8082"),
                ),
                top_n=4,
                batch_size=10,
            )
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )

        pass_question = lambda input: input["question"]
        rag_chain = (
            RunnablePassthrough.assign(context=pass_question | retriever | format_docs)
            | prompt
            | model
            | StrOutputParser()
        )

        return rag_chain.stream({"question": question, "chat_history": chat_history})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_retriever(
    name, description, client, chroma_embedding_function, embedding_svc, reranker=False
):
    collection_name = "software-slacks"
    rag = RAG(collection_name=collection_name, db_client=client)
    pages = rag.load_documents("spencer/software_slacks", num_docs=100)
    chunks = rag.chunk_doc(pages)
    rag.insert_embeddings(chunks, chroma_embedding_function)
    vector_store = Chroma(
        embedding_function=embedding_svc,
        collection_name=collection_name,
        client=client,
    )
    if reranker:
        compressor = TEIRerank(
            url="http://{host}:{port}".format(
                host=os.getenv("RERANKER_HOST", "localhost"),
                port=os.getenv("RERANKER_PORT", "8082"),
            ),
            top_n=10,
            batch_size=16,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 100}
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        info_retriever = create_retriever_tool(compression_retriever, name, description)
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        )
        info_retriever = create_retriever_tool(retriever, name, description)

    return info_retriever


@st.cache_resource
def setup_agent(_model, _prompt, _tools):
    agent = create_react_agent(
        llm=_model,
        prompt=_prompt,
        tools=_tools,
    )
    agent_executor = AgentExecutor(
        agent=agent, verbose=True, tools=_tools, handle_parsing_errors=True
    )
    return agent_executor


def main():
    client = setup_chroma_client()
    chroma_embedding_function = hf_embedding_server()
    prompt, system_instructions = load_prompt_and_system_ins()
    if os.getenv("ENABLE_PORTKEY", "False") == "True":
        model = setup_portkey_integrated_model()
    else:
        model = setup_chat_endpoint()
    embedder = setup_huggingface_embeddings()
    use_reranker = os.getenv("USE_RERANKER", "False") == "True"

    retriever_tool = create_retriever(
        "slack_conversations_retriever",
        "Useful for when you need to answer from Slack conversations.",
        client,
        chroma_embedding_function,
        embedder,
        reranker=use_reranker,
    )
    _tools = get_tools()
    _tools.append(retriever_tool)

    agent_executor = setup_agent(model, prompt, _tools)

    st.title("InSightful: Your AI Assistant for community questions")
    st.text("Made with ❤️ by InfraCloud Technologies")
    st.markdown(
        """
    InSightful is an AI assistant that helps you with your questions.
    - It can browse past conversations with your colleagues/teammates and can search StackOverflow for technical questions.
    - With access to the web, InSightful can also conduct its own research for you."""
    )

    chat_history = st.session_state.get(
        "chat_history", [{"role": "system", "content": system_instructions.content}]
    )

    for message in chat_history[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Enter your question here"):
        st.chat_message("user").markdown(question)
        chat_history.append({"role": "user", "content": question})
        with st.spinner():
            response = agent_executor.invoke(
                {
                    "input": question,
                    "chat_history": chat_history,
                }
            )["output"]
        st.chat_message("assistant").markdown(response)
        chat_history.append({"role": "assistant", "content": response})

    st.session_state["chat_history"] = chat_history


if __name__ == "__main__":
    # authenticator = authenticate()
    # if st.session_state['authentication_status']:
    #    authenticator.logout()
    main()
