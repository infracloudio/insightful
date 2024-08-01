import os
import uuid
import streamlit as st
import datasets
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import StackExchangeAPIWrapper
from langchain_community.tools.stackexchange.tool import StackExchangeTool
from langchain_core.messages import SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer

from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from tei_rerank import TEIRerank


st.set_page_config(layout="wide", page_title="InSightful")

# Set up Chroma DB client
def setup_chroma_client():
    client = chromadb.HttpClient(
        host="http://{host}:{port}".format(
            host=os.getenv("VECTORDB_HOST", "localhost"),
            port=os.getenv("VECTORDB_PORT", "8000"),
        ),
        settings=Settings(allow_reset=True, 
                          anonymized_telemetry=False)

    )
    return client

# Set up Chroma embedding function
def setup_chroma_embedding_function():
    chroma_embedding_function = HuggingFaceEmbeddingServer(
        url="http://{host}:{port}/embed".format(
            host=os.getenv("TEI_HOST", "localhost"), port=os.getenv("TEI_PORT", "8081")
        )
    )
    return chroma_embedding_function

# Set up HuggingFaceEndpoint model
def setup_huggingface_endpoint(model_id):
    llm = HuggingFaceEndpoint(
        endpoint_url="http://{host}:{port}".format(
            host=os.getenv("TGI_HOST", "localhost"), port=os.getenv("TGI_PORT", "8080")
        ),
        temperature=0.3,
        task="conversational",
        stop_sequences=[
            "<|im_end|>",
            "{your_token}".format(your_token=os.getenv("STOP_TOKEN", "<|end_of_text|>")),
        ],
    )

    model = ChatHuggingFace(llm=llm,
                            model_id=model_id)

    return model

def setup_portkey_integrated_model():
    from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
    from langchain_openai import ChatOpenAI
    portkey_headers = createHeaders(
        api_key=os.getenv("PORTKEY_API_KEY"),
        custom_host=os.getenv("PORTKEY_CUSTOM_HOST"),
        provider=os.getenv("PORTKEY_PROVIDER")
    )
    
    model = ChatOpenAI(
        api_key="None",
        base_url=PORTKEY_GATEWAY_URL,
        model="qwen2",  # Verify the exact model name
        default_headers=portkey_headers,
    )

    return model

# Set up HuggingFaceEndpointEmbeddings embedder
def setup_huggingface_embeddings():
    embedder = HuggingFaceEndpointEmbeddings(
        model="http://{host}:{port}".format(
            host=os.getenv("TEI_HOST", "localhost"), port=os.getenv("TEI_PORT", "8081")
        ),
        task="feature-extraction",
    )
    return embedder

def load_prompt_and_system_ins():
    #prompt = hub.pull("hwchase17/react-chat")
    prompt = PromptTemplate.from_file("templates/prompt_template.tmpl")
    
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
    def __init__(self, llm, embeddings, collection_name, db_client):
        self.llm = llm
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.db_client = db_client

    def load_documents(self, doc, num_docs=250):
        documents = []
        for data in datasets.load_dataset(doc, split=f"train[:{num_docs}]").to_list():
            documents.append(
                Document(
                    page_content=data["text"],
                    metadata=dict(user=data["user"], 
                                  workspace=data["workspace"]),
                )
            )
        print("Document loaded")
        return documents

    def chunk_doc(self, pages, chunk_size=512, chunk_overlap=30):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(pages)
        print("Document chunked")
        return chunks

    def insert_embeddings(self, chunks, chroma_embedding_function, embedder, batch_size=32):
        collection = self.db_client.get_or_create_collection(
            self.collection_name, embedding_function=chroma_embedding_function
        )
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            chunk_ids = [str(uuid.uuid1()) for _ in batch]
            metadatas = [chunk.metadata for chunk in batch]
            documents = [chunk.page_content for chunk in batch]
            
            collection.add(
                ids=chunk_ids,
                metadatas=metadatas,
                documents=documents
        )
        db = Chroma(
            embedding_function=embedder,
            collection_name=self.collection_name,
            client=self.db_client,
        )
        print("Embeddings inserted\n")
        return db

    def query_docs(self, model, question, vector_store, prompt):
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        answer = rag_chain.invoke(question)
        return answer

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_retriever(name, model, description, client, chroma_embedding_function, embedder):
    rag = RAG(llm=model, embeddings=embedder, collection_name="Slack", db_client=client)
    pages = rag.load_documents("spencer/software_slacks")
    chunks = rag.chunk_doc(pages)
    vector_store = rag.insert_embeddings(chunks, chroma_embedding_function, embedder)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )
    info_retriever = create_retriever_tool(retriever, name, description)
    return info_retriever

def create_reranker_retriever(name, model, description, client, chroma_embedding_function, embedder):
    rag = RAG(llm=model, embeddings=embedder, collection_name="Slack", db_client=client)
    pages = rag.load_documents("spencer/software_slacks", num_docs=100)
    chunks = rag.chunk_doc(pages)
    vector_store = rag.insert_embeddings(chunks, chroma_embedding_function, embedder)
    compressor = TEIRerank(url="http://{host}:{port}".format(host=os.getenv("RERANKER_HOST", "localhost"), 
                                                                    port=os.getenv("RERANKER_PORT", "8082")), 
                                                                    top_n=10,
                                                                    batch_size=16)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 100}
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    info_retriever = create_retriever_tool(compression_retriever, name, description)
    return info_retriever

def setup_tools(_model, _client, _chroma_embedding_function, _embedder):
    stackexchange_wrapper = StackExchangeAPIWrapper(max_results=3)
    stackexchange_tool = StackExchangeTool(api_wrapper=stackexchange_wrapper)

    web_search_tool = TavilySearchResults(max_results=10,
                                          handle_tool_error=True)

    #retriever = create_retriever(
    #    name="Slack conversations retriever",
    #    model=_model,
    #    description="Retrieves conversations from Slack for context.",
    #    client=_client,
    #    chroma_embedding_function=_chroma_embedding_function,
    #    embedder=_embedder,
    #)

    if os.getenv("USE_RERANKER", "False") == "True":
        retriever = create_reranker_retriever(
            name="slack_conversations_retriever",
            model=_model,
            description="Useful for when you need to answer from Slack conversations.",
            client=_client,
            chroma_embedding_function=_chroma_embedding_function,
            embedder=_embedder,
        )
    else:
        retriever = create_retriever(
            name="slack_conversations_retriever",
            model=_model,
            description="Useful for when you need to answer from Slack conversations.",
            client=_client,
            chroma_embedding_function=_chroma_embedding_function,
            embedder=_embedder,
        )


    return [web_search_tool, stackexchange_tool, retriever]

@st.cache_resource
def setup_agent(_model, _prompt, _client, _chroma_embedding_function, _embedder):
    tools = setup_tools(_model, _client, _chroma_embedding_function, _embedder)
    agent = create_react_agent(llm=_model, prompt=_prompt, tools=tools, )
    agent_executor = AgentExecutor(
        agent=agent, verbose=True, tools=tools, handle_parsing_errors=True
    )
    return agent_executor

def main():
    client = setup_chroma_client()
    chroma_embedding_function = setup_chroma_embedding_function()
    prompt, system_instructions = load_prompt_and_system_ins()
    if os.getenv("ENABLE_PORTKEY", "False") == "True":
        model = setup_portkey_integrated_model()
    else:
        model = setup_huggingface_endpoint(model_id="qwen/Qwen2-7B-Instruct")
    embedder = setup_huggingface_embeddings()

    agent_executor = setup_agent(
        model, prompt, client, chroma_embedding_function, embedder
    )

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
    main()
