from langchain import hub
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import StackExchangeAPIWrapper
from langchain_community.tools.stackexchange.tool import StackExchangeTool
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import datasets
import chromadb
from chromadb.config import Settings
import uuid
import streamlit as st
import os

st.set_page_config(layout="wide", page_title="InSightful")

client = chromadb.HttpClient(host="http://{host}:{port}".format(host=os.getenv("VECTORDB_HOST", "localhost"), port=os.getenv("VECTORDB_PORT", "8000")), settings=Settings(allow_reset=True))
chroma_embedding_function = HuggingFaceEmbeddingServer(url="http://{host}:{port}/embed".format(host=os.getenv("TEI_HOST", "localhost"), port=os.getenv("TEI_PORT", "8081")))

prompt = hub.pull("hwchase17/react-chat")

template = """
You are InSightful, a virtual assistant designed to help users with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model,
you are able to generate human-like text based on the input you receive, allowing you to engage in 
natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Always provide accurate and informative responses to a wide range of questions. 

You do not answer questions about personal information, such as social security numbers,
credit card numbers, or other sensitive information. You also do not provide medical, legal, or financial advice.

You will not respond to any questions that are inappropriate or offensive. You are friendly, helpful,
and you are here to assist users with any questions they may have.

Keep your answers clear and concise, and provide as much information as possible to help users understand the topic.

Use your best judgement and only use any tool if you absolutely need to.

Tools provide you with more context and up-to-date information. Use them to your advantage.

Input: {input}
"""

prompt_template = PromptTemplate.from_template(template=template)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

model = HuggingFaceEndpoint(endpoint_url="http://{host}:{port}".format(host=os.getenv("TGI_HOST", "localhost"), port=os.getenv("TGI_PORT", "8080")),
                                    temperature=0.3,
                                    task="conversational",
                                    stop_sequences=["<|eot_id|>", "{your_token}".format(os.getenv("STOP_TOKEN", "localhost"))])

embedder = HuggingFaceEndpointEmbeddings(model="http://{host}:{port}".format(host=os.getenv("TEI_HOST", "localhost"), port=os.getenv("TEI_PORT", "8081")),
                                         task="feature-extraction")

class RAG():
    def __init__(self, llm, embeddings, collection_name, db_client):
        self.llm = llm
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.db_client = db_client

    def load_documents(self, doc):
        documents: list[Document] = []
        for data in datasets.load_dataset(doc, split="train[:100]").to_list():
            documents.append(Document(page_content=data["text"], metadata=dict(
                            user=data['user'],
                        )))
        print("Document loaded")
        return documents
    
    def chunk_doc(self, pages, chunk_size=512, chunk_overlap=30):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(pages)
        print("Document chunked")
        return chunks
     
    def insert_embeddings(self, chunks):
        collection = self.db_client.get_or_create_collection(self.collection_name, embedding_function=chroma_embedding_function)#chroma_embedding_function)
        for chunk in chunks:
            collection.add(ids=[str(uuid.uuid1())], metadatas=chunk.metadata, documents=chunk.page_content)
        db = Chroma(
            embedding_function=embedder,
            collection_name=self.collection_name,
            client=self.db_client,
        )
        print("Embeddings inserted\n")
        return db
    
    def query_docs(self, question, vector_store, prompt):
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
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

stackexchange_wrapper = StackExchangeAPIWrapper(max_results=3)
stackexchange_tool = StackExchangeTool(api_wrapper=stackexchange_wrapper)
web_search_tool = TavilySearchResults(max_results=3)

@st.cache_resource
def create_retriever():
    rag = RAG(llm=model, embeddings=embedder, collection_name="Slack", db_client=client)
    pages = rag.load_documents("spencer/software_slacks")
    chunks = rag.chunk_doc(pages)#, chunk_size=4096, chunk_overlap=2048)
    vector_store = rag.insert_embeddings(chunks)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})
    return retriever

retriever = create_retriever()

GCP_info_retriever = create_retriever_tool(retriever,
                                "Slack conversations retriever",
                                "Retrieves conversations from Slack for context.")

tools = [web_search_tool, stackexchange_tool, GCP_info_retriever]

agent = create_react_agent(llm=model,
                           prompt=prompt,
                           tools=tools)

agent_executor = AgentExecutor(agent=agent,
                               verbose=True,
                               tools=tools,
                               handle_parsing_errors=True)

st.title("InSightful: Your AI Assistant for community questions")
st.markdown("""
InSightful is an AI assistant that helps you with your questions. 
- It can browse past conversations with your colleagues/teammates and can search StackOverflow for technical questions.
- With access to the web, InSightful can also conduct its own research for you.""")

chat_history = st.session_state.chat_history

for message in chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Enter your question here"):
    st.chat_message("user").markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner():
        response = agent_executor.invoke({"input": prompt_template.format(input=question), "chat_history": chat_history})['output']
    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})