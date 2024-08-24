import os
import logging
import tempfile
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import RegisterError
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
from tools import get_tools

from app import (
    setup_chroma_client,
    hf_embedding_server,
    load_prompt_and_system_ins,
    setup_huggingface_embeddings,
    setup_huggingface_endpoint,
    RAG,
    setup_agent
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

def configure_authenticator():
    auth_config = os.getenv("AUTH_CONFIG_FILE_PATH", default=".streamlit/config.yaml")
    print(f"auth_config: {auth_config}")
    with open(file=auth_config) as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["pre-authorized"],
    )
    return authenticator


def authenticate(op):
    authenticator = configure_authenticator()

    if op == "login":
        name, authentication_status, username = authenticator.login()
        st.session_state["authentication_status"] = authentication_status
        st.session_state["username"] = username
        st.session_state["name"] = name
    elif op == "register":
        try:
            (
                email_of_registered_user,
                username_of_registered_user,
                name_of_registered_user,
            ) = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                st.success("User registered successfully")
        except RegisterError as e:
            st.error(e)
    return authenticator


class MultiTenantRAG(RAG):
    def __init__(self, user_id, collection_name, db_client):
        self.user_id = user_id
        super().__init__(collection_name, db_client)

    def load_documents(self, doc):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(doc.name)[1]
        ) as tmp:
            tmp.write(doc.getvalue())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        cleaned_pages = []
        for doc in documents:
            doc.page_content = clean_extra_whitespace(doc.page_content)
            doc.page_content = group_broken_paragraphs(doc.page_content)
            cleaned_pages.append(doc)
        return cleaned_pages


def main():
    use_reranker = st.sidebar.toggle("Use reranker", False)
    use_tools = st.sidebar.toggle("Use tools", False)
    uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf"])
    question = st.chat_input("Chat with your doc")


    llm = setup_huggingface_endpoint(model_id=os.getenv("MODEL_ID"))

    embedding_svc = setup_huggingface_embeddings()

    chroma_embeddings = hf_embedding_server()

    user_id = st.session_state["username"]

    client = setup_chroma_client()
    # Set up prompt template
    template = """
    Based on the retrieved context, respond with an accurate answer.

    Be concise and always provide accurate, specific, and relevant information.
    """

    template_file_path = "templates/multi_tenant_rag_prompt_template.tmpl"
    if use_tools:
        template_file_path = "templates/multi_tenant_rag_prompt_template_tools.tmpl"
        
    prompt, system_instructions = load_prompt_and_system_ins(
        template_file_path=template_file_path,
        template=template,
    )

    chat_history = st.session_state.get(
        "chat_history", [{"role": "system", "content": system_instructions.content}]
    )

    for message in chat_history[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not user_id:
        st.error("Please login to continue")
        return

    collection = client.get_or_create_collection(
        f"user-collection-{user_id}", embedding_function=chroma_embeddings
    )


    logger = logging.getLogger(__name__)
    logger.info(f"user_id: {user_id} use_reranker: {use_reranker} use_tools: {use_tools} question: {question}")
    rag = MultiTenantRAG(user_id, collection.name, client)

    if use_tools:
        tools = get_tools()
        agent_executor = setup_agent(llm, prompt, tools)

    # prompt = hub.pull("rlm/rag-prompt")

    vectorstore = Chroma(
        embedding_function=embedding_svc,
        collection_name=collection.name,
        client=client,
    )

    if uploaded_file:
        document = rag.load_documents(uploaded_file)
        chunks = rag.chunk_doc(document)
        rag.insert_embeddings(
            chunks=chunks,
            chroma_embedding_function=chroma_embeddings,
            batch_size=32,
        )

    if question:
        st.chat_message("user").markdown(question)
        with st.spinner():
            if use_tools:
                answer = agent_executor.invoke({
                        "question": question,
                        "chat_history": chat_history,
                    })["output"]
                with st.chat_message("assistant"):
                    answer = st.write(answer)
                    logger.info(f"answer: {answer}")
            else:
                answer = rag.query_docs(
                    model=llm,
                    question=question,
                    vector_store=vectorstore,
                    prompt=prompt,
                    chat_history=chat_history,
                    use_reranker=use_reranker,
                )
                with st.chat_message("assistant"):
                    answer = st.write_stream(answer)
                    logger.info(f"answer: {answer}")

            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})
            st.session_state["chat_history"] = chat_history


if __name__ == "__main__":
    authenticator = authenticate("login")
    if st.session_state["authentication_status"]:
        st.sidebar.text(f"Welcome {st.session_state['username']}")
        authenticator.logout(location="sidebar")
        main()
