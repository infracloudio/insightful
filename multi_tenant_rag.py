import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import RegisterError, LoginError
import os
from langchain_community.vectorstores.chroma import Chroma
from app import setup_chroma_client, setup_chroma_embedding_function, load_prompt_and_system_ins
from app import setup_huggingface_embeddings, setup_huggingface_endpoint
from app import RAG
from langchain import hub
import tempfile
from langchain_community.document_loaders import PyPDFLoader

import yaml
from yaml.loader import SafeLoader

def configure_authenticator():
    with open('.streamlit/config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
    )
    return authenticator

def authenticate(op):
    authenticator = configure_authenticator()

    if op == "login":
        name, authentication_status, username = authenticator.login()
        st.session_state['authentication_status'] = authentication_status
        st.session_state['username'] = username
    elif op == "register":
        try:
            (email_of_registered_user,
            username_of_registered_user,
            name_of_registered_user) = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                st.success('User registered successfully')
        except RegisterError as e:
            st.error(e)
    return authenticator

class MultiTenantRAG(RAG):
    def __init__(self, user_id, llm, embeddings, collection_name, db_client):
        self.user_id = user_id
        super().__init__(llm, embeddings, collection_name, db_client)
    
    def load_documents(self, doc):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(doc.name)[1]) as tmp:
            tmp.write(doc.getvalue())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        return documents

def main():
    llm = setup_huggingface_endpoint(model_id="qwen/Qwen2-7B-Instruct")

    embeddings = setup_huggingface_embeddings()

    chroma_embeddings = setup_chroma_embedding_function()

    user_id = st.session_state['username']

    client = setup_chroma_client()
    # Set up prompt template
    template = """
    Based on the retrieved context, respond with an accurate answer.

    Be concise and always provide accurate, specific, and relevant information.
    """
    
    prompt, system_instructions = load_prompt_and_system_ins(template_file_path="templates/multi_tenant_rag_prompt_template.tmpl", template=template)
    
    chat_history = st.session_state.get(
        "chat_history", [{"role": "system", "content": system_instructions.content}]
    )

    for message in chat_history[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_id:

        collection = client.get_or_create_collection(f"user-collection-{user_id}",
                                                    embedding_function=chroma_embeddings)

        uploaded_file = st.file_uploader("Upload a document", type=["pdf"])

        rag = MultiTenantRAG(user_id, llm, embeddings, collection.name, client)

        # prompt = hub.pull("rlm/rag-prompt")

        if uploaded_file:
            document = rag.load_documents(uploaded_file)
            chunks = rag.chunk_doc(document)
            vectorstore = rag.insert_embeddings(chunks=chunks,
                                                chroma_embedding_function=chroma_embeddings,
                                                embedder=embeddings,
                                                batch_size=32)
        else:
            vectorstore = Chroma(embedding_function=embeddings,
                collection_name=collection.name,
                client=client)
            
        if question := st.chat_input("Chat with your doc"):
            st.chat_message("user").markdown(question)
            chat_history.append({"role": "user", "content": question})
            with st.spinner():
                answer = rag.query_docs(model=llm,
                                    question=question,
                                    vector_store=vectorstore,
                                    prompt=prompt,
                                    chat_history=chat_history)
                print("####\n#### Answer received by querying docs: " + answer + "\n####")
                st.chat_message("assistant").markdown(answer)

if __name__ == "__main__":
    authenticator = authenticate("login")
    if st.session_state['authentication_status']:
        authenticator.logout()
        main()