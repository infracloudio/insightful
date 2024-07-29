import streamlit as st
import os
from langchain_community.vectorstores.chroma import Chroma
from app import setup_chroma_client, setup_chroma_embedding_function
from app import setup_huggingface_embeddings, setup_huggingface_endpoint
from app import RAG
from langchain import hub
import tempfile
from langchain_community.document_loaders import PyPDFLoader

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
    llm = setup_huggingface_endpoint()

    embeddings = setup_huggingface_embeddings()

    chroma_embeddings = setup_chroma_embedding_function()

    user_id = st.text_input("Enter user ID")

    client = setup_chroma_client()

    if user_id:

        collection = client.get_or_create_collection(f"user-collection-{user_id}",
                                                    embedding_function=chroma_embeddings)

        uploaded_file = st.file_uploader("Upload a document", type=["pdf"])

        rag = MultiTenantRAG(user_id, llm, embeddings, collection.name, client)

        prompt = hub.pull("rlm/rag-prompt")

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
            with st.spinner():
                answer = rag.query_docs(model=llm,
                                    question=question,
                                    vector_store=vectorstore,
                                    prompt=prompt)
                st.chat_message("assistant").markdown(answer)

if __name__ == "__main__":
    main()