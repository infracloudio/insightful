import unittest
from unittest.mock import MagicMock
from InSightful import RAG, Document

class RAGTests(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.embeddings = MagicMock()
        self.collection_name = "test_collection"
        self.db_client = MagicMock()
        self.rag = RAG(self.llm, self.embeddings, self.collection_name, self.db_client)

    def test_load_documents(self):
        doc = "test_dataset"
        dataset = [
            {"text": "Document 1", "user": "User 1", "workplace": "Workplace 1"},
            {"text": "Document 2", "user": "User 2", "workplace": "Workplace 2"},
        ]
        self.db_client.get_or_create_collection.return_value = MagicMock()
        self.db_client.get_or_create_collection.return_value.add = MagicMock()

        documents = self.rag.load_documents(doc)

        self.assertEqual(len(documents), len(dataset))
        for i, data in enumerate(dataset):
            self.assertIsInstance(documents[i], Document)
            self.assertEqual(documents[i].page_content, data["text"])
            self.assertEqual(documents[i].metadata["user"], data["user"])
            self.assertEqual(documents[i].metadata["workplace"], data["workplace"])

    def test_chunk_doc(self):
        pages = ["Page 1", "Page 2", "Page 3"]
        chunk_size = 2
        chunk_overlap = 1

        chunks = self.rag.chunk_doc(pages, chunk_size, chunk_overlap)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].page_content, "Page 1")
        self.assertEqual(chunks[1].page_content, "Page 2")
        self.assertEqual(chunks[2].page_content, "Page 3")

    def test_insert_embeddings(self):
        chunks = [
            Document(page_content="Chunk 1", metadata={}),
            Document(page_content="Chunk 2", metadata={}),
            Document(page_content="Chunk 3", metadata={}),
        ]
        self.db_client.get_or_create_collection.return_value = MagicMock()
        self.db_client.get_or_create_collection.return_value.add = MagicMock()

        db = self.rag.insert_embeddings(chunks, self.embeddings, self.embeddings)

        self.assertIsInstance(db, MagicMock)
        self.db_client.get_or_create_collection.assert_called_once_with(
            self.collection_name, embedding_function=self.embeddings
        )
        self.assertEqual(
            self.db_client.get_or_create_collection.return_value.add.call_count, 3
        )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    class RAGTests(unittest.TestCase):
        ...
    
    def test_query_docs(self):
        model = MagicMock()
        question = "Test question"
        vector_store = MagicMock()
        prompt = MagicMock()
        answer = "Test answer"
        model.return_value = answer
    
        result = self.rag.query_docs(model, question, vector_store, prompt)
    
        self.assertEqual(result, answer)
        model.assert_called_once_with({"context": vector_store | format_docs, "question": MagicMock()} | prompt | self.llm | MagicMock())

if __name__ == "__main__":
    unittest.main()