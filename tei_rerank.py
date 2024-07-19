from typing import Dict, Optional, Sequence, List
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.pydantic_v1 import Extra
import requests

DEFAULT_TOP_N = 3
DEFAULT_BATCH_SIZE = 32

class TEIRerank(BaseDocumentCompressor):
    """Document compressor using a custom rerank service."""

    url: str
    """URL of the custom rerank service."""
    top_n: int = DEFAULT_TOP_N
    """Number of documents to return."""
    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size to use for reranking."""

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def rerank(self, query: str, texts: List[str]) -> List[Dict]:
        url = f"{self.url}/rerank"
        print(f"URL: {url}")
        request_body = {"query": query, "texts": texts, "truncate": True, "batch_size": self.batch_size}
        print(f"Request Body: {request_body}")
        response = requests.post(url, json=request_body)
        print(f"Response Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Response Content: {response.content}")
            raise RuntimeError(f"Failed to rerank documents, detail: {response}")
        print(f"Response JSON: {response.json()}")
        return response.json()

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        print("compress_documents called")
        if not documents:
            print("No documents to compress")
            return []

        texts = [doc.page_content for doc in documents]
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        all_results = []

        for batch in batches:
            results = self.rerank(query=query, texts=batch)
            all_results.extend(results)

        # Sort results based on scores and select top_n
        all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:self.top_n]

        final_results = []
        for result in all_results:
            index = int(result["index"])
            metadata = documents[index].metadata.copy()
            metadata["relevance_score"] = result["score"]
            final_results.append(
                Document(page_content=documents[index].page_content, metadata=metadata)
            )

        return final_results