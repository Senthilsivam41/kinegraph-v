from typing import List, Any
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

class LangChainEmbeddingWrapper(BaseEmbedding):
    _lc_embeddings: Any = PrivateAttr()

    def __init__(self, langchain_embeddings: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self._lc_embeddings = langchain_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._lc_embeddings.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._lc_embeddings.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._lc_embeddings.embed_documents(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._lc_embeddings.aembed_query(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        res = await self._lc_embeddings.aembed_documents([text])
        return res[0]
