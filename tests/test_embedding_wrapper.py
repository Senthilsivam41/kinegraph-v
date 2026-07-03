from backend.graph_ingestion.embedding_wrapper import LangChainEmbeddingWrapper
from unittest.mock import MagicMock

def test_embedding_wrapper():
    # Create mock langchain embeddings
    mock_lc_emb = MagicMock()
    mock_lc_emb.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_lc_emb.embed_documents.return_value = [[0.4, 0.5, 0.6]]
    
    wrapper = LangChainEmbeddingWrapper(mock_lc_emb)
    
    # Verify query embedding delegation
    query_res = wrapper.get_query_embedding("test query")
    assert query_res == [0.1, 0.2, 0.3]
    mock_lc_emb.embed_query.assert_called_once_with("test query")
    
    # Verify text embedding delegation
    text_res = wrapper.get_text_embedding("test text")
    assert text_res == [0.4, 0.5, 0.6]
    mock_lc_emb.embed_documents.assert_called_once_with(["test text"])
