import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.ragpipeline.retrieval.retrieveknowledge import  rerank_documents, run_retriever


@pytest.mark.asyncio
async def test_run_retriever():
    with patch('src.ragpipeline.retrieval.embeddingmodel.get_embeddings') as mock_get_embeddings, \
         patch('src.ragpipeline.retrieval.retrieveknowledge.query_pinecone_index') as mock_query_pinecone, \
         patch('src.ragpipeline.retrieval.retrieveknowledge.rerank_documents') as mock_rerank:
        
        mock_get_embeddings.return_value = [0.1, 0.2, 0.3]
        mock_query_pinecone.return_value = {
            "matches": [{"metadata": {"text": "doc1"}}, {"metadata": {"text": "doc2"}}]
        }
        mock_rerank.return_value = ["reranked_doc1", "reranked_doc2"]

        result = await run_retriever("test query")
        assert result == "<documents>\nreranked_doc1\nreranked_doc2\n</documents>"

@pytest.mark.asyncio
async def test_run_retriever_no_matches():
    with patch('src.ragpipeline.retrieval.embeddingmodel.get_embeddings'), \
         patch('src.ragpipeline.retrieval.retrieveknowledge.query_pinecone_index') as mock_query_pinecone:
        
        mock_query_pinecone.return_value = {"matches": []}

        result = await run_retriever("test query")
        assert result == "no rag data"

def test_rerank_documents():
    with patch('src.ragpipeline.retrieval.retrieveknowledge.get_cross_encoder') as mock_get_cross_encoder:
        mock_model = MagicMock()
        mock_model.rank.return_value = [
            {"text": "reranked_doc1"},
            {"text": "reranked_doc2"},
            {"text": "reranked_doc3"}
        ]
        mock_get_cross_encoder.return_value = mock_model

        result = rerank_documents("test query", ["doc1", "doc2", "doc3", "doc4"])
        assert result == ["reranked_doc1", "reranked_doc2", "reranked_doc3"]


