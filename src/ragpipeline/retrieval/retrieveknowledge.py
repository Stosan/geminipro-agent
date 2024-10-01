import asyncio
from functools import lru_cache
from typing import List, Dict, Any
from src.config.appconfig import PINECONE_API_KEY, PINECONE_INDEX
from src.ragpipeline.retrieval.embeddingmodel import get_embeddings
from sentence_transformers import CrossEncoder
from pinecone import Pinecone

# Initialize Pinecone client once
pc = Pinecone(api_key=PINECONE_API_KEY)
pc_index = pc.Index(PINECONE_INDEX)

# Cache the CrossEncoder model
@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = "mixedbread-ai/mxbai-rerank-base-v1") -> CrossEncoder:
    return CrossEncoder(model_name)

async def run_retriever(query: str) -> str:
    """
    Asynchronously retrieve and rerank documents based on the given query.
    """
    query_embeddings = await asyncio.to_thread(get_embeddings, [query])
    retrieved_results = await asyncio.to_thread(query_pinecone_index, query_embeddings)

    if not retrieved_results["matches"]:
        return "no rag data"

    data_to_rerank = [m["metadata"]["text"] for m in retrieved_results["matches"]]
    reranked_result = await asyncio.to_thread(rerank_documents, query, data_to_rerank)
    return "<documents>\n" + "\n".join(reranked_result) + "\n</documents>"

def rerank_documents(
    query: str, documents: List[str], model_name: str = "mixedbread-ai/mxbai-rerank-base-v1"
) -> List[str]:
    """
    Rerank documents based on their relevance to the query using a cached model.
    """
    model = get_cross_encoder(model_name)
    results = model.rank(query, documents, return_documents=True, top_k=3)
    return [r["text"] for r in results]

def query_pinecone_index(
    query_embeddings: List[float], top_k: int = 5, include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Query the Pinecone index using pre-initialized client.
    """
    return pc_index.query(
        vector=query_embeddings.tolist(),
        top_k=top_k,
        include_metadata=include_metadata
    )