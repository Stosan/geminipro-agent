from src.config.appconfig import PINECONE_API_KEY, PINECONE_INDEX
from src.ragpipeline.retrieval.embeddingmodel import get_embeddings
from sentence_transformers import CrossEncoder
from pinecone import Pinecone


def run_retriever(query: str) -> str:
    """
    Retrieve and rerank documents based on the given query.

    This function first retrieves documents from the Pinecone index using the query embeddings.
    It then reranks the retrieved documents based on their relevance to the query.

    Args:
        query (str): The query string to search for relevant documents.

    Returns:
        str: A string containing the reranked documents separated by newlines.
             If no documents are found, returns "no rag data".
    """
    data_to_rerank = []

    # Retrieve results from Pinecone index using query embeddings
    retrieved_results = query_pinecone_index(query_embeddings=get_embeddings([query]))

    # Check if there are any matches
    if retrieved_results["matches"] == []:
        return "no rag data"

    # Extract text from metadata of each match
    for m in retrieved_results["matches"]:
        data_to_rerank.append(m["metadata"]["text"])

    # Rerank the documents based on their relevance to the query
    reranked_result = rerank_documents(query, data_to_rerank)

    return "".join(
        "<documents>", "\n", "\n".join(reranked_result), "\n", "</documents>"
    )


def rerank_documents(
    query: str, documents: list, model_name: str = "mixedbread-ai/mxbai-rerank-base-v1"
) -> list:
    """
    Rerank documents based on their relevance to the query using a specified model.

    Args:
        query (str): The query string to match against the documents.
        documents (list): A list of documents to be reranked.
        model_name (str): The name of the model to use for reranking (default: "mixedbread-ai/mxbai-rerank-base-v1").

    Returns:
        list: A list of reranked documents based on their relevance to the query.
    """

    # Initialize the CrossEncoder model with the specified model name
    model = CrossEncoder(model_name)

    # Rank the documents based on their relevance to the query
    results = model.rank(query, documents, return_documents=True, top_k=3)
    print([r["text"] for r in results])
    return [r["text"] for r in results]


def query_pinecone_index(
    query_embeddings: list, top_k: int = 5, include_metadata: bool = True
) -> dict[str, any]:
    """
    Query a Pinecone index.

    Args:
    - index (Any): The Pinecone index object to query.
    - vectors (List[List[float]]): List of query vectors.
    - top_k (int): Number of nearest neighbors to retrieve (default: 2).
    - include_metadata (bool): Whether to include metadata in the query response (default: True).

    Returns:
    - query_response (Dict[str, Any]): Query response containing nearest neighbors.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc_index = pc.Index(PINECONE_INDEX)

    query_response = pc_index.query(
        vector=query_embeddings.tolist(), top_k=top_k, include_metadata=include_metadata
    )
    return query_response
