from sentence_transformers import SentenceTransformer


def get_embeddings(docs: list, model_name: str = "mixedbread-ai/mxbai-embed-large-v1", dimensions: int = 1024):
    """
    Generate embeddings for a list of documents using a specified model.

    Args:
    - docs (list): List of documents to encode.
    - model_name (str): The name of the model to use for encoding (default: "mixedbread-ai/mxbai-embed-large-v1").
    - dimensions (int): The dimensionality to truncate the model to (default: 512).

    Returns:
    - embeddings (list): List of embeddings for the provided documents.
    """
    # Load model
    model = SentenceTransformer(model_name, truncate_dim=dimensions)

    # Encode documents
    embeddings = model.encode(docs)
    
    return embeddings