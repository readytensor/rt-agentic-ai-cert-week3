import chromadb
from run_wk3_l4_vector_db_ingest import embed_documents


def query_db(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> dict:
    """
    Query the ChromaDB database with a string query.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to query
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)

    Returns:
        dict: Query results containing ids, documents, distances, and metadata
    """
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }
    # Embed the query using the same model used for documents
    query_embedding = embed_documents([query])[0]  # Get the first (and only) embedding

    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    return relevant_results
