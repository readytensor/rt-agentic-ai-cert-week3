import os
from dotenv import load_dotenv
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from langchain_groq import ChatGroq
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH
from run_wk3_l4_vector_db_ingest import get_db_collection, embed_documents

load_dotenv()

# To avoid tokenizer parallelism warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"

collection = get_db_collection(collection_name="publications")


def retrieve_relevant_documents(
    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> list[str]:
    """
    Query the ChromaDB database with a string query.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

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

    return relevant_results["documents"]


def respond_to_query(
    query: str,
    llm: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> str:
    """
    Respond to a query using the ChromaDB database.
    """

    relevant_documents = retrieve_relevant_documents(
        query, n_results=n_results, threshold=threshold
    )

    print("-" * 100)
    print("Relevant documents:")
    for doc in relevant_documents:
        print(doc)
        print("-" * 100)
    print("-" * 100)

    print("User's question:")
    print(query)
    print("-" * 100)
    input_data = (
        f"Relevant documents:\n\n{relevant_documents}\n\nUser's question:\n\n{query}"
    )

    rag_assistant_prompt = build_prompt_from_config(
        prompt_config["rag_assistant_prompt"], input_data=input_data
    )

    llm = ChatGroq(model=llm)

    response = llm.invoke(rag_assistant_prompt)
    return response.content


if __name__ == "__main__":
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

    vectordb_params = app_config["vectordb"]
    llm = app_config["llm"]

    exit_app = False
    while not exit_app:
        query = input(
            "Enter a question, 'config' to change the parameters, or 'exit' to quit: "
        )
        if query == "exit":
            exit_app = True
            exit()
        elif query == "config":
            threshold = float(input("Enter the retrieval threshold: "))
            n_results = int(input("Enter the Top K value: "))
            vectordb_params = {
                "threshold": threshold,
                "n_results": n_results,
            }
            continue

        print(
            respond_to_query(
                query=query,
                llm=llm,
                **vectordb_params,
            ),
            "\n\n",
        )
