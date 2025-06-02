import os
import logging
from dotenv import load_dotenv
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from langchain_groq import ChatGroq
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from run_wk3_l4_vector_db_ingest import get_db_collection, embed_documents

logger = logging.getLogger()


def setup_logging():

    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant.log"))
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


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
    logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }
    # Embed the query using the same model used for documents
    logging.info("Embedding query...")
    query_embedding = embed_documents([query])[0]  # Get the first (and only) embedding

    logging.info("Querying collection...")
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    logging.info("Filtering results...")
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
    prompt_config: dict,
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

    logging.info("-" * 100)
    logging.info("Relevant documents: \n")
    for doc in relevant_documents:
        logging.info(doc)
        logging.info("-" * 100)
    logging.info("")

    logging.info("User's question:")
    logging.info(query)
    logging.info("")
    logging.info("-" * 100)
    logging.info("")
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
    setup_logging()
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

        response = respond_to_query(
            prompt_config=prompt_config,
            query=query,
            llm=llm,
            **vectordb_params,
        )
        logging.info(response + "\n\n")
