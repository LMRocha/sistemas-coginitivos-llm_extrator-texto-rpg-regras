from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config_loader import load_config
from model_loader import load_model, send_message
from rag_pipeline import (
    load_pdfs_from_folder,
    chunk_documents,
    embed_chunks,
)


def build_rag(config: dict) -> None:

    documents = load_pdfs_from_folder()

    if not documents:
        print("No PDF documents found.")
        return

    chunks = chunk_documents(documents)

    if not chunks:
        print("No chunks generated.")
        return

    embed_chunks(chunks)


def retrieve_context(
    question: str,
    config: dict,
    top_k: int = 5,
) -> str:
    """
    Retrieve the most relevant chunks from ChromaDB.
    """

    embedding_model = SentenceTransformer(
        config["EMBEDDING_MODEL_ID"]
    )

    embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
    ).tolist()

    client = PersistentClient(
        path=config["CHROMADB_DIR"],
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_collection("docs")

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
    )

    return "\n\n".join(results["documents"][0])


def main():

    config = load_config()

    build_rag(config)

    model, tokenizer = load_model()

    print("RAG ready.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("User: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        context = retrieve_context(question, config)

        prompt = f"""
Use the following context to answer the user's question.

If the answer is not contained in the context, answer using your own knowledge.

Context:
{context}

Question:
{question}
"""

        send_message(
            model=model,
            tokenizer=tokenizer,
            usr_message=prompt,
        )


if __name__ == "__main__":
    main()