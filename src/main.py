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

    question = "" 

    while not question == "exit":

        question = input("User: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        context = retrieve_context(question, config)

        prompt = f"""
        # ROLE
            You are an role playing game expert.

            Always prioritize the retrieved context when answering.

            If the answer cannot be found in the retrieved context,
            use your own knowledge but clearly indicate that the information
            was not found in the documents.

            Be concise, accurate and well-structured.

        # CONTEXT

        The following text was retrieved from a document database.

        {context}

        # TASK

        Answer the user's request, using the retrieved context as the primary source.
        
        The user's request can be a question or a request like 'Explaing this...', 'Resume this..' for example.

        If the context contains only part of the answer:
        - Complete the answer using your own knowledge.
        - Clearly indicate which information comes from your own knowledge.

        If the context does not contain the answer:
        - State that the information was not found in the retrieved documents.
        - Then answer using your general knowledge.

        # USER REQUEST

        {question}

        # OUTPUT

        Produce:
        1. A complete answer.
        2. A well-structured explanation.
        3. Bullet points or numbered lists when appropriate.
        4. Markdown formatting.

        # CHECKLIST

        Before answering, verify:
        - The retrieved context was analyzed.
        - Information from multiple retrieved passages was combined when relevant.
        - No facts were invented as if they came from the documents.
        - The answer is complete and detailed.
        - The response directly answers the user's question.
        """

        send_message(
            model=model,
            tokenizer=tokenizer,
            usr_message=prompt
        )


if __name__ == "__main__":
    main()