import os
import shutil
from pypdf import PdfReader
from pathlib import Path
import yaml
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from config_loader import load_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

configs = load_config()

def load_pdfs_from_folder() -> list[dict]:
    """
    Load and extract text from all PDF files in a folder.
    Returns a list of dicts: [{"filename": "doc.pdf", "text": "..."}, ...]
    """
    folder_path = configs.get("DATA_DIR")
    texts = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return texts

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            path = os.path.join(folder_path, file)
            try:
                reader = PdfReader(path)
                page_texts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        page_texts.append(text)
                if page_texts:
                    full_text = "\n".join(page_texts)
                    texts.append({"filename": file, "text": full_text})
                    print(f"✅ Loaded: {file}")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
    return texts


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, str]]:
    """
    Split each document's text into overlapping chunks.

    Args:
        documents: List of dicts from load_pdfs_from_folder(),
                   each with "filename" and "text".
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of dicts, each with:
            "filename"   : str
            "chunk"      : str (the text chunk)
            "chunk_index": int (position of this chunk in its document)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    for doc in documents:
        filename = doc["filename"]
        text = doc["text"]
        if not text or not text.strip():
            continue
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": filename,
                "chunk": chunk,
                "chunk_index": idx,
            })
    return all_chunks

def embed_chunks(
    chunk_list: List[Dict[str, str]]
):
    if not chunk_list:
        print("No chunks to embed.")
        return

    texts = [item["chunk"] for item in chunk_list]
    embedding_model = SentenceTransformer(configs.get("EMBEDDING_MODEL_ID"))
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    ids = []
    metadatas = []
    for item in chunk_list:
        base = Path(item["filename"]).stem
        ids.append(f"{base}_{item['chunk_index']}")
        metadatas.append({
            "source": item["filename"],
            "chunk_index": item["chunk_index"],
        })

    client = chromadb.PersistentClient(
        path=configs.get("CHROMADB_DIR"), 
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(
        name="docs",             
        embedding_function=None
    )

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Added {len(ids)} chunks.")


if __name__ == "__main__":
    raw_texts = load_pdfs_from_folder()
    print(f"PDFs loaded: {raw_texts}")

    chunks = chunk_documents(raw_texts)
    print(f"Chunks: {chunks}")

    embeddings = embed_chunks(chunks)