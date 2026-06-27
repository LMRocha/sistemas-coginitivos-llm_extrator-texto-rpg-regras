# Local RAG Chatbot with ChromaDB

A lightweight Retrieval-Augmented Generation (RAG) application that combines a local Large Language Model (LLM) with a PDF knowledge base.

The project indexes PDF documents into a ChromaDB vector database using Sentence Transformers embeddings, retrieves the most relevant document chunks for each user query, and augments the prompt before generating an answer with a local Hugging Face model.

---

# Features

- Local LLM inference using Hugging Face Transformers
- PDF document ingestion
- Automatic document chunking
- Embedding generation using Sentence Transformers
- Persistent vector database with ChromaDB
- Retrieval-Augmented Generation (RAG)
- Interactive command-line chatbot
- Modular project architecture

---

# Project Structure

```text
project/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
│
├── chromadb/
│
├── src/
│   ├── main.py
│   ├── config_loader.py
│   ├── model_loader.py
│   └── rag_pipeline.py
│
├── requirements.txt
└── README.md
```

---

# Architecture

```text
                  User Question
                        │
                        ▼
                 main.py (Pipeline)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
 config_loader    rag_pipeline     model_loader
        │               │               │
        │               ▼               ▼
        │       ChromaDB Search     Local LLM
        │               │               │
        └───────────────┴───────────────┘
                        │
                        ▼
                 Generated Response
```

---

# Pipeline

The application executes the following steps:

1. Load the project configuration.
2. Load the embedding model.
3. Load the local language model.
4. Read all PDF documents from the configured data directory.
5. Split documents into overlapping text chunks.
6. Generate embeddings for each chunk.
7. Store embeddings in a persistent ChromaDB collection.
8. Wait for a user question.
9. Embed the user query.
10. Retrieve the most relevant document chunks.
11. Build an augmented prompt using the retrieved context.
12. Generate the final response with the LLM.

---

# Configuration

Create a `configs/config.yaml` file.

Example:

```yaml
MODEL_ID: microsoft/Phi-3-mini-4k-instruct

EMBEDDING_MODEL_ID: BAAI/bge-base-en-v1.5

DATA_DIR: data

CHROMADB_DIR: chromadb
```

---

# Installation

## Clone the repository

```bash
git clone <repository_url>

cd <repository>
```

## Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Required Packages

```text
transformers
torch
sentence-transformers
chromadb
pypdf
langchain-text-splitters
PyYAML
tqdm
```

---

# Running the Application

Place one or more PDF documents inside the configured `DATA_DIR`.

Then run:

```bash
python main.py
```

Example:

```text
User: Explain the Character Abilities

Assistant:
Character Abilities define a player's primary attributes...
```

---

# Main Components

## `config_loader.py`

Responsible for loading the project configuration from `config.yaml`.

**Responsibilities**

- Read YAML configuration
- Validate configuration file existence
- Return configuration dictionary

---

## `model_loader.py`

Responsible for loading the Hugging Face language model.

**Responsibilities**

- Load tokenizer
- Load causal language model
- Generate responses from prompts

---

## `rag_pipeline.py`

Responsible for document indexing.

**Responsibilities**

- Load PDF files
- Extract text
- Split text into chunks
- Generate embeddings
- Store vectors in ChromaDB

---

## `main.py`

Coordinates the complete application workflow.

**Responsibilities**

- Initialize the RAG pipeline
- Build or reuse the vector database
- Accept user input
- Retrieve relevant context
- Build the augmented prompt
- Generate responses with the LLM

---

# Retrieval-Augmented Generation (RAG)

The application follows a standard RAG workflow:

```text
PDF
 │
 ▼
Extract Text
 │
 ▼
Chunk Documents
 │
 ▼
Generate Embeddings
 │
 ▼
Store in ChromaDB
 │
 ▼
──────────────────────────────
 │
 ▼
User Question
 │
 ▼
Generate Query Embedding
 │
 ▼
Similarity Search
 │
 ▼
Retrieve Relevant Chunks
 │
 ▼
Build Prompt
 │
 ▼
LLM
 │
 ▼
Answer
```

---

# Performance Recommendations

For better retrieval quality:

- Use a `chunk_size` between **1500–2000** characters.
- Use a `chunk_overlap` between **200–400** characters.
- Retrieve at least **5–10** chunks (`top_k`).
- Use embedding models such as:
  - `BAAI/bge-base-en-v1.5`
  - `BAAI/bge-large-en-v1.5`
  - `intfloat/e5-large-v2`
- Increase `max_new_tokens` to at least **512** for detailed responses.
- Consider retrieving neighboring chunks to improve long-document context.

---

# Future Improvements

Possible enhancements include:

- Incremental indexing of new documents
- Duplicate document detection
- Metadata filtering
- Hybrid keyword + semantic search
- Parent Document Retrieval
- Streaming model responses
- Conversation history
- Source citations in generated answers
- GPU inference optimization
- REST API with FastAPI
- Web interface with Gradio or Streamlit

---

# License

This project is intended as a reference implementation of a local Retrieval-Augmented Generation (RAG) chatbot built with:

- Hugging Face Transformers
- Sentence Transformers
- ChromaDB
- PyPDF
- LangChain Text Splitters

It is designed as an educational and extensible foundation for building local AI assistants capable of answering questions from custom PDF knowledge bases.