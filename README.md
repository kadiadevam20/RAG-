# Enterprise Knowledge Assistant

A retrieval-augmented generation (RAG) application that allows users to
upload documents and ask questions answered strictly from the content
of those documents.

## Architecture

The application consists of the following components:

- Document Processing: Extracts text from PDF, DOCX, and TXT files and
  splits the content into overlapping chunks for retrieval.
- Embedding Model: Generates vector representations of text using the
  `sentence-transformers/all-MiniLM-L6-v2` model, a free model hosted
  on Hugging Face that runs locally.
- Vector Store: Stores and searches document embeddings using Qdrant
  in local persistent mode. No external Qdrant server is required.
- Language Model: Generates answers using `meta-llama/Llama-3.1-8B-Instruct`
  through the Hugging Face Inference Providers API (free tier). No
  model weights are downloaded or run locally, keeping memory usage
  low and making the app suitable for deployment on resource-limited
  hosting such as Streamlit Community Cloud. Responses are grounded
  in retrieved document context only.
- User Interface: A Streamlit application providing document upload,
  knowledge base management, and a chat-based question answering
  interface.

## Requirements

- Python 3.10 or later
- A free Hugging Face account and API token, available at
  https://huggingface.co/settings/tokens. Set it as the HF_TOKEN
  environment variable (locally via a .env file, or via your
  hosting platform's secrets manager). The app also functions
  without a token, subject to lower anonymous rate limits.

## Installation

```
pip install -r requirements.txt
```

## Running the Application

```
streamlit run app.py
```

On first use, the embedding model and language model are downloaded
automatically from Hugging Face and cached locally. This requires an
internet connection for the initial download only; all subsequent
processing and generation run offline.

## Usage

1. Upload one or more documents in PDF, DOCX, or TXT format.
2. Select Process Documents to build the knowledge base.
3. Ask questions in the chat interface. Answers are generated from the
   content of the uploaded documents only.
4. Select Clear Knowledge Base to remove all indexed content and start
   a new session.

## Project Structure

```
rag-app/
  app.py                     Streamlit application entry point
  requirements.txt           Python dependencies
  modules/
    document_processor.py    Text extraction and chunking
    embeddings.py            Embedding model wrapper
    vector_store.py          Qdrant vector store wrapper
    llm_handler.py            Language model handler
```

## Notes

- Document embeddings are stored locally in the `qdrant_storage`
  directory. Deleting this directory removes all indexed content.
- The application answers questions using only the content available
  in the knowledge base. If the relevant information is not present in
  the uploaded documents, the application will state this rather than
  producing an unsupported answer.
