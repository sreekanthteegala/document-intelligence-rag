# Document Intelligence System (RAG-based Q&A)

## Overview
This project is an end-to-end Document Intelligence system that enables users to upload PDF documents and ask natural language questions. The system uses Retrieval-Augmented Generation (RAG) to produce grounded, context-aware answers while minimizing hallucinations.

## Features
- PDF ingestion and text extraction
- Text cleaning, chunking, and embedding generation
- Semantic search using FAISS vector database
- Retrieval-Augmented Generation (RAG) with Hugging Face LLMs
- Hallucination control with “I don’t know” responses
- Intent-aware routing for document summaries vs fact-based QA
- Simple HTML UI served via FastAPI

## Tech Stack
- Python
- FastAPI
- LangChain
- Hugging Face Transformers & Sentence-Transformers
- FAISS
- HTML / JavaScript

## Architecture
1. PDF documents are uploaded via a FastAPI endpoint.
2. Text is extracted, cleaned, and split into chunks.
3. Chunks are embedded using sentence-transformers.
4. Embeddings are stored in a FAISS vector index.
5. User queries trigger semantic retrieval.
6. Retrieved context is passed to an LLM for grounded answer generation.

## Use Cases
- Document Q&A
- Document summarization
- Knowledge extraction from reports, emails, and research papers

## Limitations
- Currently supports text-based PDFs only
- Single-vector-store overwrite (no multi-document indexing)
- CPU-based inference

## Future Improvements
- Multi-document indexing with metadata
- OCR support for scanned PDFs
- Page-number-based citations
- Improved summarization with map-reduce strategies

## How to Run
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
Open: http://127.0.0.1:8000



In VS Code terminal
