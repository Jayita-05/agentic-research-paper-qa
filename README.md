# 📄 Agentic Research Paper Q&A Assistant

## Overview
This project is an Agentic AI system that allows users to upload research papers (PDFs) and ask questions from them.

It uses Retrieval-Augmented Generation (RAG) with a LangGraph-based agent to ensure accurate, grounded answers.

---

## Features
- Upload multiple research papers
- Extract and chunk text
- Semantic search using vector embeddings
- Agent-based reasoning (LangGraph)
- Memory-based conversation
- Source attribution
- No hallucination responses

---

## Architecture
- **Frontend**: Streamlit
- **Agent Framework**: LangGraph
- **LLM**: Groq (LLaMA 3.1)
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers

---

## Workflow
1. Upload PDF
2. Extract text + chunking
3. Store embeddings in vector DB
4. User asks question
5. Agent retrieves relevant chunks
6. LLM generates grounded answer
7. Evaluation ensures faithfulness

---

## Installation

```bash
pip install -r requirements.txt