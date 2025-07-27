# 🚀 Context-Aware Test Case Generator

This FastAPI app lets users upload documents (PDF/DOCX), enter a user story, and generate relevant test cases using Together AI. It uses semantic search via FAISS to provide context-aware results.

## Features
- Upload project-specific documents
- Enter a user story
- Generate Functional, Negative, and Edge test cases
- Semantic vector search using FAISS
- Together AI integration

## Run Locally
```bash
uvicorn main:app --reload
