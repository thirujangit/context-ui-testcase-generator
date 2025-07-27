import os
import json
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import numpy as np
import requests
from PyPDF2 import PdfReader

try:
    from docx import Document
except ImportError:
    raise ImportError("Install 'python-docx' using: pip install python-docx")

try:
    import faiss
except ImportError:
    raise ImportError("Install FAISS using: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Install sentence-transformers using: pip install sentence-transformers")

# Load environment variables
load_dotenv()

# Initialize embedding model
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Set up directories
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --------- Helper Functions --------- #

def extract_text(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            reader = PdfReader(tmp_path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        elif ext == ".docx":
            doc = Document(tmp_path)
            return "\n".join(p.text for p in doc.paragraphs)
    finally:
        os.remove(tmp_path)
    return ""


def chunk_text(text, size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def save_index(project: str, texts: List[str]):
    vecs = EMBEDDING_MODEL.encode(texts)
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(np.array(vecs))
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{project}.index"))
    with open(os.path.join(TEXT_DIR, f"{project}.json"), "w") as f:
        json.dump(texts, f)


def load_index(project: str):
    index_path = os.path.join(INDEX_DIR, f"{project}.index")
    text_path = os.path.join(TEXT_DIR, f"{project}.json")
    if not os.path.exists(index_path) or not os.path.exists(text_path):
        return None, None
    index = faiss.read_index(index_path)
    with open(text_path) as f:
        texts = json.load(f)
    return index, texts


def search_chunks(index, texts, query, k=5):
    vec = EMBEDDING_MODEL.encode([query])
    _, I = index.search(np.array(vec), k)
    return [texts[i] for i in I[0]]


def build_prompt(story, chunks):
    context = "\n".join(chunks)
    return f"""
You are a QA expert.
Generate test cases based on the user story and supporting context.

User Story:
{story}

Context:
{context}

Generate:
- 5 Functional
- 3 Negative
- 2 Edge cases
"""


def call_together_ai(prompt):
    api_key = os.getenv("TOGETHER_API_KEY")
    resp = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "messages": [
                {"role": "system", "content": "You are a QA expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def get_all_projects():
    return [f.split(".")[0] for f in os.listdir(TEXT_DIR) if f.endswith(".json")]


# --------- Routes --------- #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    projects = get_all_projects()
    return templates.TemplateResponse("index.html", {"request": request, "project_names": projects})


@app.post("/upload-docs")
async def upload_docs_ui(request: Request, project: str = Form(...), files: List[UploadFile] = File(...)):
    all_chunks = []
    for file in files:
        text = extract_text(file)
        if text:
            all_chunks.extend(chunk_text(text))
    save_index(project, all_chunks)
    projects = get_all_projects()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": f"Uploaded and indexed {len(all_chunks)} chunks for project '{project}'.",
        "project_names": projects
    })


@app.post("/generate", response_class=HTMLResponse)
async def generate_ui(request: Request, user_story: str = Form(...), project: str = Form(...)):
    index, texts = load_index(project)
    projects = get_all_projects()
    if not index:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "No document found. Please upload first.",
            "project_names": projects
        })

    top_chunks = search_chunks(index, texts, user_story)
    prompt = build_prompt(user_story, top_chunks)
    result = call_together_ai(prompt)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "story": user_story,
        "testcases": result,
        "project_names": projects
    })
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Use 10000 as fallback for local
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
