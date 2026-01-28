import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse
from PyPDF2 import PdfReader

from app.rag import (
    chunk_text,
    embed_chunks,
    retrieve_chunks,
    generate_answer
)

app = FastAPI()

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = ""

    if file.filename.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""

    elif file.filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF and TXT files are allowed"}
        )

    chunks = chunk_text(text)
    num_embeddings = embed_chunks(chunks)

    return {
        "filename": file.filename,
        "message": "File uploaded, chunked and embedded successfully",
        "text_length": len(text),
        "num_chunks": len(chunks),
        "num_embeddings": num_embeddings
    }


@app.post("/ask")
def ask_question(question: str):
    retrieved = retrieve_chunks(question)
    answer = generate_answer(question, retrieved)

    return {
        "question": question,
        "answer": answer,
        "sources_used": len(retrieved)
    }
