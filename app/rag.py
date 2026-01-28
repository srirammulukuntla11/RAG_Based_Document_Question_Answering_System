import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- EMBEDDING SETUP ----------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

EMBEDDING_DIM = 384
index = faiss.IndexFlatL2(EMBEDDING_DIM)
stored_chunks = []


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index.add(embeddings)
    stored_chunks.extend(chunks)

    return len(chunks)


def retrieve_chunks(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(stored_chunks):
            results.append(stored_chunks[idx])

    return results


# ---------------- LOCAL LLM (NO API) ----------------

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


def generate_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant information found in the document."

    context = " ".join(retrieved_chunks[:3])

    prompt = (
        "Answer the question using only the document.\n\n"
        f"Document:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
