import os
import fitz 
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from rag import embed

QDRANT_URL = os.getenv('QDRANT_URL','http://localhost:6333')
qdrant = QdrantClient(url=QDRANT_URL.replace('http://',''), prefer_grpc=False) if 'localhost' in QDRANT_URL or '127.0.0.1' in QDRANT_URL else QdrantClient(url=QDRANT_URL)

COLLECTION_NAME = 'PLMUN_Chatbot_Knowledge'

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = []
    for page in doc:
        t = page.get_text()
        if t and t.strip():
            text.append(t.strip())
    return "\n\n".join(text)

def chunk_text(text, chunk_size=400, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

def ensure_collection():
    try:
        qdrant.get_collection(COLLECTION_NAME)
    except Exception:
        qdrant.recreate_collection(collection_name=COLLECTION_NAME, vectors_config=rest_models.VectorParams(size=384, distance=rest_models.Distance.COSINE))

def ingest_file(path, doc_id=None):
    text = extract_text_from_pdf(path) if path.lower().endswith('.pdf') else open(path,'r',encoding='utf-8').read()
    chunks = chunk_text(text)
    ensure_collection()
    points = []
    for i, chunk in enumerate(chunks):
        vec = embed(chunk)
        payload = {'text': chunk, 'source': os.path.basename(path)}
        points.append(rest_models.PointStruct(id=int(f"{doc_id or 0}{i}"), vector=vec, payload=payload))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} chunks from {path}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True, help='Path to PDF or text file')
    args = p.parse_args()
    ingest_file(args.file)

