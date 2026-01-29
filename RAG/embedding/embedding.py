import torch
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


INDEX = "zh_rag"
MODEL = "BAAI/bge-small-zh-v1.5"

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("root", "123456")
)

embedder = SentenceTransformer(
    MODEL,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed(texts):
    return embedder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=True
    )

with open("resume_kb.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


chunks = chunk_text(raw_text)
vectors = embed(chunks)

actions = []
for i, (text, vec) in enumerate(zip(chunks, vectors)):
    actions.append({
        "_index": INDEX,
        "_id": i,
        "_source": {
            "content": text,
            "embedding": vec.tolist()
        }
    })

bulk(es, actions)
print("Documents indexed")
