from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import torch

INDEX = "zh_rag"
MODEL = "BAAI/bge-small-zh-v1.5"

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("es_user", "xxxxxx")
)

embedder = SentenceTransformer(
    MODEL,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def hybrid_search(query, top_k=5):
    q_vec = embedder.encode(
        [query],
        normalize_embeddings=True
    )[0]

    body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": {
                                "query": query,
                                "boost": 1.0
                            }
                        }
                    },
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.q, 'embedding') + 1.0",
                                "params": {"q": q_vec.tolist()}
                            }
                        }
                    }
                ]
            }
        }
    }

    res = es.search(index=INDEX, body=body)
    return [hit["_source"]["content"] for hit in res["hits"]["hits"]]


