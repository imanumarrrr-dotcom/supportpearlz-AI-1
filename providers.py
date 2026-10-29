"""Provider helpers shared by the dashboard and retrieval pipeline."""
import math
from langchain_openai import OpenAIEmbeddings

def make_embeddings(api_key: str, model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key=api_key, model=model, max_retries=2)

def cosine(a, b) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) ** 2 for x in a)); nb = math.sqrt(sum(float(y) ** 2 for y in b))
    return dot / (na * nb) if na and nb else 0.0
