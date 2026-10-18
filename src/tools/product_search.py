"""Retrievers over brochure page chunks. Each .search() returns [(chunk_idx, score)]
sorted best-first, with score normalised to roughly 0-1 so one abstention
threshold can be tuned per method."""
import hashlib
import json
import math
import re
from collections import Counter

import numpy as np

from src import llm
from src.config import CACHE_DIR, EMBED_MODEL

STOP = set(
    "a an and are as at be by can do does for from has have how i in is it its of on or "
    "our that the their this to we what when which who with you your".split()
)


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks if t not in STOP]


def doc_text(chunk: dict, prefix: bool) -> str:
    """Optionally prepend the product name so 'Larox' matches pages that never say it."""
    return f"{chunk['file'].replace('-', ' ')}. {chunk['text']}" if prefix else chunk["text"]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.lens.mean()
        n = len(texts)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 5):
        q = [t for t in tokenize(query) if t in self.idf]
        if not q:
            return []
        ceiling = sum(self.idf[t] for t in q) * (self.k1 + 1)  # best possible score
        scores = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            scores[i] = sum(
                self.idf[t] * d[t] * (self.k1 + 1) / (d[t] + norm) for t in q if t in d
            )
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i] / ceiling)) for i in order]

    def all_scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, s in self.search(query, k=len(self.docs)):
            out[i] = s
        return out


def _embed(texts: list[str], task: str) -> np.ndarray:
    """Gemini embeddings with an on-disk cache so evals re-run for free."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "embeddings.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    keys = [hashlib.sha1(f"{EMBED_MODEL}|{task}|{t}".encode()).hexdigest() for t in texts]
    missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in cache]
    if missing:
        from google.genai import types

        client = llm.client()
        for i in range(0, len(missing), 40):
            batch = missing[i : i + 40]
            res = llm.call(lambda batch=batch: client.models.embed_content(
                model=EMBED_MODEL, contents=[t for _, t in batch],
                config=types.EmbedContentConfig(task_type=task)), "embed")
            for (k, _), e in zip(batch, res.embeddings, strict=True):
                cache[k] = e.values
            cache_file.write_text(json.dumps(cache))  # keep progress between batches
    m = np.array([cache[k] for k in keys], dtype=float)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


class Dense:
    def __init__(self, texts: list[str]):
        self.mat = _embed(texts, "RETRIEVAL_DOCUMENT")

    def all_scores(self, query: str) -> np.ndarray:
        return self.mat @ _embed([query], "RETRIEVAL_QUERY")[0]  # cosine similarity

    def search(self, query: str, k: int = 5):
        s = self.all_scores(query)
        return [(int(i), float(s[i])) for i in np.argsort(-s)[:k]]


class Hybrid:
    """Weighted sum of BM25 (0-1) and dense cosine."""

    def __init__(self, bm25: BM25, dense: Dense, alpha: float = 0.5):
        self.bm25, self.dense, self.alpha = bm25, dense, alpha

    def all_scores(self, query: str) -> np.ndarray:
        return self.alpha * self.bm25.all_scores(query) + (1 - self.alpha) * self.dense.all_scores(query)

    def search(self, query: str, k: int = 5):
        s = self.all_scores(query)
        return [(int(i), float(s[i])) for i in np.argsort(-s)[:k]]


class ProductSearch:
    """BM25 (with product name prepended) ranks pages; dense cosine is the on-topic gate."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [doc_text(c, prefix=True) for c in chunks]
        self.bm25, self.dense = BM25(texts), Dense(texts)
        self.pages = {c["id"]: c["text"] for c in chunks}

    def search(self, query: str | list[str], k: int = 4) -> dict:
        """Accepts one query or several (e.g. the model's rewrite AND the user's own words); rankings are fused
        with reciprocal rank fusion so a good hit under any wording survives."""
        queries = [query] if isinstance(query, str) else [q for q in query if q.strip()]
        fused: dict[int, float] = {}
        for q in queries:
            for rank, (i, _) in enumerate(self.bm25.search(q, len(self.chunks)), 1):
                fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        hits = [{**self.chunks[i], "bm25": self.bm25.all_scores(queries[0])[i]} for i in top]
        gate = max(float(self.dense.all_scores(q).max()) for q in queries)
        return {"hits": hits, "gate_score": gate}
