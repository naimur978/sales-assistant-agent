"""Compare retrieval methods on the labelled query set.
Run from the project root:  python -m scripts.eval_retrieval"""
import json
import sys

import numpy as np

from src.config import BROCHURE_DIR, ROOT
from src.ingest.brochures import load_chunks
from src.tools.product_search import BM25, Dense, Hybrid, _embed, doc_text

KS = (1, 3, 5)
evalset = json.loads((ROOT / "tests" / "retrieval_eval.json").read_text())
chunks = load_chunks(BROCHURE_DIR)
ids = [c["id"] for c in chunks]
print(f"{len(chunks)} page chunks from {len({c['file'] for c in chunks})} brochures")

bad = [g for item in evalset["in_scope"] for g in item["gold"] if g not in ids]
if bad:
    sys.exit(f"gold pages not found among chunks: {bad}")


def rank_of_first_gold(ranked_ids, gold):
    for r, cid in enumerate(ranked_ids, 1):
        if cid in gold:
            return r
    return None


def best_threshold(pos, neg):
    """Threshold maximising balanced accuracy of 'answer if top score >= t'."""
    best = (0, 0)
    for t in sorted(set(pos + neg)):
        acc = (np.mean([p >= t for p in pos]) + np.mean([n < t for n in neg])) / 2
        if acc > best[0]:
            best = (acc, t)
    return best


def auroc(pos, neg):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


all_q = [i['q'] for i in evalset['in_scope'] + evalset['out_of_scope']]
_embed(all_q, 'RETRIEVAL_QUERY')  # one batch, cached

rows = []
detail = {}
margins = {}
for prefix in (False, True):
    texts = [doc_text(c, prefix) for c in chunks]
    bm25, dense = BM25(texts), Dense(texts)
    methods = {"bm25": bm25, "dense": dense, "hybrid": Hybrid(bm25, dense)}
    for name, m in methods.items():
        label = f"{name}{'+title' if prefix else ''}"
        ranks, hard_ranks, top_in, misses = [], [], [], []
        for item in evalset["in_scope"]:
            res = m.search(item["q"], k=len(chunks))
            ranked = [ids[i] for i, _ in res]
            r = rank_of_first_gold(ranked, set(item["gold"]))
            ranks.append(r)
            if item.get("hard"):
                hard_ranks.append(r)
            top_in.append(res[0][1] if res else 0.0)
            if r is None or r > 3:
                misses.append((item["q"], r, ranked[:3]))
        top_out = [(m.search(q["q"], k=1) or [(0, 0.0)])[0][1] for q in evalset["out_of_scope"]]
        hit = lambda rs, k: np.mean([r is not None and r <= k for r in rs])
        mrr = np.mean([1 / r if r else 0 for r in ranks])
        bacc, thr = best_threshold(top_in, top_out)
        margins[label] = (min(top_in), max(top_out))
        rows.append(
            [label] + [hit(ranks, k) for k in KS] + [mrr, hit(hard_ranks, 3), auroc(top_in, top_out), bacc, thr]
        )
        detail[label] = misses

hdr = ["method"] + [f"Hit@{k}" for k in KS] + ["MRR", "Hit@3 hard", "AUROC abst.", "bal.acc", "thr"]
print(("{:<14}" + "{:>11}" * (len(hdr) - 1)).format(*hdr))
for r in rows:
    print(("{:<14}" + "{:>11.3f}" * (len(r) - 1)).format(*r))

print("\nAbstention margin (lowest in-scope top score vs highest out-of-scope top score; >0 gap = separable):")
for k, (lo, hi) in margins.items():
    print(f"  {k:<14} in-scope min {lo:.3f}   out-of-scope max {hi:.3f}   gap {lo - hi:+.3f}")

print("\nQueries whose first gold page is outside the top 3:")
for label, misses in detail.items():
    print(f"\n[{label}]")
    for q, r, top in misses:
        print(f"  rank={r}  {q}\n      top3={top}")
