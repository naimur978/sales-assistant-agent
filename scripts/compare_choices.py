"""Offline comparisons behind design choices (no API calls).   python -m scripts.compare_choices"""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from src.config import BROCHURE_DIR, ROOT
from src.ingest.brochures import load_chunks
from src.ingest.services import REPORT_DIR, load_visits
from src.tools import product_search as ps
from src.tools.flotation import FE, SI
from src.tools.flotation import load as load_flotation
from src.tools.service_cost import comparable_total

ev = json.loads((ROOT / "tests" / "retrieval_eval.json").read_text())["in_scope"]
chunks = load_chunks(BROCHURE_DIR)
ids = [c["id"] for c in chunks]


def score_ranking(rank_pages):
    """rank_pages(query) -> list of page ids best-first. Returns Hit@1/3/5 and MRR over the 24 labelled queries."""
    ranks = []
    for item in ev:
        ranked = rank_pages(item["q"])
        ranks.append(next((r for r, p in enumerate(ranked, 1) if p in set(item["gold"])), None))
    hit = lambda k: np.mean([r is not None and r <= k for r in ranks])
    return [hit(1), hit(3), hit(5), np.mean([1 / r if r else 0 for r in ranks])]


def table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |")


# ---- 1. own BM25 vs the rank_bm25 package ------------------------------------------------------
texts = [ps.doc_text(c, True) for c in chunks]
own = ps.BM25(texts)
pkg = BM25Okapi([ps.tokenize(t) for t in texts])
by_own = lambda q: [ids[i] for i, _ in own.search(q, len(ids))]
by_pkg = lambda q: [ids[i] for i in np.argsort(-pkg.get_scores(ps.tokenize(q)))]
table("BM25: my implementation vs rank_bm25 package (same tokenizer, k1=1.5, b=0.75)",
      ["implementation", "Hit@1", "Hit@3", "Hit@5", "MRR"],
      [["own (src/tools/product_search.py)"] + score_ranking(by_own), ["rank_bm25.BM25Okapi"] + score_ranking(by_pkg)])

# ---- 2. tokenizer variants ---------------------------------------------------------------------
orig_tok, orig_stop = ps.tokenize, ps.STOP


def make_tok(stop: bool, stem: bool):
    def tok(text):
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return [t[:-1] if stem and len(t) > 3 and t.endswith("s") else t for t in toks if not (stop and t in orig_stop)]
    return tok


rows = []
for name, (stop, stem) in {"no stop-words, no stemming": (False, False), "stop-words only": (True, False),
                           "stemming only": (False, True), "stop-words + plural stemming (chosen)": (True, True)}.items():
    ps.tokenize = make_tok(stop, stem)
    m = ps.BM25(texts)
    rows.append([name] + score_ranking(lambda q: [ids[i] for i, _ in m.search(q, len(ids))]))
ps.tokenize = orig_tok
table("BM25 tokenizer", ["tokenizer", "Hit@1", "Hit@3", "Hit@5", "MRR"], rows)

# ---- 3. chunk size: whole page vs half page (BM25 only; page hit if any piece of the page ranks) ----
def halves(c):
    lines = c["text"].split("\n")
    cut, acc = 0, 0
    for i, ln in enumerate(lines):
        acc += len(ln)
        if acc >= len(c["text"]) / 2:
            cut = i + 1
            break
    return ["\n".join(lines[:cut]), "\n".join(lines[cut:])] if 0 < cut < len(lines) else [c["text"]]


pieces = [(c["id"], ps.doc_text({**c, "text": h}, True)) for c in chunks for h in halves(c)]
mh = ps.BM25([t for _, t in pieces])


def by_halves(q):
    seen, out = set(), []
    for i, _ in mh.search(q, len(pieces)):
        if pieces[i][0] not in seen:
            seen.add(pieces[i][0])
            out.append(pieces[i][0])
    return out


table(f"Chunk size (BM25 only; {len(chunks)} page chunks vs {len(pieces)} half-page chunks)",
      ["chunking", "Hit@1", "Hit@3", "Hit@5", "MRR"],
      [["whole page (chosen)"] + score_ranking(by_own), ["half page"] + score_ranking(by_halves)])

# ---- 4. flotation: chronological vs random split ------------------------------------------------
df = load_flotation()
X, y = df.drop(columns=["date", SI, FE]), df[SI]
n = len(df)


def fit_eval(train, test):
    mu, sd = X.iloc[train].mean(), X.iloc[train].std()
    Z = lambda idx: np.c_[np.ones(len(idx)), ((X.iloc[idx] - mu) / sd).values]
    b = np.linalg.lstsq(Z(train), y.iloc[train].values, rcond=None)[0]
    lin = r2_score(y.iloc[test], Z(test) @ b)
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0).fit(X.iloc[train], y.iloc[train])
    return lin, r2_score(y.iloc[test], gbm.predict(X.iloc[test]))


cut = int(n * 0.7)
chron = fit_eval(np.arange(cut), np.arange(cut, n))
rand = np.array([fit_eval(*(lambda p: (p[:cut], p[cut:]))(np.random.default_rng(s).permutation(n))) for s in range(5)])
table("Flotation: can process settings predict silica? (test R2)", ["split", "linear", "gradient boosting"],
      [["chronological: train first 70%, test last 30% (chosen)", chron[0], chron[1]],
       ["random 70/30, mean of 5 shuffles", rand[:, 0].mean(), rand[:, 1].mean()]])

# ---- 5. reports: layouts, and a regex parser as the alternative to LLM extraction -----------------
GOLDEN = json.loads((ROOT / "tests" / "service_golden.json").read_text())  # hand-read values

reports = {p.stem: p.read_text() for p in sorted(REPORT_DIR.glob("FSR-*.txt"))}
visits = {v["report_id"]: v for v in load_visits()}


def layout(t):
    if re.search(r"(?i)cost summary", t):
        return "itemised cost table"
    if re.search(r"(?i)\|\s*\*{0,2}total", t):
        return "table without 'Cost Summary'"
    if re.search(r"(?i)(?:^|\n)\s*(?:#|\*\*)?\s*(?:field service report)", t) and re.search(r"(?i)\n(?:customer|tech|ref|date)\w*:", t):
        return "labelled fields + prose"
    return "other"


def prose_layout(t):
    tables = bool(re.search(r"(?m)^\|.*\|\s*$", t))
    return "markdown tables" if tables else ("short free-text note" if len(t) < 1500 else "long free-text notes")


from collections import Counter

cnt = Counter(prose_layout(t) for t in reports.values())


def regex_extract(t):
    num = lambda s: float(s.replace(",", "")) if s else None
    m = (re.search(r"(?i)\|\s*\*{0,2}total[^|\n]*\*{0,2}\s*\|[^\n]*?[$€]?\s*\*{0,2}([\d][\d,]*(?:\.\d+)?)\*{0,2}\s*\|?\s*$", t, re.M)
         or re.search(r"(?i)(?:total due|grand total|total amount due|total (?:service )?cost|cost total|cost)[^\d\n]*?([\d][\d,]*(?:\.\d+)?)", t))
    on = re.search(r"(?i)(?:on-?\s?site|hours on site|time on site)[^\d\n]{0,30}(\d+(?:\.\d+)?)", t)
    tr = re.search(r"(?i)travel[^\d\n]{0,30}(\d+(?:\.\d+)?)", t)
    return {"total": num(m.group(1)) if m else None, "onsite_hours": num(on.group(1)) if on else None,
            "travel_hours": num(tr.group(1)) if tr else None}


def grade(extract):
    right = wrong = missed = 0
    for rid, gold in GOLDEN.items():
        got = extract(rid)
        for f in ("total", "onsite_hours", "travel_hours"):
            if f not in gold:
                continue
            g, x = gold[f], got.get(f)
            if g is None:
                wrong += x is not None
                right += x is None
            elif x is None:
                missed += 1
            else:
                right += abs(x - g) < 0.01
                wrong += abs(x - g) >= 0.01
    return right, wrong, missed


rx = grade(lambda rid: regex_extract(reports[rid]))
llm = grade(lambda rid: {"total": comparable_total(visits[rid]), "onsite_hours": visits[rid]["onsite_hours"],
                         "travel_hours": visits[rid]["travel_hours"]})
table("Service reports: layouts across the 50 files", ["layout", "files"], [[k, v] for k, v in cnt.most_common()])
table(f"Service reports: regex parser vs LLM extraction, on {rx[0] + rx[1] + rx[2]} hand-read values (18 reports)",
      ["method", "correct", "wrong value", "missed (value exists, none found)"],
      [["regex parser (written for this check)", *rx], ["LLM extraction + number validation (chosen)", *llm]])
