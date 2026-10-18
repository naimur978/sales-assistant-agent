"""End-to-end hallucination eval of the grounded product answerer.
  python -m scripts.eval_answers            # dev split only
  python -m scripts.eval_answers holdout    # run ONCE at the end, after tuning"""
import json
import re
import sys
from collections import Counter

from src.config import BROCHURE_DIR, CACHE_DIR, ROOT
from src.grounded import answer
from src.ingest.brochures import load_chunks
from src.tools.product_search import ProductSearch
from src.verifier import _norm_number, verify_claim

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
items = [i for i in json.loads((ROOT / "tests" / "answer_eval.json").read_text()) if split == "all" or i["split"] == split]
search = ProductSearch(load_chunks(BROCHURE_DIR))


def norm(text: str) -> str:
    text = re.sub(r"\d+(?:[.,]\d+)*", lambda m: _norm_number(m.group()), text.lower())
    return text.replace("–", "-")


def has(text: str, needle: str) -> bool:
    n = norm(needle)
    if n[0].isdigit():
        return re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", text) is not None
    return n in text


def grade(item, r):
    claims_text = norm(" ".join(c.claim for c in r.claims))
    answered = r.status == "answered"
    if item["kind"] in ("unanswerable",):
        return "WRONG" if answered else "correct_refusal"
    if item["kind"] == "ambiguous":
        return "review" if answered else "correct_refusal"
    forbidden = answered and any(has(claims_text, x) for x in item["must_not"])
    if forbidden:
        return "WRONG"
    if not answered:
        return "refused" if item["kind"] != "table" else "correct_refusal"
    ok_all = all(has(claims_text, x) for x in item["include_all"])
    ok_any = (not item["include_any"]) or any(has(claims_text, x) for x in item["include_any"])
    if ok_all and ok_any and (item["include_all"] or item["include_any"]):
        return "correct"
    return "incomplete" if (item["include_all"] or item["include_any"]) else "review"


results, counts, bad_citations, total_claims = [], Counter(), 0, 0
for item in items:
    r = answer(item["q"], search)
    g = grade(item, r)
    counts[g] += 1
    for c in r.claims:
        total_claims += 1
        bad_citations += not verify_claim(c, search.pages)[0]
    results.append({"q": item["q"], "kind": item["kind"], "grade": g, "status": r.status, "text": r.text,
                    "dropped": r.dropped})
    print(f"{g:<16}{item['kind']:<14}{item['q'][:70]}")

CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / f"results_{split}.json").write_text(json.dumps(results, indent=1, default=str))
n = len(items)
answerable = [x for x in results if x["kind"] in ("answerable", "conflict", "false_premise")]
print(f"\n== {split}: {n} questions ==")
print(f"WRONG answers          : {counts['WRONG']}   <- target 0")
print(f"correct answers        : {counts['correct']}")
print(f"incomplete (no wrong value, missing required fact): {counts['incomplete']}")
print(f"refused although answerable: {counts['refused']} of {len(answerable)}")
print(f"correct refusals       : {counts['correct_refusal']}")
print(f"left for manual review : {counts['review']}")
print(f"citations that fail re-verification: {bad_citations} of {total_claims} claims")
for x in results:
    if x["grade"] in ("WRONG", "incomplete", "review", "refused"):
        print(f"\n[{x['grade']}] {x['q']}\n{x['text']}")
