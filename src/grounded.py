"""Question -> retrieve -> LLM extracts quoted claims -> code verifies -> answer or abstain.
The final text is built from verified claims only; the LLM never writes free-form facts."""
import logging
import re
from dataclasses import dataclass, field

from google.genai import types
from pydantic import BaseModel

from src import llm
from src.config import GATE, TOP_K
from src.prompts import ENTAIL_SYSTEM, EXTRACT_SYSTEM
from src.tools.product_search import ProductSearch
from src.verifier import Claim, verify_claim

log = logging.getLogger(__name__)

# Display names come from the cited brochure's file, never from model text.
PRODUCT_NAMES = {
    "larox-pf-brochure-a": "Larox PF pressure filter", "larox-pf-brochure-b": "Larox PF pressure filter",
    "larox-pf-ds": "Larox PF-DS tower press filter", "filter-optimizer": "Metso Filter optimizer",
    "columncell": "ColumnCell", "concorde-cell": "Concorde Cell", "cellstation": "CellStation",
    "floatforce-upgrade": "FloatForce upgrade", "md-series-pump": "MD series mill discharge pump",
    "orion-series-pump": "Orion series slurry pump", "ac-filter": "AC filter", "gm-filter": "GM filter",
}


class ExtractedClaim(Claim):
    from_table: bool = False


class Recommendation(BaseModel):
    claim_index: int  # which entry of `claims` best answers the salesperson's need
    customer_phrase: str  # words copied from the salesperson's own message that this claim responds to


class Extraction(BaseModel):
    claims: list[ExtractedClaim]
    recommendation: Recommendation | None = None
    missing: str | None = None
    clarifying_question: str | None = None


class Verdict(BaseModel):
    index: int
    supported: bool


class Verdicts(BaseModel):
    items: list[Verdict]


@dataclass
class Answer:
    status: str  # answered | clarify | abstained
    text: str
    claims: list[ExtractedClaim] = field(default_factory=list)
    recommended: str | None = None  # product name, when a verified recommendation was made
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (claim, reason)
    closest: list[str] = field(default_factory=list)


def _generate(contents: str, system: str, schema, purpose: str = "extract"):
    cfg = types.GenerateContentConfig(temperature=0, system_instruction=system,
                                      response_mime_type="application/json", response_schema=schema)
    return llm.generate(contents, cfg, purpose).parsed


def _sources(hits) -> str:
    return "\n\n".join(f"[SOURCE file={h['file']} page={h['page']}]\n{h['text']}" for h in hits)


def entailed(claims: list[ExtractedClaim]) -> set[int]:
    """Second LLM pass: is each claim fully stated by its own quote? Fails closed."""
    if not claims:
        return set()
    body = "\n\n".join(f"{i}. CLAIM: {c.claim}\n   QUOTE (from brochure {c.file}, page {c.page}): {c.quote}" for i, c in enumerate(claims))
    try:
        res = _generate(body, ENTAIL_SYSTEM, Verdicts, purpose="entail")
        return {v.index for v in res.items if v.supported}
    except llm.LLMUnavailable:
        raise  # say "model unavailable", not a misleading "not in the brochures"
    except Exception as e:
        log.warning("entailment check failed, dropping all claims: %r", e)
        return set()  # cannot check -> trust nothing


def answer(question: str, search: ProductSearch, also: str = "") -> Answer:
    res = search.search([question, also] if also else question, TOP_K)
    closest = [h["id"] for h in res["hits"]]
    if res["gate_score"] < GATE or not res["hits"]:
        return Answer("abstained", "This isn't covered in the product brochures.", closest=closest)

    own_words = f"SALESPERSON'S OWN WORDS: {also}\n" if also else ""
    ext = _generate(f"QUESTION: {question}\n{own_words}\n{_sources(res['hits'])}", EXTRACT_SYSTEM, Extraction)
    if ext.clarifying_question and not ext.claims:
        return Answer("clarify", ext.clarifying_question)

    verified, dropped = [], []  # verified: (index in ext.claims, claim)
    for idx, c in enumerate(ext.claims):
        ok, why = verify_claim(c, search.pages)
        if ok:
            verified.append((idx, c))
        else:
            dropped.append((c.claim, why))
    supported = entailed([c for _, c in verified])
    final, final_pos = [], {}  # final_pos: original index -> position in final
    for j, (idx, c) in enumerate(verified):
        if j in supported:
            final_pos[idx] = len(final)
            final.append(c)
        else:
            dropped.append((c.claim, "quote does not fully support claim"))

    if not final:
        why = ext.missing or "the brochures do not state this."
        return Answer("abstained", f"I can't answer that from the brochures: {why}", dropped=dropped, closest=closest)

    return _compose(final, _checked_recommendation(ext.recommendation, final_pos, final, also or question),
                    dropped, closest)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%.]+", " ", text.lower()).strip()


def _checked_recommendation(rec: Recommendation | None, final_pos: dict, final: list, user_words: str):
    """A recommendation stands only if it points at a claim that survived verification and quotes the user's own words."""
    if rec is None or rec.claim_index not in final_pos:
        return None
    phrase = _norm(rec.customer_phrase)
    if len(phrase.split()) < 2 or phrase not in _norm(user_words):
        return None
    claim = final[final_pos[rec.claim_index]]
    if claim.file not in PRODUCT_NAMES:
        return None
    return PRODUCT_NAMES[claim.file], rec.customer_phrase.strip(), final_pos[rec.claim_index]


def _compose(final, rec, dropped, closest) -> Answer:
    def line(c):
        flag = " (from a spec table, please confirm against the PDF)" if c.from_table else ""
        return f"{c.claim}{flag} [{c.file}, p.{c.page}]"

    if rec is None:
        return Answer("answered", "\n".join(f"- {line(c)}" for c in final), claims=final, dropped=dropped, closest=closest)
    product, phrase, pos = rec
    out = [f'Recommendation: {product}. You said "{phrase}"; the brochure says: {line(final[pos])}']
    others = [c for i, c in enumerate(final) if i != pos]
    if others:
        out += ["", "Other relevant facts:"] + [f"- {line(c)}" for c in others]
    return Answer("answered", "\n".join(out), claims=final, recommended=product, dropped=dropped, closest=closest)
