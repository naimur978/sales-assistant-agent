"""Question -> retrieve -> LLM extracts quoted claims -> code verifies -> answer or abstain.
The final text is built from verified claims only; the LLM never writes free-form facts."""
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from src.config import GATE, GEN_MODEL, TOP_K
from src.prompts import ENTAIL_SYSTEM, EXTRACT_SYSTEM
from src.tools.product_search import ProductSearch
from src.verifier import Claim, verify_claim

load_dotenv()
_client = None


class ExtractedClaim(Claim):
    from_table: bool = False


class Extraction(BaseModel):
    claims: list[ExtractedClaim]
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
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (claim, reason)
    closest: list[str] = field(default_factory=list)


def _generate(contents: str, system: str, schema):
    global _client
    _client = _client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cfg = types.GenerateContentConfig(
        temperature=0,
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
    )
    for attempt in range(6):
        try:
            return _client.models.generate_content(model=GEN_MODEL, contents=contents, config=cfg).parsed
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or attempt == 5:
                raise
            time.sleep(15 * (attempt + 1))


def _sources(hits) -> str:
    return "\n\n".join(f"[SOURCE file={h['file']} page={h['page']}]\n{h['text']}" for h in hits)


def entailed(claims: list[ExtractedClaim]) -> set[int]:
    """Second LLM pass: is each claim fully stated by its own quote? Fails closed."""
    if not claims:
        return set()
    body = "\n\n".join(f"{i}. CLAIM: {c.claim}\n   QUOTE (from brochure {c.file}, page {c.page}): {c.quote}" for i, c in enumerate(claims))
    try:
        res = _generate(body, ENTAIL_SYSTEM, Verdicts)
        return {v.index for v in res.items if v.supported}
    except Exception as e:
        print(f"[entailment check failed, dropping all claims: {e!r}]")
        return set()  # cannot check -> trust nothing


def answer(question: str, search: ProductSearch) -> Answer:
    res = search.search(question, TOP_K)
    closest = [h["id"] for h in res["hits"]]
    if res["gate_score"] < GATE or not res["hits"]:
        return Answer("abstained", "This isn't covered in the product brochures.", closest=closest)

    ext = _generate(f"QUESTION: {question}\n\n{_sources(res['hits'])}", EXTRACT_SYSTEM, Extraction)
    if ext.clarifying_question and not ext.claims:
        return Answer("clarify", ext.clarifying_question)

    kept, dropped = [], []
    for c in ext.claims:
        ok, why = verify_claim(c, search.pages)
        (kept if ok else dropped).append(c if ok else (c.claim, why))
    supported = entailed(kept)
    final = []
    for i, c in enumerate(kept):
        if i in supported:
            final.append(c)
        else:
            dropped.append((c.claim, "quote does not fully support claim"))

    if not final:
        why = ext.missing or "the brochures do not state this."
        return Answer("abstained", f"I can't answer that from the brochures: {why}", dropped=dropped, closest=closest)

    lines = []
    for c in final:
        flag = " (from a spec table, please confirm against the PDF)" if c.from_table else ""
        lines.append(f"- {c.claim}{flag} [{c.file}, p.{c.page}]")
    return Answer("answered", "\n".join(lines), claims=final, dropped=dropped, closest=closest)
