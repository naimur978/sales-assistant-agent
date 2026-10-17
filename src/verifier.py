"""Deterministic checks that a claim is backed by a verbatim quote from a cited brochure page.
Everything here fails closed: if in doubt, the claim is rejected."""
import re
import unicodedata

from pydantic import BaseModel

MIN_QUOTE_TOKENS = 3
_TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


class Claim(BaseModel):
    claim: str
    quote: str
    file: str
    page: int


def _norm_number(tok: str) -> str:
    if re.fullmatch(r"\d{1,3}(,\d{3})+", tok):  # 1,300 -> 1300 (thousands separator)
        return tok.replace(",", "")
    return tok.replace(",", ".")  # 1,6 -> 1.6 (decimal comma)


def tokens(text: str) -> list[str]:
    """Lowercase words and numbers. Punctuation, dashes and line breaks are ignored,
    but 1.6 and 16 stay different tokens."""
    text = unicodedata.normalize("NFKC", text).lower().replace("­", "")
    return [_norm_number(t) if t[0].isdigit() else t for t in _TOKEN.findall(text)]


def numbers(text: str) -> list[str]:
    return [_norm_number(t) for t in _NUM.findall(unicodedata.normalize("NFKC", text))]


def _contains(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


_UNITS = r"(?:bars?|m2|m²|m3|m³|kw|hp|mm|kg|t|l|rpm|hz|v|°c|%|minutes?|min|h|m|microns?|μm|years?|installations?|units?|pcs)\b"
# A product model number: letters-code + number ("PF 60", "AC2000", "HM-100") or "Model 2025",
# not followed by a unit. These name the product; they are not facts needing a quote.
_MODEL = re.compile(rf"(?:\b[A-Z]{{1,4}}[ -]?|\bModel )(\d+(?:[.,]\d+)?)(?!\s*{_UNITS})(?![\d.,]*\d)")


def model_numbers(text: str) -> set[str]:
    return {_norm_number(m) for m in _MODEL.findall(text)}


def verify_claim(c: Claim, pages: dict[str, str]) -> tuple[bool, str]:
    """pages maps 'file:page' -> page text. Returns (ok, reason)."""
    page = pages.get(f"{c.file}:{c.page}")
    if page is None:
        return False, f"cited page {c.file}:{c.page} does not exist"
    q = tokens(c.quote)
    if len(q) < MIN_QUOTE_TOKENS:
        return False, "quote too short to be evidence"
    if not _contains(tokens(page), q):
        return False, "quote not found verbatim on the cited page"
    evidence = set(numbers(c.quote)) | model_numbers(c.claim)
    for n in numbers(c.claim):
        if _norm_number(n) not in evidence:
            return False, f"number {n} in claim is not in the quote"
    return True, "ok"
