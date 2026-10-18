import re
from pathlib import Path

PAGE_RE = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)


# Company boilerplate (copyright / "frontrunner" paragraph / contact lines) carries no
# product facts but matches any query that mentions "Metso", so drop it.
BOILERPLATE = [
    re.compile(r"(?:Copyright )?©.*?(?:\n|$)"),
    re.compile(r"Metso(?: Outotec)?(?: Corporation)? is a frontrunner.*?positive change\.", re.S),
    re.compile(r"Metso(?: Outotec)? Corporation,.*?(?:\n|$)"),
    re.compile(r"^(?:tel\.|metso\.com|mogroup\.com).*$", re.M),
]


def _clean(text: str) -> str:
    for rx in BOILERPLATE:
        text = rx.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def load_chunks(brochure_dir: Path) -> list[dict]:
    """One chunk per brochure page: {id, file, page, text}."""
    chunks = []
    for path in sorted(Path(brochure_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        parts = PAGE_RE.split(raw)  # [header, page_no, body, page_no, body, ...]
        for page_no, body in zip(parts[1::2], parts[2::2], strict=True):
            text = _clean(body)
            if len(re.findall(r"[A-Za-z]{3,}", text)) < 5:
                continue  # image-only / near-empty page
            chunks.append(
                {
                    "id": f"{path.stem}:{int(page_no)}",
                    "file": path.stem,
                    "page": int(page_no),
                    "text": text,
                }
            )
    return chunks
