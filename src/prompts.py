"""Prompts live in prompts/*.md (versioned in git). Each file starts with `version: N` then `---`.
Bump the version whenever you change a prompt; the run log records name, version and content hash."""
import hashlib
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache
def load(name: str) -> tuple[str, str]:
    """Returns (text, label) where label is 'name@version:hash' for logging."""
    raw = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    header, _, body = raw.partition("\n---\n")
    version = header.strip().removeprefix("version:").strip()
    body = body.rstrip("\n")
    return body, f"{name}@{version}:{hashlib.sha1(body.encode()).hexdigest()[:8]}"


def text(name: str) -> str:
    return load(name)[0]


def labels() -> list[str]:
    return [load(p.stem)[1] for p in sorted(PROMPT_DIR.glob("*.md"))]


AGENT_SYSTEM = text("agent_system")
EXTRACT_SYSTEM = text("extract_system")
ENTAIL_SYSTEM = text("entail_system")
SERVICE_EXTRACT_SYSTEM = text("service_extract_system")
