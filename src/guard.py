"""Checks on the agent's final reply before the user sees it. Deterministic and fail-closed.
  1. every number must appear in a tool result or in what the user said
  2. every Metso product name must appear in a tool result or in what the user said
  3. the reply must not repeat the system prompt"""
import re

from src.ingest.services import number_values
from src.prompts import AGENT_SYSTEM

PRODUCT_NAMES = [
    "larox", "pf-ds", "columncell", "concorde", "cellstation", "frothsense", "exactlevel", "floatforce",
    "orion", "md series", "mdm", "mdr", "filter optimizer", "ac filter", "gm filter", "dm filter", "sala",
    "tankcell", "skimair", "sonicsparger", "microcel",
]
SHINGLE = 10
CITATION_ANY = re.compile(r"\[page \d+ of [^\]]+\]")
OFF_TOPIC = "I can only help with Metso products, service cost estimates and flotation readings."
INJECTION = re.compile(
    r"\b(ignore|disregard|forget|override)\b[^.?!\n]{0,30}\b(instructions?|rules|prompt|guidelines)\b"
    r"|\b(reveal|show|print|repeat|output)\b[^.?!\n]{0,30}\b(system|hidden|initial)\s+(prompt|instructions?)\b"
    r"|\byou are now\b|\bact as\b[^.?!\n]{0,20}\b(dan|unrestricted|jailbroken)\b|\bdeveloper mode\b|\bjailbreak\b",
    re.I)


def looks_like_injection(text: str) -> bool:
    """Code-side screen for the common 'ignore your rules' and 'show your prompt' phrasings, before the model sees them."""
    return INJECTION.search(text) is not None


def _shingles(text: str) -> set[tuple]:
    w = re.findall(r"[a-z']+", text.lower())
    return {tuple(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1)}


# the canned refusal is meant to be said, so it must not count as leaking the prompt
_PROMPT_SHINGLES = _shingles(AGENT_SYSTEM.replace(OFF_TOPIC[2:-1], ""))


def _strip_list_numbers(text: str) -> str:
    return re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)


def check_reply(reply: str, sources: list[str], user_text: str) -> list[str]:
    """Returns a list of problems; empty means the reply may be shown."""
    problems = []
    known = number_values(" ".join(sources) + " " + user_text)
    bad = sorted({x for x in number_values(_strip_list_numbers(reply)) if not any(abs(x - k) < 1e-6 for k in known)})
    if bad:
        problems.append("numbers not found in any tool result or in the user's messages: "
                        + ", ".join(f"{x:g}" for x in bad))
    source_text = " ".join(sources)
    invented = sorted({c for c in CITATION_ANY.findall(reply.replace("\u00a0", " ")) if c not in source_text})
    if invented:
        problems.append("source references not found in any tool result (only cite what the tools gave): " + ", ".join(invented))
    corpus = (" ".join(sources) + " " + user_text).lower()
    unsourced = [n for n in PRODUCT_NAMES if re.search(rf"\b{re.escape(n)}", reply.lower()) and n not in corpus]
    if unsourced:
        problems.append("product names not found in any tool result: " + ", ".join(unsourced))
    if re.search(r"(?m)^Details:\s*$", source_text) and not re.search(r"(?m)^Details:\s*$", reply):
        problems.append('the reply dropped the "Details:" section; give the "Quick answer" first, then "Details:" with the tool\'s breakdown')
    if _shingles(reply) & _PROMPT_SHINGLES:
        problems.append("the reply repeats the system instructions")
    return problems
