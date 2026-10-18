"""Run logs that contain no private data: user text is redacted, tool results are logged as sizes and
statuses only (they can contain customer names and report details)."""
import json
import re
import time

from src.config import CACHE_DIR

PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email]"),
    (re.compile(r"AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9_-]{16,}|(?i:key|token|secret)\s*[:=]\s*\S+"), "[secret]"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "[phone]"),
]
LOG = CACHE_DIR / "traces.jsonl"


def redact(text: str) -> str:
    for rx, token in PATTERNS:
        text = rx.sub(token, text)
    return text


def log_turn(user_message: str, steps: list[dict], reply: str, usage=None, prompt_versions=None) -> None:
    record = {
        "ts": int(time.time()),
        "user": redact(user_message),
        "steps": [
            {"type": s["type"], "name": s.get("name"), "args": {k: v for k, v in (s.get("args") or {}).items()},
             "status": s.get("status"), "result_chars": len(s.get("result", "")) if "result" in s else None,
             "problems": s.get("problems")}
            for s in steps
        ],
        "reply_chars": len(reply),
        "usage": None if usage is None else {"calls": usage.calls, "retries": usage.retries, "failures": usage.failures,
                                             "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        "prompts": prompt_versions,
    }
    CACHE_DIR.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def describe_steps(steps: list[dict]) -> str:
    """One line a user can read after each reply, e.g. 'estimate_service_cost (needs_info) -> estimate_service_cost (estimate)'."""
    parts = [f"{s['name']} ({s['status']})" for s in steps if s["type"] == "tool"]
    if any(s["type"] == "guard" and s.get("stage") == "first_reply" for s in steps):
        parts.append("reply rewritten after the check")
    return " -> ".join(parts) if parts else "answered directly, no tool used"
