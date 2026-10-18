"""The one place that talks to the model API: retries, timeout, quota handling and usage counting."""
import os
import random
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src import config

load_dotenv()
sleep = time.sleep  # replaced in tests


class LLMUnavailable(Exception):
    """The model could not be reached after retries (network, timeout, server errors)."""


class QuotaExhausted(LLMUnavailable):
    """A daily or billing quota is used up. Retrying within this session is pointless."""


@dataclass
class Usage:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_purpose: dict = field(default_factory=dict)

    def snapshot(self) -> "Usage":
        return Usage(self.calls, self.retries, self.failures, self.input_tokens, self.output_tokens,
                     {k: dict(v) for k, v in self.by_purpose.items()})

    def since(self, before: "Usage") -> "Usage":
        return Usage(self.calls - before.calls, self.retries - before.retries, self.failures - before.failures,
                     self.input_tokens - before.input_tokens, self.output_tokens - before.output_tokens)

    def cost_usd(self) -> float | None:
        p = config.PRICE_PER_MILLION_TOKENS
        if p["input"] is None or p["output"] is None:
            return None
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1_000_000

    def summary(self) -> str:
        cost = self.cost_usd()
        return (f"{self.calls} model calls, {self.input_tokens:,} input + {self.output_tokens:,} output tokens, "
                f"{self.retries} retries, {self.failures} failures, "
                + (f"~${cost:.4f}" if cost is not None else "cost: set PRICE_PER_MILLION_TOKENS in config"))


USAGE = Usage()
_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                               http_options=types.HttpOptions(timeout=config.REQUEST_TIMEOUT_S * 1000))
    return _client


def _is_daily_quota(e: errors.APIError) -> bool:
    return e.code == 429 and "PerDay" in str(e)


def call(fn, purpose: str = "other"):
    """Run fn() with retries on temporary errors. Raises QuotaExhausted fast on daily quotas."""
    for attempt in range(config.MAX_ATTEMPTS):
        try:
            result = fn()
            USAGE.calls += 1
            USAGE.by_purpose.setdefault(purpose, {"calls": 0})["calls"] += 1
            return result
        except errors.APIError as e:
            if _is_daily_quota(e):
                USAGE.failures += 1
                raise QuotaExhausted("The model's daily quota is used up.") from e
            temporary = e.code in (408, 429, 500, 502, 503, 504)
        except Exception as e:  # timeouts / connection resets surface as httpx or OS errors
            temporary = any(k in type(e).__name__ for k in ("Timeout", "Connect", "Read", "Remote", "Protocol"))
            if not temporary:
                raise
        if attempt == config.MAX_ATTEMPTS - 1 or not temporary:
            USAGE.failures += 1
            raise LLMUnavailable("The model did not respond after several attempts.")
        USAGE.retries += 1
        sleep(config.BACKOFF_S[min(attempt, len(config.BACKOFF_S) - 1)] + random.random())


def generate(contents, cfg: types.GenerateContentConfig, purpose: str):
    resp = call(lambda: client().models.generate_content(model=config.GEN_MODEL, contents=contents, config=cfg), purpose)
    meta = getattr(resp, "usage_metadata", None)
    if meta:
        i, o = meta.prompt_token_count or 0, meta.candidates_token_count or 0
        USAGE.input_tokens += i
        USAGE.output_tokens += o
        p = USAGE.by_purpose[purpose]
        p["input_tokens"], p["output_tokens"] = p.get("input_tokens", 0) + i, p.get("output_tokens", 0) + o
    return resp
