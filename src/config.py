import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
BROCHURE_DIR = DATA_DIR / "brochures"
CACHE_DIR = ROOT / ".cache"

EMBED_MODEL = "gemini-embedding-001"

GEN_MODEL = "gemini-3.1-flash-lite"
TOP_K = 6
GATE = 0.70  # loose on-topic gate (dense cosine); the verifier does the real safety work

FLOTATION_CSV = DATA_DIR / "flotation_process_data_hourly.csv"

# Reliability
REQUEST_TIMEOUT_S = 60       # per model request
MAX_ATTEMPTS = 4             # per request, for temporary errors only
BACKOFF_S = (2, 5, 12)       # waits between attempts (a little jitter is added)
TURN_DEADLINE_S = 180        # whole user message, including tool calls and retries
# Set these from the provider's pricing page (USD per 1M tokens). Left empty on purpose: a made-up price would be
# a made-up cost. While empty, usage reports show tokens only.
PRICE_PER_MILLION_TOKENS = {"input": None, "output": None}

# Input handling
MAX_INPUT_CHARS = 2000       # longer messages are refused before any model call
HISTORY_TOKEN_BUDGET = 20000  # rough size of the conversation resent each call; only older whole messages are dropped beyond it
