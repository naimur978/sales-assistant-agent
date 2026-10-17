import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
BROCHURE_DIR = DATA_DIR / "brochures"
CACHE_DIR = ROOT / ".cache"

EMBED_MODEL = "gemini-embedding-001"

GEN_MODEL = "gemini-3.5-flash-lite"
TOP_K = 4
GATE = 0.70  # loose on-topic gate (dense cosine); the verifier does the real safety work
