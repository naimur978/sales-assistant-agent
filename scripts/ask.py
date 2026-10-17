"""Try the grounded product answerer:  python -m scripts.ask "your question" """
import sys

from src.config import BROCHURE_DIR
from src.grounded import answer
from src.ingest.brochures import load_chunks
from src.tools.product_search import ProductSearch

search = ProductSearch(load_chunks(BROCHURE_DIR))
r = answer(" ".join(sys.argv[1:]), search)
print(f"[{r.status}]\n{r.text}")
if r.dropped:
    print("\ndropped by verifier:")
    for c, why in r.dropped:
        print(f"  - {c}  ({why})")
print("\nclosest pages:", r.closest)
