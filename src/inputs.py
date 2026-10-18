"""Clean and limit what the user types before it reaches the model."""
import re
import unicodedata


def clean_input(text: str) -> str:
    """Drop control characters (keeping newlines and tabs), collapse runs of spaces, trim."""
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"[ \t]+", " ", text).strip()
