"""
Shared text preprocessing.

Important: the exact same cleaning function must be used at training time
and at prediction time, otherwise the vectorizer sees different-looking
text than what it was trained on and accuracy silently drops.
"""
import re

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_PATTERN = re.compile(r"<.*?>")
_NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/punctuation/numbers, collapse whitespace."""
    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = _NON_ALPHA_PATTERN.sub(" ", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text
