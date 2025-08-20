# src/data/cleaning.py
from __future__ import annotations
import re

BOILERPLATE_PATTERNS = [
    r"thank you for your help",
    r"please contact me",
    r"i have attached",
]

_WHITESPACE_RE = re.compile(r"\s+")

# Fix: Move the (?i) flag to the beginning of each pattern
BOILERPLATE_PATTERNS_WITH_FLAGS = [
    r"(?i)thank you for your help",
    r"(?i)please contact me", 
    r"(?i)i have attached",
]

# Only compile if patterns exist
if BOILERPLATE_PATTERNS_WITH_FLAGS:
    _BOILER_RE = re.compile("|".join(BOILERPLATE_PATTERNS_WITH_FLAGS))
else:
    _BOILER_RE = None

def clean_text(text: str) -> str:
    if text is None:
        return ""
    
    # Normalize whitespace
    normalized = _WHITESPACE_RE.sub(" ", str(text)).strip()
    
    # Convert to lowercase
    normalized = normalized.lower()
    
    # Remove boilerplate patterns if regex exists
    if _BOILER_RE:
        normalized = _BOILER_RE.sub(" ", normalized)
    
    return normalized.strip()