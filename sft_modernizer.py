"""Rule-based archaic→modern English substitution for Shakespeare text.

Used to build synthetic SFT training pairs:
  source  = original Shakespeare line (archaic)
  modern  = modernized version produced by these substitution rules

Rules are ordered so longer/more-specific patterns are matched first
(e.g. 'wouldst' before a hypothetical 'woul').
"""

import re


# (regex pattern, modern replacement) — order matters
_RULES = [
    # Contractions first (most specific) — no leading \b since ' is non-word
    (r"'tis\b",         "it is"),
    (r"'twas\b",        "it was"),
    (r"\bo'er\b",       "over"),
    (r"\bne'er\b",      "never"),
    (r"\be'er\b",       "ever"),
    # Multi-word / long patterns
    (r"\bwherefore\b",  "why"),
    (r"\bmethinks\b",   "I think"),
    (r"\bprithee\b",    "please"),
    (r"\bforsooth\b",   "indeed"),
    (r"\bverily\b",     "truly"),
    (r"\bnaught\b",     "nothing"),
    (r"\bperchance\b",  "perhaps"),
    (r"\bmayhap\b",     "maybe"),
    # Verb conjugations (longer forms first)
    (r"\bwouldst\b",    "would"),
    (r"\bcouldst\b",    "could"),
    (r"\bshouldst\b",   "should"),
    (r"\bcanst\b",      "can"),
    (r"\bdidst\b",      "did"),
    (r"\bhadst\b",      "had"),
    (r"\bwilt\b",       "will"),
    (r"\bhast\b",       "have"),
    (r"\bhath\b",       "has"),
    (r"\bdoth\b",       "does"),
    (r"\bdost\b",       "do"),
    (r"\bart\b",        "are"),
    # Pronouns
    (r"\bthine\b",      "your"),
    (r"\bthy\b",        "your"),
    (r"\bthee\b",       "you"),
    (r"\bthou\b",       "you"),
    # Misc
    (r"\bere\b",        "before"),
    (r"\bnay\b",        "no"),
    (r"\byea\b",        "yes"),
    (r"\boft\b",        "often"),
]

# Pre-compile all patterns for speed
_COMPILED = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in _RULES]


def _preserve_case(matched_word: str, replacement: str) -> str:
    """Capitalize replacement if the first alphabetic char of the match is uppercase."""
    for ch in matched_word:
        if ch.isalpha():
            if ch.isupper():
                return replacement[0].upper() + replacement[1:]
            break
    return replacement


def modernize(text: str) -> str:
    """Return a modernized version of the given Shakespeare text."""
    result = text
    for compiled, replacement in _COMPILED:
        def _sub(m, rep=replacement):
            return _preserve_case(m.group(0), rep)
        result = compiled.sub(_sub, result)
    return result


def has_archaic(text: str) -> bool:
    """Return True if text contains at least one archaic pattern."""
    for compiled, _ in _COMPILED:
        if compiled.search(text):
            return True
    return False
