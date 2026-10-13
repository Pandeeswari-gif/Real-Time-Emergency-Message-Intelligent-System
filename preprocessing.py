"""
preprocessing.py
================
Text preprocessing pipeline for EmergencyNLP.

Pipeline (in order):
    Input Text -> Text Cleaning -> Lowercase -> Tokenization
               -> Stop-word Removal -> Lemmatization

Every step is a small function, so you can test or explain each one on its own.
The spaCy English model is used for lemmatization (and later for NER). If the
model is not installed, a simple rule-based lemmatizer takes over, so the
project never crashes because of a missing model.
"""

import importlib
import logging
import re
import subprocess
import sys
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SPACY_MODEL_NAME = "en_core_web_sm"

# --------------------------------------------------------------------------
# Stop words
# --------------------------------------------------------------------------
# A compact English stop-word list. Words such as "no", "not", "help",
# "please" are intentionally NOT removed because they carry meaning in
# emergency messages. (spaCy's own list also removes words like "serious".)
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "as", "until", "while", "s", "t",
    "also", "please", "pls", "plz", "kindly", "hello", "hi", "sir", "madam", "near",
}

# Irregular word forms used by the fallback lemmatizer.
IRREGULAR_LEMMAS = {
    "people": "person", "persons": "person", "men": "man", "women": "woman",
    "children": "child", "injuries": "injury", "was": "be", "were": "be",
    "trapped": "trap", "stranded": "strand", "flooded": "flood",
    "burning": "burn", "burnt": "burn", "bleeding": "bleed",
    "running": "run", "hit": "hit", "fell": "fall", "fallen": "fall",
    "lost": "lose", "stolen": "steal", "broken": "break", "unconscious": "unconscious",
}


# --------------------------------------------------------------------------
# spaCy loader (shared with nlp_utils.py)
# --------------------------------------------------------------------------
_NLP = None
_NLP_LOAD_ATTEMPTED = False


def get_nlp(auto_download: bool = True):
    """
    Load the spaCy English model once and reuse it.

    Returns the spaCy pipeline, or None when spaCy / the model is unavailable.
    If the model is missing and auto_download is True, we try to download it
    (needs internet, done only once per program run).
    """
    global _NLP, _NLP_LOAD_ATTEMPTED
    if _NLP is not None or _NLP_LOAD_ATTEMPTED:
        return _NLP
    _NLP_LOAD_ATTEMPTED = True

    try:
        import spacy
    except ImportError:
        logger.warning("spaCy is not installed. Using rule-based fallbacks.")
        return None

    try:
        _NLP = spacy.load(SPACY_MODEL_NAME)
        return _NLP
    except OSError:
        logger.warning("spaCy model '%s' not found.", SPACY_MODEL_NAME)

    if auto_download:
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", SPACY_MODEL_NAME],
                check=True, capture_output=True, timeout=600,
            )
            importlib.invalidate_caches()
            _NLP = spacy.load(SPACY_MODEL_NAME)
            return _NLP
        except Exception as exc:  # no internet, pip error, timeout ...
            logger.warning("Could not download spaCy model: %s", exc)

    return None


def spacy_available() -> bool:
    """True when the spaCy English model is loaded and usable."""
    return get_nlp() is not None


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Step 1 - Text cleaning: remove URLs, HTML tags, symbols and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # links
    text = re.sub(r"<[^>]+>", " ", text)                    # html tags
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)             # punctuation, emojis
    text = re.sub(r"\s+", " ", text)                        # extra whitespace
    return text.strip()


def to_lowercase(text: str) -> str:
    """Step 2 - Lowercase conversion."""
    return text.lower()


def tokenize(text: str) -> List[str]:
    """Step 3 - Tokenization: split text into word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def remove_stop_words(tokens: List[str]) -> List[str]:
    """Step 4 - Stop-word removal."""
    return [t for t in tokens if t not in STOP_WORDS]


def _simple_lemma(word: str) -> str:
    """Very small rule-based lemmatizer used when spaCy is not available."""
    if word in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def lemmatize(tokens: List[str]) -> List[str]:
    """
    Step 5 - Lemmatization: reduce words to their base form
    (injured -> injure, trapped -> trap, buildings -> building).
    """
    if not tokens:
        return []
    nlp = get_nlp()
    if nlp is not None:
        try:
            # Only the components needed for lemmas are run (faster).
            doc = nlp(" ".join(tokens), disable=["parser", "ner"])
            lemmas = [tok.lemma_.lower() for tok in doc if not tok.is_space]
            return [l for l in lemmas if l.strip()]
        except Exception as exc:
            logger.warning("spaCy lemmatization failed, using fallback: %s", exc)
    return [_simple_lemma(t) for t in tokens]


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Run the whole pipeline and return one clean string (used for TF-IDF)."""
    cleaned = clean_text(text)
    tokens = tokenize(to_lowercase(cleaned))
    tokens = remove_stop_words(tokens)
    return " ".join(lemmatize(tokens))


def preprocess_steps(text: str) -> Dict[str, object]:
    """Return the output of every pipeline step (shown on the dashboard)."""
    cleaned = clean_text(text)
    lowered = to_lowercase(cleaned)
    tokens = tokenize(lowered)
    no_stop = remove_stop_words(tokens)
    lemmas = lemmatize(no_stop)
    return {
        "Original text": text,
        "Cleaned text": cleaned,
        "Lowercase": lowered,
        "Tokens": tokens,
        "After stop-word removal": no_stop,
        "Lemmas": lemmas,
        "Final text for TF-IDF": " ".join(lemmas),
    }


def is_valid_message(text: Optional[str], min_chars: int = 5, max_chars: int = 1000):
    """
    Validate user input. Returns (is_valid, error_message).
    Rejects empty text, too-short/too-long text and text with no letters.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return False, "Please enter an emergency message."
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False, f"The message is too short. Please write at least {min_chars} characters."
    if len(stripped) > max_chars:
        return False, f"The message is too long. Please keep it under {max_chars} characters."
    if len(re.findall(r"[A-Za-z]{2,}", stripped)) < 2:
        return False, "The message needs at least two readable words."
    return True, ""
