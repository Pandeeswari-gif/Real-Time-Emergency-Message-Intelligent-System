"""
nlp_utils.py
============
The "brain" of EmergencyNLP. It contains:

    1. Keyword lexicons (critical / high / medium / de-escalation words)
    2. Named Entity Recognition with spaCy (+ fallbacks)
    3. Location extraction (spaCy + place-word rules + fallbacks)
    4. Affected-people extraction
    5. Required-service detection (rule based, easy to replace with an ML model)
    6. Keyword extraction
    7. Priority scoring (category + keywords + people count)
    8. Recommended response text
    9. EmergencyAnalyzer - one class that runs the complete pipeline

The ML category classifier itself is trained in train_model.py and used here.
"""

import logging
import re
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from preprocessing import get_nlp, is_valid_message, preprocess_steps

logger = logging.getLogger(__name__)

# ==========================================================================
# 1. LEXICONS
# ==========================================================================
CATEGORIES = [
    "Accident", "Fire", "Flood", "Medical Emergency",
    "Earthquake", "Crime", "Rescue", "Other",
]
PRIORITY_LEVELS = ["Critical", "High", "Medium", "Low"]

# Critical words: life-threatening situations (each adds +5 to the score).
CRITICAL_KEYWORDS = [
    "dying", "dead", "death", "unconscious", "not breathing", "stopped breathing",
    "cannot breathe", "struggling to breathe", "difficulty breathing", "no pulse",
    "turning blue", "choking", "choked", "heart attack", "cardiac arrest", "trapped",
    "buried", "severe injury", "severe injuries", "serious injury", "serious injuries",
    "seriously injured", "severely injured", "critically injured", "badly injured",
    "critical condition", "heavy bleeding", "bleeding heavily", "explosion", "blast",
    "fire", "burning", "blaze", "life threatening", "drowning", "swept away",
    "gunshot", "gunshots", "shot", "shooting", "stabbed", "kidnapped", "hostage",
    "bomb", "acid attack", "overdose", "suicide", "jump off", "electrocuted",
    "open well", "borewell", "immediately", "immediate",
]

# High words: serious but not always life-threatening (each adds +3).
HIGH_KEYWORDS = [
    "injured", "injury", "injuries", "hurt", "wounded", "bleeding", "accident",
    "crash", "collision", "collided", "rescue", "danger", "dangerous", "dangerously",
    "serious", "seriously", "severe", "severely", "urgent", "urgently", "hurry",
    "quickly", "as soon as possible", "asap", "attack", "attacked", "attacking",
    "robbery", "weapon", "weapons", "knife", "gun", "threat", "threatened",
    "threatening", "stalking", "harassing", "harassed", "snatched", "snatching",
    "kidnap", "gas leak", "smoke", "short circuit", "sparks", "collapsed",
    "collapse", "stranded", "drowned", "chest pain", "fracture", "broken leg",
    "fainted", "seizure", "stroke", "poisoning", "overturned", "hit and run",
    "evacuate", "evacuation", "panic", "panicking", "screaming", "break in",
    "break into", "breaking into", "broke into",
]

# Medium words: help is needed but the situation is not clearly dangerous (+2).
MEDIUM_KEYWORDS = [
    "need help", "help", "problem", "issue", "assistance", "water problem",
    "water logging", "waterlogging", "power cut", "stuck", "fever", "vomiting",
    "theft", "stolen", "missing", "suspicious", "complaint", "fight", "leak",
    "leaking", "nuisance", "blocked", "smell", "overflow", "overflowing",
]

# Topic words: they describe WHAT happened. Weight 0 (shown as keywords only).
TOPIC_KEYWORDS = [
    "flood", "flooded", "flooding", "earthquake", "tremor", "aftershock",
    "cyclone", "landslide", "storm", "ambulance", "police", "fire brigade",
    "fire service", "doctor", "doctors", "robbers", "thief", "murder",
    "harassment", "hospital", "well", "lift", "boat", "cylinder", "debris",
    "burnt", "burns", "pregnant", "labour", "breathing", "vomit", "collapsed building",
]

# Mild de-escalation words: the situation is probably small (-2 each).
MILD_LOW_KEYWORDS = [
    "minor", "small", "slight", "slightly", "mild", "not serious",
    "nothing serious", "no damage",
]
# Strong de-escalation words: it is over, or only a question (-4 each).
STRONG_LOW_KEYWORDS = [
    "under control", "extinguished", "put out", "false alarm", "safe now",
    "just asking", "just informing", "information", "enquiry", "inquiry",
    "query", "general", "advice",
]
LOW_KEYWORDS = MILD_LOW_KEYWORDS + STRONG_LOW_KEYWORDS

# Words that cancel the automatic "life-threat floor" (see calculate_priority).
FLOOR_CANCEL_TERMS = set(LOW_KEYWORDS) | {"slight", "slightly"}
# Critical words that do NOT by themselves prove a life threat.
NOT_A_LIFE_THREAT = {"immediately", "immediate"}

# Phrases that contain a critical word but do not describe a real emergency.
# They are removed before the priority is scored ("fire station" is not a fire).
NEUTRAL_PHRASES = [
    r"\bfire (?:station|extinguisher|extinguishers|safety|drill|alarm|certificate|noc)\b",
    r"\bsmell of burning\b",
    r"\bburning (?:garbage|waste|trash|leaves|rubbish)\b",
    r"\b(?:small|little|young|tiny)\s+(?=(?:child|children|boy|girl|baby|kid|kids|boys|girls)\b)",
]

# Phrases like "no one is injured" are replaced by this marker (score -1).
NO_DAMAGE_MARKER = "nodamage"
NEGATION_PATTERNS = [
    r"\b(?:no one|nobody|noone)\s+(?:is|was|got|were|are|has been|have been)?\s*(?:injured|hurt|trapped|affected)\b",
    r"\bno\s+(?:injuries|injury|casualties|casualty)\b",
    r"\bnot\s+(?:injured|hurt|trapped)\b",
]

# Base score for each category (a fire or earthquake is worse than "Other").
CATEGORY_BASE_SCORE = {
    "Accident": 2, "Fire": 2, "Flood": 2, "Medical Emergency": 2,
    "Earthquake": 2, "Crime": 2, "Rescue": 2, "Other": 0,
}

# Score needed for each priority level.
PRIORITY_THRESHOLDS = [("Critical", 14), ("High", 5), ("Medium", 2)]  # else "Low"


# ==========================================================================
# Helper: keyword matching
# ==========================================================================
def normalize_text(text: str) -> str:
    """Lowercase text and replace symbols by spaces, so matching is simple."""
    text = text.lower().replace("\u2019", "'").replace("can't", "cannot")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=None)
def _term_regex(term: str):
    """Regex for a word or phrase (allows a plural 's' / 'es' at the end)."""
    words = r"\s+".join(re.escape(w) for w in term.split())
    return re.compile(r"\b" + words + r"(?:s|es)?\b")


def find_terms(norm_text: str, terms: Dict[str, object]) -> List[Tuple[int, str, str, object]]:
    """
    Find every lexicon term in the text.

    Longer phrases are matched first and "used up", so "need help" is not
    counted again as "help" and "heart attack" is not counted as "attack".
    Returns a list of (position, term, matched_text, value) sorted by position.
    """
    working = norm_text
    hits = []
    for term in sorted(terms, key=lambda t: (-len(t), t)):
        for m in list(_term_regex(term).finditer(working)):
            hits.append((m.start(), term, m.group(), terms[term]))
            working = working[:m.start()] + " " * (m.end() - m.start()) + working[m.end():]
    return sorted(hits, key=lambda h: h[0])


def _apply_negations(norm_text: str) -> str:
    """Replace phrases such as 'no one is injured' with a neutral marker."""
    for pattern in NEGATION_PATTERNS:
        norm_text = re.sub(pattern, f" {NO_DAMAGE_MARKER} ", norm_text)
    return norm_text


def _keyword_weights() -> Dict[str, int]:
    """One dictionary: term -> weight (used by priority scoring & keywords)."""
    weights = {t: 5 for t in CRITICAL_KEYWORDS}
    weights.update({t: 3 for t in HIGH_KEYWORDS})
    weights.update({t: 2 for t in MEDIUM_KEYWORDS})
    weights.update({t: 0 for t in TOPIC_KEYWORDS})
    weights.update({t: -2 for t in MILD_LOW_KEYWORDS})
    weights.update({t: -4 for t in STRONG_LOW_KEYWORDS})
    weights[NO_DAMAGE_MARKER] = -1
    return weights


KEYWORD_WEIGHTS = _keyword_weights()


# ==========================================================================
# 2. NAMED ENTITY RECOGNITION (spaCy)
# ==========================================================================
ENTITY_LABEL_MAP = {
    "GPE": "Location", "LOC": "Location", "FAC": "Location",
    "ORG": "Organization", "PERSON": "Person", "DATE": "Date", "TIME": "Time",
}


def _empty_entities() -> Dict[str, List[str]]:
    return {"Location": [], "Organization": [], "Person": [], "Date": [], "Time": []}


def get_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract Location, Organization, Person, Date and Time entities with spaCy.
    If spaCy is not available, simple regular expressions find dates and times.
    This function never raises an error: on failure it returns empty lists.
    """
    entities = _empty_entities()
    if not isinstance(text, str) or not text.strip():
        return entities

    nlp = get_nlp()
    if nlp is not None:
        try:
            # spaCy NER works best with normal capitalisation. For all-lowercase
            # messages we also try a title-cased copy.
            candidates = [text]
            if text == text.lower():
                candidates.append(text.title())
            for candidate in candidates:
                doc = nlp(candidate, disable=["lemmatizer"])
                for ent in doc.ents:
                    label = ENTITY_LABEL_MAP.get(ent.label_)
                    if label and ent.text.strip() not in entities[label]:
                        # Ignore junk entities from the title-cased copy.
                        if candidate is not text and label != "Location":
                            continue
                        entities[label].append(ent.text.strip())
            return entities
        except Exception as exc:
            logger.warning("spaCy NER failed, using regex fallback: %s", exc)

    # ---- Fallback (no spaCy): only dates and times can be found by regex.
    for m in re.finditer(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\b(?:noon|midnight)\b", text, re.I):
        entities["Time"].append(m.group().strip())
    for m in re.finditer(
        r"\b(?:today|yesterday|tonight|tomorrow|this (?:morning|evening|afternoon))\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text, re.I):
        entities["Date"].append(m.group().strip())
    return entities


def entities_to_rows(entities: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Flatten the entity dictionary into table rows for the dashboard."""
    return [{"Entity": value, "Type": label}
            for label, values in entities.items() for value in values]


# ==========================================================================
# 3. LOCATION EXTRACTION
# ==========================================================================
# Words that usually end a location phrase ("... near the railway station").
PLACE_NOUNS = [
    "railway station", "railway junction", "bus stand", "bus stop", "bus station",
    "bus depot", "metro station", "railway gate", "railway crossing", "level crossing",
    "toll gate", "toll plaza", "petrol bunk", "petrol pump", "police station",
    "government hospital", "medical college", "shopping mall", "main road",
    "cross road", "cross street", "high way", "highway", "bypass road", "bypass",
    "flyover", "bridge", "junction", "signal", "road", "street", "lane", "nagar",
    "colony", "layout", "village", "town", "city", "district", "hospital",
    "clinic", "school", "college", "university", "campus", "hostel", "canteen",
    "market", "bazaar", "mall", "supermarket", "temple", "church", "mosque",
    "apartment", "apartments", "flats", "building", "tower", "complex", "factory",
    "godown", "warehouse", "shop", "restaurant", "hotel", "lodge", "theatre",
    "cinema", "stadium", "ground", "park", "garden", "beach", "lake", "river",
    "canal", "dam", "well", "pond", "forest", "hill", "airport", "port", "harbour",
    "office", "bank", "atm", "showroom", "hall", "house", "home", "terrace",
    "subway", "underpass", "square", "circle", "gate", "crossing", "stand",
    "textiles", "mill", "mills", "industries", "company", "pharmacy", "store", "stores",
    "station", "area", "locality", "neighbourhood", "neighborhood", "quarters",
]
_PLACE_ALT = "|".join(re.escape(p) for p in sorted(PLACE_NOUNS, key=len, reverse=True))
_PREPS = (r"at|near|in|on|opposite|outside|inside|beside|behind|around|from|to|towards|"
          r"nearby|by|along|across|next to|close to|infront of|in front of|of")
# Words that may NOT appear between the preposition and the place word.
_BLOCKED = (r"to|the|of|and|is|are|was|were|for|with|please|has|have|had|from|in|at|on|"
            r"near|by|but|or|that|which|who|since|because|after|before|when|while|then|"
            r"send|come|need|help|there|it|they|we|i")

# Pattern 1: preposition + (up to 3 name words) + place word
PLACE_PHRASE_RE = re.compile(
    r"(?i:\b(?:" + _PREPS + r")\b)\s+(?:(?i:the|a|an|our|my|this|that|their)\s+)?"
    r"((?:(?!(?i:" + _BLOCKED + r")\b)[A-Za-z0-9'\-]+\s+){0,3}?(?i:" + _PLACE_ALT + r")(?:\s+(?i:" + _PLACE_ALT + r"))?)\b"
)
# A city written right after the place: "Sunrise Apartments in Guindy"
TRAILING_CITY_RE = re.compile(r"\s*(?:,|\b(?:in|at|near|of)\b)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)")
# Pattern 2: highway numbers such as "NH 44"
HIGHWAY_RE = re.compile(r"\b(?:NH|SH)[\s\-]?\d{1,3}[A-Z]?\b", re.I)
# Pattern 3: preposition + Capitalised Words ("near Anna Nagar")
CAPITALISED_RE = re.compile(
    r"(?i:\b(?:at|near|in|on|opposite|outside|inside|beside|behind|around|from|to|"
    r"towards|nearby|by|along|across|next to|close to)\b)\s+(?:(?i:the|our|my)\s+)?"
    r"([A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*){0,3})"
)
# Pattern 4: preposition + up to 3 lowercase words (last resort)
LOWERCASE_RE = re.compile(
    r"\b(?:at|near|in|on|opposite|outside|inside|beside|behind|around|nearby|along|"
    r"across|next to|close to)\s+(?:(?:the|a|an|our|my)\s+)?"
    r"([a-z0-9'\-]+(?:\s+[a-z0-9'\-]+){0,2})"
)
# Capitalised words that are NOT part of a place name.
STOP_CAPS = {
    "Please", "Kindly", "Send", "Help", "Hurry", "Urgent", "There", "It", "I", "We",
    "They", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "One", "Many", "Several", "The", "A", "An", "And", "But", "Immediately",
    "Ambulance", "Police", "Fire", "Need", "Come", "Someone", "Also", "Now",
}
# Lowercase phrases that look like a place after "in/on/at" but are not.
NOT_PLACE_WORDS = {
    "danger", "pain", "fire", "trouble", "need", "critical", "serious", "condition",
    "urgent", "help", "time", "hurry", "panic", "emergency", "immediately", "now",
    "progress", "flames", "distress", "shock", "fear", "morning", "evening",
    "night", "afternoon", "today", "yesterday", "tonight", "hours", "minutes",
    "minute", "hour", "day", "days", "week", "case", "front", "a", "an", "the",
    "very", "huge", "big", "heavy", "severe", "bad", "great", "large", "our", "my",
    "this", "that", "his", "her", "their", "some", "any", "many", "all", "both",
}
BOUNDARY_WORDS = {
    "and", "but", "please", "with", "where", "who", "which", "since", "because",
    "after", "is", "are", "was", "were", "has", "have", "had", "need", "needs",
    "send", "come", "help", "people", "persons", "person", "someone", "i", "we",
    "they", "it", "there", "that", "urgent", "immediately", "now", "also", "so",
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "one",
    "many", "several", "kindly", "hurry", "asap", "as", "because", "or", "while",
}


def _format_place(place: str, whole_text_is_upper: bool) -> str:
    """Nice capitalisation: 'madurai bus stand' -> 'Madurai Bus Stand'."""
    words = []
    for w in place.split():
        if w.isupper() and len(w) <= 4 and not whole_text_is_upper:
            words.append(w)                    # keep short acronyms such as NH, KCET
        else:
            words.append(w.capitalize())
    return " ".join(words)


def extract_location(text: str, entities: Optional[Dict[str, List[str]]] = None) -> Tuple[str, str]:
    """
    Find where the emergency is happening.
    Returns (location, method). If nothing is found: ("Not specified", "none").

    Order of attempts:
        1. Preposition + place word rule  ("near the railway station") and,
           when spaCy found a city (GPE) that is not in the phrase, we add it.
        2. Highway number (NH 44)
        3. spaCy Location entities (GPE / LOC / FAC)
        4. Preposition + Capitalised words ("near Anna Nagar")
        5. Preposition + lowercase words (last resort)
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return "Not specified", "none"
        entities = entities or _empty_entities()
        upper = text.isupper()
        spacy_locations = entities.get("Location", [])

        # 1. place-word rule ------------------------------------------------
        #    Several phrases may match; the one that contains a proper name
        #    ("Sunrise Apartments") beats a generic one ("apartment building").
        best = None
        for m in PLACE_PHRASE_RE.finditer(text):
            named = any(w[:1].isupper() for w in m.group(1).split())
            if best is None or (named and not best[0]):
                best = (named, m)
        if best:
            m = best[1]
            place = _format_place(m.group(1).strip(), upper)
            trailing = TRAILING_CITY_RE.match(text[m.end():])
            if trailing:
                city = trailing.group(1).split()
                while city and city[-1] in STOP_CAPS:
                    city.pop()
                if city and city[0] not in STOP_CAPS and " ".join(city).lower() not in place.lower():
                    place = f"{place}, {_format_place(' '.join(city), upper)}"
            for ent in spacy_locations:              # add a city if missing
                if ent.lower() not in place.lower() and ent.lower() not in NOT_PLACE_WORDS \
                        and ent[:1].isupper() and len(ent) > 2:
                    place = f"{place}, {_format_place(ent, upper)}"
                    break
            return place, "place-word rule + spaCy" if spacy_locations else "place-word rule"

        # 2. highway number -------------------------------------------------
        m = HIGHWAY_RE.search(text)
        if m:
            return m.group().upper().replace("-", " "), "highway rule"

        # 3. spaCy entities -------------------------------------------------
        for ent in spacy_locations:
            if ent.lower() not in NOT_PLACE_WORDS and len(ent) > 2:
                return _format_place(ent, upper), "spaCy NER"

        # 4. capitalised words after a preposition -----------------------------
        source = text.title() if upper else text
        m = CAPITALISED_RE.search(source)
        if m:
            words = []
            for w in m.group(1).split():
                if w in STOP_CAPS:
                    break
                words.append(w)
            if words:
                return _format_place(" ".join(words), upper), "capitalised-word fallback"

        # 5. lowercase words after a preposition -----------------------------
        m = LOWERCASE_RE.search(text.lower())
        if m:
            words = []
            for w in m.group(1).split():
                if w in BOUNDARY_WORDS:
                    break
                words.append(w)
            if words and words[0] not in NOT_PLACE_WORDS and not words[0].isdigit():
                return _format_place(" ".join(words), upper), "keyword fallback"
    except Exception as exc:
        logger.warning("Location extraction failed: %s", exc)
    return "Not specified", "none"


def reconcile_entities(entities: Dict[str, List[str]], location: str) -> Dict[str, List[str]]:
    """
    The small spaCy model often labels Indian place names as PERSON or ORG
    ("Anna Nagar" -> Person). If such an entity is part of the location we
    found, it is moved to the Location list.
    """
    fixed = {label: list(values) for label, values in entities.items()}
    loc_lower = location.lower()
    if loc_lower == "not specified":
        return fixed
    for label in ("Person", "Organization"):
        for value in list(fixed[label]):
            if value.lower() in loc_lower:
                fixed[label].remove(value)
                if value not in fixed["Location"]:
                    fixed["Location"].append(value)
    return fixed


# ==========================================================================
# 4. PEOPLE AFFECTED
# ==========================================================================
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19, "dozen": 12}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}
_NUM = (r"(?:\d{1,4}|(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+)?"
        r"(?:one|two|three|four|five|six|seven|eight|nine)|" +
        "|".join(list(_TEENS) + list(_TENS)) + r")")
_PEOPLE_NOUNS = (r"(?:people|persons|person|victims|victim|students|student|children|child|"
                 r"kids|kid|passengers|passenger|workers|worker|labourers|laborers|men|man|"
                 r"women|woman|patients|patient|villagers|residents|families|family|"
                 r"members|employees|tourists|pilgrims|devotees|elders|girls|boys|"
                 r"travellers|travelers|youths|females|males|pedestrians|riders|drivers)")
_STATES = (r"(?:injured|hurt|dead|trapped|stranded|missing|unconscious|affected|bleeding|"
           r"killed|dying|sick|drowned|burnt|wounded|stuck|buried|fainted)")
_FILLER = r"(?:(?!" + _NUM + r"\b)[a-z]+\s+){0,2}?"

_RE_NUM_PEOPLE = re.compile(r"\b(" + _NUM + r")\s+" + _FILLER + _PEOPLE_NOUNS + r"\b")
_RE_NUM_STATE = re.compile(
    r"\b(" + _NUM + r")\s+(?:(?:are|were|is|was|have been|has been|got)\s+)?"
    r"(?:(?:seriously|severely|critically|badly|still)\s+)?" + _STATES + r"\b")
_RE_FAMILY_OF = re.compile(r"\b(?:family|group|team|party) of (" + _NUM + r")\b")
_RE_COUPLE = re.compile(r"\b(?:a couple of|couple of)\b")
_RE_QUANTIFIER = re.compile(
    r"\b(several|many|multiple|numerous|dozens of|hundreds of|a few|crowd of|large number of)\s+"
    + _FILLER + r"(?:" + _PEOPLE_NOUNS + r"|" + _STATES + r")\b")
_SINGULAR_NOUNS = (r"(?:man|woman|child|boy|girl|person|baby|patient|driver|rider|passenger|"
                   r"student|worker|kid|lady|gentleman|farmer|pedestrian|infant|toddler|"
                   r"youth|labourer|laborer|victim|shopkeeper|watchman|guard)")
_RE_SINGULAR = re.compile(r"\b(?:a|an|one)\s+(?:(?!" + _NUM + r"\b)[a-z]+\s+){0,2}?" + _SINGULAR_NOUNS + r"\b")
_RE_PRONOUN = re.compile(r"\b(?:he|she)\s+(?:is|was|has|had|feels|fell|cannot|can not)\b")
_RE_SOMEONE = re.compile(
    r"\b(?:someone|somebody|my (?:father|mother|son|daughter|husband|wife|brother|sister|"
    r"friend|grandfather|grandmother|grandmom|grandpa|grandma|baby|child|uncle|aunt|neighbour|neighbor)"
    r"|our (?:neighbour|neighbor|driver|watchman))\b")
_QUANT_HINT = {"several": 3, "many": 5, "multiple": 3, "numerous": 5, "dozens of": 24,
               "hundreds of": 200, "a few": 3, "crowd of": 20, "large number of": 20}


def _words_to_number(chunk: str) -> Optional[int]:
    """Convert '3', 'three' or 'twenty five' into an integer."""
    chunk = chunk.strip().lower().replace("-", " ")
    if chunk.isdigit():
        return int(chunk)
    total = 0
    for word in chunk.split():
        total += {**_UNITS, **_TEENS, **_TENS}.get(word, 0)
    return total or None


def extract_people_affected(text: str) -> Dict[str, object]:
    """
    Find how many people are affected.
    Returns {"count": int or None, "label": "3" / "Several" / "Not mentioned",
             "score_hint": number used by the priority scorer}.
    """
    result = {"count": None, "label": "Not mentioned", "score_hint": 0}
    try:
        norm = _apply_negations(normalize_text(text))
        numbers: List[int] = []
        for regex in (_RE_NUM_PEOPLE, _RE_NUM_STATE, _RE_FAMILY_OF):
            for m in regex.finditer(norm):
                value = _words_to_number(m.group(1))
                if value and value < 10000:
                    numbers.append(value)
        if _RE_COUPLE.search(norm):
            numbers.append(2)

        if numbers:                                  # use the biggest number found
            count = max(numbers)
            result.update(count=count, label=str(count), score_hint=count)
            return result

        if _RE_SINGULAR.search(norm) or _RE_SOMEONE.search(norm) or _RE_PRONOUN.search(norm):
            result.update(count=1, label="1", score_hint=1)
            return result

        m = _RE_QUANTIFIER.search(norm)
        if m:
            word = m.group(1)
            result.update(count=None, label=word.replace(" of", "").capitalize(),
                          score_hint=_QUANT_HINT.get(word, 3))
    except Exception as exc:
        logger.warning("People extraction failed: %s", exc)
    return result


# ==========================================================================
# 5. REQUIRED SERVICE DETECTION
# ==========================================================================
SERVICE_ORDER = ["Ambulance", "Fire Service", "Police", "Rescue Team",
                 "Medical Team", "Disaster Management Team"]

# Words that show which service is needed. Both "explicit requests"
# (the user names the service) and "implicit clues" live in one table.
SERVICE_KEYWORDS = {
    "Ambulance": [
        "ambulance", "paramedic", "paramedics", "injured", "injury", "injuries", "bleeding",
        "unconscious", "fainted", "not breathing", "stopped breathing", "cannot breathe",
        "heart attack", "cardiac arrest", "chest pain", "stroke", "seizure", "wounded",
        "fracture", "broken leg", "burnt", "burns", "overdose", "suicide", "stabbed",
        "gunshot", "gunshots", "hurt", "critical condition", "in labour", "labour pain",
        "seriously injured", "severely injured", "critically injured", "badly injured",
    ],
    "Fire Service": [
        "fire rescue team", "fire rescue", "fire service", "fire brigade", "fire engine",
        "fire truck", "fire station", "firefighter", "firefighters", "fire fighters",
        "fire", "smoke", "blaze", "burning", "flames", "explosion", "blast", "gas leak",
        "short circuit", "cylinder blast",
    ],
    "Police": [
        "police", "cops", "patrol", "robbery", "robbed", "robbers", "theft", "thief",
        "stolen", "murder", "attack", "attacked", "weapon", "weapons", "knife", "gun",
        "kidnap", "kidnapped", "harassment", "harassing", "assault", "threatening",
        "fighting", "riot", "criminal", "suspicious", "hit and run", "break in",
        "broke into", "snatching", "gang", "drunk",
    ],
    "Rescue Team": [
        "rescue team", "rescue teams", "rescue squad", "rescuers", "ndrf", "sdrf",
        "rescue boat", "rescue", "trapped", "stuck", "stranded", "drowning", "drowned",
        "buried", "building collapsed", "wall collapsed", "roof collapsed",
        "house collapsed", "bridge collapsed", "building collapse", "collapsed building",
        "swept away", "fell into", "fallen into", "landslide", "lost in the forest",
    ],
    "Medical Team": [
        "medical team", "medical help", "medical camp", "doctor", "doctors", "nurse",
        "fever", "vomiting", "food poisoning", "poisoning", "epidemic", "outbreak",
        "dehydration", "diarrhoea", "diarrhea", "infection", "oxygen", "first aid",
        "medicine", "medicines", "diabetic", "sugar level",
    ],
    "Disaster Management Team": [
        "disaster management", "disaster team", "disaster response", "ndma", "relief team",
        "flood", "flooded", "flooding", "earthquake", "cyclone", "tsunami", "disaster",
        "evacuate", "evacuation", "tremor", "aftershock", "storm", "water level",
    ],
}
# If the ML model says the category is X, this service is always needed.
CATEGORY_DEFAULT_SERVICES = {
    "Accident": ["Ambulance"],
    "Fire": ["Fire Service"],
    "Flood": ["Disaster Management Team"],
    "Medical Emergency": ["Ambulance"],
    "Earthquake": ["Disaster Management Team"],
    "Crime": ["Police"],
    "Rescue": ["Rescue Team"],
    "Other": [],
}


class ServiceDetector:
    """
    Base class. Any service detector must offer detect(text, category, priority).
    To use a machine-learning model later, write a new class (for example
    MLServiceDetector) that inherits from this class and pass an object of it
    to EmergencyAnalyzer(service_detector=...). Nothing else has to change.
    """

    def detect(self, text: str, category: str, priority: Optional[str] = None) -> List[str]:
        raise NotImplementedError("Subclasses must implement detect().")


class RuleBasedServiceDetector(ServiceDetector):
    """Keyword/rule based detector: keywords + default service of the category."""

    def __init__(self):
        # term -> service (flattened table)
        self.term_to_service = {t: s for s, terms in SERVICE_KEYWORDS.items() for t in terms}

    def detect(self, text: str, category: str, priority: Optional[str] = None) -> List[str]:
        services = set(CATEGORY_DEFAULT_SERVICES.get(category, []))
        try:
            norm = _apply_negations(normalize_text(text))
            for _, _, _, service in find_terms(norm, self.term_to_service):
                services.add(service)
        except Exception as exc:
            logger.warning("Service detection failed: %s", exc)
        return [s for s in SERVICE_ORDER if s in services]


# ==========================================================================
# 6. KEYWORD EXTRACTION
# ==========================================================================
def extract_keywords(text: str, processed_text: str = "", vectorizer=None,
                     max_keywords: int = 10, exclude_text: str = "") -> List[str]:
    """
    Return the important words of a message.
    Step 1: emergency words from our lexicons (in the order they appear).
    Step 2: if fewer than 3 were found, add the highest-IDF words known to
            the TF-IDF vectorizer (place names excluded) so the list is never empty.
    """
    keywords: List[str] = []
    try:
        norm = _apply_negations(normalize_text(text))
        for _, _, surface, weight in find_terms(norm, KEYWORD_WEIGHTS):
            if weight >= 0 and surface not in keywords and surface != NO_DAMAGE_MARKER:
                keywords.append(surface)

        if len(keywords) < 3 and vectorizer is not None and processed_text:
            vocab = getattr(vectorizer, "vocabulary_", {})
            idf = getattr(vectorizer, "idf_", None)
            if idf is not None:
                skip = set(normalize_text(exclude_text).split())   # place names are not keywords
                candidates = [(idf[vocab[w]], w) for w in set(processed_text.split())
                              if w in vocab and len(w) > 3 and not w.isdigit() and w not in skip]
                for _, word in sorted(candidates, reverse=True):
                    if word not in keywords and len(keywords) < 5:
                        keywords.append(word)
    except Exception as exc:
        logger.warning("Keyword extraction failed: %s", exc)
    return keywords[:max_keywords]


# ==========================================================================
# 7. PRIORITY DETECTION
# ==========================================================================
def _prepare_for_scoring(text: str) -> str:
    """Normalise the text and remove phrases that only look like emergencies."""
    norm = _apply_negations(normalize_text(text))
    for pattern in NEUTRAL_PHRASES:
        norm = re.sub(pattern, " ", norm)
    return norm


def calculate_priority(text: str, category: str, people_hint: int = 0) -> Dict[str, object]:
    """
    Priority scoring - it does NOT rely on keywords only.

        score = category base score                      (0 - 2)
              + critical keywords  (+5 each, max +15)
              + high keywords      (+3 each, max +6)
              + medium keywords    (+2 each, max +2)
              - de-escalation words such as "minor" (-2) or "under control" (-4)
              + number of people affected (2-4 people +2, 5 or more +3)

    Then:  score >= 14 Critical | >= 5 High | >= 2 Medium | else Low.
    Safety rule: a real life-threat word (fire, trapped, unconscious, explosion...)
    makes the priority at least Critical unless the message also says the
    situation is small / minor / under control / only a question.
    """
    reasons: List[str] = []
    try:
        norm = _prepare_for_scoring(text)
        # Each keyword is counted once, even if the message repeats it.
        hits = list({h[1]: h for h in find_terms(norm, KEYWORD_WEIGHTS)}.values())

        critical = [h[2] for h in hits if h[3] == 5]
        high = [h[2] for h in hits if h[3] == 3]
        medium = [h[2] for h in hits if h[3] == 2]
        low = [(h[2], -h[3]) for h in hits if h[3] < -1]
        no_damage = any(h[1] == NO_DAMAGE_MARKER for h in hits)

        base = CATEGORY_BASE_SCORE.get(category, 0)
        score = base
        reasons.append(f"Category '{category}' base score: +{base}")

        if critical:
            points = min(5 * len(critical), 15)
            score += points
            reasons.append(f"Critical keywords ({', '.join(dict.fromkeys(critical))}): +{points}")
        if high:
            points = min(3 * len(high), 6)
            score += points
            reasons.append(f"High-priority keywords ({', '.join(dict.fromkeys(high))}): +{points}")
        if medium:
            points = 2
            score += points
            reasons.append(f"Medium keywords ({', '.join(dict.fromkeys(medium))}): +{points}")
        if low:
            points = min(sum(p for _, p in low), 8)
            score -= points
            reasons.append(f"Minor / informational words ({', '.join(dict.fromkeys(w for w, _ in low))}): -{points}")
        if no_damage:
            score -= 1
            reasons.append("Message says nobody is injured: -1")

        if people_hint >= 5:
            score += 3
            reasons.append(f"Many people affected ({people_hint}): +3")
        elif people_hint >= 2:
            score += 2
            reasons.append(f"{people_hint} people affected: +2")

        score = max(score, 0)
        priority = "Low"
        for level, minimum in PRIORITY_THRESHOLDS:
            if score >= minimum:
                priority = level
                break

        # Safety rule ("life-threat floor")
        life_threats = [h[2] for h in hits if h[3] == 5 and h[1] not in NOT_A_LIFE_THREAT]
        cancelled = any(h[1] in FLOOR_CANCEL_TERMS for h in hits)
        if life_threats and not cancelled and priority != "Critical":
            priority = "Critical"
            reasons.append(f"Safety rule: life-threat word '{life_threats[0]}' -> at least Critical")

        return {"priority": priority, "score": score, "reasons": reasons}
    except Exception as exc:
        logger.warning("Priority calculation failed: %s", exc)
        return {"priority": "Medium", "score": 0,
                "reasons": ["Priority could not be calculated - defaulting to Medium."]}


# ==========================================================================
# 8. RECOMMENDED RESPONSE
# ==========================================================================
URGENCY_TEXT = {
    "Critical": "Dispatch immediately and keep the caller on the line.",
    "High": "Dispatch as soon as possible and confirm the details with the caller.",
    "Medium": "Schedule a response and monitor the situation.",
    "Low": "Log the message and give guidance or information. No dispatch needed now.",
}
PLAYBOOK = {
    "Accident": "Send {services}. Ask the caller to keep injured people still, not to move them unless there is danger, and to keep the road clear.",
    "Fire": "Send {services}. Ask everyone to leave the building, stay low under smoke, avoid lifts and keep away from the fire.",
    "Flood": "Alert {services}. Ask people to move to higher ground, switch off electricity at the mains and avoid walking through flood water.",
    "Medical Emergency": "Send {services}. Keep the patient calm and lying down, check breathing and follow the first-aid guidance of the dispatcher.",
    "Earthquake": "Alert {services}. Advise people to stay away from damaged buildings, protect their heads and expect aftershocks.",
    "Crime": "Send {services}. Ask the caller to move to a safe place, not to confront the offender and to note descriptions or vehicle numbers.",
    "Rescue": "Send {services}. Ask the caller not to attempt a risky rescue alone and to stay in contact.",
    "Other": "Send {services}. Route the message to a human dispatcher and confirm the details with the caller.",
}


def get_recommended_response(category: str, priority: str, services: List[str]) -> str:
    """Build a short, human-readable recommended response."""
    service_text = " + ".join(services) if services else "the appropriate local team"
    playbook = PLAYBOOK.get(category, PLAYBOOK["Other"]).format(services=service_text)
    return f"{URGENCY_TEXT.get(priority, '')} {playbook}"


# ==========================================================================
# Sample messages (used by the dashboard and the README)
# ==========================================================================
SAMPLE_MESSAGES = {
    "Road accident": "There is a major accident near the railway station. Two people are seriously injured. Please send an ambulance immediately.",
    "Fire accident": "There is a fire accident at Madurai bus stand. Three people are injured. Please send an ambulance and fire rescue team immediately.",
    "Flood": "Flood water has entered our street in Velachery. Four people are stranded on the first floor. Please send a rescue team.",
    "Medical emergency": "My father has severe chest pain and is not breathing properly. Please send an ambulance to Anna Nagar immediately.",
    "Earthquake": "Strong earthquake at Bhuj. A building has collapsed and several people are trapped under the debris. Send help immediately.",
    "Crime": "A man is threatening shopkeepers with a knife near the Trichy bus stand. Please send the police urgently.",
    "Rescue": "A child has fallen into an open well near the village school. Please send a rescue team immediately.",
    "Minor / general": "Street light is not working near the temple. Please repair it when possible.",
    "No location given": "Someone is unconscious and bleeding. Please send help immediately.",
    "No people mentioned": "Fire in our apartment building, smoke is spreading to all floors of Sunrise Apartments in Guindy.",
}


# ==========================================================================
# 9. THE COMPLETE ANALYZER
# ==========================================================================
class EmergencyAnalyzer:
    """
    Runs the complete pipeline:
    clean -> tokenize -> stop words -> lemmas -> TF-IDF -> classifier ->
    NER -> location -> people -> priority -> service -> keywords -> response.
    """

    LOW_CONFIDENCE = 0.40  # below this we warn the user

    def __init__(self, model=None, vectorizer=None, service_detector: Optional[ServiceDetector] = None):
        if model is None or vectorizer is None:
            # Imported here (not at the top) to avoid a circular import.
            from train_model import load_or_train_model
            model, vectorizer = load_or_train_model()
        self.model = model
        self.vectorizer = vectorizer
        self.service_detector = service_detector or RuleBasedServiceDetector()

    # ---- ML classification ------------------------------------------------
    def predict_category(self, processed_text: str) -> Tuple[str, float, Dict[str, float]]:
        """Return (category, confidence 0-1, probability of every category)."""
        features = self.vectorizer.transform([processed_text])
        if features.nnz == 0:
            # None of the words are known to the model.
            return "Other", 0.0, {}
        probabilities = self.model.predict_proba(features)[0]
        best = int(probabilities.argmax())
        classes = list(self.model.classes_)
        return classes[best], float(probabilities[best]), \
            {c: float(p) for c, p in zip(classes, probabilities)}

    # ---- main entry -----------------------------------------------------------
    def analyze(self, message: str) -> Dict[str, object]:
        """Analyse one message. Always returns a dict; check result['success']."""
        valid, error = is_valid_message(message)
        if not valid:
            return {"success": False, "error": error}
        try:
            message = message.strip()
            steps = preprocess_steps(message)
            processed = steps["Final text for TF-IDF"]

            category, confidence, probabilities = self.predict_category(processed)
            entities = get_entities(message)
            location, location_method = extract_location(message, entities)
            entities = reconcile_entities(entities, location)
            people = extract_people_affected(message)
            priority = calculate_priority(message, category, int(people["score_hint"]))
            services = self.service_detector.detect(message, category, priority["priority"])
            keywords = extract_keywords(message, processed, self.vectorizer, exclude_text=location)

            warnings = []
            if confidence < self.LOW_CONFIDENCE:
                warnings.append("Low model confidence - please review this message manually.")
            if not probabilities:
                warnings.append("The message contains no words known to the model.")

            return {
                "success": True,
                "message": message,
                "emergency_type": category,
                "priority": priority["priority"],
                "priority_score": priority["score"],
                "priority_reasons": priority["reasons"],
                "location": location,
                "location_method": location_method,
                "people_affected": people["label"],
                "people_count": people["count"],
                "required_service": " + ".join(services) if services else "Not determined (dispatcher review)",
                "services": services,
                "confidence": round(confidence * 100, 1),
                "keywords": keywords,
                "entities": entities,
                "recommended_response": get_recommended_response(category, priority["priority"], services),
                "probabilities": probabilities,
                "pipeline_steps": steps,
                "warnings": warnings,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as exc:
            logger.exception("Analysis failed")
            return {"success": False, "error": f"Sorry, the message could not be analysed ({exc})."}
