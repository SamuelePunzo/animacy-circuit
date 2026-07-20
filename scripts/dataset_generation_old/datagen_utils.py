from collections import defaultdict
from transformers import AutoTokenizer
import re
from lemminflect import getInflection


IRREGULAR_VBN = {
    "arise": "arisen",
    "awake": "awoken",
    "be": "been",
    "bear": "borne",
    "beat": "beaten",
    "become": "become",
    "begin": "begun",
    "bend": "bent",
    "bet": "bet",
    "bind": "bound",
    "bite": "bitten",
    "bleed": "bled",
    "blow": "blown",
    "break": "broken",
    "bring": "brought",
    "build": "built",
    "buy": "bought",
    "catch": "caught",
    "choose": "chosen",
    "come": "come",
    "cost": "cost",
    "cut": "cut",
    "do": "done",
    "draw": "drawn",
    "drink": "drunk",
    "drive": "driven",
    "eat": "eaten",
    "fall": "fallen",
    "feed": "fed",
    "feel": "felt",
    "fight": "fought",
    "find": "found",
    "fly": "flown",
    "forget": "forgotten",
    "freeze": "frozen",
    "get": "gotten",
    "give": "given",
    "go": "gone",
    "grow": "grown",
    "hang": "hung",
    "have": "had",
    "hear": "heard",
    "hide": "hidden",
    "hold": "held",
    "keep": "kept",
    "know": "known",
    "lead": "led",
    "leave": "left",
    "lose": "lost",
    "make": "made",
    "meet": "met",
    "pay": "paid",
    "put": "put",
    "read": "read",
    "ride": "ridden",
    "ring": "rung",
    "rise": "risen",
    "run": "run",
    "say": "said",
    "see": "seen",
    "sell": "sold",
    "send": "sent",
    "set": "set",
    "shake": "shaken",
    "shine": "shone",
    "shoot": "shot",
    "show": "shown",
    "shut": "shut",
    "sing": "sung",
    "sink": "sunk",
    "sit": "sat",
    "sleep": "slept",
    "speak": "spoken",
    "spend": "spent",
    "split": "split",
    "stand": "stood",
    "steal": "stolen",
    "swear": "sworn",
    "swim": "swum",
    "take": "taken",
    "teach": "taught",
    "tear": "torn",
    "tell": "told",
    "think": "thought",
    "throw": "thrown",
    "understand": "understood",
    "wear": "worn",
    "win": "won",
    "write": "written",
}

CORRUPT_BLOCKLIST = {
    "distilled", "varnished", "japanned", "enamelled",
    "schlepped", "lugged", "towed", "hauled",
}

PHYSICAL_TYPES = {
    "concrete",
    "solid",
    "substance",
    "location",
    "artifact",
    "machine",
    "vehicle",
    "body_part",
}

def tok_len(word: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer.encode(" " + word))


def group_by_tok(words: list[str], tokenizer: AutoTokenizer) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for word in words:
        grouped[tok_len(word, tokenizer)].append(word)
    return {k: sorted(set(v)) for k, v in sorted(grouped.items())}

#VerbNet is stored as XML. Each verb class has thematic roles (Agent, Patient, Theme, Cause…) and each role can have selectional restrictions — constraints like +animate, -animate, +artifact. 
# The following four functions form a small parsing stack: 
#   - collect_selrestrs reads the XML nodes, 
#   - class_role_restrictions organizes them per role,
#   - has_restriction lets you ask "does the Agent of this class require +animate?".

# cleans raw VerbNet member strings
def normalize_member_name(name: str | None) -> str | None:
    if not name:
        return None
    norm = name.strip().lower().replace("_", " ")
    norm = re.sub(r"[^a-z\s-]", "", norm)
    if " " in norm or "-" in norm or not norm:
        return None
    return norm

# Recursively extracts selectiona restrictions
def collect_selrestrs(node) -> list[tuple[str, str]]:
    if node is None:
        return []
    out: list[tuple[str, str]] = []
    for sel in node.findall("SELRESTR"):
        out.append((sel.attrib.get("Value", ""), sel.attrib.get("type", "").lower()))
    for nested in node.findall("SELRESTRS"):
        out.extend(collect_selrestrs(nested))
    return out

# Maps thamtic roles to their selectional restrictions for a given VerbNet class
def class_role_restrictions(vn_class_xml) -> dict[str, list[tuple[str, str]]]:
    role_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for role in vn_class_xml.findall("THEMROLES/THEMROLE"):
        role_name = (role.attrib.get("type") or "").lower()
        role_map[role_name].extend(collect_selrestrs(role.find("SELRESTRS")))
    return role_map

#Quries the map of role restrictions for a given role name, value, and type
def has_restriction(
    role_map: dict[str, list[tuple[str, str]]],
    role_name: str,
    value: str,
    restr_type: str,
) -> bool:
    return any(v == value and t == restr_type for v, t in role_map.get(role_name, []))

# returns the past participle form of a verb lemma, using lemminflect if available, 
# otherwise applying common English rules and handling irregular verbs.
def to_past_participle(lemma: str) -> str:
    forms = getInflection(lemma, tag="VBN")
    if forms:
        return forms[0].lower()
    # fallback for any lemma lemminflect doesn't recognize
    if lemma in IRREGULAR_VBN:
        return IRREGULAR_VBN[lemma]
    return lemma + "ed"

def is_transitive_verb(lemma: str) -> bool:
    """Return True if WordNet annotates this lemma with a core transitive frame."""
    # 8: Somebody ----s something
    # 9: Somebody ----s somebody
    # 10: Something ----s somebody
    # 11: Something ----s something
    # (Removed 26, which let in clausal complement verbs like 'complain' or 'think')
    TRANSITIVE_FRAME_IDS = {8, 9, 10, 11}
    
    for synset in wn.synsets(lemma, pos=wn.VERB):
        for wn_lemma in synset.lemmas():
            if wn_lemma.name().lower().replace("_", " ") == lemma.lower():
                if any(fid in TRANSITIVE_FRAME_IDS for fid in wn_lemma.frame_ids()):
                    return True
    return False


def has_transitive_frame(vnclass) -> bool:
    """Return True if the VerbNet class has a direct transitive NP V NP frame."""
    # 1. Check formal descriptions first (highly reliable in VerbNet)
    for frame in vnclass.findall("FRAMES/FRAME/DESCRIPTION"):
        primary = frame.attrib.get("primary", "")
        if "NP V NP" in primary:
            return True

    # 2. Fallback to manual SYNTAX parsing
    for frame in vnclass.findall("FRAMES/FRAME/SYNTAX"):
        nodes = frame.findall("*")
        types = [node.tag for node in nodes]
        
        if "VERB" in types:
            verb_idx = types.index("VERB")
            post_verb_types = types[verb_idx + 1:]
            
            if "NP" in post_verb_types:
                np_idx_after_verb = post_verb_types.index("NP")
                
                # Check if there is a PREP between the VERB and the NP
                # If there is, this is likely a prepositional object, not a direct object.
                if "PREP" not in post_verb_types[:np_idx_after_verb]:
                    # Ensure there was also an NP before the VERB (the subject)
                    if "NP" in types[:verb_idx]:
                        return True
                        
    return False

#The folloeing three functions define what makes a verb "clean" 
# (requires animate/human agent → e.g. "The soldier was praised by the general") 
# vs. 
# "corrupt" (inanimate cause → e.g. "The building was destroyed by the earthquake"). 
# The fallback is needed because VerbNet 3.4 often doesn't explicitly annotate Cause with -animate, 
# so it infers it from physical Patient/Theme restrictions instead.

# Agent has +animate or +human -> clean
def is_clean_class(role_map: dict[str, list[tuple[str, str]]]) -> bool:
    return has_restriction(role_map, "agent", "+", "animate") or has_restriction(
        role_map, "agent", "+", "human"
    )

# Cause has -animate -> corrupt (strict)
def is_corrupt_class_strict(role_map: dict[str, list[tuple[str, str]]]) -> bool:
    return has_restriction(role_map, "cause", "-", "animate")

# This is a heuristic fallback for classes that don't meet the strict corrupt condition but still have strong signals of non-animacy, 
# based on a combination of role restrictions and the presence of a Cause role.
def is_corrupt_class_fallback(role_map):
    has_cause_role = "cause" in role_map

    patient_or_theme_physical = False
    for role_name in ("patient", "theme"):
        for value, typ in role_map.get(role_name, []):
            if (value == "-" and typ == "animate") or (value == "+" and typ in PHYSICAL_TYPES):
                patient_or_theme_physical = True

    instrument_physical = any(
        value == "+" and typ in {"concrete", "solid", "artifact"}
        for value, typ in role_map.get("instrument", [])
    )

    agent_intentional = any(
        value == "+" and typ in {"int_control", "animate", "human"}
        for value, typ in role_map.get("agent", [])
    )

    return patient_or_theme_physical and (has_cause_role or instrument_physical or agent_intentional)

# Maps an entire {lemma → VerbNet classes} dict into {past_participle → VerbNet classes}, 
# dropping any lemma whose participle contains non-alpha characters.
def convert_pool(lemma_to_classes: dict[str, set[str]]) -> tuple[dict[str, set[str]], list[dict[str, str]]]:
    participle_to_classes: dict[str, set[str]] = defaultdict(set)
    dropped: list[dict[str, str]] = []

    for lemma, class_ids in lemma_to_classes.items():
        participle = to_past_participle(lemma)

        if not re.fullmatch(r"[a-z]+", participle):
            dropped.append({"lemma": lemma, "reason": "non_alpha_participle", "candidate": participle})
            continue

        participle_to_classes[participle].update(class_ids)

    return participle_to_classes, dropped

#----------------------------------------------------------------------------
# The following are utility functions for the noun pool generation
#----------------------------------------------------------------------------

from nltk.corpus import wordnet as wn
import inflect
from better_profanity import profanity

profanity.load_censor_words()

# The keywords below are used to filter out artifact synsets that are likely 
# to be abstract rather than concrete, based on their definitions.
ABSTRACT_ARTIFACT_KEYWORDS = {
    "software",
    "program",
    "document",
    "record",
    "symbol",
    "idea",
    "concept",
    "theory",
    "plan",
    "method",
    "formula",
    "message",
    "language",
    "code",
}

# The ARTIFACT_PHYSICAL_ANCHORS are the WordNet synsets that serve as roots for the artifact subtree.
ARTIFACT_PHYSICAL_ANCHORS = [
    "vehicle.n.01",
    "structure.n.01",
    "tool.n.01",
    "instrumentality.n.03",
    "container.n.01",
    "weapon.n.01",
]

#This is an iterative depth-first traversal of WordNet's hyponym tree. 
# WordNet is organized as a DAG where each synset can have many specific subtypes 
# (hyponyms). Starting from e.g. person.n.01, it walks down to collect every more-specific
#  concept (doctor, child, athlete, etc.). 
# The seen set prevents revisiting synsets that appear in multiple paths
def all_hyponyms(root_synset):
    seen = set()
    stack = [root_synset]
    while stack:
        syn = stack.pop()
        if syn in seen:
            continue
        seen.add(syn)
        stack.extend(syn.hyponyms())
    return seen

#WordNet stores lemma names like "police_officer" or "free-agent" 
# normalize_lemma_name filters those out, keeping only single lowercase alpha tokens. 
def normalize_lemma_name(lemma_name: str) -> str | None:
    token = lemma_name.lower().replace("_", " ").strip()
    if " " in token or "-" in token:
        return None
    if not re.fullmatch(r"[a-z]+", token):
        return None
    return token

# singularize_word uses the inflect library to convert plurals to singular.
# The inflect.singular_noun call returns False when the word is already singular, 
# so the isinstance(singular, str) and singular check handles both the 
# "already singular" and "successfully singularized" cases
def singularize_word(word: str, infl: inflect.engine) -> str | None:
    singular = infl.singular_noun(word)
    candidate = singular if isinstance(singular, str) and singular else word
    if not re.fullmatch(r"[a-z]+", candidate):
        return None
    return candidate.lower()

#heuristic filter: it checks if any keyword from the blocklist appears in the WordNet definition text
def is_concrete_artifact_synset(synset) -> bool:
    definition = synset.definition().lower()
    return not any(keyword in definition for keyword in ABSTRACT_ARTIFACT_KEYWORDS)

#for each synset, optionally apply the concreteness filter, 
# then collect all lemma names, normalize and singularize them.
def words_from_synsets(synsets, infl: inflect.engine, concrete_artifact_filter: bool = False) -> set[str]:
    words: set[str] = set()
    for syn in synsets:
        if concrete_artifact_filter and not is_concrete_artifact_synset(syn):
            continue
        for lemma in syn.lemma_names():
            norm = normalize_lemma_name(lemma)
            if norm is None:
                continue
            sg = singularize_word(norm, infl)
            if sg is None:
                continue
            words.add(sg)
    return words

def serialize_pool(by_tok: dict[int, list[str]], note: str) -> dict:
    return {
        "note": note,
        "by_token_length": {str(k): {"words": v} for k, v in sorted(by_tok.items())},
    }

def is_valid_patient(lemma: str) -> bool:
    # 1. Reject short strings (abbreviations, pronouns like "mp", "si", "nan")
    if len(lemma) < 4:
        return False
    
    # 2. Reject pronouns and determiners explicitly
    BLOCKLIST = {"someone", "anyone", "nobody", "everybody", "something", "whoever"}
    if lemma.lower() in BLOCKLIST:
        return False
    
    # 3. Reject offensive lemmas
    if profanity.contains_profanity(lemma):
        return False
    
    # 4. Must be a hyponym of person.n.01
    person_synset = wn.synset("person.n.01")
    for synset in wn.synsets(lemma, pos=wn.NOUN):
        if person_synset in synset.closure(lambda s: s.hypernyms()):
            return True
    
    return False

#----------------------------------------------------------------------------
#Phase 3 domain grouping utilities
#----------------------------------------------------------------------------
from collections import Counter
from difflib import SequenceMatcher

def canonical_vn_class(class_id: str) -> str:
    # Keep only the canonical VerbNet class stem, removing subclass suffixes like "-1".
    m = re.match(r"^(.+?-\d+(?:\.\d+)*)", class_id)
    return m.group(1) if m else class_id


def class_family(class_id: str) -> int | None:
    canonical = canonical_vn_class(class_id)
    if "-" not in canonical:
        return None
    numeric = canonical.rsplit("-", 1)[-1]
    head = numeric.split(".")[0]
    return int(head) if head.isdigit() else None

def domain_label(class_id: str) -> str:
    # "judgement-33.1-1" → "judgement"
    canonical = canonical_vn_class(class_id)
    return canonical.rsplit("-", 1)[0] if "-" in canonical else canonical


def representative_class(class_ids: list[str]) -> str | None:
    if not class_ids:
        return None
    canonical = [canonical_vn_class(c) for c in class_ids]
    return Counter(canonical).most_common(1)[0][0]


def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()