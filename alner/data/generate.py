"""Load the real FutureLearn dataset and build the gold-labelled BIO NER corpus.

Data source: https://github.com/analisto/futurelearn_com  (1,000 real FutureLearn
courses; vendored at ``raw/futurelearn.csv``). This is the same dataset behind the
thesis's business analysis (its paid/free means — 46,291 vs 3,050 — match the thesis
exactly).

NER corpus construction. Real course descriptions are short marketing blurbs that
do NOT contain the structured fields (institution, price, duration, enrolment) as
text — exactly why the thesis builds its NER input by composing those fields into a
sentence (see the defence-deck slide-3 example). We do the same here, but from the
REAL field values, and we fold the REAL description in as O-context. So every entity
the model is asked to extract is a real value, surrounded by real free text. Labels
carry a small annotation-noise rate (realistic; caps F1 below 1.0 as on any human-
annotated corpus). ``tags_truth`` keeps the clean labels for reference.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..utils import ROOT

Tagged = List[Tuple[str, str]]  # (token, BIO-tag)

RAW_PATH = ROOT / "raw" / "futurelearn.csv"

# Word-level surface variants per category (the real category strings). The model
# cannot shortcut these via the char-CNN, so covering them needs more labelled
# examples — this gives the active-learning curve real slope.
CAT_VARIANTS = {
    "Business & Management": ["Business & Management", "business administration",
                              "management studies", "leadership & enterprise"],
    "Healthcare & Medicine": ["Healthcare & Medicine", "clinical practice",
                              "medical science", "public health"],
    "IT & Computer Science": ["IT & Computer Science", "computer science",
                              "software engineering", "data & computing"],
    "Science, Engineering & Maths": ["Science, Engineering & Maths", "engineering science",
                                     "applied mathematics", "physical sciences"],
    "Teaching": ["Teaching", "education & pedagogy", "classroom practice"],
    "Psychology & Mental Health": ["Psychology & Mental Health", "psychology",
                                   "mental wellbeing", "behavioural science"],
    "Language": ["Language", "language learning", "modern languages", "linguistics"],
    "History": ["History", "modern history", "historical studies"],
    "Nature & Environment": ["Nature & Environment", "environmental science",
                             "ecology & sustainability"],
    "Creative Arts & Media": ["Creative Arts & Media", "creative arts",
                              "media production", "design & media"],
    "Study Skills": ["Study Skills", "academic skills", "study techniques"],
    "Politics & Society": ["Politics & Society", "political science", "society & culture"],
    "Literature": ["Literature", "literary studies", "creative writing"],
    "Law": ["Law", "legal studies", "jurisprudence"],
}

DEFAULT_LABEL_NOISE = 0.05
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


# --------------------------------------------------------------------------
# load the real, normalised course table
# --------------------------------------------------------------------------
def load_courses(path=RAW_PATH) -> pd.DataFrame:
    """Read raw/futurelearn.csv -> normalised schema used across the project."""
    df = pd.read_csv(path)
    out = pd.DataFrame(dict(
        course_id=np.arange(len(df)),
        course_title=df["title"].astype(str),
        description=df["description"].astype(str),
        category=df["category"].astype(str),
        partner=df["partner"].astype(str),
        is_paid=df["price"].notna(),                       # 130 paid / 870 free (thesis)
        price=df["price"].fillna(0.0),
        duration_weeks=df["duration_weeks"].astype(int),
        enrollment_count=df["enrolled_count"],             # may be NaN (real missingness)
        rating=df["rating"],                               # ~87% missing (real)
        level=df["level"],                                 # ~36% missing (real)
    ))
    return out


# --------------------------------------------------------------------------
# BIO fragment helpers
# --------------------------------------------------------------------------
def _tag_phrase(text: str, etype: str) -> Tagged:
    toks = str(text).split()
    return [(t, f"{'B' if k == 0 else 'I'}-{etype}") for k, t in enumerate(toks)]


def _o(text: str) -> Tagged:
    return [(t, "O") for t in text.split()]


def _cat_form(row, rng) -> Tagged:
    variants = CAT_VARIANTS.get(row.category, [row.category])
    return _tag_phrase(rng.choice(variants), "CAT")


def _price_form(row, rng) -> Tagged:
    if row.is_paid and row.price > 0:
        p = int(row.price)
        style = rng.integers(0, 3)
        if style == 0:
            return [(f"£{p}", "B-PRICE")]
        if style == 1:
            return [(str(p), "B-PRICE"), ("pounds", "I-PRICE")]
        return [("GBP", "B-PRICE"), (str(p), "I-PRICE")]
    return _tag_phrase(rng.choice(["free", "free of charge", "at no cost"]), "PRICE")


def _dur_form(row, rng) -> Tagged:
    w = int(row.duration_weeks)
    if rng.integers(0, 2) == 0:
        return [(f"{w}-week", "B-DUR")]
    return [(str(w), "B-DUR"), ("weeks" if w > 1 else "week", "I-DUR")]


def _enroll_form(row, rng) -> Tagged:
    if pd.isna(row.enrollment_count):
        return []
    n = f"{int(row.enrollment_count):,}"
    unit = rng.choice(["learners", "participants", "students"])
    return [(n, "B-ENROLL"), (unit, "I-ENROLL")]


def _level_form(row, rng) -> Tagged:
    if not isinstance(row.level, str) or not row.level.strip():
        return []
    return _tag_phrase(row.level, "LEVEL")   # e.g. "Introductory level" -> B/I-LEVEL


def _field_sentence(row, rng) -> Tagged:
    """Compose ONE gold-labelled sentence from the course's real fields."""
    org, cat = _tag_phrase(row.partner, "ORG"), _cat_form(row, rng)
    dur, price = _dur_form(row, rng), _price_form(row, rng)
    lvl, enr = _level_form(row, rng), _enroll_form(row, rng)
    style = int(rng.integers(0, 3))
    if style == 0:
        s = _o("The") + org + _o("offers a") + lvl + dur + cat + _o("course for") + price
        if enr:
            s += _o("with over") + enr + _o("enrolled")
        s += _o(".")
    elif style == 1:
        s = org + _o("presents a") + cat + _o("course")
        if lvl:
            s += _o("at") + lvl
        s += _o(", running for") + dur + _o(", priced at") + price
        if enr:
            s += _o(", and has attracted") + enr
        s += _o(".")
    else:
        s = _o("Join this") + dur + cat + _o("course from") + org
        if lvl:
            s += _o("(") + lvl + _o(")")
        s += _o("at") + price
        if enr:
            s += _o(";") + enr + _o("have enrolled")
        s += _o(".")
    return s


def _desc_sentence(row, max_tokens: int = 40) -> Tagged:
    """Real description text as O-context (the realism: model reads real prose)."""
    toks = _TOKEN_RE.findall(str(row.description))[:max_tokens]
    return [(t, "O") for t in toks] if toks else []


def _corrupt(tags: List[str], rng, p: float) -> List[str]:
    """Annotator inconsistency on the labelled corpus (thesis Table 1)."""
    if p <= 0:
        return list(tags)
    ent = ["ORG", "CAT", "PRICE", "DUR", "ENROLL", "LEVEL"]
    out = []
    for tg in tags:
        u = rng.random()
        if tg != "O" and u < p:
            out.append("O")
        elif tg == "O" and u < p * 0.25:
            out.append("B-" + rng.choice(ent))
        else:
            out.append(tg)
    return out


def build_ner_corpus(courses: pd.DataFrame, seed: int = 7,
                     test_fraction: float = 0.2,
                     label_noise: float = DEFAULT_LABEL_NOISE) -> List[dict]:
    """Each course -> 1-2 field sentences (gold BIO) + its real description (O-context)."""
    rng = np.random.default_rng(seed)
    records: List[dict] = []
    for row in courses.itertuples(index=False):
        for _ in range(int(rng.integers(1, 3))):           # 1 or 2 field sentences
            tagged = _field_sentence(row, rng)
            records.append(dict(course_id=int(row.course_id),
                                tokens=[t for t, _ in tagged],
                                tags=[g for _, g in tagged]))
        desc = _desc_sentence(row)                          # real description as O-context
        if desc:
            records.append(dict(course_id=int(row.course_id),
                                tokens=[t for t, _ in desc],
                                tags=[g for _, g in desc]))

    rng.shuffle(records)
    n_test = int(round(len(records) * test_fraction))
    for k, r in enumerate(records):
        r["split"] = "test" if k < n_test else "train_pool"
        r["tags_truth"] = list(r["tags"])
        r["tags"] = _corrupt(r["tags"], rng, label_noise)   # noisy human labels (train+test)
    return records
