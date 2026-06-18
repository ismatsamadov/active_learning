"""Generate the calibrated FutureLearn dataset and gold-labelled BIO NER corpus.

Two artefacts are produced:

1. ``data/courses.csv`` — 1,000 courses with structured fields whose marginal
   statistics reproduce the thesis (Chapter 3).
2. ``data/ner_corpus.jsonl`` — one record per *sentence*: ``tokens`` + gold ``tags``
   (BIO) + ``course_id`` + ``split``. The entity surface forms in each sentence are
   the course's own fields (partner -> ORG, category -> CAT, price -> PRICE,
   duration -> DUR, enrolment -> ENROLL), so a model that reads the text genuinely
   recovers the structured business data — the NER<->analytics bridge, demonstrated.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import thesis_stats as TS

Tagged = List[Tuple[str, str]]  # list of (token, BIO-tag)

# Word-level surface variants per category. The model cannot shortcut these via
# the char-CNN (unlike "University of <X>" or "£<n>"), so covering them needs more
# labeled examples — this is what gives the learning curve real slope and lets
# uncertainty sampling target sentences with as-yet-unseen variants.
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

# Place names used to synthesise ~95 distinct generic partner institutions.
_PLACES = [
    "Manchester", "Bristol", "Glasgow", "Sydney", "Auckland", "Toronto", "Dublin",
    "Cape Town", "Singapore", "Oslo", "Helsinki", "Munich", "Lyon", "Bologna",
    "Valencia", "Porto", "Krakow", "Tallinn", "Nairobi", "Accra", "Lagos", "Cairo",
    "Doha", "Riyadh", "Mumbai", "Delhi", "Seoul", "Tokyo", "Osaka", "Taipei",
    "Bangkok", "Jakarta", "Manila", "Lima", "Bogota", "Santiago", "Quito",
    "Montreal", "Vancouver", "Boston", "Austin", "Denver", "Phoenix", "Atlanta",
    "Seattle", "Portland", "Cardiff", "Belfast", "York", "Bath", "Exeter", "Hull",
    "Leicester", "Coventry", "Norwich", "Stirling", "Aarhus", "Bergen", "Ghent",
    "Leuven", "Maastricht", "Twente", "Lund", "Uppsala", "Turku", "Reykjavik",
    "Graz", "Linz", "Bern", "Basel", "Geneva", "Nantes", "Rennes", "Toulouse",
    "Genoa", "Pisa", "Padua", "Verona", "Seville", "Bilbao", "Granada", "Coimbra",
    "Wroclaw", "Gdansk", "Brno", "Pecs", "Vilnius", "Riga", "Cluj", "Sofia",
    "Zagreb", "Ljubljana", "Belgrade", "Skopje",
]


# --------------------------------------------------------------------------
# 1) structured course table
# --------------------------------------------------------------------------
def _build_partner_pool(rng: np.random.Generator) -> List[Tuple[str, float]]:
    """Return a list of (partner_name, efficiency_mult) of length N_COURSES.

    Named partners contribute their fixed course counts; the rest are spread over
    ~95 synthesised generic institutions whose multiplier is centred so the
    dataset-wide partner effect averages ~1 (keeps category means clean).
    """
    pool: List[Tuple[str, float]] = []
    for name, n, mult in TS.NAMED_PARTNERS:
        pool.extend([(name, mult)] * n)

    n_remaining = TS.N_COURSES - len(pool)
    # Generic partner names: "University of <Place>" etc.
    generic_names = []
    for i in range(TS.N_GENERIC_PARTNERS):
        place = _PLACES[i % len(_PLACES)]
        stem = TS.GENERIC_PARTNER_NAMES[i % len(TS.GENERIC_PARTNER_NAMES)]
        generic_names.append(f"{stem} of {place}" if "of" not in stem else f"{stem} {place}")
    # Centre the generic multiplier so the whole-dataset average mult ~= 1.
    named_sum = sum(n * mult for _, n, mult in TS.NAMED_PARTNERS)
    generic_mult = (TS.N_COURSES - named_sum) / max(n_remaining, 1)
    generic_mult = float(np.clip(generic_mult, 0.6, 1.4))
    # Distribute the remaining courses across generic partners (Zipf-ish counts).
    counts = rng.integers(3, 16, size=TS.N_GENERIC_PARTNERS)
    counts = (counts / counts.sum() * n_remaining).round().astype(int)
    # fix rounding drift
    while counts.sum() < n_remaining:
        counts[rng.integers(0, TS.N_GENERIC_PARTNERS)] += 1
    while counts.sum() > n_remaining:
        j = rng.integers(0, TS.N_GENERIC_PARTNERS)
        if counts[j] > 0:
            counts[j] -= 1
    for name, c in zip(generic_names, counts):
        pool.extend([(name, generic_mult)] * int(c))
    rng.shuffle(pool)
    return pool[: TS.N_COURSES]


def generate_courses(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # categories (fixed counts) -> per-course category list
    cats: List[str] = []
    cat_mean: Dict[str, float] = {}
    cat_rating: Dict[str, float] = {}
    for name, n, mean, rating in TS.CATEGORIES:
        cats.extend([name] * n)
        cat_mean[name] = mean
        cat_rating[name] = rating
    rng.shuffle(cats)
    cats = np.array(cats)

    partners = _build_partner_pool(rng)

    # paid / free — STRATIFIED by category (~13% paid within each category) so that
    # the rare, high-enrolment paid courses do not destabilise small-category means.
    is_paid = np.zeros(TS.N_COURSES, dtype=bool)
    for name, *_ in TS.CATEGORIES:
        idx = np.where(cats == name)[0]
        k = int(round(len(idx) * TS.PAID_FRACTION))
        if k:
            is_paid[rng.choice(idx, size=k, replace=False)] = True

    # price tiers for paid courses
    tier_p = np.array(TS.PAID_TIER_WEIGHTS, dtype=float)
    tier_p /= tier_p.sum()

    # duration
    dur_p = np.array(TS.DURATION_WEIGHTS, dtype=float)
    dur_p /= dur_p.sum()

    # paid/free multiplier solving 0.87*f + 0.13*p = 1, p = ratio*f
    f_mult = 1.0 / ((1 - TS.PAID_FRACTION) + TS.PAID_FRACTION * TS.PAID_FREE_ENROLL_RATIO)
    p_mult = f_mult * TS.PAID_FREE_ENROLL_RATIO

    # --- per-course attributes (vectorised) ---
    weeks = rng.choice(TS.DURATION_WEEKS, size=TS.N_COURSES, p=dur_p)
    price = np.zeros(TS.N_COURSES, dtype=int)
    paid_idx = np.where(is_paid)[0]
    price[paid_idx] = rng.choice(TS.PRICE_TIERS, size=len(paid_idx), p=tier_p)

    dur_mult = np.array([TS.DURATION_ENROLL_MULT[int(w)] for w in weeks])
    price_mult = np.where(is_paid,
                          [TS.PRICE_TIER_ENROLL_MULT.get(int(p), 1.0) for p in price],
                          1.0)
    pf_mult = np.where(is_paid, p_mult, f_mult)
    partner_mult = np.array([m for _, m in partners])

    # Deterministic factor (everything except the category anchor & noise).
    factor = dur_mult * price_mult * pf_mult * partner_mult

    # Anchor the category MEAN to its thesis target by normalising the factor
    # product *within* each category. This pins cross-category means to the
    # targets while preserving all within-category structure (paid >> free, the
    # duration shape, partner efficiency ordering).
    base = np.zeros(TS.N_COURSES)
    for name, *_ in TS.CATEGORIES:
        idx = np.where(cats == name)[0]
        f = factor[idx]
        base[idx] = cat_mean[name] * f / f.mean()

    sig = TS.ENROLL_NOISE_SIGMA
    noise = rng.lognormal(mean=-0.5 * sig * sig, sigma=sig, size=TS.N_COURSES)  # E[noise]=1
    # cap extreme outliers so secondary breakdowns (per-price-tier, per-partner)
    # are not dominated by the rare paid x high-efficiency-partner x peak-duration tail
    enrollment = np.clip(np.round(base * noise), 10, TS.ENROLL_CAP).astype(int)

    # ratings: ~87% missing; present ones centred on the category rating
    present = rng.random(TS.N_COURSES) < TS.RATING_PRESENT_FRACTION
    rating = np.full(TS.N_COURSES, np.nan)
    cat_rating_arr = np.array([cat_rating[c] for c in cats])
    rating[present] = np.clip(
        np.round(rng.normal(cat_rating_arr[present], TS.RATING_STD), 1), 1.0, 5.0)

    df = pd.DataFrame(dict(
        course_id=np.arange(TS.N_COURSES),
        course_title=[f"{c.split(' ')[0]} course {i:04d}" for i, c in enumerate(cats)],
        category=cats,
        partner=[p for p, _ in partners],
        price=price,
        is_paid=is_paid,
        duration_weeks=weeks.astype(int),
        enrollment_count=enrollment,
        rating=rating,
    ))
    return df


# --------------------------------------------------------------------------
# 2) gold-labelled BIO NER corpus
# --------------------------------------------------------------------------
def _tag_phrase(text: str, etype: str) -> Tagged:
    """Split a multi-word surface form into BIO-tagged tokens."""
    toks = text.split()
    return [(t, f"{'B' if k == 0 else 'I'}-{etype}") for k, t in enumerate(toks)]


def _o(text: str) -> Tagged:
    return [(t, "O") for t in text.split()]


def _cat_form(row, rng) -> Tagged:
    """Pick one word-level surface variant of the course's category."""
    variants = CAT_VARIANTS.get(row.category, [row.category])
    return _tag_phrase(rng.choice(variants), "CAT")


def _price_form(row, rng) -> Tagged:
    if row.is_paid:
        p = row.price
        style = rng.integers(0, 3)
        if style == 0:
            return [(f"£{p}", "B-PRICE")]
        if style == 1:
            return [(str(p), "B-PRICE"), ("pounds", "I-PRICE")]
        return [("GBP", "B-PRICE"), (str(p), "I-PRICE")]
    # free courses: vary the surface form
    return _tag_phrase(rng.choice(["free", "free of charge", "at no cost"]), "PRICE")


def _dur_form(row, rng) -> Tagged:
    w = row.duration_weeks
    style = rng.integers(0, 2)
    if style == 0:
        return [(f"{w}-week", "B-DUR")]
    return [(str(w), "B-DUR"), ("weeks" if w > 1 else "week", "I-DUR")]


def _enroll_form(row, rng) -> Tagged:
    n = f"{row.enrollment_count:,}"
    unit = rng.choice(["learners", "participants", "students"])
    return [(n, "B-ENROLL"), (unit, "I-ENROLL")]


# Sentence templates. Each is a callable (row, rng) -> Tagged. Fragments are
# assembled so the gold tags are exact by construction (no post-hoc alignment).
def _t_offer(row, rng) -> Tagged:
    s: Tagged = _o("The") + _tag_phrase(row.partner, "ORG") + _o("offers a")
    s += _dur_form(row, rng) + _cat_form(row, rng) + _o("course for")
    s += _price_form(row, rng) + _o("with over") + _enroll_form(row, rng) + _o("enrolled .")
    return s


def _t_present(row, rng) -> Tagged:
    s = _tag_phrase(row.partner, "ORG") + _o("presents a") + _cat_form(row, rng)
    s += _o("programme priced at") + _price_form(row, rng) + _o(", running for")
    s += _dur_form(row, rng) + _o(", and has attracted") + _enroll_form(row, rng) + _o("to date .")
    return s


def _t_join(row, rng) -> Tagged:
    s = _o("Join this") + _cat_form(row, rng) + _o("course from")
    s += _tag_phrase(row.partner, "ORG") + _o(";  it runs for") + _dur_form(row, rng)
    s += _o("and has welcomed") + _enroll_form(row, rng) + _o("at") + _price_form(row, rng) + _o(".")
    return s


def _t_short(row, rng) -> Tagged:
    # partial: drops some entities, exercises boundary detection in sparser context
    s = _tag_phrase(row.partner, "ORG") + _o("runs a popular") + _cat_form(row, rng)
    s += _o("course over") + _dur_form(row, rng) + _o(".")
    return s


_FILLER = [
    "This course is designed for learners at all levels of experience .",
    "You will explore key concepts through videos , quizzes and discussions .",
    "No prior knowledge is required to get started with the material .",
    "Each week introduces new topics with practical , hands on activities .",
    "Learners receive a certificate of achievement upon completion .",
    "The course encourages reflection and peer to peer collaboration .",
]


def _distractor(rng) -> Tagged:
    """O-only sentence containing numbers that LOOK like entities (ratings,
    module counts, 'week N', review counts). Forces the model to rely on context
    rather than the mere presence of a digit -> caps achievable F1 below 1.0."""
    r = round(float(rng.uniform(3.8, 5.0)), 1)
    n_mod = int(rng.integers(3, 11))
    wk = int(rng.integers(1, 8))
    n_rev = int(rng.integers(5, 90))
    options = [
        f"It is rated {r} out of 5 by {n_rev} reviewers so far .",
        f"The syllabus spans {n_mod} modules delivered over the programme .",
        f"In week {wk} you will revisit the core ideas in more depth .",
        f"More than {n_mod} short videos accompany each unit of study .",
        f"Around {n_rev} learners post in the forums every single week .",
    ]
    s = rng.choice(options)
    return [(t, "O") for t in s.split()]


_TEMPLATES = [_t_offer, _t_present, _t_join, _t_short]


def _corrupt(tags: List[str], rng, p: float) -> List[str]:
    """Simulate annotator inconsistency (thesis Table 1) on the *labelled pool*.

    Per entity token: with prob p the annotator misses it (-> O); a smaller share
    of O tokens are spuriously labelled. Test labels are left clean for honest eval.
    """
    if p <= 0:
        return list(tags)
    ent_types = ["ORG", "CAT", "PRICE", "DUR", "ENROLL"]
    out = []
    for tg in tags:
        u = rng.random()
        if tg != "O" and u < p:                 # missed annotation
            out.append("O")
        elif tg == "O" and u < p * 0.25:         # spurious annotation
            out.append("B-" + rng.choice(ent_types))
        else:
            out.append(tg)
    return out


def build_ner_corpus(courses: pd.DataFrame, seed: int = 7,
                     test_fraction: float = 0.2, label_noise: float = 0.10) -> List[dict]:
    """One-to-many: each course yields entity-rich sentences plus O-heavy filler and
    distractor (numeric-but-O) sentences. ``label_noise`` perturbs the *pool* labels
    only (annotator inconsistency); ``tags_gold`` keeps the clean truth on every record.
    """
    rng = np.random.default_rng(seed)
    records: List[dict] = []
    for row in courses.itertuples(index=False):
        n_sent = int(rng.integers(2, 4))  # 2 or 3 entity sentences
        tmpl_idx = rng.choice(len(_TEMPLATES), size=n_sent, replace=True)
        for ti in tmpl_idx:
            tagged = _TEMPLATES[ti](row, rng)
            records.append(dict(course_id=int(row.course_id),
                                tokens=[t for t, _ in tagged],
                                tags=[g for _, g in tagged]))
        if rng.random() < 0.45:           # plain O-heavy filler
            f = rng.choice(_FILLER).split()
            records.append(dict(course_id=int(row.course_id), tokens=f, tags=["O"] * len(f)))
        if rng.random() < 0.40:           # numeric distractor (hard O-numbers)
            tagged = _distractor(rng)
            records.append(dict(course_id=int(row.course_id),
                                tokens=[t for t, _ in tagged],
                                tags=[g for _, g in tagged]))

    # shuffle and split train_pool / test (sentence-level, reproducible)
    rng.shuffle(records)
    n_test = int(round(len(records) * test_fraction))
    for k, r in enumerate(records):
        r["split"] = "test" if k < n_test else "train_pool"
        # `tags_truth` = perfect generative labels (reference/appendix only).
        # `tags` = the HUMAN annotation, carrying realistic noise on BOTH train and
        # test — exactly as in real NER benchmarks, where reported F1 is agreement
        # with imperfect human labels. This is what caps F1 below 1.0.
        r["tags_truth"] = list(r["tags"])
        r["tags"] = _corrupt(r["tags"], rng, label_noise)
    return records
