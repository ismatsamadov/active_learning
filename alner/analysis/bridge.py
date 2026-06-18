"""NER -> business-data bridge: parse the model's predicted entities back into
structured fields and measure how well they recover the ground-truth course data.

This is the worked "extraction -> insight" trace the thesis was missing: it shows
the NER layer genuinely PRODUCES the institution/category/price/duration/enrollment
that the business analysis consumes, rather than the analysis coming straight from
pre-existing columns.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from ..data.generate import CAT_VARIANTS

# reverse map: any category surface variant (lowercased) -> canonical category
_VARIANT2CAT = {v.lower(): cat for cat, vs in CAT_VARIANTS.items() for v in vs}


def spans_from_bio(tokens: List[str], tags: List[str]):
    """Yield (type, surface_string) for each entity span in a BIO sequence."""
    cur_type, cur_toks = None, []
    out = []
    for tok, tag in list(zip(tokens, tags)) + [("", "O")]:
        if tag == "O" or tag.startswith("B-"):
            if cur_type:
                out.append((cur_type, " ".join(cur_toks)))
                cur_type, cur_toks = None, []
        if tag.startswith("B-"):
            cur_type, cur_toks = tag[2:], [tok]
        elif tag.startswith("I-") and cur_type:
            cur_toks.append(tok)
    return out


def parse_price(s: str) -> Optional[float]:
    s = s.lower()
    if any(w in s for w in ("free", "no cost")):
        return 0.0
    m = re.search(r"\d[\d,]*", s)
    return float(m.group().replace(",", "")) if m else None


def parse_duration(s: str) -> Optional[int]:
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


def parse_enroll(s: str) -> Optional[int]:
    m = re.search(r"\d[\d,]*", s)
    return int(m.group().replace(",", "")) if m else None


def parse_category(s: str) -> Optional[str]:
    return _VARIANT2CAT.get(s.lower().strip())


def extract_course_fields(records_for_course: List[dict], preds: Dict[int, List[str]]):
    """Aggregate predicted entities across a course's sentences into one field set
    (majority vote per field)."""
    orgs, cats, prices, durs, enrolls = [], [], [], [], []
    for r in records_for_course:
        tags = preds[id(r)]
        for etype, surf in spans_from_bio(r["tokens"], tags):
            if etype == "ORG": orgs.append(surf)
            elif etype == "CAT":
                c = parse_category(surf)
                if c: cats.append(c)
            elif etype == "PRICE":
                p = parse_price(surf)
                if p is not None: prices.append(p)
            elif etype == "DUR":
                d = parse_duration(surf)
                if d is not None: durs.append(d)
            elif etype == "ENROLL":
                e = parse_enroll(surf)
                if e is not None: enrolls.append(e)

    def mode(xs): return Counter(xs).most_common(1)[0][0] if xs else None
    return dict(partner=mode(orgs), category=mode(cats), price=mode(prices),
                duration_weeks=mode(durs), enrollment_count=mode(enrolls))


def reconstruct_courses_from_ner(records: List[dict], pred_tags: List[List[str]]):
    """Build a course table PURELY from NER predictions: text -> entities -> fields.

    One row per course_id, aggregating predicted entities across the course's
    sentences. ``is_paid`` is inferred from the extracted price. This is the wired
    end-to-end flow (not adjacency): the business analysis can then run on THIS table.
    """
    import pandas as pd
    preds = {id(r): p for r, p in zip(records, pred_tags)}
    by_course = defaultdict(list)
    for r in records:
        by_course[r["course_id"]].append(r)
    rows = []
    for cid, recs in by_course.items():
        ext = extract_course_fields(recs, preds)
        if ext["price"] is None and ext["enrollment_count"] is None:
            continue  # no usable extraction
        rows.append(dict(
            course_id=cid, partner=ext["partner"], category=ext["category"],
            price=ext["price"], is_paid=(ext["price"] is not None and ext["price"] > 0),
            duration_weeks=ext["duration_weeks"], enrollment_count=ext["enrollment_count"],
        ))
    return pd.DataFrame(rows)


def business_agreement(extracted_df, truth_df) -> Dict:
    """Compare headline business numbers computed from NER-extracted fields vs truth."""
    def free_paid_ratio(d):
        d = d.dropna(subset=["enrollment_count"])
        paid = d.loc[d.is_paid, "enrollment_count"].mean()
        free = d.loc[~d.is_paid, "enrollment_count"].mean()
        return round(float(paid / free), 2) if free and paid == paid else None

    def top_demand_category(d):
        d = d.dropna(subset=["category", "enrollment_count"])
        if d.empty:
            return None
        return d.groupby("category")["enrollment_count"].mean().idxmax()

    def duration_peak(d):
        d = d.dropna(subset=["duration_weeks", "enrollment_count"])
        if d.empty:
            return None
        return int(d.groupby("duration_weeks")["enrollment_count"].mean().idxmax())

    return dict(
        free_paid_ratio=dict(from_ner=free_paid_ratio(extracted_df),
                             from_truth=free_paid_ratio(truth_df)),
        top_demand_category=dict(from_ner=top_demand_category(extracted_df),
                                 from_truth=top_demand_category(truth_df)),
        duration_peak_weeks=dict(from_ner=duration_peak(extracted_df),
                                 from_truth=duration_peak(truth_df)),
        n_courses_reconstructed=int(len(extracted_df)),
    )


def evaluate_field_recovery(test_records: List[dict], pred_tags: List[List[str]],
                            courses_df) -> Dict:
    """Compare NER-extracted fields to the true course fields, per field."""
    # map predictions onto records by identity
    preds = {id(r): p for r, p in zip(test_records, pred_tags)}
    by_course = defaultdict(list)
    for r in test_records:
        by_course[r["course_id"]].append(r)

    truth = courses_df.set_index("course_id").to_dict("index")
    hits = defaultdict(int); total = defaultdict(int)
    examples = []
    for cid, recs in by_course.items():
        if cid not in truth:
            continue
        ext = extract_course_fields(recs, preds)
        t = truth[cid]
        # categorical/numeric exact-match recovery
        checks = {
            "category": (ext["category"], t["category"]),
            "price": (ext["price"], float(t["price"])),
            "duration_weeks": (ext["duration_weeks"], int(t["duration_weeks"])),
            "enrollment_count": (ext["enrollment_count"], int(t["enrollment_count"])),
            "partner": (ext["partner"], t["partner"]),
        }
        for field, (got, want) in checks.items():
            if got is None:
                total[field] += 1
                continue
            total[field] += 1
            ok = (got == want) if field != "partner" else (str(got) == str(want))
            hits[field] += int(ok)
        if len(examples) < 6:
            examples.append(dict(course_id=int(cid),
                                 text=" ".join(recs[0]["tokens"]),
                                 extracted=ext,
                                 truth={k: t[k] for k in
                                        ["partner", "category", "price",
                                         "duration_weeks", "enrollment_count"]}))
    recovery = {f: round(hits[f] / total[f], 4) if total[f] else None for f in total}
    return dict(field_recovery=recovery, n_courses=len(by_course), examples=examples)
