"""Generate the calibrated dataset + NER corpus, then validate against thesis numbers.

Usage:
    python scripts/01_generate_data.py
Writes:
    data/courses.csv, data/ner_corpus.jsonl, results/data_validation.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alner.data.generate import generate_courses, build_ner_corpus
from alner.data import thesis_stats as TS
from alner.utils import DATA_DIR, RESULTS_DIR, write_jsonl, write_json


def validate(courses: pd.DataFrame, corpus) -> dict:
    """Compare achieved marginals to the thesis targets; print a side-by-side table."""
    rep = {}

    # free vs paid enrolment ratio (~15x)
    paid_mean = courses.loc[courses.is_paid, "enrollment_count"].mean()
    free_mean = courses.loc[~courses.is_paid, "enrollment_count"].mean()
    rep["paid_mean_enroll"] = round(float(paid_mean), 1)
    rep["free_mean_enroll"] = round(float(free_mean), 1)
    rep["paid_free_ratio"] = round(float(paid_mean / free_mean), 2)
    rep["pct_paid"] = round(float(courses.is_paid.mean() * 100), 1)

    # category means vs targets
    cat_rows = []
    for name, n, target, _ in TS.CATEGORIES:
        got = courses.loc[courses.category == name, "enrollment_count"].mean()
        cat_rows.append(dict(category=name, n=int((courses.category == name).sum()),
                             target_mean=target, got_mean=round(float(got)),
                             err_pct=round(100 * (got - target) / target, 1)))
    rep["categories"] = cat_rows

    # duration curve
    dur = courses.groupby("duration_weeks")["enrollment_count"].mean().round().astype(int)
    rep["duration_mean_enroll"] = {int(k): int(v) for k, v in dur.items()}

    # price tier curve (paid only)
    price = courses.loc[courses.is_paid].groupby("price")["enrollment_count"].mean().round().astype(int)
    rep["price_mean_enroll"] = {int(k): int(v) for k, v in price.items()}

    # partner efficiency (named ones)
    prow = []
    for name, *_ in TS.NAMED_PARTNERS:
        sub = courses.loc[courses.partner == name]
        prow.append(dict(partner=name, n=int(len(sub)),
                         mean_enroll=round(float(sub.enrollment_count.mean())),
                         total_enroll=int(sub.enrollment_count.sum())))
    rep["named_partners"] = prow

    # ratings
    rep["pct_rating_present"] = round(float(courses.rating.notna().mean() * 100), 1)
    rep["mean_rating"] = round(float(courses.rating.mean()), 2)

    # corpus stats
    n_tok = sum(len(r["tokens"]) for r in corpus)
    n_ent_tok = sum(sum(1 for t in r["tags"] if t != "O") for r in corpus)
    rep["corpus"] = dict(
        sentences=len(corpus),
        tokens=n_tok,
        entity_tokens=n_ent_tok,
        pct_entity_tokens=round(100 * n_ent_tok / n_tok, 1),
        train_pool=sum(1 for r in corpus if r["split"] == "train_pool"),
        test=sum(1 for r in corpus if r["split"] == "test"),
    )
    return rep


def main() -> None:
    courses = generate_courses(seed=42)
    corpus = build_ner_corpus(courses, seed=7, test_fraction=0.2, label_noise=0.05)

    courses.to_csv(DATA_DIR / "courses.csv", index=False)
    write_jsonl(DATA_DIR / "ner_corpus.jsonl", corpus)

    rep = validate(courses, corpus)
    write_json(RESULTS_DIR / "data_validation.json", rep)

    # pretty print the key comparisons
    print("\n=== DATA VALIDATION vs THESIS ===")
    print(f"Courses: {len(courses)} | paid {rep['pct_paid']}% "
          f"(target 13%) | rating present {rep['pct_rating_present']}% (target ~13%)")
    print(f"Paid mean enrol {rep['paid_mean_enroll']:,} vs free {rep['free_mean_enroll']:,} "
          f"-> ratio {rep['paid_free_ratio']}x (thesis ~15x)")
    print(f"Mean rating {rep['mean_rating']} (thesis 4.70)\n")
    print(f"{'Category':30s} {'n':>4s} {'target':>8s} {'got':>8s} {'err%':>6s}")
    for r in rep["categories"]:
        print(f"{r['category']:30s} {r['n']:>4d} {r['target_mean']:>8,} {r['got_mean']:>8,} {r['err_pct']:>6.1f}")
    print("\nDuration -> mean enrol (thesis: peak 2-5wk; 5wk~15,343, 1wk~1,564, 8wk~1,029):")
    for w, v in rep["duration_mean_enroll"].items():
        print(f"  {w:>2d} wk: {v:>8,}")
    print("\nNamed partners (thesis: Groningen ~80,435/course; Leeds biggest total):")
    for r in rep["named_partners"]:
        print(f"  {r['partner']:32s} n={r['n']:>3d} mean={r['mean_enroll']:>8,} total={r['total_enroll']:>10,}")
    print(f"\nCorpus: {rep['corpus']['sentences']} sentences, {rep['corpus']['tokens']:,} tokens, "
          f"{rep['corpus']['pct_entity_tokens']}% entity tokens | "
          f"pool={rep['corpus']['train_pool']} test={rep['corpus']['test']}")
    print(f"\nWrote {DATA_DIR/'courses.csv'} and {DATA_DIR/'ner_corpus.jsonl'}")


if __name__ == "__main__":
    main()
