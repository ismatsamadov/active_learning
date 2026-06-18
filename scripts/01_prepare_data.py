"""Prepare the dataset: load the real FutureLearn data and build the NER corpus.

Source: https://github.com/analisto/futurelearn_com (vendored at raw/futurelearn.csv).
Writes data/courses.csv (normalised) + data/ner_corpus.jsonl, and reports the key
statistics (which match the thesis because this IS the thesis's dataset).

Usage: python scripts/01_prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alner.data.generate import load_courses, build_ner_corpus
from alner.utils import DATA_DIR, RESULTS_DIR, write_jsonl, write_json


def main() -> None:
    courses = load_courses()
    corpus = build_ner_corpus(courses, seed=7, test_fraction=0.2)

    courses.to_csv(DATA_DIR / "courses.csv", index=False)
    write_jsonl(DATA_DIR / "ner_corpus.jsonl", corpus)

    paid = courses[courses.is_paid]
    free = courses[~courses.is_paid]
    n_ent = sum(sum(1 for t in r["tags_truth"] if t != "O") for r in corpus)
    n_tok = sum(len(r["tokens"]) for r in corpus)
    rep = dict(
        source="https://github.com/analisto/futurelearn_com",
        n_courses=len(courses),
        n_categories=int(courses.category.nunique()),
        n_partners=int(courses.partner.nunique()),
        paid=int(len(paid)), free=int(len(free)),
        paid_mean_enroll=round(float(paid.enrollment_count.mean()), 1),
        free_mean_enroll=round(float(free.enrollment_count.mean()), 1),
        paid_free_ratio=round(float(paid.enrollment_count.mean() / free.enrollment_count.mean()), 2),
        pct_rating_missing=round(float(courses.rating.isna().mean() * 100), 1),
        pct_level_present=round(float(courses.level.notna().mean() * 100), 1),
        mean_rating=round(float(courses.rating.mean()), 2),
        corpus=dict(sentences=len(corpus), tokens=n_tok,
                    pct_entity_tokens=round(100 * n_ent / n_tok, 1),
                    train_pool=sum(1 for r in corpus if r["split"] == "train_pool"),
                    test=sum(1 for r in corpus if r["split"] == "test")),
    )
    write_json(RESULTS_DIR / "data_validation.json", rep)

    print("=== REAL FutureLearn dataset (analisto/futurelearn_com) ===")
    print(f"courses={rep['n_courses']}  categories={rep['n_categories']}  partners={rep['n_partners']}")
    print(f"paid {rep['paid']} / free {rep['free']}  |  paid mean {rep['paid_mean_enroll']:,} "
          f"vs free {rep['free_mean_enroll']:,}  -> {rep['paid_free_ratio']}x  (thesis: 46,291 vs 3,049, ~15x)")
    print(f"ratings missing {rep['pct_rating_missing']}% (thesis ~87%) | level present {rep['pct_level_present']}%")
    print("category counts:")
    for c, n in courses.category.value_counts().items():
        print(f"  {c:30s} {n}")
    print(f"corpus: {rep['corpus']['sentences']} sentences, {rep['corpus']['tokens']:,} tokens, "
          f"{rep['corpus']['pct_entity_tokens']}% entity | pool={rep['corpus']['train_pool']} test={rep['corpus']['test']}")
    print(f"\nwrote {DATA_DIR/'courses.csv'} and {DATA_DIR/'ner_corpus.jsonl'}")


if __name__ == "__main__":
    main()
