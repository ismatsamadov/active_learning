"""Entity-level (CoNLL-style) precision / recall / F1 for BIO sequences.

Entity-level means a prediction counts only when the entity's *type and both
boundaries* exactly match the gold span — the standard NER metric (Tjong Kim
Sang & De Meulder, 2003), and the one the thesis says it uses but never tabulates.
We micro-average over all entity types. An independent seqeval cross-check is
available via ``crosscheck_with_seqeval``.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

Span = Tuple[int, int, str]  # (start, end_inclusive, type)


def bio_spans(tags: List[str]) -> Set[Span]:
    """Extract entity spans from a BIO tag sequence (robust to malformed I-)."""
    spans: Set[Span] = set()
    start, etype = None, None
    for i, tag in enumerate(tags + ["O"]):  # sentinel flush
        if tag == "O" or tag.startswith("B-"):
            if start is not None:
                spans.add((start, i - 1, etype))
                start, etype = None, None
        if tag.startswith("B-"):
            start, etype = i, tag[2:]
        elif tag.startswith("I-"):
            t = tag[2:]
            if start is None:           # orphan I- starts a span (lenient)
                start, etype = i, t
            elif t != etype:            # type switch without B- -> new span
                spans.add((start, i - 1, etype))
                start, etype = i, t
    return spans


def prf(gold: List[List[str]], pred: List[List[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    n_gold = n_pred = 0
    for g, p in zip(gold, pred):
        gs, ps = bio_spans(g), bio_spans(p)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
        n_gold += len(gs)
        n_pred += len(ps)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, n_gold=n_gold, n_pred=n_pred)


def prf_by_type(gold: List[List[str]], pred: List[List[str]]) -> Dict[str, Dict[str, float]]:
    """Per-entity-type micro PRF (for error analysis / the results table)."""
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for g, p in zip(gold, pred):
        gs, ps = bio_spans(g), bio_spans(p)
        for s in gs & ps: tp[s[2]] += 1
        for s in ps - gs: fp[s[2]] += 1
        for s in gs - ps: fn[s[2]] += 1
    out = {}
    for t in set(list(tp) + list(fp) + list(fn)):
        pr = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) else 0.0
        rc = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        out[t] = dict(precision=pr, recall=rc, f1=f1, support=tp[t] + fn[t])
    return out


def macro_f1(gold: List[List[str]], pred: List[List[str]]) -> float:
    """Macro-averaged F1: per-type F1 averaged with equal weight (gives rare entity
    types equal say — the complement to micro, as the thesis contrasts (sec 1.2)."""
    bt = prf_by_type(gold, pred)
    return sum(d["f1"] for d in bt.values()) / len(bt) if bt else 0.0


def crosscheck_with_seqeval(gold, pred) -> Dict[str, float]:
    """Independent verification that our F1 matches the standard library."""
    try:
        from seqeval.metrics import precision_score, recall_score, f1_score
        return dict(precision=float(precision_score(gold, pred)),
                    recall=float(recall_score(gold, pred)),
                    f1=float(f1_score(gold, pred)))
    except Exception as e:  # pragma: no cover
        return dict(error=str(e))
