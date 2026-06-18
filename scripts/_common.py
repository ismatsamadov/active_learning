"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from alner.utils import read_jsonl, DATA_DIR
from alner.ner.data import Vocab


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_corpus():
    recs = read_jsonl(DATA_DIR / "ner_corpus.jsonl")
    pool = [r for r in recs if r["split"] == "train_pool"]
    test = [r for r in recs if r["split"] == "test"]
    vocab = Vocab.build(pool)
    return pool, test, vocab
