"""Shared utilities: reproducible seeding, JSONL/JSON IO, timing."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

# Project paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Deterministic given a fixed seed on CPU with
    num_workers=0; GPU/MPS ops are not guaranteed bit-identical (cuDNN is pinned
    but torch.use_deterministic_algorithms is not forced). For hash determinism set
    PYTHONHASHSEED in the launching environment (see run_all.sh) — assigning it here
    has no effect on the already-running interpreter."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))  # affects child processes only
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN where available (no-op on CPU/MPS).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def write_jsonl(path: os.PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: os.PathLike) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: os.PathLike, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    print(f"[{label}] {time.time() - t0:.1f}s")
