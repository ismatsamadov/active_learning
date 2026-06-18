"""Active-learning query strategies for sequence labelling.

All strategies pick ``k`` indices from the *unlabeled* pool to be annotated next.

* random       — uniform baseline (the control the thesis was missing)
* uncertainty  — sequence-level MNLP (Shen et al., 2018): label where the model
                 is least confident in its best tag path
* hybrid       — uncertainty + diversity (Lewis & Gale, 1994; Sener & Savarese,
                 2018): take the top ``factor*k`` uncertain sentences, then use
                 k-center-greedy (core-set) over their BiLSTM embeddings to pick a
                 diverse, non-redundant batch of ``k``
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..ner.data import NERDataset, Vocab, make_collate


@torch.no_grad()
def score_pool(model, records: List[dict], vocab: Vocab, device: str,
               batch_size: int = 64, want_emb: bool = True, unc_kind: str = "mnlp"):
    """Return (uncertainty[N], embeddings[N,H]) aligned to ``records`` order."""
    model.eval()
    loader = DataLoader(NERDataset(records, vocab), batch_size=batch_size,
                        shuffle=False, collate_fn=make_collate(vocab))
    N = len(records)
    unc = np.zeros(N, dtype=np.float64)
    emb: Optional[np.ndarray] = None
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        idx = batch["indices"].cpu().numpy()
        u = model.uncertainty(batch, kind=unc_kind).cpu().numpy()   # higher = more uncertain
        unc[idx] = u
        if want_emb:
            e = model.sentence_embeddings(batch).cpu().numpy()
            if emb is None:
                emb = np.zeros((N, e.shape[1]), dtype=np.float32)
            emb[idx] = e
    return unc, emb


def k_center_greedy(cand: np.ndarray, k: int, init: Optional[np.ndarray] = None) -> List[int]:
    """Greedy k-center / core-set selection (Sener & Savarese, 2018).

    Repeatedly pick the candidate farthest from the set of already-selected points
    (and from ``init``, the existing labelled embeddings, if given). Maximises
    coverage of the embedding space -> diverse, non-redundant batch.
    """
    n = len(cand)
    k = min(k, n)
    # min distance from each candidate to the current selected/init set
    if init is not None and len(init):
        d = _min_sqdist(cand, init)
    else:
        d = np.full(n, np.inf)
    selected: List[int] = []
    for _ in range(k):
        i = int(np.argmax(d))
        selected.append(i)
        di = ((cand - cand[i]) ** 2).sum(axis=1)
        d = np.minimum(d, di)
    return selected


def _min_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Min squared L2 distance from each row of a to any row of b (chunked).

    Computed in float64 with FP warnings silenced — BiLSTM embeddings can be large
    enough that the |a|^2 - 2a.b + |b|^2 expansion overflows float32 harmlessly
    (the relative ordering used by k-center selection is unaffected)."""
    a = a.astype(np.float64, copy=False)
    out = np.full(len(a), np.inf)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for start in range(0, len(b), 512):
            chunk = b[start:start + 512].astype(np.float64, copy=False)
            d = (a ** 2).sum(1)[:, None] - 2 * a @ chunk.T + (chunk ** 2).sum(1)[None, :]
            out = np.minimum(out, np.nan_to_num(d, nan=np.inf).min(axis=1))
    return out


def select(strategy: str, *, model, pool_records, pool_indices, labeled_emb,
           vocab, device, k: int, rng, factor: int = 4) -> List[int]:
    """Return a list of indices INTO ``pool_indices`` to label next."""
    n = len(pool_indices)
    k = min(k, n)

    if strategy == "random":
        return list(rng.choice(n, size=k, replace=False))

    # 'least_confidence' = thesis sec 2.3 literal wording; 'uncertainty'/'hybrid' = MNLP
    unc_kind = "least_confidence" if strategy == "least_confidence" else "mnlp"
    unc, emb = score_pool(model, pool_records, vocab, device,
                          want_emb=(strategy == "hybrid"), unc_kind=unc_kind)

    if strategy in ("uncertainty", "least_confidence"):
        return list(np.argsort(-unc)[:k])

    if strategy == "hybrid":
        m = min(n, factor * k)
        cand_local = np.argsort(-unc)[:m]            # most-uncertain candidates
        chosen_in_cand = k_center_greedy(emb[cand_local], k, init=labeled_emb)
        return [int(cand_local[i]) for i in chosen_in_cand]

    raise ValueError(f"unknown strategy {strategy}")
