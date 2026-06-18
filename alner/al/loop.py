"""The pool-based active-learning loop (the thesis's core framework, made real).

seed set -> train BiLSTM-CRF -> evaluate -> score pool -> query batch -> annotate
(reveal gold labels = oracle) -> add to labelled set -> retrain ... until budget.

All strategies share the SAME seed set for a given ``seed`` so any divergence in
the learning curve is attributable purely to the query strategy.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..ner.data import Vocab
from ..ner.train import evaluate, train_model
from . import strategies as S


@torch.no_grad()
def _embeddings_for(model, records, vocab, device):
    _, emb = S.score_pool(model, records, vocab, device, want_emb=True)
    return emb


def run_active_learning(pool: List[dict], test: List[dict], vocab: Vocab, cfg,
                        strategy: str, seed: int, device: str,
                        verbose: bool = False) -> Dict:
    """Run one AL trajectory; return per-iteration test metrics + label counts."""
    rng = np.random.default_rng(seed)
    al = cfg.al
    n_pool = len(pool)

    # shared initial seed set (same for every strategy at this `seed`)
    perm = rng.permutation(n_pool)
    labeled_mask = np.zeros(n_pool, dtype=bool)
    labeled_mask[perm[: al.seed_size]] = True

    curve = []
    for it in range(al.n_iterations + 1):
        labeled_idx = np.where(labeled_mask)[0]
        labeled = [pool[i] for i in labeled_idx]

        model, info = train_model(labeled, vocab, cfg, device, seed=seed)
        m = evaluate(model, test, vocab, device)
        point = dict(iteration=it, n_labeled=int(len(labeled_idx)),
                     f1=m["f1"], precision=m["precision"], recall=m["recall"],
                     best_epoch=info["best_epoch"])
        curve.append(point)
        if verbose:
            print(f"  [{strategy} seed{seed}] it{it} n={point['n_labeled']:4d} "
                  f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

        if it == al.n_iterations:
            break

        # --- query the next batch from the unlabeled pool ---
        pool_local = np.where(~labeled_mask)[0]
        if len(pool_local) == 0:
            break
        # subsample the pool for scoring (standard AL speedup; random sees full pool)
        cap = getattr(al, "score_pool_cap", 0)
        if strategy != "random" and cap and len(pool_local) > cap:
            pool_local = rng.choice(pool_local, size=cap, replace=False)
        pool_records = [pool[i] for i in pool_local]
        labeled_emb = None
        if strategy == "hybrid":
            labeled_emb = _embeddings_for(model, labeled, vocab, device)
        chosen_local = S.select(strategy, model=model, pool_records=pool_records,
                                pool_indices=pool_local, labeled_emb=labeled_emb,
                                vocab=vocab, device=device, k=al.query_size, rng=rng,
                                factor=al.uncertainty_pool_factor)
        labeled_mask[pool_local[np.array(chosen_local, dtype=int)]] = True

    return dict(strategy=strategy, seed=seed, curve=curve)
