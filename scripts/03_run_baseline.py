"""Full-supervised baseline: train BiLSTM-CRF on the entire labelled pool.

This is the upper-reference the active-learning curves are compared against.
Reports entity-level P/R/F1 (overall + per type) averaged over seeds, with an
independent seqeval cross-check.

Usage: python scripts/03_run_baseline.py [--fast]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import load_corpus, pick_device
from alner.config import fast_config, full_config
from alner.ner.train import train_model, evaluate
from alner.ner.metrics import crosscheck_with_seqeval, macro_f1
from alner.utils import RESULTS_DIR, write_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = fast_config() if args.fast else full_config()
    device = pick_device(args.device)
    cfg.device = device
    pool, test, vocab = load_corpus()
    print(f"device={device}  pool={len(pool)}  test={len(test)}  "
          f"vocab={vocab.n_words}w/{vocab.n_chars}c  tags={vocab.n_tags}")

    f1s, ps, rs, macros = [], [], [], []
    by_type_acc = {}
    last = None
    for seed in cfg.al.seeds:
        set_seed(seed)
        t0 = time.time()
        model, info = train_model(pool, vocab, cfg, device, seed=seed)
        m = evaluate(model, test, vocab, device, by_type=True)
        f1s.append(m["f1"]); ps.append(m["precision"]); rs.append(m["recall"])
        macros.append(macro_f1(m["_gold"], m["_pred"]))
        for t, d in m["by_type"].items():
            by_type_acc.setdefault(t, []).append(d["f1"])
        last = m
        print(f"  seed {seed}: F1={m['f1']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f}  ({time.time()-t0:.0f}s, ep{info['best_epoch']})")

    xcheck = crosscheck_with_seqeval(last["_gold"], last["_pred"])
    # The README advertises seqeval as an independent cross-check — enforce it in code
    # rather than in prose. If seqeval is installed, our entity-F1 must match it; if it
    # is absent, warn loudly so the advertised check doesn't silently disappear.
    if "f1" in xcheck:
        assert abs(xcheck["f1"] - last["f1"]) < 1e-4, (
            f"in-house F1 {last['f1']:.6f} disagrees with seqeval {xcheck['f1']:.6f}")
    else:
        print(f"!! WARNING: seqeval cross-check unavailable ({xcheck.get('error')}) — "
              f"install seqeval to verify the entity-F1 independently.")
    result = dict(
        n_pool=len(pool), n_test=len(test), seeds=cfg.al.seeds,
        f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
        macro_f1_mean=float(np.mean(macros)),
        precision_mean=float(np.mean(ps)), recall_mean=float(np.mean(rs)),
        by_type={t: dict(f1_mean=float(np.mean(v))) for t, v in by_type_acc.items()},
        seqeval_crosscheck=xcheck,
    )
    write_json(RESULTS_DIR / "baseline.json", result)
    print(f"\nFULL-DATA BASELINE: F1 = {result['f1_mean']:.4f} ± {result['f1_std']:.4f} "
          f"(micro) | macro-F1 = {result['macro_f1_mean']:.4f} "
          f"(P {result['precision_mean']:.4f} / R {result['recall_mean']:.4f})")
    print("Per-type F1:", {t: round(d["f1_mean"], 3) for t, d in result["by_type"].items()})
    print("seqeval cross-check:", {k: round(v, 4) for k, v in xcheck.items() if isinstance(v, float)})
    print(f"-> wrote {RESULTS_DIR/'baseline.json'}")


if __name__ == "__main__":
    main()
