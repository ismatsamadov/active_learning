"""The active-learning experiment: random vs uncertainty vs hybrid, multiple seeds.

Runs each strategy x seed through the pool-based AL loop, aggregates per-budget
mean +/- std, plots the learning curve against the full-data baseline, and computes
the headline label-efficiency numbers (how few labels each strategy needs to reach
the baseline). This is the real, reproducible version of the thesis's headline.

Usage: python scripts/04_run_active_learning.py [--fast]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import load_corpus, pick_device
from alner.config import fast_config, full_config
from alner.al.loop import run_active_learning
from alner.analysis.curves import aggregate, plot_learning_curves, label_efficiency
from alner.utils import RESULTS_DIR, read_json, write_json


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
          f"strategies={cfg.al.strategies}  seeds={cfg.al.seeds}")
    print(f"budget: seed={cfg.al.seed_size}, +{cfg.al.query_size}/iter x "
          f"{cfg.al.n_iterations} -> max {cfg.al.seed_size + cfg.al.query_size*cfg.al.n_iterations}")

    runs = []
    t0 = time.time()
    for strat in cfg.al.strategies:
        for seed in cfg.al.seeds:
            ts = time.time()
            r = run_active_learning(pool, test, vocab, cfg, strat, seed, device)
            runs.append(r)
            final = r["curve"][-1]
            print(f"  {strat:11s} seed {seed}: final F1={final['f1']:.4f} "
                  f"@ {final['n_labeled']} labels  ({time.time()-ts:.0f}s)")
    print(f"total AL time {time.time()-t0:.0f}s")

    agg = aggregate(runs)

    # baseline (from script 03 if available, else final full-pool point is not it —
    # use the saved baseline)
    bpath = RESULTS_DIR / "baseline.json"
    baseline = read_json(bpath)["f1_mean"] if bpath.exists() else max(
        max(s["mean"]) for s in agg.values())

    fig = plot_learning_curves(agg, baseline)
    eff = label_efficiency(agg, baseline, tol=0.01)
    pool_n = len(pool)
    for strat, e in eff.items():
        if e["budget_to_reach"] is not None:
            e["pct_of_pool"] = round(100 * e["budget_to_reach"] / pool_n, 1)

    result = dict(baseline_f1=baseline, pool_size=pool_n, config=cfg.to_dict(),
                  aggregate=agg, runs=runs, label_efficiency=eff)
    write_json(RESULTS_DIR / "al_results.json", result)

    print(f"\n=== LABEL EFFICIENCY (reach within 0.01 F1 of baseline {baseline:.3f}) ===")
    for strat in ["random", "uncertainty", "hybrid"]:
        if strat in eff:
            e = eff[strat]
            b = e["budget_to_reach"]
            print(f"  {strat:11s}: {'%d labels (%.1f%% of pool)'%(b, e['pct_of_pool']) if b else 'did not reach'}"
                  f"   final={e['final_mean']:.4f}")
    print(f"-> wrote {RESULTS_DIR/'al_results.json'} and {fig}")


if __name__ == "__main__":
    main()
