"""Plot active-learning learning curves (mean +/- std across seeds)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..utils import FIGURES_DIR

COLORS = {"random": "#94a3b8", "least_confidence": "#16a34a",
          "uncertainty": "#f59e0b", "hybrid": "#2563eb"}
LABELS = {"random": "Random (baseline control)",
          "least_confidence": "Least-confidence (thesis §2.3)",
          "uncertainty": "Uncertainty (MNLP)",
          "hybrid": "Hybrid (uncertainty + diversity)"}
_ORDER = ["random", "least_confidence", "uncertainty", "hybrid"]


def aggregate(runs: List[dict]) -> Dict[str, dict]:
    """runs: list of {strategy, seed, curve:[{n_labeled,f1,...}]} -> per-strategy stats."""
    by_strat = defaultdict(lambda: defaultdict(list))   # strat -> n_labeled -> [f1,...]
    for r in runs:
        for p in r["curve"]:
            by_strat[r["strategy"]][p["n_labeled"]].append(p["f1"])
    out = {}
    for strat, d in by_strat.items():
        ns = sorted(d)
        mean = np.array([np.mean(d[n]) for n in ns])
        std = np.array([np.std(d[n]) for n in ns])
        out[strat] = dict(n=ns, mean=mean.tolist(), std=std.tolist())
    return out


def plot_learning_curves(agg: Dict[str, dict], baseline: float,
                         fname: str = "learning_curve.png") -> str:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axhline(baseline, color="#dc2626", ls="--", lw=1.8,
               label=f"Full-data baseline (F1={baseline:.3f})")
    for strat in _ORDER:
        if strat not in agg:
            continue
        s = agg[strat]
        n = np.array(s["n"]); mean = np.array(s["mean"]); std = np.array(s["std"])
        ax.plot(n, mean, marker="o", color=COLORS[strat], lw=2.3, label=LABELS[strat])
        ax.fill_between(n, mean - std, mean + std, color=COLORS[strat], alpha=0.15)
    ax.set_xlabel("Labeled sentences (annotation budget)")
    ax.set_ylabel("Test entity-level F1")
    ax.set_title("Active learning vs random sampling for NER\n(BiLSTM-CRF; mean ± std over seeds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = FIGURES_DIR / fname
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def label_efficiency(agg: Dict[str, dict], baseline: float, tol: float = 0.01) -> Dict:
    """Smallest budget at which each strategy reaches (baseline - tol)."""
    target = baseline - tol
    out = {}
    for strat, s in agg.items():
        n, mean = s["n"], s["mean"]
        hit = next((n[i] for i, v in enumerate(mean) if v >= target), None)
        out[strat] = dict(budget_to_reach=hit, target=round(target, 4),
                          final_mean=round(mean[-1], 4))
    return out
