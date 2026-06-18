"""Synthesise all results into a thesis-ready Markdown report (reports/RESULTS.md).

Pulls together the data validation, full-data baseline, active-learning learning
curve + label efficiency, the NER->fields extraction bridge, and the business
analysis — with the real numbers that replace the thesis's illustrative ones.

Usage: python scripts/07_make_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alner.utils import RESULTS_DIR, REPORTS_DIR, read_json


def _try(name):
    p = RESULTS_DIR / name
    return read_json(p) if p.exists() else None


def main():
    base = _try("baseline.json")
    al = _try("al_results.json")
    biz = _try("business_analysis.json")
    bridge = _try("extraction_bridge.json")
    dataval = _try("data_validation.json")

    L = []
    w = L.append
    w("# Active Learning for NER — Real Experimental Results\n")
    w("_Auto-generated from the implementation in this repository. These numbers "
      "replace the thesis's illustrative values with measured, reproducible ones._\n")

    # ---- headline ----
    if al and base:
        b = al["baseline_f1"]; eff = al["label_efficiency"]; pool = al["pool_size"]
        w("## Headline\n")
        hyb = eff.get("hybrid", {}); rnd = eff.get("random", {})
        if hyb.get("budget_to_reach"):
            w(f"- **Hybrid active learning reaches the full-data baseline "
              f"(F1 = {b:.3f}) using only {hyb['budget_to_reach']} labelled "
              f"sentences — {hyb['pct_of_pool']}% of the {pool}-sentence pool.**")
        if rnd.get("budget_to_reach"):
            w(f"- Random sampling reaches the same bar at "
              f"{rnd['budget_to_reach']} labels ({rnd['pct_of_pool']}% of pool).")
        elif rnd:
            w(f"- Random sampling does **not** reach within 0.01 F1 of the baseline "
              f"within the budget (final F1 = {rnd.get('final_mean')}).")
        w("")

    # ---- baseline ----
    if base:
        w("## 1. Full-data supervised baseline (BiLSTM-CRF)\n")
        w(f"Trained on all {base['n_pool']} pool sentences, evaluated on "
          f"{base['n_test']} held-out test sentences, averaged over seeds {base['seeds']}.\n")
        w("| Metric | Value |")
        w("| --- | --- |")
        w(f"| Entity-level F1 (micro) | **{base['f1_mean']:.4f} ± {base['f1_std']:.4f}** |")
        if "macro_f1_mean" in base:
            w(f"| Entity-level F1 (macro) | {base['macro_f1_mean']:.4f} |")
        w(f"| Precision | {base['precision_mean']:.4f} |")
        w(f"| Recall | {base['recall_mean']:.4f} |")
        xc = base.get("seqeval_crosscheck", {})
        if "f1" in xc:
            w(f"| seqeval F1 (independent check) | {xc['f1']:.4f} |")
        w("")
        if base.get("by_type"):
            w("**Per-entity-type F1:**\n")
            w("| Type | F1 |")
            w("| --- | --- |")
            for t, d in sorted(base["by_type"].items()):
                w(f"| {t} | {d['f1_mean']:.3f} |")
            w("")

    # ---- AL curve ----
    if al:
        agg = al["aggregate"]
        w("## 2. Active-learning learning curve\n")
        w("Test entity-F1 (mean ± std over seeds) at each annotation budget.\n")
        strats = [s for s in ["random", "least_confidence", "uncertainty", "hybrid"] if s in agg]
        ns = agg[strats[0]]["n"]
        w("| Labels | " + " | ".join(strats) + " |")
        w("| " + " --- |" * (len(strats) + 1))
        for i, n in enumerate(ns):
            row = [f"{n}"]
            for s in strats:
                a = agg[s]
                row.append(f"{a['mean'][i]:.3f} ± {a['std'][i]:.3f}")
            w("| " + " | ".join(row) + " |")
        w("")
        w("![learning curve](../figures/learning_curve.png)\n")

    # ---- bridge ----
    if bridge:
        w("## 3. NER → business-data bridge (does NER actually produce the fields?)\n")
        w(f"The trained model is run over the test descriptions; predicted entities "
          f"are parsed back into course fields and compared to ground truth "
          f"(test entity-F1 = {bridge.get('test_entity_f1', float('nan')):.3f}).\n")
        w("| Field | Recovery accuracy |")
        w("| --- | --- |")
        for f, v in bridge["field_recovery"].items():
            w(f"| {f} | {v:.3f} |" if v is not None else f"| {f} | n/a |")
        w("")
        if bridge.get("examples"):
            ex = bridge["examples"][0]
            w("**Worked example:**\n")
            w(f"> {ex['text']}\n")
            w(f"- extracted: `{ex['extracted']}`")
            w(f"- truth: `{ex['truth']}`\n")

    # ---- business ----
    if biz:
        w("## 4. Business-oriented analysis (Figures 5–11)\n")
        fp = biz["fig6_7"]
        w(f"- Paid courses average **{fp['paid_mean']:,}** enrolments vs **{fp['free_mean']:,}** "
          f"for free — a **{fp['ratio']}×** gap.")
        dur = {k: v for k, v in biz["fig9"].items() if v is not None}
        peak = max(dur, key=dur.get)
        w(f"- Engagement peaks at **{peak}-week** courses ({dur[peak]:,} avg); "
          f"1-week and 8+-week courses are far lower (2–5 week window).")
        w(f"- Star categories (high demand + high satisfaction): {', '.join(biz['fig11']['stars'])}.")
        w("\n**Executive decision summary:**\n")
        w("| Priority | Finding | Recommended action |")
        w("| --- | --- | --- |")
        for r in biz["executive_summary"]:
            w(f"| {r['priority']} | {r['finding']} | {r['action']} |")
        w("\nFigures: `figures/fig5…fig11`.\n")

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "RESULTS.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"-> wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
