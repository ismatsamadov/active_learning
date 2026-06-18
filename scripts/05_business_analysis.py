"""Reproduce the business-analysis figures (5-11) + executive decision table.

Usage: python scripts/05_business_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alner.analysis import business as B
from alner.utils import DATA_DIR, RESULTS_DIR, FIGURES_DIR, write_json


def main():
    df = pd.read_csv(DATA_DIR / "courses.csv")
    stats = {}
    stats["fig5"] = B.fig5_category(df)
    stats["fig6_7"] = B.fig6_7_free_paid(df)
    stats["fig8"] = B.fig8_price(df)
    stats["fig9"] = B.fig9_duration(df)
    stats["fig10"] = B.fig10_partner(df)
    stats["fig11"] = B.fig11_opportunity(df)
    stats["executive_summary"] = B.executive_summary(df, stats)

    write_json(RESULTS_DIR / "business_analysis.json", stats)
    print("Business analysis written. Figures in", FIGURES_DIR)
    print(f"  free vs paid: {stats['fig6_7']['free_mean']:,} vs "
          f"{stats['fig6_7']['paid_mean']:,} ({stats['fig6_7']['ratio']}x)")
    print(f"  duration peak: {max(stats['fig9'], key=stats['fig9'].get)} weeks")
    print(f"  star categories (Fig 11): {stats['fig11']['stars']}")
    print("  executive summary:")
    for r in stats["executive_summary"]:
        print(f"    [{r['priority']}] {r['finding']}")


if __name__ == "__main__":
    main()
