"""Demonstrate the NER -> structured-fields bridge on the held-out test set.

Trains the BiLSTM-CRF on the full pool, runs it over the test descriptions, parses
predicted entities back into course fields, and measures field-level recovery vs
the true data. Writes worked examples (text -> extracted -> truth).

Usage: python scripts/06_extraction_bridge.py [--fast]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import load_corpus, pick_device
from alner.config import fast_config, full_config
from alner.ner.train import train_model, evaluate
from alner.analysis.bridge import evaluate_field_recovery
from alner.utils import DATA_DIR, RESULTS_DIR, write_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = fast_config() if args.fast else full_config()
    device = pick_device(args.device)

    pool, test, vocab = load_corpus()
    set_seed(42)
    model, _ = train_model(pool, vocab, cfg, device, seed=42)
    m = evaluate(model, test, vocab, device)

    # predict per record in test order so predictions align with `test` indices
    from alner.ner.data import NERDataset, make_collate
    from torch.utils.data import DataLoader
    import torch
    pred_by_index = {}
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(NERDataset(test, vocab), batch_size=64, shuffle=False,
                                collate_fn=make_collate(vocab)):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            paths = model.predict(batch)
            for k, path in enumerate(paths):
                idx = int(batch["indices"][k]); n = int(batch["lengths"][k])
                pred_by_index[idx] = [vocab.id2tag[t] for t in path[:n]]
    pred_tags = [pred_by_index[i] for i in range(len(test))]

    courses = pd.read_csv(DATA_DIR / "courses.csv")
    rep = evaluate_field_recovery(test, pred_tags, courses)
    rep["test_entity_f1"] = m["f1"]

    # --- WIRED end-to-end flow: NER over the WHOLE corpus -> reconstruct the course
    #     table -> recompute the headline business numbers FROM model output, and
    #     compare to the ground-truth-derived numbers. (Closes the integration gap.)
    from alner.utils import read_jsonl
    from alner.analysis.bridge import reconstruct_courses_from_ner, business_agreement
    all_recs = read_jsonl(DATA_DIR / "ner_corpus.jsonl")
    all_pred = {}
    with torch.no_grad():
        for batch in DataLoader(NERDataset(all_recs, vocab), batch_size=64, shuffle=False,
                                collate_fn=make_collate(vocab)):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            for k, path in enumerate(model.predict(batch)):
                idx = int(batch["indices"][k]); n = int(batch["lengths"][k])
                all_pred[idx] = [vocab.id2tag[t] for t in path[:n]]
    all_pred_tags = [all_pred[i] for i in range(len(all_recs))]
    extracted = reconstruct_courses_from_ner(all_recs, all_pred_tags)
    rep["integration"] = business_agreement(extracted, courses)
    write_json(RESULTS_DIR / "extraction_bridge.json", rep)

    print(f"Test entity-F1 = {m['f1']:.4f}")
    print("Field recovery (NER-extracted field == true field):")
    for f, v in rep["field_recovery"].items():
        print(f"  {f:18s}: {v:.3f}" if v is not None else f"  {f:18s}: n/a")
    print("\nWorked examples (text -> extracted vs truth):")
    for ex in rep["examples"][:3]:
        print(f"  TEXT: {ex['text']}")
        print(f"    extracted: {ex['extracted']}")
        print(f"    truth    : {ex['truth']}")
    print("\nEnd-to-end integration (business numbers FROM NER vs FROM truth):")
    for k, v in rep["integration"].items():
        print(f"  {k}: {v}")
    print(f"-> wrote {RESULTS_DIR/'extraction_bridge.json'}")


if __name__ == "__main__":
    main()
