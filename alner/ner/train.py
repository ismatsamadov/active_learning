"""Train and evaluate the BiLSTM-CRF (with early stopping on dev entity-F1)."""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import NERDataset, Vocab, make_collate
from .metrics import prf, prf_by_type
from .model import BiLSTMCRF


def _loader(records, vocab, batch_size, shuffle):
    return DataLoader(NERDataset(records, vocab), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=make_collate(vocab))


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: BiLSTMCRF, records: List[dict], vocab: Vocab, device: str,
             batch_size: int = 64, by_type: bool = False) -> Dict:
    model.eval()
    gold: List[List[str]] = []
    pred: List[List[str]] = []
    for batch in _loader(records, vocab, batch_size, shuffle=False):
        batch = _to_device(batch, device)
        paths = model.predict(batch)
        lengths = batch["lengths"].tolist()
        idxs = batch["indices"].tolist()
        for k, path in enumerate(paths):
            n = lengths[k]
            rec = records[idxs[k]]
            pred.append([vocab.id2tag[t] for t in path[:n]])
            # score against the human-annotated labels (standard NER practice)
            gold.append(rec["tags"][:n])
    out = prf(gold, pred)
    if by_type:
        out["by_type"] = prf_by_type(gold, pred)
    out["_gold"], out["_pred"] = gold, pred
    return out


def train_model(labeled: List[dict], vocab: Vocab, cfg, device: str, seed: int,
                dev: Optional[List[dict]] = None, verbose: bool = False
                ) -> Tuple[BiLSTMCRF, Dict]:
    """Train on `labeled`; early-stop on a dev split (carved out if not given)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if dev is None:
        idx = rng.permutation(len(labeled))
        n_dev = max(1, int(round(len(labeled) * cfg.dev_fraction)))
        dev = [labeled[i] for i in idx[:n_dev]]
        train = [labeled[i] for i in idx[n_dev:]]
    else:
        train = labeled

    model = BiLSTMCRF(vocab, cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)

    best_f1, best_state, best_epoch, since = -1.0, None, -1, 0
    history = []
    for epoch in range(1, cfg.train.max_epochs + 1):
        model.train()
        total = 0.0
        for batch in _loader(train, vocab, cfg.train.batch_size, shuffle=True):
            batch = _to_device(batch, device)
            opt.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            total += loss.item() * batch["words"].size(0)
        dev_m = evaluate(model, dev, vocab, device, batch_size=64)
        history.append(dict(epoch=epoch, train_loss=total / max(len(train), 1),
                            dev_f1=dev_m["f1"]))
        if verbose:
            print(f"  ep{epoch:02d} loss={total/max(len(train),1):.3f} dev_f1={dev_m['f1']:.4f}")
        if dev_m["f1"] > best_f1:
            best_f1, best_epoch, since = dev_m["f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= cfg.train.patience and epoch >= cfg.train.min_epochs:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dict(history=history, best_dev_f1=best_f1, best_epoch=best_epoch,
                       n_train=len(train), n_dev=len(dev))
