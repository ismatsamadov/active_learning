"""Brute-force correctness tests for the from-scratch CRF.

Validates the forward-algorithm partition and the Viterbi decode against explicit
enumeration of all tag sequences on small examples. Run:
    .venv/bin/python tests/test_crf.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alner.ner.crf import CRF


def brute_force_logZ(crf, emissions, length):
    """log sum_exp over every possible tag path of the given length."""
    K = crf.num_tags
    em = emissions[0]  # (T,K)
    scores = []
    for path in itertools.product(range(K), repeat=length):
        s = crf.start_transitions[path[0]] + em[0, path[0]]
        for t in range(1, length):
            s = s + crf.transitions[path[t - 1], path[t]] + em[t, path[t]]
        s = s + crf.end_transitions[path[-1]]
        scores.append(s)
    return torch.logsumexp(torch.stack(scores), dim=0)


def brute_force_best(crf, emissions, length):
    K = crf.num_tags
    em = emissions[0]
    best_s, best_p = None, None
    for path in itertools.product(range(K), repeat=length):
        s = crf.start_transitions[path[0]] + em[0, path[0]]
        for t in range(1, length):
            s = s + crf.transitions[path[t - 1], path[t]] + em[t, path[t]]
        s = s + crf.end_transitions[path[-1]]
        if best_s is None or s > best_s:
            best_s, best_p = s, list(path)
    return best_s, best_p


def main():
    torch.manual_seed(0)
    K, T = 4, 5
    crf = CRF(K)
    # randomise transitions so the test is non-trivial
    with torch.no_grad():
        crf.start_transitions.uniform_(-1, 1)
        crf.end_transitions.uniform_(-1, 1)
        crf.transitions.uniform_(-1, 1)

    emissions = torch.randn(1, T, K)
    mask = torch.ones(1, T, dtype=torch.bool)

    # 1) partition
    logZ = crf._partition(emissions, mask).item()
    logZ_bf = brute_force_logZ(crf, emissions, T).item()
    assert abs(logZ - logZ_bf) < 1e-4, f"partition mismatch {logZ} vs {logZ_bf}"
    print(f"[ok] partition: forward={logZ:.5f}  brute={logZ_bf:.5f}")

    # 2) Viterbi decode
    path = crf.decode(emissions, mask)[0]
    best_s_bf, path_bf = brute_force_best(crf, emissions, T)
    assert path == path_bf, f"viterbi path mismatch {path} vs {path_bf}"
    print(f"[ok] viterbi: {path} (best score {best_s_bf.item():.4f})")

    # 3) variable length / padding: pad to T+2 with mask, must match length-T result
    pad = 2
    em_pad = torch.cat([emissions, torch.randn(1, pad, K)], dim=1)
    mask_pad = torch.cat([mask, torch.zeros(1, pad, dtype=torch.bool)], dim=1)
    logZ_pad = crf._partition(em_pad, mask_pad).item()
    assert abs(logZ_pad - logZ_bf) < 1e-4, f"masked partition {logZ_pad} vs {logZ_bf}"
    path_pad = crf.decode(em_pad, mask_pad)[0]
    assert path_pad == path_bf, f"masked viterbi {path_pad} vs {path_bf}"
    print(f"[ok] masking: partition & viterbi unaffected by padding")

    # 4) gold score + nll sanity: nll >= 0 and finite
    tags = torch.tensor([path_bf])
    nll = crf(emissions, tags, mask, reduction="mean").item()
    assert nll == nll and nll >= -1e-4, f"bad nll {nll}"
    # nll of the *best* path should be the smallest among all paths
    nlls = []
    for p in itertools.product(range(K), repeat=T):
        nlls.append(crf(emissions, torch.tensor([list(p)]), mask, "mean").item())
    assert abs(min(nlls) - nll) < 1e-4, "best path is not the min-nll path"
    print(f"[ok] nll: best-path nll={nll:.5f} is the minimum over all {K**T} paths")

    print("\nALL CRF TESTS PASSED")


if __name__ == "__main__":
    main()
