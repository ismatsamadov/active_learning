"""Linear-chain Conditional Random Field, implemented from scratch in PyTorch.

This is the CRF decoding/training layer of the BiLSTM-CRF (Lample et al., 2016).
Implementing it here — rather than importing a library — means the thesis's stated
architecture genuinely lives in the repository, and it lets us expose the
sequence-level probabilities the active-learning query strategies need
(e.g. MNLP, Shen et al., 2018).

Shapes (batch_first):
    emissions : (B, T, K)   float  — per-token tag scores from the BiLSTM
    tags      : (B, T)      long   — gold tag ids
    mask      : (B, T)      bool   — 1 for real tokens, 0 for padding

The implementation mirrors the well-established forward-algorithm / Viterbi
formulation and is unit-tested against brute-force enumeration in tests/.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))  # [from, to]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    # -- log-likelihood ----------------------------------------------------
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor,
                mask: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Return the NEGATIVE log-likelihood (a loss to minimise)."""
        mask = mask.to(torch.bool)
        numerator = self._score(emissions, tags, mask)        # (B,)
        denominator = self._partition(emissions, mask)        # (B,)
        llh = numerator - denominator                          # log p(tags | x)
        nll = -llh
        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        if reduction == "mean":
            return nll.mean()
        if reduction == "token_mean":
            return nll.sum() / mask.sum().clamp(min=1)
        raise ValueError(reduction)

    def _score(self, emissions, tags, mask) -> torch.Tensor:
        """Unnormalised log-score of the gold tag sequence."""
        B, T, K = emissions.shape
        mask = mask.to(emissions.dtype)
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[torch.arange(B), 0, tags[:, 0]]
        for t in range(1, T):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[torch.arange(B), t, tags[:, t]]
            score = score + (trans + emit) * mask[:, t]
        # add end transition from the last real tag of each sequence
        seq_lens = mask.sum(dim=1).long() - 1            # index of last token
        last_tags = tags[torch.arange(B), seq_lens]
        score = score + self.end_transitions[last_tags]
        return score

    def _partition(self, emissions, mask) -> torch.Tensor:
        """log sum_exp over all tag sequences (forward algorithm)."""
        B, T, K = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]   # (B,K)
        for t in range(1, T):
            # broadcast: (B,K_from,1) + (K_from,K_to) + (B,1,K_to)
            emit = emissions[:, t].unsqueeze(1)                 # (B,1,K)
            trans = self.transitions.unsqueeze(0)               # (1,K,K)
            next_alpha = torch.logsumexp(alpha.unsqueeze(2) + trans + emit, dim=1)  # (B,K)
            m = mask[:, t].unsqueeze(1).to(emissions.dtype)     # (B,1)
            alpha = m * next_alpha + (1 - m) * alpha            # keep alpha past padding
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)                    # (B,)

    # -- decoding ----------------------------------------------------------
    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Viterbi: most likely tag sequence per example (variable length)."""
        mask = mask.to(torch.bool)
        B, T, K = emissions.shape
        history = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]   # (B,K)
        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B,K_from,K_to)
            best_score, best_prev = broadcast.max(dim=1)                    # (B,K_to)
            best_score = best_score + emissions[:, t]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, best_score, score)
            history.append(best_prev)
        score = score + self.end_transitions.unsqueeze(0)

        seq_lens = mask.sum(dim=1).long()
        best_paths: List[List[int]] = []
        for b in range(B):
            n = int(seq_lens[b].item())
            last_tag = int(score[b].argmax().item())
            best = [last_tag]
            # history[t] corresponds to transition into position t+1
            for t in range(n - 2, -1, -1):
                last_tag = int(history[t][b, last_tag].item())
                best.append(last_tag)
            best.reverse()
            best_paths.append(best)
        return best_paths

    # -- uncertainty signals ----------------------------------------------
    @torch.no_grad()
    def _best_logprob_and_len(self, emissions, mask):
        """Return (log P(best path), length) per example: best Viterbi score - logZ."""
        mask = mask.to(torch.bool)
        B, T, K = emissions.shape
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, _ = broadcast.max(dim=1)
            best_score = best_score + emissions[:, t]
            m = mask[:, t].unsqueeze(1).to(emissions.dtype)
            score = m * best_score + (1 - m) * score
        best = (score + self.end_transitions.unsqueeze(0)).max(dim=1).values  # (B,)
        logZ = self._partition(emissions, mask)
        lengths = mask.sum(dim=1).clamp(min=1).to(emissions.dtype)
        return best - logZ, lengths

    def best_path_normalized_logprob(self, emissions, mask):
        """MNLP signal (Shen et al., 2018): log P(best path) / length, per example.
        Returns (B,); HIGHER = more confident. Length-normalised to remove the
        trivial bias toward short sequences."""
        logp, lengths = self._best_logprob_and_len(emissions, mask)
        return logp / lengths

    def least_confidence(self, emissions, mask):
        """Sequence-level least-confidence (Lewis & Gale, 1994; thesis sec 2.3's
        'lower maximum probability'): 1 - P(best path). Returns (B,); HIGHER = more
        uncertain. Not length-normalised, so it reads the thesis wording literally."""
        logp, _ = self._best_logprob_and_len(emissions, mask)
        return 1.0 - torch.exp(logp.clamp(max=0.0))
