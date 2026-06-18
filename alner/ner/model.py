"""BiLSTM-CRF NER model with a character-CNN.

Backbone: word embeddings + a character-level CNN (Ma & Hovy, 2016) feeding a
bidirectional LSTM, decoded by the from-scratch linear-chain CRF (Lample et al.,
2016). The character CNN matters for this corpus because prices ("£39"),
durations ("5-week") and enrolment counts ("24,000") are open-vocabulary tokens
that word embeddings alone would treat as <unk>.

Beyond loss/decoding the model exposes the two signals the active learner needs:
* ``uncertainty`` — sequence-level MNLP (Shen et al., 2018)
* ``sentence_embeddings`` — mean-pooled BiLSTM states, for diversity/core-set
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .crf import CRF
from .data import Vocab


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab: Vocab, cfg):
        super().__init__()
        self.vocab = vocab
        m = cfg
        self.word_emb = nn.Embedding(vocab.n_words, m.word_emb_dim, padding_idx=0)
        self.char_emb = nn.Embedding(vocab.n_chars, m.char_emb_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(m.char_emb_dim, m.char_channels,  # char-CNN: Ma & Hovy (2016)
                                  kernel_size=m.char_kernel, padding=m.char_kernel // 2)
        self.dropout = nn.Dropout(m.dropout)
        lstm_in = m.word_emb_dim + m.char_channels
        # LSTM: Hochreiter & Schmidhuber (1997); bidirectional: Graves & Schmidhuber (2005)
        self.lstm = nn.LSTM(lstm_in, m.lstm_hidden, num_layers=m.lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=m.dropout if m.lstm_layers > 1 else 0.0)
        self.hidden2tag = nn.Linear(2 * m.lstm_hidden, vocab.n_tags)
        self.crf = CRF(vocab.n_tags)

    # -- shared feature extractor -----------------------------------------
    def _features(self, words, chars, mask):
        B, T, W = chars.shape
        # char-CNN: (B*T, cdim, W) -> conv -> max over W -> (B*T, channels)
        ce = self.char_emb(chars.view(B * T, W)).transpose(1, 2)   # (B*T, cdim, W)
        cc = torch.relu(self.char_cnn(ce))                          # (B*T, ch, W)
        cc, _ = cc.max(dim=2)                                       # (B*T, ch)
        cc = cc.view(B, T, -1)
        we = self.word_emb(words)                                  # (B,T,wd)
        x = self.dropout(torch.cat([we, cc], dim=-1))
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        return self.dropout(out)                                   # (B,T,2H)

    def emissions(self, words, chars, mask):
        return self.hidden2tag(self._features(words, chars, mask))  # (B,T,K)

    # -- training / inference ---------------------------------------------
    def loss(self, batch):
        em = self.emissions(batch["words"], batch["chars"], batch["mask"])
        return self.crf(em, batch["tags"], batch["mask"], reduction="mean")

    @torch.no_grad()
    def predict(self, batch):
        em = self.emissions(batch["words"], batch["chars"], batch["mask"])
        return self.crf.decode(em, batch["mask"])

    # -- active-learning signals ------------------------------------------
    @torch.no_grad()
    def uncertainty(self, batch, kind: str = "mnlp"):
        """Higher = more uncertain. ``kind='mnlp'`` (Shen et al. 2018, length-norm,
        default) or ``kind='least_confidence'`` (thesis sec 2.3's literal
        'lower maximum probability', i.e. 1 - P(best path))."""
        em = self.emissions(batch["words"], batch["chars"], batch["mask"])
        if kind == "least_confidence":
            return self.crf.least_confidence(em, batch["mask"])
        mnlp = self.crf.best_path_normalized_logprob(em, batch["mask"])  # higher=confident
        return -mnlp

    @torch.no_grad()
    def sentence_embeddings(self, batch):
        feats = self._features(batch["words"], batch["chars"], batch["mask"])  # (B,T,2H)
        mask = batch["mask"].unsqueeze(-1).to(feats.dtype)
        summed = (feats * mask).sum(1)
        return summed / mask.sum(1).clamp(min=1)                  # (B,2H) mean-pool
