"""Vocabulary + tensorisation for the NER corpus."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .. import BIO_TAGS

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    """Word, character and tag vocabularies built from the training records."""

    def __init__(self, word2id, char2id, tag2id):
        self.word2id = word2id
        self.char2id = char2id
        self.tag2id = tag2id
        self.id2tag = {i: t for t, i in tag2id.items()}

    @classmethod
    def build(cls, records: List[dict], min_word_freq: int = 1) -> "Vocab":
        from collections import Counter
        wc, cc = Counter(), Counter()
        for r in records:
            for tok in r["tokens"]:
                wc[tok.lower()] += 1
                for ch in tok:
                    cc[ch] += 1
        word2id = {PAD: 0, UNK: 1}
        for w, c in wc.most_common():
            if c >= min_word_freq:
                word2id[w] = len(word2id)
        char2id = {PAD: 0, UNK: 1}
        for ch, _ in cc.most_common():
            char2id[ch] = len(char2id)
        tag2id = {t: i for i, t in enumerate(BIO_TAGS)}
        return cls(word2id, char2id, tag2id)

    @property
    def n_words(self): return len(self.word2id)
    @property
    def n_chars(self): return len(self.char2id)
    @property
    def n_tags(self): return len(self.tag2id)

    def encode_word(self, tok: str) -> int:
        return self.word2id.get(tok.lower(), self.word2id[UNK])

    def encode_chars(self, tok: str) -> List[int]:
        return [self.char2id.get(ch, self.char2id[UNK]) for ch in tok]


class NERDataset(Dataset):
    def __init__(self, records: List[dict], vocab: Vocab):
        self.records = records
        self.vocab = vocab

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        toks = r["tokens"]
        words = [self.vocab.encode_word(t) for t in toks]
        chars = [self.vocab.encode_chars(t) for t in toks]
        tags = [self.vocab.tag2id[g] for g in r["tags"]]
        return dict(words=words, chars=chars, tags=tags, length=len(toks), index=i)


def make_collate(vocab: Vocab):
    pad_w = vocab.word2id[PAD]
    pad_c = vocab.char2id[PAD]

    def collate(batch):
        B = len(batch)
        T = max(b["length"] for b in batch)
        W = max((len(c) for b in batch for c in b["chars"]), default=1)
        W = max(W, vocab_kernel_min)  # ensure >= conv kernel width

        words = torch.full((B, T), pad_w, dtype=torch.long)
        chars = torch.full((B, T, W), pad_c, dtype=torch.long)
        tags = torch.zeros((B, T), dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.bool)
        lengths = torch.zeros(B, dtype=torch.long)
        indices = torch.zeros(B, dtype=torch.long)

        for i, b in enumerate(batch):
            n = b["length"]
            lengths[i] = n
            indices[i] = b["index"]
            words[i, :n] = torch.tensor(b["words"], dtype=torch.long)
            tags[i, :n] = torch.tensor(b["tags"], dtype=torch.long)
            mask[i, :n] = True
            for j, ch in enumerate(b["chars"]):
                chars[i, j, : len(ch)] = torch.tensor(ch, dtype=torch.long)
        return dict(words=words, chars=chars, tags=tags, mask=mask,
                    lengths=lengths, indices=indices)

    return collate


# minimum word width so the char-CNN kernel (default 3) always fits
vocab_kernel_min = 3
