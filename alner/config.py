"""Experiment configuration with FAST (smoke) and FULL presets."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class ModelConfig:
    word_emb_dim: int = 100
    char_emb_dim: int = 25
    char_channels: int = 30      # char-CNN output channels (Ma & Hovy, 2016 style)
    char_kernel: int = 3
    lstm_hidden: int = 128       # per direction
    lstm_layers: int = 1
    dropout: float = 0.35


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 60
    patience: int = 10           # early-stopping on dev entity-F1
    grad_clip: float = 5.0
    min_epochs: int = 15         # warmup past the all-O basin before stopping


@dataclass
class ALConfig:
    seed_size: int = 120         # initial labelled sentences (reliably escapes all-O)
    query_size: int = 40         # sentences annotated per AL iteration
    n_iterations: int = 13       # -> labelled budget 120,160,...,640
    uncertainty_pool_factor: int = 4   # hybrid: take top (factor*query_size) uncertain, then diversify
    # least_confidence = thesis sec 2.3 literal wording; uncertainty = MNLP (Shen 2018)
    strategies: List[str] = field(default_factory=lambda: ["random", "least_confidence", "uncertainty", "hybrid"])
    seeds: List[int] = field(default_factory=lambda: [13, 29, 42])


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    al: ALConfig = field(default_factory=ALConfig)
    device: str = "cpu"          # set to "mps"/"cuda" by scripts if available
    dev_fraction: float = 0.15   # carved out of the labelled set for early stopping

    def to_dict(self):
        return asdict(self)


def fast_config() -> Config:
    """Tiny preset for a quick end-to-end smoke test (seconds, not minutes)."""
    c = Config()
    c.model.word_emb_dim = 50
    c.model.lstm_hidden = 64
    c.train.max_epochs = 8
    c.train.patience = 3
    c.train.min_epochs = 2
    c.al.seed_size = 80
    c.al.query_size = 120
    c.al.n_iterations = 4
    c.al.strategies = ["random", "least_confidence", "uncertainty", "hybrid"]
    c.al.seeds = [13]
    return c


def full_config() -> Config:
    return Config()
