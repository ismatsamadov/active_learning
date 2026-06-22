"""alner — Active Learning for Named Entity Recognition.

A faithful, reproducible implementation of the master thesis
"Active Learning for Named Entity Recognition: Reducing Labeled Data Requirements"
(Samadov, UNEC, 2026).

Components
----------
- data    : real FutureLearn course dataset + gold-labelled BIO NER corpus
- ner     : BiLSTM-CRF sequence labeller (CRF implemented from scratch)
- al      : pool-based active-learning loop (random / uncertainty / diversity / hybrid)
- analysis: business-oriented analysis of the extracted entities
"""

__version__ = "1.0.0"

# Canonical BIO tag inventory (entity-level, CoNLL-style).
ENTITY_TYPES = ["ORG", "CAT", "PRICE", "DUR", "ENROLL", "LEVEL"]

BIO_TAGS = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ("B", "I")]
# -> 13 tags: ['O', 'B-ORG', 'I-ORG', 'B-CAT', 'I-CAT', 'B-PRICE', 'I-PRICE',
#     'B-DUR', 'I-DUR', 'B-ENROLL', 'I-ENROLL', 'B-LEVEL', 'I-LEVEL']
