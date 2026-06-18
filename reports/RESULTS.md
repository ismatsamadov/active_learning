# Active Learning for NER — Real Experimental Results

_Auto-generated from the implementation in this repository. These numbers replace the thesis's illustrative values with measured, reproducible ones._

## Headline

- **Hybrid active learning reaches the full-data baseline (F1 = 0.861) using only 200 labelled sentences — 10.1% of the 1982-sentence pool.**
- Random sampling reaches the same bar at 440 labels (22.2% of pool).

## 1. Full-data supervised baseline (BiLSTM-CRF)

Trained on all 1982 pool sentences, evaluated on 495 held-out test sentences, averaged over seeds [13, 29].

| Metric | Value |
| --- | --- |
| Entity-level F1 (micro) | **0.8615 ± 0.0003** |
| Entity-level F1 (macro) | 0.8648 |
| Precision | 0.8889 |
| Recall | 0.8358 |
| seqeval F1 (independent check) | 0.8612 |

**Per-entity-type F1:**

| Type | F1 |
| --- | --- |
| CAT | 0.845 |
| DUR | 0.927 |
| ENROLL | 0.885 |
| LEVEL | 0.890 |
| ORG | 0.787 |
| PRICE | 0.854 |

## 2. Active-learning learning curve

Test entity-F1 (mean ± std over seeds) at each annotation budget.

| Labels | random | least_confidence | uncertainty | hybrid |
|  --- | --- | --- | --- | --- |
| 120 | 0.792 ± 0.018 | 0.792 ± 0.018 | 0.792 ± 0.018 | 0.792 ± 0.018 |
| 160 | 0.831 ± 0.011 | 0.846 ± 0.007 | 0.846 ± 0.001 | 0.844 ± 0.001 |
| 200 | 0.827 ± 0.001 | 0.856 ± 0.001 | 0.855 ± 0.002 | 0.854 ± 0.002 |
| 240 | 0.833 ± 0.005 | 0.855 ± 0.001 | 0.859 ± 0.001 | 0.853 ± 0.001 |
| 280 | 0.849 ± 0.000 | 0.858 ± 0.003 | 0.860 ± 0.002 | 0.856 ± 0.001 |
| 320 | 0.846 ± 0.001 | 0.860 ± 0.001 | 0.861 ± 0.002 | 0.857 ± 0.000 |
| 360 | 0.850 ± 0.005 | 0.861 ± 0.001 | 0.860 ± 0.001 | 0.855 ± 0.001 |
| 400 | 0.850 ± 0.001 | 0.861 ± 0.001 | 0.861 ± 0.000 | 0.857 ± 0.000 |
| 440 | 0.857 ± 0.000 | 0.861 ± 0.001 | 0.862 ± 0.000 | 0.858 ± 0.000 |
| 480 | 0.853 ± 0.001 | 0.861 ± 0.000 | 0.862 ± 0.000 | 0.859 ± 0.001 |
| 520 | 0.854 ± 0.002 | 0.862 ± 0.000 | 0.862 ± 0.001 | 0.858 ± 0.000 |
| 560 | 0.856 ± 0.001 | 0.862 ± 0.000 | 0.862 ± 0.001 | 0.859 ± 0.000 |
| 600 | 0.858 ± 0.002 | 0.862 ± 0.000 | 0.862 ± 0.001 | 0.860 ± 0.001 |

![learning curve](../figures/learning_curve.png)

## 3. NER → business-data bridge (does NER actually produce the fields?)

The trained model is run over the test descriptions; predicted entities are parsed back into course fields and compared to ground truth (test entity-F1 = 0.862).

| Field | Recovery accuracy |
| --- | --- |
| partner | 0.612 |
| category | 0.612 |
| price | 0.609 |
| duration_weeks | 0.612 |
| enrollment_count | 0.611 |
| level | 0.633 |

**Worked example:**

> Join this 2-week leadership & enterprise course from Starweaver at free .

- extracted: `{'partner': 'Starweaver', 'category': 'Business & Management', 'price': 0.0, 'duration_weeks': 2, 'enrollment_count': None, 'level': None}`
- truth: `{'partner': 'Starweaver', 'category': 'Business & Management', 'price': 0.0, 'duration_weeks': 2, 'enrollment_count': None, 'level': None}`

## 4. Business-oriented analysis (Figures 5–11)

- Paid courses average **46,291** enrolments vs **3,050** for free — a **15.18×** gap.
- Engagement peaks at **5-week** courses (15,344 avg); 1-week and 8+-week courses are far lower (2–5 week window).
- Star categories (high demand + high satisfaction): Healthcare & Medicine, History, Language, Psychology & Mental Health.

**Executive decision summary:**

| Priority | Finding | Recommended action |
| --- | --- | --- |
| 🔴 Critical | Language shows the highest demand (~34,673/course) on only 58 courses | Commission more Language courses with high-performing partners |
| 🔴 Critical | Paid courses average 15.18x the enrollment of free courses | Evaluate converting top-performing free courses into paid offerings |
| 🔴 Critical | ~87.0% of courses lack learner ratings | Implement a systematic learner feedback/review mechanism |
| 🟠 High | CIPD - Chartered Institute of Personnel and Development is the most efficient partner (~121,693/course) | Expand collaboration with high-efficiency partners |
| 🟠 High | The £79 price tier underperforms (~35,891/course) vs the £59 tier (~64,821) | Reposition mid-tier pricing toward clearer value tiers |
| 🟠 High | 5-week courses achieve the highest engagement (~15,344) | Prioritise course designs in the 2-5 week range |
| 🟡 Medium | Advanced content is priced similarly to introductory content | Introduce differentiated pricing for advanced-level courses (£79-£99) |
| 🟡 Medium | Nature & Environment has high supply (84 courses) but modest demand (~3,967) | Review content focus; refocus on high-demand subdomains |
| 🟡 Medium | Business & Management is the largest catalogue (216 courses) | Quality/relevance assessment before further expansion |
| 🟢 Ongoing | A small set of courses drive disproportionate enrollment (top-decile >22,995) | Maintain and extend high-performing courses with updates/advanced modules |

Figures: `figures/fig5…fig11`.
