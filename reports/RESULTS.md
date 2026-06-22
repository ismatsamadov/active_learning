# Active Learning for NER — Real Experimental Results

_Auto-generated from the implementation in this repository. These numbers replace the thesis's illustrative values with measured, reproducible ones._

## Headline

- Random sampling does **not** reach within 0.01 F1 of the baseline within the budget (final F1 = 0.8679).

## 1. Full-data supervised baseline (BiLSTM-CRF)

Trained on all 1985 pool sentences, evaluated on 492 held-out test sentences, averaged over seeds [13].

| Metric | Value |
| --- | --- |
| Entity-level F1 (micro) | **0.9961 ± 0.0000** |
| Entity-level F1 (macro) | 0.9966 |
| Precision | 0.9948 |
| Recall | 0.9974 |
| seqeval F1 (independent check) | 0.9961 |

**Per-entity-type F1:**

| Type | F1 |
| --- | --- |
| CAT | 1.000 |
| DUR | 0.998 |
| ENROLL | 1.000 |
| LEVEL | 1.000 |
| ORG | 0.981 |
| PRICE | 1.000 |

## 2. Active-learning learning curve

Test entity-F1 (mean ± std over seeds) at each annotation budget.

| Labels | random | least_confidence | uncertainty | hybrid |
|  --- | --- | --- | --- | --- |
| 80 | 0.063 ± 0.000 | 0.063 ± 0.000 | 0.063 ± 0.000 | 0.063 ± 0.000 |
| 200 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 320 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 440 | 0.785 ± 0.000 | 0.778 ± 0.000 | 0.814 ± 0.000 | 0.759 ± 0.000 |
| 560 | 0.868 ± 0.000 | 0.942 ± 0.000 | 0.880 ± 0.000 | 0.895 ± 0.000 |

![learning curve](../figures/learning_curve.png)

## 3. NER → business-data bridge (does NER actually produce the fields?)

The trained model is run over the test descriptions; predicted entities are parsed back into course fields and compared to ground truth (test entity-F1 = 0.999).

| Field | Recovery accuracy |
| --- | --- |
| partner | 0.990 |
| category | 1.000 |
| price | 1.000 |
| duration_weeks | 1.000 |
| enrollment_count | 1.000 |
| level | 1.000 |

**Worked example:**

> Develop tools for assessing , managing , and treating pain to help ease distress for palliative care patients .

- extracted: `{'partner': 'University of Colorado', 'category': 'Healthcare & Medicine', 'price': 0.0, 'duration_weeks': 5, 'enrollment_count': 321, 'level': None}`
- truth: `{'partner': 'University of Colorado', 'category': 'Healthcare & Medicine', 'price': 0.0, 'duration_weeks': 5, 'enrollment_count': 321.0, 'level': None}`

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
