"""Thesis-calibrated distributions for the synthetic FutureLearn dataset.

Every constant below is traceable to a figure reported in the thesis
(Chapter 3, "Findings and Discussion") or its defence deck. Generating the
dataset from these targets makes the business-analysis figures reproduce the
thesis narrative while remaining fully synthetic and reproducible.

Key thesis facts encoded here
-----------------------------
* 1,000 courses, 14 categories, 100+ partner institutions          (Table 4)
* 870 free (87%) / 130 paid (13%); price range GBP 0-120           (Table 4)
* paid courses ~15x the enrolment of free courses                  (sec 3.3, line 353/394)
* Language under-supplied (58 courses, avg ~34,673)                (line 351)
* Business & Management over-supplied (216 courses, avg ~10,981)   (line 350)
* IT & Computer Science: 113 courses, avg ~5,153                   (line 352)
* duration engagement peaks at 2-5 weeks (5-wk ~15,343)            (line 368/375)
* Groningen most efficient (~80,435/course); Leeds biggest reach   (line 376/380)
* avg learner rating 4.70/5.0 but ~87% of ratings missing          (Table 4 / Table 5 #3)
"""
from __future__ import annotations

# --- Categories: (name, n_courses, target_mean_enrollment, target_mean_rating) ---
# n_courses sums to 1000. Means anchor the Figure-5 demand/supply story.
CATEGORIES = [
    ("Business & Management",       216, 10_981, 4.62),
    ("Healthcare & Medicine",       130,  9_132, 4.66),
    ("IT & Computer Science",       113,  5_153, 4.68),
    ("Science, Engineering & Maths", 80,  6_775, 4.55),  # "danger zone": low rating
    ("Teaching",                     60,  8_500, 4.65),
    ("Psychology & Mental Health",   60, 12_500, 4.74),
    ("Language",                     58, 34_673, 4.76),  # star: high demand+satisfaction
    ("History",                      55, 13_381, 4.70),
    ("Nature & Environment",         45,  7_800, 4.63),
    ("Creative Arts & Media",        45,  6_200, 4.60),
    ("Study Skills",                 40, 14_195, 4.72),
    ("Politics & Society",           40,  9_500, 4.58),
    ("Literature",                   30,  8_800, 4.69),
    ("Law",                          28,  7_000, 4.61),
]
assert sum(c[1] for c in CATEGORIES) == 1000

# --- Partners: (name, n_courses, efficiency_multiplier) ---
# Named partners encode the Figure-10 reach-vs-efficiency split. The multiplier
# scales a course's enrolment on top of category/duration/price effects.
# Multipliers are relative to the dataset-wide mean enrolment (~10.7k), so e.g.
# Groningen 7.2 -> ~77k/course (thesis ~80,435), Leeds 2.2 -> ~24k (thesis ~24,135).
NAMED_PARTNERS = [
    ("University of Leeds",            94, 2.20),  # scale leader: biggest total reach
    ("Taipei Medical University",      54, 0.38),  # high volume, low efficiency (~4.1k)
    ("University of Reading",          12, 4.60),  # ~51k/course
    ("Vrije Universiteit Amsterdam",   10, 6.00),  # ~69k/course
    ("University of Groningen",         8, 7.20),  # efficiency leader (~80k/course)
]
# The remaining ~822 courses are spread over generic partners (see generate.py).
N_GENERIC_PARTNERS = 95
GENERIC_PARTNER_NAMES = [
    "University", "Institute of Technology", "College", "School of Business",
    "Academy", "Polytechnic", "Medical School", "School of Arts",
]  # combined with place names to synthesise ~95 distinct institutions

# --- Free / paid split & pricing -------------------------------------------
N_COURSES = 1000
PAID_FRACTION = 0.13            # -> 130 paid, 870 free
PAID_FREE_ENROLL_RATIO = 16.0  # paid courses draw ~15x free (offsets the enroll cap)

# Discrete price tiers (GBP) with relative popularity weights among PAID courses.
# The 39/59/79 story: 39 and 79 perform well, 59 underperforms, 99-120 is niche.
# Weights concentrate paid courses on the three key tiers so per-tier means are
# stable enough (~25-30 courses each) for the £59 dip to show cleanly.
PRICE_TIERS = [19, 24, 29, 39, 49, 59, 69, 79, 89, 99, 109, 120]
PAID_TIER_WEIGHTS = [3, 3, 4, 28, 5, 24, 5, 24, 4, 3, 2, 2]  # 39, 59, 79 dominate
# Per-tier enrolment modifier (the £59 dip; clear cheap/premium positioning wins).
PRICE_TIER_ENROLL_MULT = {
    19: 0.85, 24: 0.90, 29: 0.95, 39: 1.30, 49: 1.05, 59: 0.78,
    69: 0.95, 79: 1.25, 89: 1.05, 99: 0.80, 109: 0.70, 120: 0.65,
}

# --- Duration: weeks 1-11, engagement peaks at 2-5 weeks -------------------
DURATION_WEEKS = list(range(1, 12))           # 1..11
DURATION_WEIGHTS = [6, 16, 18, 15, 14, 10, 7, 5, 4, 3, 2]  # how common each length is
DURATION_ENROLL_MULT = {                       # sharp 2-5wk peak, 1wk & 8wk troughs
    1: 0.16, 2: 1.20, 3: 1.40, 4: 1.50, 5: 1.62,
    6: 0.92, 7: 0.52, 8: 0.13, 9: 0.13, 10: 0.11, 11: 0.10,
}

# --- Ratings ---------------------------------------------------------------
RATING_PRESENT_FRACTION = 0.13   # ~87% missing (Table 5 #3)
RATING_GLOBAL_MEAN = 4.70        # Table 4
RATING_STD = 0.18

# --- Enrolment generative model -------------------------------------------
# Lognormal multiplicative noise applied on top of the deterministic factors
# (de-biased to E[noise]=1 in generate.py).
ENROLL_NOISE_SIGMA = 0.40
ENROLL_CAP = 250_000   # clip the heavy upper tail so per-tier/per-partner means are stable
