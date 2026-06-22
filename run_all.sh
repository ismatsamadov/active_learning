#!/usr/bin/env bash
# Reproduce the whole study end to end. Pass --fast for a quick smoke run.
set -euo pipefail
cd "$(dirname "$0")"

# Pin hash randomisation for the child interpreters (must be set before they start;
# setting it inside Python is too late). Makes set-iteration order reproducible.
export PYTHONHASHSEED=0

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
FAST="${1:-}"

echo "### 0. CRF correctness tests"
$PY tests/test_crf.py

echo "### 1. Prepare real dataset + build NER corpus"
$PY scripts/01_prepare_data.py

echo "### 2. Full-data baseline"
$PY scripts/03_run_baseline.py $FAST --device cpu

echo "### 3. Active learning (random vs uncertainty vs hybrid)"
$PY scripts/04_run_active_learning.py $FAST --device cpu

echo "### 4. Business analysis (Figures 5-11)"
$PY scripts/05_business_analysis.py

echo "### 5. NER -> business-fields extraction bridge"
$PY scripts/06_extraction_bridge.py $FAST --device cpu

echo "### 6. Thesis-ready report"
$PY scripts/07_make_report.py

echo ""
echo "Done. See reports/RESULTS.md, plus figures/ and results/."
