#!/usr/bin/env bash
# Reproduce every homework 1 result with one command, from the repository root.
#
# Usage: ./run_all.sh [--required]
#   --required   only the runs behind the report's main table (about 30 min on one H800);
#                without it every extra experiment runs too (about 2 h on one H800).
#
# Environment:
#   PYTHON                 Python command (default "uv run python"; if uv is not installed,
#                          .venv/bin/python is used when present).
#   CUDA_VISIBLE_DEVICES   GPU for the BERT runs, e.g. CUDA_VISIBLE_DEVICES=0 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  echo "Usage: $0 [--required]" >&2
  exit 2
}

REQUIRED_ONLY=0
if [ $# -gt 1 ]; then
  usage
fi
case "${1:-}" in
  "") ;;
  --required) REQUIRED_ONLY=1 ;;
  -h | --help)
    sed -n '2,12p' "$0"
    exit 0
    ;;
  *) usage ;;
esac

if [ -z "${PYTHON:-}" ]; then
  if command -v uv > /dev/null 2>&1; then
    PYTHON="uv run python"
  elif [ -x .venv/bin/python ]; then
    PYTHON=".venv/bin/python"
  else
    echo "error: uv not found and .venv/bin/python missing; install uv and run 'uv sync' (see README), or set PYTHON" >&2
    exit 1
  fi
fi

DATA_DIR=hw1/dataset
missing=0
for file in nyt.csv ag.csv glove.6B.100d.txt; do
  if [ ! -f "$DATA_DIR/$file" ]; then
    echo "error: missing $DATA_DIR/$file (see the data section of README.md)" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  exit 1
fi

CODE=hw1/code
SEEDS=(42 43 44)
# Word2Vec is only bit-for-bit reproducible with a fixed hash seed.
export PYTHONHASHSEED=0
start_seconds=$SECONDS

run() {
  echo
  echo "==> $* [$(((SECONDS - start_seconds) / 60)) min elapsed]"
  $PYTHON "$@"
}

echo "Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset, all visible>}"
echo "Mode: $([ "$REQUIRED_ONLY" -eq 1 ] && echo required || echo full)"

PYTHONPATH=$CODE $PYTHON -c "from dataloader import ensure_punkt; ensure_punkt()"
if [ "$($PYTHON -c 'import torch; print(torch.cuda.is_available())')" != "True" ]; then
  echo "warning: no CUDA GPU visible, BERT will run on CPU and be very slow" >&2
fi

# Required: dataset statistics and the six settings of the main table.
run $CODE/stats.py
for method in binary frequency; do
  run $CODE/BoW.py --method "$method"
done
run $CODE/word2Vec.py --method glove
for seed in "${SEEDS[@]}"; do
  run $CODE/word2Vec.py --method ag --train --seed "$seed"
  run $CODE/word2Vec.py --method nyt --train --seed "$seed"
done
for seed in "${SEEDS[@]}"; do
  run $CODE/bert.py --max_length 64 --seed "$seed"
done

if [ "$REQUIRED_ONLY" -eq 0 ]; then
  # Extra: TF-IDF and the inverse regularization sweep for all BoW variants.
  run $CODE/BoW.py --method tfidf
  for method in binary frequency tfidf; do
    for inverse_reg in 0.01 0.1 10 100; do
      run $CODE/BoW.py --method "$method" --C "$inverse_reg"
    done
  done

  # Extra: NYT Word2Vec trained on all text, the leakage ablation.
  for seed in "${SEEDS[@]}"; do
    run $CODE/word2Vec.py --method nyt --train --nyt_all_text --seed "$seed"
  done

  # Extra: BERT maximum length sweep (64 already ran above).
  for length in 32 96 128 192 256 384 512; do
    for seed in "${SEEDS[@]}"; do
      run $CODE/bert.py --max_length "$length" --seed "$seed"
    done
  done

  # Extra: learning curve on stratified subsets of the training split.
  for fraction in 0.1 0.25 0.5; do
    run $CODE/BoW.py --method frequency --train_fraction "$fraction"
    run $CODE/word2Vec.py --method glove --train_fraction "$fraction"
    for seed in "${SEEDS[@]}"; do
      run $CODE/bert.py --max_length 64 --seed "$seed" --train_fraction "$fraction"
    done
  done

  # Extra: which part of a long document BERT keeps.
  for length in 64 128; do
    for truncation in tail head_tail; do
      for seed in "${SEEDS[@]}"; do
        run $CODE/bert.py --max_length "$length" --seed "$seed" --truncation "$truncation"
      done
    done
  done
fi

# Rebuild results/summary.md; in --required mode the extra rows come from the committed JSON files.
run $CODE/summarize.py
echo
echo "Done in $(((SECONDS - start_seconds) / 60)) min. Tables: hw1/results/summary.md"
