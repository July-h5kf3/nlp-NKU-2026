#!/usr/bin/env bash
# Reproduce every result in hw1/results. Override PYTHON to use another interpreter, GPU to pick a device.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-"uv run python"}
GPU=${GPU:-0}
SEEDS=(42 43 44)
export PYTHONHASHSEED=0

$PYTHON hw1/code/stats.py

for method in binary frequency tfidf; do
  $PYTHON hw1/code/BoW.py --method "$method"
  for inverse_reg in 0.01 0.1 10 100; do
    $PYTHON hw1/code/BoW.py --method "$method" --C "$inverse_reg"
  done
done

$PYTHON hw1/code/word2Vec.py --method glove
for seed in "${SEEDS[@]}"; do
  $PYTHON hw1/code/word2Vec.py --method ag --train --seed "$seed"
  $PYTHON hw1/code/word2Vec.py --method nyt --train --seed "$seed"
  $PYTHON hw1/code/word2Vec.py --method nyt --train --nyt_all_text --seed "$seed"
done

for length in 64 32 96 128 192 256 384 512; do
  for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON hw1/code/bert.py --max_length "$length" --seed "$seed"
  done
done

$PYTHON hw1/code/summarize.py
