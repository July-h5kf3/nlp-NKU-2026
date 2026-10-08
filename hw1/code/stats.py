"""Dataset statistics used in the report: split sizes, class counts and document lengths."""

import csv
import logging
from collections import Counter

import numpy as np
from nltk.tokenize import word_tokenize
from transformers import BertTokenizerFast

from bert import MODEL_NAME
from dataloader import DATASET_DIR, SPLIT_SEED, ensure_punkt, load_csv_splits, save_results

BERT_LENGTHS = [32, 64, 96, 128, 192, 256, 384, 512]
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

logger = logging.getLogger(__name__)


def main() -> None:
    ensure_punkt()
    splits = load_csv_splits(DATASET_DIR / "nyt.csv", seed=SPLIT_SEED)
    with (DATASET_DIR / "ag.csv").open(newline="", encoding="utf-8") as handle:
        ag_texts = [row["text"] for row in csv.DictReader(handle)]
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    split_stats: dict[str, dict[str, object]] = {}
    for name, split in splits.items():
        texts = [row["text"] for row in split.rows]
        word_lengths = np.array([len(word_tokenize(text)) for text in texts])
        # Lengths include [CLS] and [SEP], matching how truncation counts tokens.
        wordpiece_lengths = np.array([len(ids) for ids in tokenizer(texts)["input_ids"]])
        split_stats[name] = {
            "num_docs": len(texts),
            "label_counts": dict(sorted(Counter(row["label"] for row in split.rows).items())),
            "nltk_tokens_quantiles": {str(q): float(np.quantile(word_lengths, q)) for q in QUANTILES},
            "wordpiece_tokens_quantiles": {str(q): float(np.quantile(wordpiece_lengths, q)) for q in QUANTILES},
            "num_truncated_by_length": {str(length): int((wordpiece_lengths > length).sum()) for length in BERT_LENGTHS},
        }
        logger.info(f"{name}: {len(texts)} documents")

    save_results(
        "stats",
        {
            "nyt_num_docs": sum(len(split.rows) for split in splits.values()),
            "nyt_splits": split_stats,
            "ag_num_docs": len(ag_texts),
            "ag_has_uppercase_docs": sum(text != text.lower() for text in ag_texts),
            "nyt_has_uppercase_docs": sum(row["text"] != row["text"].lower() for split in splits.values() for row in split.rows),
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
