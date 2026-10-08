"""Aggregate the JSON results into results/summary.md and collect BERT error examples."""

import json
import logging

import numpy as np

from dataloader import DATASET_DIR, RESULTS_DIR, SPLIT_SEED, load_csv_splits, save_results

SEEDS = [42, 43, 44]
BERT_LENGTHS = [32, 64, 96, 128, 192, 256, 384, 512]
INVERSE_REGS = ["0.01", "0.1", "1", "10", "100"]
REQUIRED_LENGTH = 64
NUM_EXAMPLES = 8
EXAMPLE_CHARS = 400

logger = logging.getLogger(__name__)


def load(name: str) -> dict[str, object]:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> str:
    """Mean and sample standard deviation over seeds."""
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"


def main() -> None:
    lines = ["# Homework 1 results (NYT test, 1152 documents)", ""]
    lines += ["## Main results", "", "| Method | Accuracy | Macro-F1 |", "|---|---|---|"]
    single_runs = {
        "Binary BoW": "bow_binary",
        "Word frequency BoW": "bow_frequency",
        "TF-IDF BoW (extra)": "bow_tfidf",
        "GloVe 6B 100d": "emb_glove",
        "Word2Vec AG News (seed 42)": "emb_ag_seed42",
        "Word2Vec NYT train split (seed 42)": "emb_nyt_seed42",
        "Word2Vec NYT all text, leaky ablation (seed 42)": "emb_nyt_all_seed42",
    }
    per_class_rows: list[str] = []
    for label, name in single_runs.items():
        test = load(name)["test"]
        lines.append(f"| {label} | {test['accuracy']:.4f} | {test['macro_f1']:.4f} |")
        per_class_rows.append(f"| {label} | " + " | ".join(f"{test['per_class'][c]['f1']:.4f}" for c in test["labels"]) + " |")
    for method, label in [("ag", "Word2Vec AG News"), ("nyt", "Word2Vec NYT train split"), ("nyt_all", "Word2Vec NYT all text")]:
        runs = [load(f"emb_{method}_seed{seed}")["test"] for seed in SEEDS]
        lines.append(
            f"| {label} (3 seeds) | {mean_std([run['accuracy'] for run in runs])} | "
            f"{mean_std([run['macro_f1'] for run in runs])} |"
        )
    for key, label in [("test_last_epoch", "last epoch"), ("test_best_dev", "best dev epoch")]:
        runs = [load(f"bert_len{REQUIRED_LENGTH}_seed{seed}")[key] for seed in SEEDS]
        lines.append(
            f"| BERT max_length {REQUIRED_LENGTH}, {label} (3 seeds) | {mean_std([run['accuracy'] for run in runs])} | "
            f"{mean_std([run['macro_f1'] for run in runs])} |"
        )
        for seed, run in zip(SEEDS, runs):
            per_class_rows.append(
                f"| BERT-{REQUIRED_LENGTH} {label} seed {seed} | "
                + " | ".join(f"{run['per_class'][c]['f1']:.4f}" for c in run["labels"])
                + " |"
            )

    labels = load("bow_binary")["test"]["labels"]
    lines += ["", "## Per-class F1", "", "| Method | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    lines += per_class_rows

    lines += [
        "",
        "## BERT max_length sweep (3 seeds, mean ± sample std)",
        "",
        "| max_length | last acc | last macro-F1 | best-dev acc | best-dev macro-F1 | best epochs |",
        "|---|---|---|---|---|---|",
    ]
    for length in BERT_LENGTHS:
        runs = [load(f"bert_len{length}_seed{seed}") for seed in SEEDS]
        cells = [
            mean_std([run[key][metric] for run in runs])
            for key in ["test_last_epoch", "test_best_dev"]
            for metric in ["accuracy", "macro_f1"]
        ]
        best_epochs = ",".join(str(run["best_epoch"]) for run in runs)
        lines.append(f"| {length} | " + " | ".join(cells) + f" | {best_epochs} |")

    lines += [
        "",
        "## BoW inverse regularization C (val macro-F1 / test macro-F1)",
        "",
        "| Method | " + " | ".join(f"C={value}" for value in INVERSE_REGS) + " |",
        "|---" * (len(INVERSE_REGS) + 1) + "|",
    ]
    for method in ["binary", "frequency", "tfidf"]:
        runs = [load(f"bow_{method}" if value == "1" else f"bow_{method}_C{value}") for value in INVERSE_REGS]
        lines.append(
            f"| {method} | "
            + " | ".join(f"{run['val']['macro_f1']:.4f} / {run['test']['macro_f1']:.4f}" for run in runs)
            + " |"
        )

    lines += ["", "## Word embedding coverage on NYT test", "", "| Embedding | vocab | token OOV | type OOV |", "|---|---|---|---|"]
    for name in ["emb_glove", "emb_ag_seed42", "emb_nyt_seed42", "emb_nyt_all_seed42"]:
        run = load(name)
        lines.append(f"| {name} | {run['vocab_size']} | {run['test_token_oov_rate']:.4f} | {run['test_type_oov_rate']:.4f} |")

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {RESULTS_DIR / 'summary.md'}")

    test_rows = load_csv_splits(DATASET_DIR / "nyt.csv", seed=SPLIT_SEED)["test"].rows
    bow_predictions = load("bow_frequency")["test_predictions"]
    bert_predictions = load(f"bert_len{REQUIRED_LENGTH}_seed{SPLIT_SEED}")["test_predictions_last_epoch"]
    bert_only_errors = [
        {
            "test_index": index,
            "gold": row["label"],
            "bert_prediction": bert_predictions[index],
            "text_head": row["text"][:EXAMPLE_CHARS],
        }
        for index, row in enumerate(test_rows)
        if bert_predictions[index] != row["label"] and bow_predictions[index] == row["label"]
    ]
    save_results(
        "error_examples",
        {
            "bert_run": f"bert_len{REQUIRED_LENGTH}_seed{SPLIT_SEED} last epoch",
            "bow_run": "bow_frequency",
            "num_bert_wrong": sum(pred != row["label"] for pred, row in zip(bert_predictions, test_rows)),
            "num_bow_wrong": sum(pred != row["label"] for pred, row in zip(bow_predictions, test_rows)),
            "num_both_wrong": sum(
                bert != row["label"] and bow != row["label"]
                for bert, bow, row in zip(bert_predictions, bow_predictions, test_rows)
            ),
            "num_bert_wrong_bow_right": len(bert_only_errors),
            "examples": bert_only_errors[:NUM_EXAMPLES],
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
