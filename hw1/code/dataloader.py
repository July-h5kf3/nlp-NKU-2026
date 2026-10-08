"""Shared data loading and evaluation helpers for homework 1."""

import csv
import json
import logging
from pathlib import Path

import nltk
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

HW1_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = HW1_DIR / "dataset"
RESULTS_DIR = HW1_DIR / "results"
SPLIT_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

logger = logging.getLogger(__name__)


def load_csv_splits(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """Stratified train/val/test rows of a labelled CSV, always split with SPLIT_SEED."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train_rows, heldout_rows = train_test_split(
        rows,
        train_size=TRAIN_RATIO,
        random_state=SPLIT_SEED,
        stratify=[row["label"] for row in rows],
    )
    val_rows, test_rows = train_test_split(
        heldout_rows,
        train_size=VAL_RATIO / (1.0 - TRAIN_RATIO),
        random_state=SPLIT_SEED,
        stratify=[row["label"] for row in heldout_rows],
    )
    return {"train": train_rows, "val": val_rows, "test": test_rows}


def subsample_rows(rows: list[dict[str, str]], fraction: float) -> list[dict[str, str]]:
    """Stratified subset of the training rows, used for learning-curve runs."""
    if fraction == 1.0:
        return rows
    subset, _ = train_test_split(
        rows,
        train_size=fraction,
        random_state=SPLIT_SEED,
        stratify=[row["label"] for row in rows],
    )
    return subset


def fraction_tag(fraction: float) -> str:
    return "" if fraction == 1.0 else f"_frac{fraction:g}"


def check_fraction(fraction: float) -> None:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")


def ensure_punkt() -> None:
    """Download the NLTK tokenizer data used by word_tokenize if it is missing."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt_tab")
        nltk.download("punkt_tab", quiet=True)


def classification_metrics(gold: list[str], predicted: list[str]) -> dict[str, object]:
    """Accuracy, macro F1, per-class P/R/F1 and the confusion matrix (rows are gold labels)."""
    labels = sorted(set(gold))
    precision, recall, f1, support = precision_recall_fscore_support(gold, predicted, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro")),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(gold, predicted, labels=labels).tolist(),
    }


def save_results(name: str, payload: dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"Saved results to {path}")
