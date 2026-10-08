import argparse
import logging
import time

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from dataloader import DATASET_DIR, SPLIT_SEED, classification_metrics, ensure_punkt, load_csv_splits, save_results

MAX_ITER = 10000
TOP_FEATURES = 10

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bag-of-words features with logistic regression on NYT.")
    parser.add_argument("--method", choices=["binary", "frequency", "tfidf"], required=True)
    parser.add_argument("--C", type=float, default=1.0, help="inverse L2 regularization strength")
    return parser.parse_args()


def main(method: str, inverse_reg: float) -> None:
    seed = SPLIT_SEED
    ensure_punkt()
    splits = load_csv_splits(DATASET_DIR / "nyt.csv", seed=seed)
    texts = {name: [row["text"] for row in split.rows] for name, split in splits.items()}
    labels = {name: [row["label"] for row in split.rows] for name, split in splits.items()}

    if method == "binary":
        vectorizer = CountVectorizer(binary=True, tokenizer=word_tokenize, token_pattern=None)
    elif method == "frequency":
        vectorizer = CountVectorizer(binary=False, tokenizer=word_tokenize, token_pattern=None)
    else:
        vectorizer = TfidfVectorizer(binary=False, tokenizer=word_tokenize, token_pattern=None)

    start = time.perf_counter()
    train_features = vectorizer.fit_transform(texts["train"])
    classifier = LogisticRegression(C=inverse_reg, solver="saga", max_iter=MAX_ITER, random_state=seed)
    classifier.fit(train_features, labels["train"])
    train_seconds = time.perf_counter() - start
    num_iter = int(classifier.n_iter_.max())
    logger.info(f"Vocabulary size {len(vectorizer.vocabulary_)}, saga iterations {num_iter}, {train_seconds:.1f}s")

    val_metrics = classification_metrics(labels["val"], classifier.predict(vectorizer.transform(texts["val"])).tolist())
    test_predictions = classifier.predict(vectorizer.transform(texts["test"])).tolist()
    test_metrics = classification_metrics(labels["test"], test_predictions)

    feature_names = vectorizer.get_feature_names_out()
    top_features = {
        str(label): [str(feature_names[index]) for index in np.argsort(weights)[::-1][:TOP_FEATURES]]
        for label, weights in zip(classifier.classes_, classifier.coef_)
    }
    suffix = "" if inverse_reg == 1.0 else f"_C{inverse_reg:g}"
    save_results(
        f"bow_{method}{suffix}",
        {
            "method": method,
            "C": inverse_reg,
            "vocab_size": len(vectorizer.vocabulary_),
            "saga_iterations": num_iter,
            "converged": num_iter < MAX_ITER,
            "train_seconds": round(train_seconds, 1),
            "val": val_metrics,
            "test": test_metrics,
            "top_features": top_features,
            "test_predictions": test_predictions,
        },
    )
    print(f"Val accuracy: {val_metrics['accuracy']:.4f}, val macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()
    main(args.method, args.C)
