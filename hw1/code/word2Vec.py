import argparse
import csv
import logging
import time

import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression

from dataloader import DATASET_DIR, SPLIT_SEED, classification_metrics, ensure_punkt, load_csv_splits, save_results

DIM = 100
MAX_ITER = 10000

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Averaged word embeddings with logistic regression on NYT.")
    parser.add_argument("--method", choices=["glove", "ag", "nyt"], required=True)
    parser.add_argument("--train", action="store_true", help="train the ag or nyt Word2Vec model before evaluating")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="Word2Vec training seed")
    parser.add_argument(
        "--nyt_all_text",
        action="store_true",
        help="ablation: train the nyt model on all splits (leaks validation and test text)",
    )
    return parser.parse_args()


def train_w2v(sentences: list[list[str]], output_name: str, *, seed: int) -> None:
    """Skip-gram with negative sampling; deterministic only with workers=1 and a fixed PYTHONHASHSEED."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=DIM,
        window=5,
        min_count=5,
        sg=1,
        negative=5,
        workers=1,
        seed=seed,
        epochs=5,
    )
    model.save(str(DATASET_DIR / output_name))
    logger.info(f"Trained {output_name} on {len(sentences)} documents, vocabulary size {len(model.wv)}")


def embed_doc(
    tokenized_texts: list[list[str]],
    vectors: dict[str, np.ndarray],
) -> np.ndarray:
    features = np.zeros((len(tokenized_texts), DIM), dtype=np.float32)
    for row, tokens in enumerate(tokenized_texts):
        found = [vectors[token] for token in tokens if token in vectors]
        if found:
            features[row] = np.mean(found, axis=0)
    return features


def main() -> None:
    args = parse_args()
    if args.train and args.method == "glove":
        raise ValueError("--train only applies to --method ag or nyt")
    if args.nyt_all_text and args.method != "nyt":
        raise ValueError("--nyt_all_text only applies to --method nyt")
    seed = SPLIT_SEED
    ensure_punkt()
    splits = load_csv_splits(DATASET_DIR / "nyt.csv", seed=seed)
    tokens = {name: [word_tokenize(row["text"]) for row in split.rows] for name, split in splits.items()}
    labels = {name: [row["label"] for row in split.rows] for name, split in splits.items()}

    corpus_tag = "_all" if args.nyt_all_text else ""
    model_name = f"{args.method}{corpus_tag}_seed{args.seed}.w2v"
    if args.train and args.method == "ag":
        with (DATASET_DIR / "ag.csv").open(newline="", encoding="utf-8") as handle:
            ag_sentences = [word_tokenize(row["text"].lower()) for row in csv.DictReader(handle)]
        train_w2v(ag_sentences, model_name, seed=args.seed)
    elif args.train and args.method == "nyt" and args.nyt_all_text:
        train_w2v(tokens["train"] + tokens["val"] + tokens["test"], model_name, seed=args.seed)
    elif args.train and args.method == "nyt":
        # Only the training split, so validation and test text never shapes the embeddings.
        train_w2v(tokens["train"], model_name, seed=args.seed)

    if args.method == "glove":
        vectors: dict[str, np.ndarray] = {}
        with (DATASET_DIR / "glove.6B.100d.txt").open(encoding="utf-8") as handle:
            for line in handle:
                word, *values = line.rstrip().split(" ")
                vectors[word] = np.asarray(values, dtype=np.float32)
        result_name = "emb_glove"
    else:
        model = Word2Vec.load(str(DATASET_DIR / model_name))
        vectors = {word: model.wv.get_vector(word) for word in model.wv.index_to_key}
        result_name = f"emb_{args.method}{corpus_tag}_seed{args.seed}"

    test_tokens = [token for doc in tokens["test"] for token in doc]
    test_types = set(test_tokens)
    oov = {
        "vocab_size": len(vectors),
        "test_token_oov_rate": sum(token not in vectors for token in test_tokens) / len(test_tokens),
        "test_type_oov_rate": sum(token not in vectors for token in test_types) / len(test_types),
        "test_empty_docs": sum(all(token not in vectors for token in doc) for doc in tokens["test"]),
    }

    start = time.perf_counter()
    classifier = LogisticRegression(solver="saga", max_iter=MAX_ITER, random_state=seed)
    classifier.fit(embed_doc(tokens["train"], vectors), labels["train"])
    train_seconds = time.perf_counter() - start
    num_iter = int(classifier.n_iter_.max())
    val_metrics = classification_metrics(labels["val"], classifier.predict(embed_doc(tokens["val"], vectors)).tolist())
    test_predictions = classifier.predict(embed_doc(tokens["test"], vectors)).tolist()
    test_metrics = classification_metrics(labels["test"], test_predictions)

    save_results(
        result_name,
        {
            "method": args.method,
            "w2v_seed": None if args.method == "glove" else args.seed,
            "nyt_all_text": args.nyt_all_text,
            **oov,
            "saga_iterations": num_iter,
            "converged": num_iter < MAX_ITER,
            "train_seconds": round(train_seconds, 1),
            "val": val_metrics,
            "test": test_metrics,
            "test_predictions": test_predictions,
        },
    )
    print(f"Token OOV rate on test: {oov['test_token_oov_rate']:.4f}")
    print(f"Val accuracy: {val_metrics['accuracy']:.4f}, val macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
