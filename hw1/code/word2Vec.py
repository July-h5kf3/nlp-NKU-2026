import argparse
import csv
from pathlib import Path

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from dataloader import load_csv_splits

DIM = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_set", type=str)
    parser.add_argument("--method", type=str, required=True)
    return parser.parse_args()

def train_w2v(
    dataset: str,
    *,
    seed: int = 42,
) -> None:
    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    with (dataset_dir / f"{dataset}.csv").open(newline="", encoding="utf-8") as handle:
        sentences = [word_tokenize(row["text"]) for row in csv.DictReader(handle)]
    model = Word2Vec(
        sentences=sentences,
        vector_size=DIM,
        window=5,
        min_count=5,
        sg=1,
        workers=1,
        seed=seed,
        epochs=5,
    )
    model.save(str(dataset_dir / f"{dataset}.w2v"))

def embed_doc(
    texts: list[str],
    vectors: dict[str, np.ndarray],
) -> np.ndarray:
    features = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        found = [vectors[token] for token in word_tokenize(text) if token in vectors]
        if found:
            features[row] = np.mean(found, axis=0)
    return features

def main() -> None:
    seed = 42
    args = parse_args()
    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    if args.train:
        if args.train_set is None:
            raise ValueError("train_set is required when --train is set")
        train_w2v(args.train_set, seed=seed)
    splits = load_csv_splits(dataset_dir / "nyt.csv", seed=seed)

    train_rows = splits["train"].rows
    test_rows = splits["test"].rows

    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["label"] for row in train_rows]

    test_texts = [row["text"] for row in test_rows]
    test_labels = [row["label"] for row in test_rows]

    if args.method == "glove":
        vectors: dict[str, np.ndarray] = {}
        with (dataset_dir / "glove.6B.100d.txt").open(encoding="utf-8") as handle:
            for line in handle:
                word, *values = line.rstrip().split(" ")
                embedding = np.asarray(values, dtype=np.float32)
                vectors[word] = embedding
    elif args.method == "ag":
        model = Word2Vec.load(str(dataset_dir / "ag.w2v"))
        vectors = {word: model.wv.get_vector(word) for word in model.wv.index_to_key}
    elif args.method == "nyt":
        model = Word2Vec.load(str(dataset_dir / "nyt.w2v"))
        vectors = {word: model.wv.get_vector(word) for word in model.wv.index_to_key}
    else:
        raise ValueError(f"Invalid method: {args.method}")
    train_features = embed_doc(train_texts, vectors)
    test_features = embed_doc(test_texts, vectors)

    classifier = LogisticRegression(solver="saga", max_iter=10000, random_state=seed)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    acc = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
if __name__ == "__main__":
    main()