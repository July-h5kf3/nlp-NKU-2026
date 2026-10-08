from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from dataloader import load_csv_splits

import argparse

from nltk.tokenize import word_tokenize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    return parser.parse_args()

def classify_bow(
    vectorizer: CountVectorizer | TfidfVectorizer,
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    *,
    seed: int = 42,
) -> tuple[float, float]:
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)

    classifier = LogisticRegression(solver="saga", max_iter=10000, random_state=seed)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    acc = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average="macro")
    return acc, macro_f1

def main(method: str) -> None:
    seed = 42
    nyt_path = Path(__file__).resolve().parent.parent / "dataset" / "nyt.csv"
    splits = load_csv_splits(nyt_path, seed=seed)

    train_rows = splits["train"].rows
    test_rows = splits["test"].rows

    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["label"] for row in train_rows]

    test_texts = [row["text"] for row in test_rows]
    test_labels = [row["label"] for row in test_rows]

    if method == "binary":
        vectorizer = CountVectorizer(binary=True, tokenizer=word_tokenize)
    elif method == "frequency":
        vectorizer = CountVectorizer(binary=False, tokenizer=word_tokenize)
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(binary=False, tokenizer=word_tokenize)
    else:
        raise ValueError(f"Invalid method: {method}")
    acc, macro_f1 = classify_bow(
        vectorizer,
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        seed=seed,
    )
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args.method)