import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from dataloader import load_csv_splits

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 64
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    return parser.parse_args()


class EncodedSplit(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        item = {key: self.encodings[key][index] for key in self.encodings}
        item["labels"] = self.labels[index]
        return item


def encode_split(
    texts: list[str],
    labels: list[str],
    tokenizer: BertTokenizerFast,
    label2id: dict[str, int],
    max_length: int,
) -> EncodedSplit:
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
    )
    encoded_labels = [label2id[label] for label in labels]
    return EncodedSplit(encodings, encoded_labels)


def evaluate(
    model: BertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    predictions: list[int] = []
    gold: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            gold.extend(labels.tolist())
    accuracy = accuracy_score(gold, predictions)
    macro_f1 = f1_score(gold, predictions, average="macro")
    return accuracy, macro_f1


def train(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        val_acc, val_macro_f1 = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}: "
            f"train loss {total_loss / len(train_loader):.4f}, "
            f"val accuracy {val_acc:.4f}, "
            f"val macro F1 {val_macro_f1:.4f}"
        )
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)


def main() -> None:
    seed = 42
    args = parse_args()
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nyt_path = Path(__file__).resolve().parent.parent / "dataset" / "nyt.csv"
    splits = load_csv_splits(nyt_path, seed=seed)
    train_rows = splits["train"].rows
    val_rows = splits["val"].rows
    test_rows = splits["test"].rows

    label_names = sorted({row["label"] for row in train_rows})
    label2id = {name: index for index, name in enumerate(label_names)}
    id2label = {index: name for name, index in label2id.items()}

    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    model = BertForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    print(f"max_length: {args.max_length}")
    train_set = encode_split(
        [row["text"] for row in train_rows],
        [row["label"] for row in train_rows],
        tokenizer,
        label2id,
        args.max_length,
    )
    val_set = encode_split(
        [row["text"] for row in val_rows],
        [row["label"] for row in val_rows],
        tokenizer,
        label2id,
        args.max_length,
    )
    test_set = encode_split(
        [row["text"] for row in test_rows],
        [row["label"] for row in test_rows],
        tokenizer,
        label2id,
        args.max_length,
    )

    collator = DataCollatorWithPadding(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    train(model, train_loader, val_loader, device)
    acc, macro_f1 = evaluate(model, test_loader, device)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
