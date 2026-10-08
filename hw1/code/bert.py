import argparse
import logging
import time
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

from dataloader import (
    DATASET_DIR,
    SPLIT_SEED,
    check_fraction,
    classification_metrics,
    fraction_tag,
    load_csv_splits,
    save_results,
    subsample_rows,
)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 64
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
NUM_SPECIAL_TOKENS = 2  # [CLS] and [SEP]
HEAD_SHARE = 128 / 510  # Sun et al. (2019): head 128 + tail 382 content tokens at length 512

Truncation = Literal["head", "tail", "head_tail"]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on NYT.")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="training seed; the data split always uses 42")
    parser.add_argument(
        "--truncation",
        choices=["head", "tail", "head_tail"],
        default="head",
        help="which part of a long document to keep",
    )
    parser.add_argument("--train_fraction", type=float, default=1.0, help="stratified share of the training split")
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
    truncation: Truncation,
) -> EncodedSplit:
    budget = max_length - NUM_SPECIAL_TOKENS
    head_budget = round(HEAD_SHARE * budget)
    input_ids: list[list[int]] = []
    for token_ids in tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]:
        if len(token_ids) > budget and truncation == "head":
            token_ids = token_ids[:budget]
        elif len(token_ids) > budget and truncation == "tail":
            token_ids = token_ids[-budget:]
        elif len(token_ids) > budget:
            token_ids = token_ids[:head_budget] + token_ids[len(token_ids) - (budget - head_budget) :]
        input_ids.append([tokenizer.cls_token_id, *token_ids, tokenizer.sep_token_id])
    encodings = {
        "input_ids": input_ids,
        "token_type_ids": [[0] * len(ids) for ids in input_ids],
        "attention_mask": [[1] * len(ids) for ids in input_ids],
    }
    encoded_labels = [label2id[label] for label in labels]
    return EncodedSplit(encodings, encoded_labels)


def predict(
    model: BertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> list[int]:
    model.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions.append(model(**batch).logits.argmax(dim=-1))
    return torch.cat(predictions).cpu().tolist()


def train(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_labels: list[str],
    id2label: dict[int, str],
    device: torch.device,
) -> tuple[list[dict[str, float]], int, dict[str, torch.Tensor]]:
    """Train for EPOCHS epochs; return per-epoch history, the best-dev epoch and its weights."""
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
    history: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.detach()

        val_predictions = [id2label[index] for index in predict(model, val_loader, device)]
        val_metrics = classification_metrics(val_labels, val_predictions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss.item() / len(train_loader),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        logger.info(
            f"Epoch {epoch}: train loss {history[-1]['train_loss']:.4f}, "
            f"val accuracy {val_metrics['accuracy']:.4f}, val macro F1 {val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return history, best_epoch, best_state


def main() -> None:
    args = parse_args()
    check_fraction(args.train_fraction)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_csv_splits(DATASET_DIR / "nyt.csv", seed=SPLIT_SEED)
    splits["train"].rows = subsample_rows(splits["train"].rows, args.train_fraction, seed=SPLIT_SEED)
    texts = {name: [row["text"] for row in split.rows] for name, split in splits.items()}
    labels = {name: [row["label"] for row in split.rows] for name, split in splits.items()}

    label_names = sorted(set(labels["train"]))
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
    logger.info(
        f"max_length {args.max_length}, truncation {args.truncation}, train fraction {args.train_fraction}, "
        f"seed {args.seed}, device {device}"
    )

    collator = DataCollatorWithPadding(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            encode_split(texts[name], labels[name], tokenizer, label2id, args.max_length, args.truncation),
            batch_size=BATCH_SIZE,
            shuffle=name == "train",
            collate_fn=collator,
            generator=generator if name == "train" else None,
        )
        for name in splits
    }

    start = time.perf_counter()
    history, best_epoch, best_state = train(model, loaders["train"], loaders["val"], labels["val"], id2label, device)
    train_seconds = time.perf_counter() - start

    last_predictions = [id2label[index] for index in predict(model, loaders["test"], device)]
    model.load_state_dict(best_state)
    best_predictions = [id2label[index] for index in predict(model, loaders["test"], device)]
    last_metrics = classification_metrics(labels["test"], last_predictions)
    best_metrics = classification_metrics(labels["test"], best_predictions)

    truncation_tag = "" if args.truncation == "head" else f"_{args.truncation}"
    save_results(
        f"bert_len{args.max_length}_seed{args.seed}{truncation_tag}{fraction_tag(args.train_fraction)}",
        {
            "model": args.model,
            "max_length": args.max_length,
            "truncation": args.truncation,
            "train_fraction": args.train_fraction,
            "num_train": len(labels["train"]),
            "seed": args.seed,
            "epochs": EPOCHS,
            "train_seconds": round(train_seconds, 1),
            "history": history,
            "best_epoch": best_epoch,
            "test_last_epoch": last_metrics,
            "test_best_dev": best_metrics,
            "test_predictions_last_epoch": last_predictions,
            "test_predictions_best_dev": best_predictions,
        },
    )
    print(f"Last epoch ({EPOCHS}) accuracy: {last_metrics['accuracy']:.4f}, macro F1: {last_metrics['macro_f1']:.4f}")
    print(f"Best dev epoch ({best_epoch}) accuracy: {best_metrics['accuracy']:.4f}, macro F1: {best_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
