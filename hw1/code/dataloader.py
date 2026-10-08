import csv
from sklearn.model_selection import train_test_split

from pathlib import Path

class CsvRows:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

def load_csv_splits(
    csv_path: Path | str,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, CsvRows]:
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    
    stratify = [row["label"] for row in rows] if "label" in rows[0] else None
    train_rows, heldout_rows = train_test_split(
        rows,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify,
    )
    heldout_labels = [row["label"] for row in heldout_rows] if stratify is not None else None
    val_rows, test_rows = train_test_split(
        heldout_rows,
        train_size= val_ratio / (1.0 - train_ratio),
        random_state=seed,
        stratify=heldout_labels,
    )
    splits = {
        "train": CsvRows(train_rows),
        "val": CsvRows(val_rows),
        "test": CsvRows(test_rows),
    }
    return splits

