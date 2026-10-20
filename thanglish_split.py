"""Deterministic train / held-out split of train.jsonl, shared by train_lora.py and evaluate_model.py.

The held-out records are never used for training, so evaluate_model.py measures
generalisation to unseen sentences. Both scripts must use the same seed and size.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FILE = HERE / "train.jsonl"
TEST_SIZE = 90      # 10% of 903
SEED = 42


def load_records(path=DEFAULT_FILE):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split(records, test_size=TEST_SIZE, seed=SEED):
    """Return (train_records, test_records); order within each part is the file order."""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    test_idx = set(idx[:test_size])
    train = [r for i, r in enumerate(records) if i not in test_idx]
    test = [r for i, r in enumerate(records) if i in test_idx]
    return train, test


def fields(record):
    """(system prompt, Thanglish input, Tamil target)"""
    return (
        record["prompt"][0]["content"],
        record["prompt"][1]["content"],
        record["completion"][0]["content"],
    )
