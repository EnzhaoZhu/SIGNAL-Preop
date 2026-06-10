"""
dataset.py

PyTorch Dataset utilities for SIGNAL-Preop-style QA sequences.

Expected input text file:
    participant_id,input_text

Expected label file:
    participant_id,label_DD,label_GAD,label_IND,label_SR

The dataset tokenizes ordered QA sequences using a Longformer-compatible
tokenizer and returns inputs for multitask binary classification.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_TASKS = ["DD", "GAD", "IND", "SR"]
DEFAULT_LABEL_COLUMNS = {
    "DD": "label_DD",
    "GAD": "label_GAD",
    "IND": "label_IND",
    "SR": "label_SR",
}


class SIGNALPreopDataset(Dataset):
    """
    Dataset for ordered QA sequences.

    Each sample returns:
    - participant_id
    - input_ids
    - attention_mask
    - labels, if available

    Labels are returned as a float tensor with shape [num_tasks],
    suitable for BCEWithLogitsLoss or focal loss.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        tasks: Sequence[str] = DEFAULT_TASKS,
        label_columns: Optional[Dict[str, str]] = None,
        has_labels: bool = True,
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tasks = list(tasks)
        self.label_columns = label_columns or DEFAULT_LABEL_COLUMNS
        self.has_labels = has_labels

        self._validate()

    def _validate(self) -> None:
        required = {"participant_id", "input_text"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Input data missing required columns: {missing}")

        if self.has_labels:
            required_labels = {self.label_columns[t] for t in self.tasks}
            missing_labels = required_labels - set(self.data.columns)
            if missing_labels:
                raise ValueError(f"Label columns missing: {missing_labels}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.data.iloc[idx]
        text = str(row["input_text"])

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor | str] = {
            "participant_id": str(row["participant_id"]),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if self.has_labels:
            labels = [
                float(row[self.label_columns[task]])
                for task in self.tasks
            ]
            item["labels"] = torch.tensor(labels, dtype=torch.float32)

        return item


def load_input_texts(input_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load preprocessed QA sequences.

    If input_path is None, synthetic QA sequences are generated.
    """
    if input_path is None:
        return simulate_input_texts()

    df = pd.read_csv(input_path)
    required = {"participant_id", "input_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input text file missing required columns: {missing}")

    return df


def load_labels(
    label_path: Optional[str] = None,
    tasks: Sequence[str] = DEFAULT_TASKS,
    label_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load multitask binary labels.

    If label_path is None, synthetic labels are generated.
    """
    label_columns = label_columns or DEFAULT_LABEL_COLUMNS

    if label_path is None:
        return simulate_labels()

    df = pd.read_csv(label_path)
    required = {"participant_id"} | {label_columns[t] for t in tasks}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Label file missing required columns: {missing}")

    for task in tasks:
        col = label_columns[task]
        df[col] = df[col].astype(int)

    return df


def merge_texts_and_labels(
    text_df: pd.DataFrame,
    label_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge input texts with labels by participant_id.

    If label_df is None, the returned dataframe is used for inference.
    """
    if label_df is None:
        return text_df.copy()

    merged = text_df.merge(label_df, on="participant_id", how="inner")

    if len(merged) != len(text_df):
        missing_ids = set(text_df["participant_id"]) - set(merged["participant_id"])
        raise ValueError(
            f"Some participants in input_texts have no labels: {sorted(missing_ids)}"
        )

    return merged


def build_dataset(
    input_path: Optional[str] = None,
    label_path: Optional[str] = None,
    tokenizer_name: str = "schen/longformer-chinese-base-4096",
    max_length: int = 2048,
    tasks: Sequence[str] = DEFAULT_TASKS,
    has_labels: bool = True,
) -> SIGNALPreopDataset:
    """
    Build SIGNALPreopDataset from files or synthetic examples.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    text_df = load_input_texts(input_path)

    if has_labels:
        label_df = load_labels(label_path, tasks=tasks)
        data_df = merge_texts_and_labels(text_df, label_df)
    else:
        data_df = merge_texts_and_labels(text_df, None)

    dataset = SIGNALPreopDataset(
        data=data_df,
        tokenizer=tokenizer,
        max_length=max_length,
        tasks=tasks,
        has_labels=has_labels,
    )

    return dataset


def build_dataloader(
    dataset: SIGNALPreopDataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def simulate_input_texts() -> pd.DataFrame:
    """
    Synthetic QA sequences for demonstration only.

    These are not derived from real participants or real questionnaire records.
    """
    rows = [
        {
            "participant_id": "P001",
            "input_text": (
                "[ITEM] DM [DOMAIN] DD [QUESTION] Core screening for depressed mood. "
                "[ANSWER] No. "
                "[ITEM] LOIOA [DOMAIN] DD [QUESTION] Core screening for loss of interest. "
                "[ANSWER] Yes, I have lost interest recently. "
                "[ITEM] SD [DOMAIN] IND [QUESTION] Core screening for sleep disturbance. "
                "[ANSWER] Yes, sleep has been poor."
            ),
        },
        {
            "participant_id": "P002",
            "input_text": (
                "[ITEM] DM [DOMAIN] DD [QUESTION] Core screening for depressed mood. "
                "[ANSWER] No. "
                "[ITEM] EWAMA [DOMAIN] GAD [QUESTION] Core screening for excessive worry. "
                "[ANSWER] Yes, I worry about several things. "
                "[ITEM] DCW [DOMAIN] GAD [QUESTION] Difficulty controlling worry. "
                "[ANSWER] It is hard to stop worrying."
            ),
        },
    ]
    return pd.DataFrame(rows)


def simulate_labels() -> pd.DataFrame:
    """
    Synthetic multitask labels for demonstration only.
    """
    rows = [
        {
            "participant_id": "P001",
            "label_DD": 1,
            "label_GAD": 0,
            "label_IND": 1,
            "label_SR": 0,
        },
        {
            "participant_id": "P002",
            "label_DD": 0,
            "label_GAD": 1,
            "label_IND": 0,
            "label_SR": 0,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SIGNAL-Preop dataset and inspect tokenized batches."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to preprocessed input_text CSV. If omitted, synthetic inputs are used.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to label CSV. If omitted, synthetic labels are used.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="schen/longformer-chinese-base-4096",
        help="Tokenizer name or local tokenizer path.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size.",
    )
    args = parser.parse_args()

    dataset = build_dataset(
        input_path=args.input,
        label_path=args.labels,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        has_labels=True,
    )
    dataloader = build_dataloader(dataset, batch_size=args.batch_size)

    batch = next(iter(dataloader))

    print("Batch keys:", batch.keys())
    print("participant_id:", batch["participant_id"])
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", batch["labels"].shape)
    print("labels:", batch["labels"])


if __name__ == "__main__":
    main()