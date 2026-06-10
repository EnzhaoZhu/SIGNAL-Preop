"""
train_multitask.py

Toy multitask training script for SIGNAL-Preop-style Longformer modeling.

This script is intended for reproducibility demonstration using synthetic or
user-provided de-identified data. It does not reproduce the reported study
performance without the original training data.

Expected input:
    participant_id,input_text

Expected labels:
    participant_id,label_DD,label_GAD,label_IND,label_SR
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.dataset import (
    DEFAULT_TASKS,
    SIGNALPreopDataset,
    load_input_texts,
    load_labels,
    merge_texts_and_labels,
)
from src.losses import build_multitask_loss
from src.model_longformer import build_multitask_model


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Select CUDA if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_dataset(
    dataset: Dataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Split dataset into training and validation sets.

    For very small toy datasets, validation is skipped.
    """
    n_total = len(dataset)

    if n_total < 5:
        return dataset, None

    n_val = max(1, int(round(n_total * val_fraction)))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    return train_set, val_set


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Move tensor fields in a batch to device.
    """
    output = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value

    return output


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    gradient_clip_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    """
    model.train()
    criterion.train()

    running_loss = 0.0
    running_task_losses = {task: 0.0 for task in DEFAULT_TASKS}
    n_batches = 0

    progress = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss_dict = criterion(
            logits=outputs["logits"],
            labels=batch["labels"],
        )

        loss = loss_dict["loss"]
        loss.backward()

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                max_norm=gradient_clip_norm,
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += float(loss.detach().cpu())
        for task in DEFAULT_TASKS:
            key = f"loss_{task}"
            if key in loss_dict:
                running_task_losses[task] += float(loss_dict[key].detach().cpu())

        n_batches += 1
        progress.set_postfix(loss=running_loss / max(n_batches, 1))

    metrics = {
        "loss": running_loss / max(n_batches, 1),
    }

    for task in DEFAULT_TASKS:
        metrics[f"loss_{task}"] = running_task_losses[task] / max(n_batches, 1)

    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate validation loss.
    """
    model.eval()
    criterion.eval()

    running_loss = 0.0
    running_task_losses = {task: 0.0 for task in DEFAULT_TASKS}
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        batch = move_batch_to_device(batch, device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss_dict = criterion(
            logits=outputs["logits"],
            labels=batch["labels"],
        )

        running_loss += float(loss_dict["loss"].detach().cpu())
        for task in DEFAULT_TASKS:
            key = f"loss_{task}"
            if key in loss_dict:
                running_task_losses[task] += float(loss_dict[key].detach().cpu())

        n_batches += 1

    metrics = {
        "loss": running_loss / max(n_batches, 1),
    }

    for task in DEFAULT_TASKS:
        metrics[f"loss_{task}"] = running_task_losses[task] / max(n_batches, 1)

    return metrics


def save_checkpoint(
    output_dir: str,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    tokenizer,
    config: Dict,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    """
    Save model checkpoint, tokenizer, and training metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "multitask_longformer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        checkpoint_path,
    )

    tokenizer.save_pretrained(output_dir)

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Saved checkpoint to: {checkpoint_path}")


def build_training_dataset(
    input_path: Optional[str],
    label_path: Optional[str],
    tokenizer_name: str,
    max_length: int,
    tasks: Sequence[str],
) -> Tuple[SIGNALPreopDataset, object]:
    """
    Build tokenized dataset and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    text_df = load_input_texts(input_path)
    label_df = load_labels(label_path, tasks=tasks)
    data_df = merge_texts_and_labels(text_df, label_df)

    dataset = SIGNALPreopDataset(
        data=data_df,
        tokenizer=tokenizer,
        max_length=max_length,
        tasks=tasks,
        has_labels=True,
    )

    return dataset, tokenizer


def train(
    input_path: Optional[str] = None,
    label_path: Optional[str] = None,
    model_name_or_path: str = "schen/longformer-chinese-base-4096",
    output_dir: str = "outputs/checkpoints",
    max_length: int = 2048,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 2,
    warmup_ratio: float = 0.1,
    dropout: float = 0.1,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    use_uncertainty_weighting: bool = True,
    val_fraction: float = 0.2,
    seed: int = 42,
    freeze_encoder: bool = False,
    gradient_checkpointing: bool = False,
    gradient_clip_norm: Optional[float] = 1.0,
) -> None:
    """
    Train multitask Longformer model.
    """
    set_seed(seed)
    device = get_device()

    tasks = DEFAULT_TASKS

    dataset, tokenizer = build_training_dataset(
        input_path=input_path,
        label_path=label_path,
        tokenizer_name=model_name_or_path,
        max_length=max_length,
        tasks=tasks,
    )

    train_set, val_set = split_dataset(
        dataset,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    model = build_multitask_model(
        model_name_or_path=model_name_or_path,
        tasks=tasks,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)

    criterion = build_multitask_loss(
        tasks=tasks,
        alpha=focal_alpha,
        gamma=focal_gamma,
        use_uncertainty_weighting=use_uncertainty_weighting,
    ).to(device)

    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = max(1, len(train_loader) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    config = {
        "model_name_or_path": model_name_or_path,
        "tasks": tasks,
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "warmup_ratio": warmup_ratio,
        "dropout": dropout,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "use_uncertainty_weighting": use_uncertainty_weighting,
        "val_fraction": val_fraction,
        "seed": seed,
        "freeze_encoder": freeze_encoder,
        "gradient_checkpointing": gradient_checkpointing,
    }

    best_metric = float("inf")
    best_epoch = -1
    best_metrics = {}

    print(f"Device: {device}")
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {0 if val_set is None else len(val_set)}")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )

        print("Train:", train_metrics)

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                criterion=criterion,
                dataloader=val_loader,
                device=device,
            )
            print("Validation:", val_metrics)
            current_metric = val_metrics["loss"]
            save_metrics = {"train": train_metrics, "validation": val_metrics}
        else:
            current_metric = train_metrics["loss"]
            save_metrics = {"train": train_metrics}

        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_metrics = save_metrics

            save_checkpoint(
                output_dir=output_dir,
                model=model,
                criterion=criterion,
                tokenizer=tokenizer,
                config=config,
                epoch=epoch,
                metrics=save_metrics,
            )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best loss: {best_metric:.4f}")
    print("Best metrics:", best_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a multitask Longformer model for SIGNAL-Preop-style QA inputs."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input_text CSV. If omitted, synthetic inputs are used.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to label CSV. If omitted, synthetic labels are used.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="schen/longformer-chinese-base-4096",
        help="Longformer checkpoint name or local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory for saving checkpoint.",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--no_uncertainty_weighting",
        action="store_true",
        help="Disable uncertainty-based task weighting.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze Longformer encoder parameters.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory use.",
    )

    args = parser.parse_args()

    train(
        input_path=args.input,
        label_path=args.labels,
        model_name_or_path=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_uncertainty_weighting=not args.no_uncertainty_weighting,
        val_fraction=args.val_fraction,
        seed=args.seed,
        freeze_encoder=args.freeze_encoder,
        gradient_checkpointing=args.gradient_checkpointing,
    )


if __name__ == "__main__":
    main()