"""
inference.py

Inference utilities for SIGNAL-Preop-style Longformer multitask prediction.

Expected input:
    participant_id,input_text

Output:
    participant_id,P_DD,P_GAD,P_IND,P_SR,pred_DD,pred_GAD,pred_IND,pred_SR

This script can run in two modes:
1. With a trained checkpoint from train_multitask.py.
2. Without a checkpoint, using randomly initialized weights for pipeline testing only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset import DEFAULT_TASKS, SIGNALPreopDataset, load_input_texts
from src.model_longformer import build_multitask_model
from src.utils import ensure_dir, find_checkpoint, get_device, load_checkpoint, load_yaml, set_seed


def build_inference_dataset(
    input_path: Optional[str],
    tokenizer_name_or_path: str,
    max_length: int = 2048,
    tasks: Sequence[str] = DEFAULT_TASKS,
) -> SIGNALPreopDataset:
    """
    Build dataset for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    text_df = load_input_texts(input_path)

    dataset = SIGNALPreopDataset(
        data=text_df,
        tokenizer=tokenizer,
        max_length=max_length,
        tasks=tasks,
        has_labels=False,
    )

    return dataset


def load_multitask_model_for_inference(
    base_model_name_or_path: str,
    checkpoint_path: Optional[str | Path] = None,
    tasks: Sequence[str] = DEFAULT_TASKS,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
):
    """
    Load Longformer multitask model and optionally restore trained weights.
    """
    device = device or get_device()

    model = build_multitask_model(
        model_name_or_path=base_model_name_or_path,
        tasks=tasks,
        dropout=dropout,
        freeze_encoder=False,
        gradient_checkpointing=False,
    )

    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict(
    input_path: Optional[str] = None,
    output_path: str = "outputs/toy_predictions.csv",
    base_model_name_or_path: str = "schen/longformer-chinese-base-4096",
    tokenizer_name_or_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    max_length: int = 2048,
    batch_size: int = 2,
    threshold: float = 0.5,
    dropout: float = 0.1,
    seed: int = 42,
    device_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run multitask inference.
    """
    set_seed(seed)
    device = get_device(device_name)

    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = base_model_name_or_path

    if checkpoint_path is None and checkpoint_dir is not None:
        checkpoint_path = find_checkpoint(checkpoint_dir)

    dataset = build_inference_dataset(
        input_path=input_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_length=max_length,
        tasks=DEFAULT_TASKS,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_multitask_model_for_inference(
        base_model_name_or_path=base_model_name_or_path,
        checkpoint_path=checkpoint_path,
        tasks=DEFAULT_TASKS,
        dropout=dropout,
        device=device,
    )

    rows = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs["logits"]
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        participant_ids = batch["participant_id"]

        for i, participant_id in enumerate(participant_ids):
            row: Dict[str, float | int | str] = {
                "participant_id": participant_id,
            }

            for task_idx, task in enumerate(DEFAULT_TASKS):
                prob = float(probs[i, task_idx])
                row[f"P_{task}"] = prob
                row[f"pred_{task}"] = int(prob >= threshold)

            rows.append(row)

    pred_df = pd.DataFrame(rows)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved predictions to: {output_path}")
    print(pred_df.head())

    return pred_df


def predict_from_config(config_path: str) -> pd.DataFrame:
    """
    Run inference from YAML config.
    """
    config = load_yaml(config_path)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})
    runtime_cfg = config.get("runtime", {})

    return predict(
        input_path=data_cfg.get("inference_input", None),
        output_path=inference_cfg.get("output_path", "outputs/toy_predictions.csv"),
        base_model_name_or_path=model_cfg.get(
            "base_model_name_or_path",
            "schen/longformer-chinese-base-4096",
        ),
        tokenizer_name_or_path=model_cfg.get("tokenizer_name_or_path", None),
        checkpoint_path=inference_cfg.get("checkpoint_path", None),
        checkpoint_dir=inference_cfg.get("checkpoint_dir", None),
        max_length=model_cfg.get("max_length", 2048),
        batch_size=inference_cfg.get("batch_size", 2),
        threshold=inference_cfg.get("threshold", 0.5),
        dropout=model_cfg.get("dropout", 0.1),
        seed=runtime_cfg.get("seed", 42),
        device_name=runtime_cfg.get("device", None),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with a SIGNAL-Preop Longformer multitask model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. If provided, other arguments are ignored.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input_text CSV. If omitted, synthetic inputs are used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/toy_predictions.csv",
        help="Output prediction CSV.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="schen/longformer-chinese-base-4096",
        help="Base Longformer checkpoint name or local path.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or local path. Defaults to base_model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to multitask_longformer.pt.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing multitask_longformer.pt.",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu, or None for automatic selection.",
    )

    args = parser.parse_args()

    if args.config is not None:
        predict_from_config(args.config)
        return

    predict(
        input_path=args.input,
        output_path=args.output,
        base_model_name_or_path=args.base_model,
        tokenizer_name_or_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        threshold=args.threshold,
        dropout=args.dropout,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()