"""
model_longformer.py

Longformer-based models for SIGNAL-Preop-style multitask psychiatric screening.

Architecture aligned with the manuscript:
1. A shared Longformer encoder processes ordered QA sequences.
2. The first-token representation is used as the shared patient representation.
3. Four task-specific linear heads predict DD, GAD, IND, and SR.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoConfig, LongformerModel


DEFAULT_TASKS = ["DD", "GAD", "IND", "SR"]


class SIGNALLongformerMultiTask(nn.Module):
    """
    Longformer multitask model.

    Outputs logits for four binary outcomes:
    - DD: depressive disorder
    - GAD: generalized anxiety disorder
    - IND: insomnia disorder
    - SR: suicide risk
    """

    def __init__(
        self,
        model_name_or_path: str = "schen/longformer-chinese-base-4096",
        tasks: Sequence[str] = DEFAULT_TASKS,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.tasks = list(tasks)

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = LongformerModel.from_pretrained(
            model_name_or_path,
            config=self.config,
        )

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(hidden_size, 1)
                for task in self.tasks
            }
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _build_global_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Longformer uses local attention by default.

        We assign global attention to the first token, which serves as the
        sequence-level representation token.
        """
        global_attention_mask = torch.zeros_like(input_ids)

        if attention_mask is None:
            global_attention_mask[:, 0] = 1
        else:
            valid_first_token = attention_mask[:, 0] > 0
            global_attention_mask[valid_first_token, 0] = 1

        return global_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Returns:
            logits: Tensor of shape [batch_size, num_tasks]
            logits_by_task: Dict mapping task name to logits
            features: Optional shared representation
        """
        if global_attention_mask is None:
            global_attention_mask = self._build_global_attention_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )

        # First-token representation as shared patient-level representation.
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        logits_by_task = {
            task: self.heads[task](pooled).squeeze(-1)
            for task in self.tasks
        }

        logits = torch.stack(
            [logits_by_task[task] for task in self.tasks],
            dim=1,
        )

        output: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {
            "logits": logits,
            "logits_by_task": logits_by_task,
        }

        if return_features:
            output["features"] = pooled

        return output

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Return task-specific predicted probabilities.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            logits_by_task = outputs["logits_by_task"]

            probabilities = {
                task: torch.sigmoid(logits_by_task[task])
                for task in self.tasks
            }

        return probabilities


class SIGNALLongformerSingleTask(nn.Module):
    """
    Single-task Longformer model.

    This is useful for ablation analysis comparing multitask learning with
    separately trained single-task Longformer models.
    """

    def __init__(
        self,
        model_name_or_path: str = "schen/longformer-chinese-base-4096",
        task_name: str = "DD",
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.task_name = task_name

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = LongformerModel.from_pretrained(
            model_name_or_path,
            config=self.config,
        )

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _build_global_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        global_attention_mask = torch.zeros_like(input_ids)

        if attention_mask is None:
            global_attention_mask[:, 0] = 1
        else:
            valid_first_token = attention_mask[:, 0] > 0
            global_attention_mask[valid_first_token, 0] = 1

        return global_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if global_attention_mask is None:
            global_attention_mask = self._build_global_attention_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )

        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled).squeeze(-1)

        output = {
            "logits": logits,
        }

        if return_features:
            output["features"] = pooled

        return output

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            probabilities = torch.sigmoid(outputs["logits"])

        return probabilities


def build_multitask_model(
    model_name_or_path: str = "schen/longformer-chinese-base-4096",
    tasks: Sequence[str] = DEFAULT_TASKS,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
    gradient_checkpointing: bool = False,
) -> SIGNALLongformerMultiTask:
    """
    Convenience function for building the multitask model.
    """
    return SIGNALLongformerMultiTask(
        model_name_or_path=model_name_or_path,
        tasks=tasks,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        gradient_checkpointing=gradient_checkpointing,
    )


def build_single_task_model(
    model_name_or_path: str = "schen/longformer-chinese-base-4096",
    task_name: str = "DD",
    dropout: float = 0.1,
    freeze_encoder: bool = False,
    gradient_checkpointing: bool = False,
) -> SIGNALLongformerSingleTask:
    """
    Convenience function for building a single-task model.
    """
    return SIGNALLongformerSingleTask(
        model_name_or_path=model_name_or_path,
        task_name=task_name,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        gradient_checkpointing=gradient_checkpointing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Longformer multitask model architecture."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="schen/longformer-chinese-base-4096",
        help="Longformer checkpoint name or local path.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Synthetic sequence length for shape checking.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Synthetic batch size for shape checking.",
    )
    args = parser.parse_args()

    model = build_multitask_model(model_name_or_path=args.model_name)
    model.eval()

    vocab_size = model.config.vocab_size

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, args.max_length),
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    print(model)
    print("Logits shape:", outputs["logits"].shape)
    for task, logits in outputs["logits_by_task"].items():
        print(f"{task} logits shape:", logits.shape)


if __name__ == "__main__":
    main()