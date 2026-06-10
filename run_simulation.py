"""
run_simulation.py

End-to-end toy run for the SIGNAL-Preop reproducibility template.

This script runs the complete simulated pipeline:

1. Build synthetic structured questionnaire responses.
2. Convert responses into ordered QA sequences.
3. Train a Longformer-based multitask model on synthetic labels.
4. Run multitask inference.
5. Save toy predictions.

No real participant data are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocess import (
    build_qa_sequences,
    load_questionnaire_schema,
    load_responses,
    save_outputs,
)
from src.train_multitask import train
from src.inference import predict
from src.utils import ensure_dir, set_seed


def run_simulation(
    model_name_or_path: str,
    output_root: str = "outputs/simulation",
    max_length: int = 2048,
    batch_size: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 2e-5,
    seed: int = 42,
    device: str | None = None,
    skip_training: bool = False,
) -> None:
    """
    Run the full toy pipeline.

    Args:
        model_name_or_path:
            HuggingFace model name or local Longformer path.
            For offline use, provide a local path such as:
            models/longformer-base-4096

        output_root:
            Directory for all simulation outputs.

        max_length:
            Maximum token length. The manuscript uses 2048.

        batch_size:
            Batch size for toy training and inference.

        num_epochs:
            Number of toy training epochs.

        learning_rate:
            Learning rate for toy training.

        seed:
            Random seed.

        device:
            "cpu", "cuda", or None for automatic selection.

        skip_training:
            If True, skips checkpoint training and runs inference with randomly
            initialized weights. This is only for checking code execution.
    """
    set_seed(seed)

    output_root = Path(output_root)
    input_dir = output_root / "inputs"
    checkpoint_dir = output_root / "checkpoints"
    prediction_dir = output_root / "predictions"

    ensure_dir(input_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(prediction_dir)

    input_text_path = input_dir / "toy_input_text.csv"
    prediction_path = prediction_dir / "toy_predictions.csv"

    print("=" * 80)
    print("Step 1. Building synthetic questionnaire responses")
    print("=" * 80)

    schema_df = load_questionnaire_schema(schema_path=None)
    response_df = load_responses(response_path=None)

    print(f"Synthetic questionnaire items: {len(schema_df)}")
    print(f"Synthetic response rows: {len(response_df)}")
    print(f"Participants: {response_df['participant_id'].nunique()}")

    print("\n" + "=" * 80)
    print("Step 2. Converting structured responses into ordered QA sequences")
    print("=" * 80)

    qa_df = build_qa_sequences(
        schema_df=schema_df,
        response_df=response_df,
    )

    save_outputs(
        df=qa_df,
        output_path=str(input_text_path),
    )

    print(f"Saved toy QA sequences to: {input_text_path}")
    print("\nExample QA sequence:")
    print(qa_df.iloc[0]["input_text"][:1000])

    if not skip_training:
        print("\n" + "=" * 80)
        print("Step 3. Training Longformer multitask model on synthetic labels")
        print("=" * 80)

        train(
            input_path=str(input_text_path),
            label_path=None,
            model_name_or_path=model_name_or_path,
            output_dir=str(checkpoint_dir),
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            seed=seed,
            val_fraction=0.2,
            dropout=0.1,
            focal_alpha=0.25,
            focal_gamma=2.0,
            use_uncertainty_weighting=True,
            freeze_encoder=False,
            gradient_checkpointing=False,
        )

        checkpoint_dir_for_inference = str(checkpoint_dir)

    else:
        print("\n" + "=" * 80)
        print("Step 3. Skipping training")
        print("=" * 80)
        print(
            "Inference will use randomly initialized model weights. "
            "This mode is only for code execution checks."
        )
        checkpoint_dir_for_inference = None

    print("\n" + "=" * 80)
    print("Step 4. Running multitask inference")
    print("=" * 80)

    predict(
        input_path=str(input_text_path),
        output_path=str(prediction_path),
        base_model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=model_name_or_path,
        checkpoint_dir=checkpoint_dir_for_inference,
        max_length=max_length,
        batch_size=batch_size,
        threshold=0.5,
        dropout=0.1,
        seed=seed,
        device_name=device,
    )

    print("\n" + "=" * 80)
    print("Simulation completed")
    print("=" * 80)
    print(f"Input text: {input_text_path}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Predictions: {prediction_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full SIGNAL-Preop toy simulation pipeline."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="schen/longformer-chinese-base-4096",
        help=(
            "HuggingFace model name or local Longformer path. "
            "Use a local path if HuggingFace cannot be accessed."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/simulation",
        help="Directory for simulation outputs.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length. The manuscript uses 2048.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Toy batch size.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of toy training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu, cuda, or None for automatic selection.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help=(
            "Skip training and run inference with randomly initialized weights. "
            "Only for checking code execution."
        ),
    )

    args = parser.parse_args()

    run_simulation(
        model_name_or_path=args.model_name,
        output_root=args.output_root,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    main()