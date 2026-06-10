"""
preprocess.py

Preprocessing utilities for SIGNAL-Preop-style structured QA inputs.

This script converts structured questionnaire responses into ordered QA sequences
for language modeling. It is intended for reproducibility demonstration using
synthetic or user-provided data only.

Key design choices aligned with the manuscript:
1. Preserve original participant responses after light text normalization.
2. Encode skipped/bypassed modules as negative responses.
3. Concatenate questions and answers using fixed separators.
4. Preserve the original interview order.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


QUESTION_SEP = "[QUESTION]"
ANSWER_SEP = "[ANSWER]"
ITEM_SEP = "[ITEM]"
DOMAIN_SEP = "[DOMAIN]"


REQUIRED_SCHEMA_COLUMNS = {
    "item_code",
    "domain",
    "item_order",
    "concept",
    "question_text",
    "negative_response",
}

REQUIRED_RESPONSE_COLUMNS = {
    "participant_id",
    "item_code",
    "response_text",
    "is_skipped",
}


def normalize_text(text: object) -> str:
    """
    Light text normalization only.

    This removes formatting/control artifacts and normalizes whitespace,
    but does not semantically rewrite participant responses.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_questionnaire_schema(schema_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load questionnaire schema.

    Expected schema format:
    {
      "items": [
        {
          "item_code": "DM",
          "domain": "DD",
          "item_order": 1,
          "concept": "Depressed mood",
          "question_text": "Core screening for depressed mood",
          "negative_response": "No depressed mood endorsed."
        }
      ]
    }

    If no schema_path is provided, a small synthetic schema is created.
    """
    if schema_path is None:
        return simulate_questionnaire_schema()

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    if "items" not in schema:
        raise ValueError("Schema JSON must contain an 'items' list.")

    df = pd.DataFrame(schema["items"])
    missing = REQUIRED_SCHEMA_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Schema is missing required columns: {missing}")

    df["item_order"] = df["item_order"].astype(int)
    return df.sort_values("item_order").reset_index(drop=True)


def simulate_questionnaire_schema() -> pd.DataFrame:
    """
    Create a minimal synthetic questionnaire schema.

    This does not reproduce the exact SIGNAL-Preop item wording.
    It only demonstrates the expected structure and branching-compatible format.
    """
    items = [
        {
            "item_code": "DM",
            "domain": "DD",
            "item_order": 1,
            "concept": "Depressed mood",
            "question_text": "Core screening for depressed mood.",
            "negative_response": "No depressed mood endorsed.",
        },
        {
            "item_code": "LOIOA",
            "domain": "DD",
            "item_order": 2,
            "concept": "Loss of interest or anhedonia",
            "question_text": "Core screening for loss of interest or pleasure.",
            "negative_response": "No loss of interest or anhedonia endorsed.",
        },
        {
            "item_code": "EWAMA",
            "domain": "GAD",
            "item_order": 3,
            "concept": "Excessive worry about multiple areas",
            "question_text": "Core screening for excessive worry.",
            "negative_response": "No excessive worry endorsed.",
        },
        {
            "item_code": "DCW",
            "domain": "GAD",
            "item_order": 4,
            "concept": "Difficulty controlling worry",
            "question_text": "Follow-up for difficulty controlling worry.",
            "negative_response": "No difficulty controlling worry endorsed.",
        },
        {
            "item_code": "SD",
            "domain": "IND",
            "item_order": 5,
            "concept": "Sleep disturbances",
            "question_text": "Core screening for sleep disturbance.",
            "negative_response": "No sleep disturbance endorsed.",
        },
        {
            "item_code": "SOD",
            "domain": "IND",
            "item_order": 6,
            "concept": "Sleep onset difficulty",
            "question_text": "Follow-up for difficulty falling asleep.",
            "negative_response": "No sleep onset difficulty endorsed.",
        },
        {
            "item_code": "IODL",
            "domain": "General",
            "item_order": 7,
            "concept": "Impact on daily life",
            "question_text": "Assessment of impact on daily life.",
            "negative_response": "No functional impact endorsed.",
        },
        {
            "item_code": "OAR",
            "domain": "General",
            "item_order": 8,
            "concept": "Onset and recurrence",
            "question_text": "Assessment of symptom onset and recurrence.",
            "negative_response": "No recurrent symptoms endorsed.",
        },
    ]
    return pd.DataFrame(items)


def load_responses(response_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load structured questionnaire responses.

    Expected columns:
    - participant_id
    - item_code
    - response_text
    - is_skipped

    If no response_path is provided, synthetic responses are created.
    """
    if response_path is None:
        return simulate_responses()

    df = pd.read_csv(response_path)
    missing = REQUIRED_RESPONSE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Response file is missing required columns: {missing}")

    df["is_skipped"] = df["is_skipped"].astype(int)
    return df


def simulate_responses() -> pd.DataFrame:
    """
    Create minimal synthetic participant responses.

    These examples are only for demonstrating code execution.
    They are not derived from real participants or real questionnaire records.
    """
    rows = [
        {
            "participant_id": "P001",
            "item_code": "DM",
            "response_text": "No.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P001",
            "item_code": "LOIOA",
            "response_text": "Yes, I have lost interest recently.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P001",
            "item_code": "EWAMA",
            "response_text": "No.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P001",
            "item_code": "DCW",
            "response_text": "",
            "is_skipped": 1,
        },
        {
            "participant_id": "P001",
            "item_code": "SD",
            "response_text": "Yes, sleep has been poor.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P001",
            "item_code": "SOD",
            "response_text": "It usually takes a long time to fall asleep.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P001",
            "item_code": "IODL",
            "response_text": "It affects my daytime energy.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P001",
            "item_code": "OAR",
            "response_text": "Started several weeks ago.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P002",
            "item_code": "DM",
            "response_text": "No.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P002",
            "item_code": "LOIOA",
            "response_text": "No.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P002",
            "item_code": "EWAMA",
            "response_text": "Yes, I worry about several things.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P002",
            "item_code": "DCW",
            "response_text": "It is hard to stop worrying.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P002",
            "item_code": "SD",
            "response_text": "No.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P002",
            "item_code": "SOD",
            "response_text": "",
            "is_skipped": 1,
        },
        {
            "participant_id": "P002",
            "item_code": "IODL",
            "response_text": "It sometimes distracts me.",
            "is_skipped": 0,
        },
        {
            "participant_id": "P002",
            "item_code": "OAR",
            "response_text": "Symptoms come and go.",
            "is_skipped": 0,
        },
    ]
    return pd.DataFrame(rows)


def merge_schema_and_responses(
    schema_df: pd.DataFrame,
    response_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge questionnaire schema with responses and preserve questionnaire order.
    """
    merged = response_df.merge(
        schema_df,
        on="item_code",
        how="left",
        validate="many_to_one",
    )

    if merged["item_order"].isna().any():
        missing_items = merged.loc[merged["item_order"].isna(), "item_code"].unique()
        raise ValueError(f"Responses contain item codes not found in schema: {missing_items}")

    merged["item_order"] = merged["item_order"].astype(int)
    merged = merged.sort_values(["participant_id", "item_order"]).reset_index(drop=True)
    return merged


def build_answer(row: pd.Series) -> str:
    """
    Return participant response or negative response for skipped items.
    """
    if int(row["is_skipped"]) == 1:
        return normalize_text(row["negative_response"])

    response = normalize_text(row["response_text"])
    if response == "":
        return normalize_text(row["negative_response"])

    return response


def build_qa_sequence(participant_df: pd.DataFrame) -> str:
    """
    Build one ordered QA sequence for a participant.
    """
    chunks: List[str] = []

    for _, row in participant_df.iterrows():
        domain = normalize_text(row["domain"])
        item_code = normalize_text(row["item_code"])
        concept = normalize_text(row["concept"])
        question = normalize_text(row["question_text"])
        answer = build_answer(row)

        chunk = (
            f"{ITEM_SEP} {item_code} "
            f"{DOMAIN_SEP} {domain} "
            f"{QUESTION_SEP} {question} "
            f"{ANSWER_SEP} {answer}"
        )
        chunks.append(chunk)

    return " ".join(chunks)


def build_qa_sequences(
    schema_df: pd.DataFrame,
    response_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert structured responses into one text sequence per participant.
    """
    merged = merge_schema_and_responses(schema_df, response_df)

    records = []
    for participant_id, group in merged.groupby("participant_id", sort=False):
        input_text = build_qa_sequence(group)
        records.append(
            {
                "participant_id": participant_id,
                "input_text": input_text,
            }
        )

    return pd.DataFrame(records)


def save_outputs(df: pd.DataFrame, output_path: str) -> None:
    """
    Save QA sequences to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ordered QA sequences for SIGNAL-Preop-style inputs."
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to questionnaire_schema.json. If omitted, a synthetic schema is used.",
    )
    parser.add_argument(
        "--responses",
        type=str,
        default=None,
        help="Path to response CSV. If omitted, synthetic responses are used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/toy_input_text.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    schema_df = load_questionnaire_schema(args.schema)
    response_df = load_responses(args.responses)
    qa_df = build_qa_sequences(schema_df, response_df)

    save_outputs(qa_df, args.output)

    print(f"Saved {len(qa_df)} QA sequences to: {args.output}")
    print("\nExample input_text:\n")
    print(qa_df.iloc[0]["input_text"])


if __name__ == "__main__":
    main()