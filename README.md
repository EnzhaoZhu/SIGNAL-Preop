# SIGNAL-Preop

This repository provides a reproducibility template for **SIGNAL-Preop**, a structured interview–driven language modeling pipeline for preoperative psychiatric screening.

The code demonstrates how structured questionnaire responses can be converted into ordered question–answer sequences, encoded using a Longformer-based model, trained with multitask learning, and used for inference.

This repository uses **synthetic toy data only**. No real participant data are included.

---

## Overview

SIGNAL-Preop consists of four main steps:

1. **Preprocessing**
   Structured questionnaire responses are converted into ordered question–answer sequences.

2. **Modeling**
   A shared Longformer encoder is used to extract patient-level text representations.

3. **Multitask training**
   Four task-specific heads predict:

   * DD: depressive disorder
   * GAD: generalized anxiety disorder
   * IND: insomnia disorder
   * SR: suicide risk

4. **Inference**
   The trained model outputs task-specific probabilities and binary predictions.

---

## Repository structure

```text
SIGNAL-Preop/
├── config/
│   └── config.yml
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model_longformer.py
│   ├── losses.py
│   ├── train_multitask.py
│   ├── inference.py
│   └── utils.py
├── run_simulation.py
├── SIGNAL-Preop Questionnaire Structure.md
└── README.md
```

---

## Requirements

Install the required Python packages:

```bash
pip install torch transformers pandas numpy scikit-learn pyyaml tqdm
```

The default model checkpoint is:

```text
schen/longformer-chinese-base-4096
```

If HuggingFace cannot be accessed directly, download the model manually and provide a local path.

Example local path:

```text
models/longformer-base-4096
```

---

## Quick start

Run the full synthetic simulation:

```bash
python run_simulation.py \
  --model_name schen/longformer-chinese-base-4096 \
  --num_epochs 1 \
  --batch_size 2 \
  --max_length 2048
```

For offline use, provide a local Longformer path:

```bash
python run_simulation.py \
  --model_name ./models/longformer-base-4096 \
  --num_epochs 1 \
  --batch_size 2 \
  --max_length 2048
```

On Windows, an absolute path can also be used:

```bash
python run_simulation.py ^
  --model_name "G:\华西\SIGNAL-Preop\models\longformer-base-4096" ^
  --num_epochs 1 ^
  --batch_size 2 ^
  --max_length 2048
```

---

## Smoke test

For a faster code check on CPU, use a shorter sequence length:

```bash
python run_simulation.py \
  --model_name ./models/longformer-base-4096 \
  --num_epochs 1 \
  --batch_size 2 \
  --max_length 256
```

Note: the manuscript used `max_length = 2048`. The shorter length is only for checking whether the code runs.

---

## Outputs

The simulation creates the following files:

```text
outputs/simulation/
├── inputs/
│   └── toy_input_text.csv
├── checkpoints/
│   └── multitask_longformer.pt
└── predictions/
    └── toy_predictions.csv
```

The prediction file contains:

```text
participant_id,P_DD,P_GAD,P_IND,P_SR,pred_DD,pred_GAD,pred_IND,pred_SR
```

---

## Main scripts

### `src/preprocess.py`

Builds ordered question–answer sequences from structured questionnaire responses.

Key features:

* Preserves the original response order
* Uses fixed question–answer separators
* Encodes skipped modules as negative responses
* Outputs one text sequence per participant

### `src/dataset.py`

Tokenizes QA sequences using a Longformer-compatible tokenizer.

Key settings:

* Padding and truncation are applied
* Maximum sequence length is set to 2048 by default
* Labels are formatted for multitask binary classification

### `src/model_longformer.py`

Defines the Longformer-based model architecture.

Model structure:

```text
Ordered QA sequence
        ↓
Shared Longformer encoder
        ↓
Patient-level representation
        ↓
Task-specific heads for DD, GAD, IND, and SR
```

### `src/losses.py`

Implements:

* Binary focal loss
* Uncertainty-based multitask loss weighting

### `src/train_multitask.py`

Trains the Longformer multitask model using synthetic or user-provided de-identified data.

### `src/inference.py`

Runs multitask prediction and outputs probabilities and binary predictions for DD, GAD, IND, and SR.

---

## Questionnaire structure

The file `SIGNAL-Preop Questionnaire Structure.md` provides a conceptual description of the questionnaire domains, item concepts, and branching logic.

It does not reproduce the exact clinical wording of the original questionnaire.

---

## Data availability

This repository does not include individual-level patient data because of privacy and institutional data-protection requirements.

All example inputs are synthetic and are provided only to demonstrate code execution.

Researchers may adapt the code to their own de-identified data using the expected input formats.

---

## Disclaimer

This code is intended for research reproducibility and method demonstration only. It is not a standalone diagnostic tool and should not be used for clinical decision-making without appropriate validation, governance, and clinical oversight.
