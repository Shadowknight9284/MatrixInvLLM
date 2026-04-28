# MatrixInverse-LLM

## Project Goal

Build a Python research pipeline that tests how accurately a small LLM can output the inverse of dense square matrices after supervised LoRA fine-tuning on synthetic data. The system must generate valid random invertible matrices, compute trusted inverse targets with numerical linear algebra, evaluate model outputs numerically, fine-tune the model, and compare pre/post-training performance.[cite:166][cite:169][cite:175]

## Core Research Question

How accurately can a small LoRA-tuned LLM produce the inverse of dense real-valued matrices, and how does accuracy change as matrix size and condition number increase?[cite:166][cite:177]

## MVP Scope

The MVP should focus on a working end-to-end experiment rather than a polished research platform.[cite:175][cite:178]

### In scope
- Python project only
- Synthetic dataset generation
- Dense square real-valued matrices
- Matrix sizes configurable, with MVP defaults of 3x3, 4x4, and 5x5
- Held-out validation and test sets
- Strict text serialization of matrices for LLM input/output
- Numeric parsing and evaluation
- One baseline evaluation of the base model
- One LoRA fine-tuning run
- One post-training evaluation
- Summary metrics broken down by matrix size and tolerance

### Out of scope for MVP
- Web UI or dashboard
- RL or agentic hyperparameter search
- Sparse matrices
- Symbolic inversion
- Complex-valued matrices
- Distributed training
- Automatic multi-loop self-improvement

## Scientific Framing

The gold inverse should be treated as a trusted numerical target, not exact symbolic truth, because floating-point inversion is inherently approximate. Ground truth must therefore be validated using residual checks against the identity matrix rather than described as mathematically certain.[cite:166][cite:169]

## System Overview

The pipeline has six major stages:

1. Generate random dense invertible matrices.
2. Compute trusted inverse targets with numerical linear algebra.
3. Serialize matrices and inverses into a strict machine-readable text format.
4. Evaluate the base model on a frozen held-out set.
5. Fine-tune the model with LoRA on the synthetic train set.
6. Re-evaluate and compare baseline vs fine-tuned performance.[cite:169][cite:175][cite:178]

## Dataset Specification

### Matrix generation

Generate matrices \(A \in \mathbb{R}^{n \times n}\) for a configurable list of sizes. Entries should be sampled from a bounded continuous distribution such as uniform[-1, 1]. Reject matrices that are singular or near-singular using determinant and condition-number filters so the inverse targets are numerically stable.[cite:166][cite:169]

### Acceptance criteria for generated matrices
- Matrix is square and dense
- Matrix rank is full
- Absolute determinant is above a configurable threshold
- Condition number is below a configurable maximum
- Inverse computation succeeds
- Residual checks for the gold inverse pass

### Stored metadata per example
Each example should store:
- `id`
- `matrix_size`
- `matrix`
- `inverse`
- `determinant`
- `condition_number`
- `input_text`
- `target_text`
- `split`
- random seed or generation metadata where useful

## Prompt and Output Format

The text format must be extremely strict so parsing is deterministic and malformed generations are easy to detect.[cite:175]

### Input format

```text
MatrixSize: 3
Matrix:
0.250000, -1.000000, 2.500000
1.500000, 0.750000, -0.500000
3.000000, 1.250000, 0.125000
```

### Required target/output format

```text
Inverse:
a11, a12, a13
a21, a22, a23
a31, a32, a33
```

### Formatting rules
- Fixed decimal precision from config
- One row per line
- Comma-separated values
- No prose in normal mode
- Parser should trim whitespace but otherwise be strict

## Evaluation Specification

The predicted inverse \(X_{pred}\) must be scored both by direct distance to the gold inverse and by functional correctness as an inverse.[cite:166][cite:169]

### Per-example metrics
- Parse success
- Correct shape
- Mean absolute error against gold inverse
- Mean squared error against gold inverse
- Frobenius norm of \(X_{pred} - X_{gold}\)
- Left residual norm of \(I - AX_{pred}\)
- Right residual norm of \(I - X_{pred}A\)
- Pass/fail at configured tolerances

### Primary success definition
A prediction counts as a success only if:
- parsing succeeds
- output matrix shape is correct
- residual norm is below the configured threshold

### Required tolerance buckets
- strict: 1e-2
- medium: 5e-2
- loose: 1e-1

### Aggregate reports
Evaluation must report:
- overall parse rate
- overall valid-shape rate
- overall success rate per tolerance
- metrics by matrix size
- metrics by condition-number bucket
- baseline vs post-train comparison

## Fine-Tuning Specification

Use supervised LoRA fine-tuning on prompt/target pairs. LoRA is appropriate because it reduces training cost and complexity while still enabling targeted adaptation for narrow tasks.[cite:125][cite:180]

### Training inputs
- `input_text` as prompt
- `target_text` as completion

### Configurable training parameters
- base model name
- LoRA rank
- LoRA alpha
- LoRA dropout
- learning rate
- batch size
- max steps
- max sequence length
- optimizer settings

## Repo Structure

```text
matrix-inverse-llm/
├── config.yaml
├── README.md
├── data/
│   ├── raw/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── eval_results/
├── checkpoints/
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   └── plots/
└── src/
    ├── generate_matrices.py
    ├── serialize_dataset.py
    ├── build_splits.py
    ├── inference.py
    ├── parse_prediction.py
    ├── score_prediction.py
    ├── evaluate_model.py
    ├── train_lora.py
    ├── run_experiment.py
    └── utils.py
```

## Module Responsibilities

| File | Responsibility |
|---|---|
| `generate_matrices.py` | Sample dense matrices, filter for stability, compute trusted inverses |
| `serialize_dataset.py` | Convert numeric examples into strict prompt/target text |
| `build_splits.py` | Build train/val/test JSONL datasets with reproducible splits |
| `inference.py` | Run baseline or post-train model generation |
| `parse_prediction.py` | Parse text into numeric matrices and validate shape |
| `score_prediction.py` | Compute residuals, norm errors, and pass/fail metrics |
| `evaluate_model.py` | Aggregate metrics over a held-out set |
| `train_lora.py` | Run LoRA fine-tuning |
| `run_experiment.py` | Orchestrate baseline eval → train → post-train eval |
| `utils.py` | Shared helpers for IO, matrix formatting, seeding, logging |

## Recommended Build Order

### Phase 1
- `config.yaml`
- `src/utils.py`
- `src/generate_matrices.py`

Goal: generate valid dense invertible matrices and trusted inverse targets.

### Phase 2
- `src/serialize_dataset.py`
- `src/build_splits.py`

Goal: produce reproducible train/val/test JSONL datasets.

### Phase 3
- `src/parse_prediction.py`
- `src/score_prediction.py`

Goal: parse model outputs and compute robust numeric evaluation metrics.

### Phase 4
- `src/inference.py`
- `src/evaluate_model.py`

Goal: run baseline evaluation on the held-out test set.

### Phase 5
- `src/train_lora.py`

Goal: fine-tune the model with LoRA on the synthetic train set.

### Phase 6
- `src/run_experiment.py`

Goal: run baseline → train → post-train evaluation end to end.

## MVP Success Criteria

The MVP is successful if it can:
- generate a valid synthetic dataset
- train on it
- evaluate the base model and the fine-tuned model on a frozen test set
- show quantitative metrics comparing the two
- break out results by matrix size and tolerance bucket.[cite:175][cite:178]

## Deadline Strategy

Because time is limited, the implementation must prioritize a working experiment on 3x3 to 5x5 matrices first. Larger sizes like 8x8 or 10x10 should be treated as stretch goals after the core pipeline works.[cite:177][cite:178]
