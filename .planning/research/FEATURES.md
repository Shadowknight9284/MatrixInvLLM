# Features Research: MatrixInverse-LLM

## Summary
The pipeline focuses on data quality and rigorous numerical evaluation of LLM-generated matrix inverses.

## Table Stakes (Must Have)

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Stable Generation** | Filter matrices by determinant and condition number to ensure numerical stability. | Low |
| **Strict Serialization** | CSV-like format with fixed decimal precision to minimize tokenization ambiguity. | Low |
| **Residual Validation** | Scoring based on $||I - AX||$ rather than just element-wise distance. | Medium |
| **Tinker Sampling** | Integration with `tinker.SamplingClient` for baseline and evaluation tasks. | Medium |
| **Tinker LoRA** | Supervised fine-tuning using `tinker.LoraTrainingClient`. | High |

## Differentiators

| Feature | Description |
|---------|-------------|
| **Tolerance Bucketing** | Categorizing results into 'Strict', 'Medium', and 'Loose' success tiers. |
| **Conditioning Analysis** | Correlating LLM accuracy with the condition number of the input matrix. |
| **Direct Tinker Control** | Low-level control over training loops using Tinker primitives for maximum transparency. |

## Anti-Features (Deliberately Not Building)

- **Variable Precision**: We will use fixed precision (e.g., 5 decimals) to keep token counts predictable.
- **Natural Language Explanations**: The model should output *only* the matrix to simplify parsing.
- **Multi-GPU Local Training**: Entirely offloaded to Tinker.

## Feature Dependencies

- **Evaluation** depends on **Parsing**.
- **Parsing** depends on **Serialization Format**.
- **Training** depends on **Dataset Serialization**.
- **Run Experiment** depends on all modules being functional.

---
*Last updated: 2026-04-28*
