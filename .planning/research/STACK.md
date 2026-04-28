# Stack Research: MatrixInverse-LLM

## Summary
The project uses a high-performance stack centered around the Tinker API for model operations and standard Python numerical libraries for data generation and validation.

## Core Stack

| Component | Choice | Rationale | Confidence |
|-----------|--------|-----------|------------|
| **Base Model** | `meta-llama/Llama-3.2-3B` | Small enough for fast iteration, yet capable of following strict formatting. | High |
| **Backend API** | **Tinker API** | Mandatory requirement. Provides low-latency sampling and efficient LoRA training orchestration. | High |
| **Numerical Engine**| `NumPy` / `SciPy` | Gold standard for matrix operations and linear algebra. | High |
| **Serialization** | Strict CSV-style format | Maximizes parseability for both models and regex-based output parsers. | High |
| **Environment** | Python 3.10+ | Compatibility with `tinker` client and modern async-io patterns. | High |

## Specific Libraries & Versions

- `tinker-python`: Latest (assumed `>0.1.0`)
- `numpy`: `^1.24.0`
- `scipy`: `^1.10.0`
- `pyyaml`: For configuration management.
- `pydantic`: For data validation and schema management (recommended for strict serialization).

## What NOT to use
- **Generic Adapter Layers**: Avoid `langchain` or `llama-index` for model calls to maintain direct Tinker integration as requested.
- **Symbolic Math Libraries**: Avoid `sympy` for the core pipeline as we are testing numerical accuracy, not algebraic manipulation.
- **Heavyweight Training Frameworks**: Avoid `transformers` / `peft` locally, as Tinker handles the training logic in the cloud.

---
*Last updated: 2026-04-28*
