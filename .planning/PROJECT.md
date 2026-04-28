# MatrixInverse-LLM

A modular Python research pipeline to test LLM accuracy in computing matrix inverses after LoRA fine-tuning on synthetic data, using the Tinker API as the primary backend for both training and inference.

## Context

The project aims to evaluate the ability of small LLMs (e.g., Llama 3.2 3B) to perform numerical linear algebra tasks, specifically dense square matrix inversion. The core of the implementation relies on the Tinker API for efficient LoRA fine-tuning and high-throughput sampling.

## Core Value

Demonstrate a clear performance delta between a baseline LLM and a LoRA-tuned LLM on a well-defined numerical task (3x3 to 5x5 matrix inversion) within a 2-day MVP timeframe.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] [Phase 1] Generate synthetic dataset of invertible 3x3, 4x4, 5x5 matrices.
- [ ] [Phase 1] Compute trusted inverses using NumPy/SciPy and perform residual validation.
- [ ] [Phase 2] Serialize matrices into strict machine-readable CSV-style text.
- [ ] [Phase 2] Build Reproducible train/val/test JSONL splits.
- [ ] [Phase 3] Implement strict parser and numeric scoring (MAE, Residuals).
- [ ] [Phase 4] Implement Tinker-first inference client for baseline sampling.
- [ ] [Phase 4] Run baseline evaluation on the test set.
- [ ] [Phase 5] Implement Tinker-first LoRA training client and run fine-tuning.
- [ ] [Phase 6] Run post-train evaluation and generate comparison report.

### Out of Scope

- Web UI or dashboard.
- Sparse or complex-valued matrices.
- Agentic hyperparameter search.
- Multi-loop self-improvement.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Tinker API | Specified as the mandatory backend for both training and sampling. | Pending |
| NumPy/SciPy | Standard, trusted libraries for generating gold inverse targets. | Pending |
| Strict CSV Format | Minimize parsing errors and ensure deterministic evaluation. | Pending |
| LoRA Fine-tuning | Efficient adaptation of small models for specific numeric tasks. | Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-28 after initialization*
