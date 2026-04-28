# Phase 4: Tinker LoRA Training - Context

**Gathered:** 2026-04-28
**Status:** Implementation-only (Phase 4 tightly scoped)

<domain>
## Phase Boundary
This phase implements the Tinker-first LoRA fine-tuning pipeline. The sole deliverable is `src/train_lora.py`.

- [ ] `src/train_lora.py`
</domain>

<decisions>
## Implementation Decisions

### Tinker Training Client
- **Decision 1**: Initialize `tinker.ServiceClient()` from `TINKER_API_KEY` in environment.
- **Decision 2**: Create a LoRA training client via `service_client.lora_training(...)` using the base model from `config.yaml model.base_model`.
- **Decision 3**: Pass LoRA hyperparameters directly from `config.yaml lora` (rank, alpha, dropout).
- **Decision 4**: Pass optimizer settings from `config.yaml optimizer` (type, lr, weight_decay, betas).
- **Decision 5**: Pass training settings from `config.yaml training` (batch_size, max_steps, max_seq_length, logging_steps, eval_every_steps).

### Dataset Loading
- **Decision 6**: Load training examples from `data/train.jsonl` as `(input_text, target_text)` pairs.
- **Decision 7**: Optionally load `data/val.jsonl` for validation callbacks if the API supports it.

### Model Persistence
- **Decision 8**: After training completes, capture and persist the resulting fine-tuned model ID or handle to `outputs/finetuned_model_id.txt` so `src/inference.py` can reference it later when running post-train evaluation.

### agent's Discretion
- How to handle Tinker LoRA API response schema details.
- Progress logging strategy during training.
</decisions>

<canonical_refs>
## Canonical References
- [config.yaml](file:///c:/dev/MatrixInvLLM/config.yaml) — All hyperparameters (lora, optimizer, training sections).
- [src/inference.py](file:///c:/dev/MatrixInvLLM/src/inference.py) — `sample_model()` already accepts `model_id` param for post-train eval.
- [.planning/REQUIREMENTS.md](file:///c:/dev/MatrixInvLLM/.planning/REQUIREMENTS.md) — TINK-03.
</canonical_refs>

<deferred>
## Deferred Ideas
- Post-train evaluation (Phase 5).
- Multi-run hyperparameter sweeps.
- Any non-Tinker training backend.
</deferred>

---
*Phase: 04-tinker-lora-training*
