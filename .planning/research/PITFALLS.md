# Pitfalls Research: MatrixInverse-LLM

## Summary
The primary risks involve numerical precision loss during serialization and the high sensitivity of residuals to small parsing errors.

## Critical Risks

| Pitfall | Prevention Strategy | Detection |
|---------|---------------------|-----------|
| **Tokenization Noise** | Use a strict CSV-like format with fixed decimal length (e.g., `5.31200`) to ensure consistent tokenization and prevent the model from getting "lost" in numbers. | Manual inspection of token IDs during early sampling. |
| **Numerical Instability** | Filter out matrices with high condition numbers ($> 50$) or small determinants ($< 0.05$). Residuals for near-singular matrices become meaningless. | Check condition numbers during generation phase. |
| **Parsing Delicacy** | Use robust regex-based row extractors that can handle minor whitespace variations while remaining strict about comma counts. | Unit tests for `parse_prediction.py` with malformed strings. |
| **Tinker Credentials** | Ensure `TINKER_API_KEY` is loaded before any client initialization to avoid immediate crashes. | Early health check on startup. |
| **Training Duration** | Since we have a 2-day deadline, use a small LoRA rank (e.g., 16) and a limited number of steps (e.g., 400) to ensure at least one full run completes. | Track training steps vs time. |

## Prevention by Phase

- **Phase 1 (Gen)**: Enforce residual checks on the gold inverse targets to ensure NumPy didn't fail.
- **Phase 3 (Eval)**: Report both Mean Absolute Error (MAE) and Functional Residuals. Sometimes a model gets the numbers "close" but the residual is huge due to sign errors.
- **Phase 5 (Train)**: Start with 3x3 matrices only to verify the Tinker training loop before scaling to 5x5.

---
*Last updated: 2026-04-28*
