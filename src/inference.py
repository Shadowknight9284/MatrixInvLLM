import os
import logging
import re

try:
    from tinker import ServiceClient, ModelInput, EncodedTextChunk, SamplingParams
except ImportError:
    ServiceClient = None
    ModelInput = None
    EncodedTextChunk = None
    SamplingParams = None
    logger = logging.getLogger(__name__)
    logger.warning("tinker library not found. Inference will return mock data.")

logger = logging.getLogger(__name__)

def build_prompt(raw_prompt: str, matrix_size: int) -> str:
    """
    Append the 'Inverse:' section header so the prompt ends mid-section.

    Ending with an open section cues base models to continue generating
    (avoiding the immediate-EOS failure caused by prompts that look
    like complete documents).  The finetuned model produces the same rows
    it learned during training regardless of what precedes the header.
    """
    return raw_prompt + "\nInverse:\n"

def _truncate_to_first_inverse_block(text: str, matrix_size: int = 0) -> str:
    """
    Return the first 'Inverse:' header plus at most `matrix_size` numeric rows.
    Stops early at a second 'Inverse:' header if one appears first.

    Also handles the case where the model glues a second 'Inverse:' directly onto
    the last numeric row without a newline (e.g. "1.2, 3.4, 5.6Inverse:").  In
    that case only the numeric prefix is kept and the inline header triggers the
    same early-stop as a standalone header line.

    No-header path: when the prompt already ends with 'Inverse:\\n' the model
    only emits the numeric rows.  If an inline 'Inverse:' appears later (the
    start of a repetition loop), everything before it is returned as the block.
    Trailing alphabetic tokens that the decoder leaves between the last digit
    and the inline header (e.g. 'g', 'ing' from a stripped <|eot_id|>) are
    stripped before returning.

    When matrix_size == 0 the function falls back to header-boundary detection only.
    Logs a WARNING whenever any content is dropped.
    """
    _NUM_RE = re.compile(r"[-+]?\d")
    _HDR_RE = re.compile(r"(?im)^inverse\s*:?[ \t]*$")
    _INLINE_HDR_RE = re.compile(r"(?i)inverse\s*:")

    # Any 2+ letter word followed by ':' that appears after a digit — covers both
    # 'Inverse:' and BPE-corruption variants like 'ING:', 'ings:', 'Ing:'.
    _ANY_HDR_RE = re.compile(r'[A-Za-z]{2,}\s*:')

    first = _HDR_RE.search(text)
    if not first:
        # No standalone Inverse: header.  The prompt likely already ends with
        # 'Inverse:\n', so the model output is pure rows.  Find the earliest
        # header-like marker (Inverse: or any corrupted BPE variant) and return
        # everything before it, stripping trailing alphabetic garbage.
        candidates = [m for m in (_INLINE_HDR_RE.search(text), _ANY_HDR_RE.search(text)) if m]
        inline_first = min(candidates, key=lambda m: m.start()) if candidates else None
        if inline_first:
            before = text[:inline_first.start()]
            # Strip trailing alphabetic garbage (e.g. 'g', 'ing', 'ING')
            # left when <|eot_id|> is removed between the last digit and the marker.
            before = re.sub(r'[A-Za-z]+\s*$', '', before).rstrip()
            if before:
                logger.warning(
                    "No standalone Inverse: header; truncated before inline "
                    "header (%d chars dropped).", len(text) - len(before)
                )
                return before
        return text

    header = text[first.start():first.end()]
    after_header = text[first.end():]

    kept_lines = []
    row_count = 0
    truncated = False

    for line in after_header.splitlines(keepends=True):
        stripped = line.rstrip()

        # Clean standalone header → stop.
        if _HDR_RE.match(stripped):
            truncated = True
            break

        # Inline header glued to end of a numeric row (e.g. "1.2, 3.4Inverse:").
        # Take only the numeric prefix so the stored prediction stays clean.
        inline_m = _INLINE_HDR_RE.search(stripped)
        if inline_m:
            prefix = stripped[:inline_m.start()].rstrip()
            if prefix:
                kept_lines.append(prefix + "\n")
                if _NUM_RE.search(prefix):
                    row_count += 1
            truncated = True
            break

        kept_lines.append(line)
        if line.strip() and _NUM_RE.search(line):
            row_count += 1
            if matrix_size > 0 and row_count >= matrix_size:
                truncated = True
                break

    result = (header + "".join(kept_lines)).rstrip()
    if truncated:
        dropped = len(text) - first.start() - len(result) + len(header.rstrip())
        logger.warning(
            "Truncated model output to first Inverse: block (%d chars dropped).", dropped
        )
    return result


def _extract_matrix_size_from_prompt(prompt: str) -> int:
    """
    Read the MatrixSize field from the prompt.
    Returns 0 if not found.
    """
    m = re.search(r"MatrixSize:\s*(\d+)", prompt)
    return int(m.group(1)) if m else 0


def get_tinker_client():
    """
    Initialize and return the Tinker ServiceClient.
    Expects TINKER_API_KEY to be set in environment.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        logger.warning("TINKER_API_KEY not found in environment.")

    return ServiceClient(api_key=api_key)


def sample_model(prompt, config, model_id=None):
    """
    Sample from a Tinker model using the SamplingClient.
    Supports both base models and adapter paths.

    Automatically appends strict format instructions derived from the
    MatrixSize field embedded in the prompt.

    Returns the cleaned decoded string (special tokens stripped).
    For richer diagnostics use sample_model_with_diagnostics().
    """
    cleaned_text, _ = sample_model_with_diagnostics(prompt, config, model_id=model_id)
    return cleaned_text


def sample_model_with_diagnostics(prompt, config, model_id=None):
    """
    Instrumented variant of sample_model().

    Returns
    -------
    tuple[str, dict]
        cleaned_text : str
            Model output with special tokens stripped (same as sample_model).
        diag : dict
            Diagnostic metadata for failure-mode classification:

            first_token_id  – int | None   ID of first generated token.
            token_count     – int          Total generated tokens.
            stop_reason     – str          'eos_only' | 'max_tokens' | 'normal' | 'error' | 'mock'
            adapter_status  – str          'base_model' | 'adapter_loaded' | 'adapter_error:<msg>'
            raw_text        – str          Decoded with special tokens kept.
            cleaned_text    – str          Decoded with special tokens stripped.
            is_eos_only     – bool         cleaned_text is empty; raw_text was EOS-only.
    """
    inf_cfg = config.get("inference", {})
    diag_cfg = config.get("diagnostics", {})
    max_raw_log = diag_cfg.get("max_raw_log_chars", 500)

    if model_id is None:
        model_id = inf_cfg.get("base_model", "meta-llama/Llama-3.2-3B")

    # ── Format-enforcement injection ──────────────────────────────────────
    matrix_size = _extract_matrix_size_from_prompt(prompt)
    if matrix_size > 0:
        prompt = build_prompt(prompt, matrix_size)
        logger.debug("Format instruction injected for %dx%d matrix.", matrix_size, matrix_size)
    else:
        logger.warning("MatrixSize not found in prompt — skipping format instruction injection.")
    # ─────────────────────────────────────────────────────────────────────

    # ── Mock path ─────────────────────────────────────────────────────────
    if ServiceClient is None:
        mock_text = "Inverse:\n0.10000, 0.20000\n0.30000, 0.40000"
        diag = {
            "first_token_id": None,
            "token_count": len(mock_text),
            "stop_reason": "mock",
            "adapter_status": "mock",
            "raw_text": mock_text,
            "cleaned_text": mock_text,
            "is_eos_only": False,
        }
        return mock_text, diag
    # ─────────────────────────────────────────────────────────────────────

    client = get_tinker_client()
    adapter_status = "base_model"

    try:
        sampling_client = None

        # Heuristic: base models have '/' and aren't prefixed with tinker://
        if "/" in model_id and "lora" not in model_id.lower() and not model_id.startswith("tinker://"):
            try:
                sampling_client = client.create_sampling_client(base_model=model_id)
                adapter_status = "base_model"
            except Exception:
                pass

        if sampling_client is None:
            path = model_id
            if not any(path.startswith(p) for p in ["tinker://", "./", "/", "C:", "c:"]):
                path = f"tinker://{path}"
            try:
                sampling_client = client.create_sampling_client(model_path=path)
                adapter_status = "adapter_loaded"
            except Exception as e:
                adapter_status = f"adapter_error:{e}"
                logger.error("Failed to load adapter '%s': %s", path, e)
                error_diag = {
                    "first_token_id": None,
                    "token_count": 0,
                    "stop_reason": "error",
                    "adapter_status": adapter_status,
                    "raw_text": "",
                    "cleaned_text": "",
                    "is_eos_only": False,
                }
                return "", error_diag

        if diag_cfg.get("log_adapter_status", True):
            logger.info("adapter_status=%s  model_id=%s", adapter_status, model_id)

        tokenizer = sampling_client.get_tokenizer()
        tokens = tokenizer.encode(prompt)

        model_input = ModelInput(chunks=[EncodedTextChunk(tokens=tokens)])
        params = SamplingParams(
            max_new_tokens=inf_cfg.get("max_new_tokens", 512),
            temperature=inf_cfg.get("temperature", 0.0),
            stop_sequences=inf_cfg.get("stop_sequences", []),
        )

        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=params,
        )
        result = future.result()

        if not (hasattr(result, "sequences") and result.sequences):
            logger.warning("No sequences returned from model for model_id=%s", model_id)
            empty_diag = {
                "first_token_id": None,
                "token_count": 0,
                "stop_reason": "eos_only",
                "adapter_status": adapter_status,
                "raw_text": "",
                "cleaned_text": "",
                "is_eos_only": True,
            }
            return "", empty_diag

        gen_tokens = result.sequences[0].tokens
        token_count = len(gen_tokens)
        first_token_id = int(gen_tokens[0]) if token_count > 0 else None

        # Double-decode: keep special tokens first to detect EOS-only outputs
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        cleaned_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        is_eos_only = bool(raw_text.strip() and not cleaned_text.strip())

        # Clip repeated Inverse: blocks before any downstream parsing sees them
        if not is_eos_only:
            cleaned_text = _truncate_to_first_inverse_block(cleaned_text, matrix_size)

        # Determine stop reason heuristic
        if is_eos_only or token_count == 0:
            stop_reason = "eos_only"
        elif token_count >= inf_cfg.get("max_new_tokens", 512):
            stop_reason = "max_tokens"
        else:
            stop_reason = "normal"

        if diag_cfg.get("log_first_token_id", True):
            logger.info(
                "first_token_id=%s  token_count=%d  stop_reason=%s",
                first_token_id, token_count, stop_reason,
            )
        if diag_cfg.get("log_raw_output", True):
            logger.info("raw_text (first %d chars): %r", max_raw_log, raw_text[:max_raw_log])
        if diag_cfg.get("log_cleaned_output", True):
            logger.info("cleaned_text (first %d chars): %r", max_raw_log, cleaned_text[:max_raw_log])

        diag = {
            "first_token_id": first_token_id,
            "token_count": token_count,
            "stop_reason": stop_reason,
            "adapter_status": adapter_status,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "is_eos_only": is_eos_only,
        }
        return cleaned_text, diag

    except Exception as e:
        logger.error("Tinker inference error for model %s: %s", model_id, e)
        err_diag = {
            "first_token_id": None,
            "token_count": 0,
            "stop_reason": "error",
            "adapter_status": adapter_status,
            "raw_text": "",
            "cleaned_text": "",
            "is_eos_only": False,
        }
        return "", err_diag
