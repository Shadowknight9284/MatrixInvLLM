from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DummySequence:
    tokens: List[int]


@dataclass
class DummyResult:
    sequences: List[DummySequence]


class DummyTokenizer:
    def decode(self, tokens: List[int]) -> str:
        return ''.join(chr(t) for t in tokens)


def build_same_prompt_we_have_been_using(raw_prompt: str, matrix_size: int) -> str:
    """
    Standalone version that mirrors the old style prompt you were using:
    it includes explanatory text and an example-oriented format section.

    If you want this to match your exact production prompt byte-for-byte,
    replace the body of this function with your current build_prompt.
    """
    return (
        f"{raw_prompt}\n\n"
        f"The numbers should be in the same order as in the Matrix. "
        f"For example, if the first row of the Matrix is row 1, then the first row of the Inverse should correspond to row 1 as well.\n"
        f"Return the inverse in row-major order.\n"
        f"Output should begin with the word Inverse and then the numbers.\n"
        f"MatrixSize {matrix_size}.\n"
    )


def make_dummy_result_from_text(text: str) -> DummyResult:
    return DummyResult(sequences=[DummySequence(tokens=[ord(c) for c in text])])


def inspect_generation(prompt: str, result: DummyResult, tokenizer: DummyTokenizer) -> None:
    logger.info('PROMPT START')
    logger.info(prompt)
    logger.info('PROMPT END')

    if not result.sequences:
        logger.error('No sequences returned from model.')
        return

    tokens = result.sequences[0].tokens
    logger.info('Generated token count: %d', len(tokens))
    logger.info('First 80 token ids: %s', tokens[:80])

    raw_text = tokenizer.decode(tokens)
    logger.info('RAW MODEL OUTPUT START')
    logger.info(raw_text)
    logger.info('RAW MODEL OUTPUT END')

    cleaned = (
        raw_text.replace('<|end_of_text|>', '')
        .replace('<|eot_id|>', '')
        .replace('endoftext', '')
        .strip()
    )

    logger.info('CLEANED MODEL OUTPUT START')
    logger.info(cleaned)
    logger.info('CLEANED MODEL OUTPUT END')


def main() -> None:
    matrix_size = 3
    raw_prompt = (
        'MatrixSize 3\n'
        'Matrix\n'
        '0.26706, 0.07155, -0.81942\n'
        '0.67060, -0.35844, -0.62696\n'
        '-0.91845, 0.18179, 0.35513'
    )

    prompt = build_same_prompt_we_have_been_using(raw_prompt, matrix_size)

    dummy_good_output = (
        'Inverse\n'
        '-0.06993, -0.91545, -1.77754\n'
        '1.77288, -3.45330, -2.00592\n'
        '-1.08836, -0.59989, -0.75447\n'
        '<|end_of_text|>'
    )

    dummy_bad_output = (
        'The numbers should be in the same order as in the Matrix. '
        'For example, if the first row of the Matrix is row 1, then the first row of the Inverse should correspond to row 1 as well.\n'
        'beginoftext Inverse\n'
        '0.00000, 0.00000, 0.00000\n'
        '0.00000, 0.00000, 0.00000\n'
        '0.00000, 0.00000, 0.00000\n'
        'endoftext'
    )

    tokenizer = DummyTokenizer()

    logger.info('===== GOOD DUMMY CASE =====')
    good_result = make_dummy_result_from_text(dummy_good_output)
    inspect_generation(prompt, good_result, tokenizer)

    logger.info('===== BAD DUMMY CASE =====')
    bad_result = make_dummy_result_from_text(dummy_bad_output)
    inspect_generation(prompt, bad_result, tokenizer)


if __name__ == '__main__':
    main()