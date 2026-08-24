"""GSM8K scoring pins."""

from training_sessions_cookbook.rl.envs.gsm8k import (
    _gsm8k_compute_score,
    _gsm8k_extract_answer,
)


def test_numerically_equal_answers_score_correct():
    """``7.0`` vs ``7`` (and ``.5`` vs ``0.5``) are the same number.

    String equality graded these as WRONG, punishing a numerically correct
    model for decimal formatting. Non-numeric values still require exact match.
    """
    assert _gsm8k_compute_score("#### 7.0", "7") == 1.0
    assert _gsm8k_compute_score("#### 7", "7.0") == 1.0
    assert _gsm8k_compute_score("#### 0.5", ".5") == 1.0
    assert _gsm8k_compute_score("#### 1000", "1,000") == 1.0
    assert _gsm8k_compute_score("#### 8", "7") == 0.0


def test_answer_survives_a_long_trailing_tail():
    """An answer followed by >300 chars of prose must still extract.

    The old 300-char tail clip silently scored such responses 0.0; last-match
    semantics already prefer the model's final stated answer, so scanning the
    full text loses nothing.
    """
    text = "#### 42\n" + ("more prose after the answer. " * 20)
    assert _gsm8k_extract_answer(text) == "42"
    revised = "first guess #### 41 ... on reflection: #### 42"
    assert _gsm8k_extract_answer(revised) == "42"
