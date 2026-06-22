import pytest

from promptforge.metrics import exact_match, f1_token


# ----- exact_match -----

def test_exact_match_identical():
    assert exact_match("Paris", "Paris") == 1.0


def test_exact_match_case_insensitive():
    assert exact_match("paris", "Paris") == 1.0


def test_exact_match_strips_punctuation():
    assert exact_match("Paris!", "Paris") == 1.0


def test_exact_match_strips_whitespace():
    assert exact_match("  Paris  ", "Paris") == 1.0


def test_exact_match_different():
    assert exact_match("London", "Paris") == 0.0


def test_exact_match_empty_both():
    assert exact_match("", "") == 1.0


# ----- f1_token -----

def test_f1_identical():
    assert f1_token("the cat sat", "the cat sat") == 1.0


def test_f1_perfect_precision_low_recall():
    score = f1_token("the cat", "the cat sat on the mat")
    assert 0.0 < score < 1.0


def test_f1_no_overlap():
    assert f1_token("hello world", "foo bar baz") == 0.0


def test_f1_both_empty():
    assert f1_token("", "") == 1.0


def test_f1_one_empty():
    assert f1_token("hello", "") == 0.0
    assert f1_token("", "hello") == 0.0


def test_f1_symmetric():
    a, b = "quick brown fox", "brown fox jumps high"
    assert abs(f1_token(a, b) - f1_token(b, a)) < 1e-9


def test_f1_partial():
    score = f1_token("sports", "sports technology")
    assert 0.0 < score < 1.0
