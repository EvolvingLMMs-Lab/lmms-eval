"""Parametrized determinism detection tests.

Replaces the individual TestDeterminismDetection methods from
test_response_cache.py with concise parametrized equivalents.
"""

import pytest

from lmms_eval.caching.response_cache import is_deterministic


@pytest.mark.parametrize(
    "request_type, gen_kwargs, expected",
    [
        # loglikelihood is always deterministic
        ("loglikelihood", {"temperature": 999}, True),
        ("loglikelihood", {}, True),
        ("loglikelihood", None, True),
        # temp=0 or missing -> deterministic
        ("generate_until", {"temperature": 0}, True),
        ("generate_until", {"temperature": 0.0}, True),
        ("generate_until", {}, True),
        ("generate_until", None, True),
        # positive temperature -> non-deterministic
        ("generate_until", {"temperature": 0.7}, False),
        ("generate_until", {"temperature": 1}, False),
        ("generate_until", {"temperature": 0.01}, False),
        # do_sample=True overrides temp=0
        ("generate_until", {"temperature": 0, "do_sample": True}, False),
        # multi-return keys
        ("generate_until", {"n": 3}, False),
        ("generate_until", {"best_of": 2}, False),
        ("generate_until", {"num_return_sequences": 5}, False),
    ],
    ids=[
        "loglikelihood_high_temp",
        "loglikelihood_empty",
        "loglikelihood_none",
        "gen_temp_0_int",
        "gen_temp_0_float",
        "gen_empty_kwargs",
        "gen_none_kwargs",
        "gen_temp_0.7",
        "gen_temp_1",
        "gen_temp_0.01",
        "gen_do_sample_overrides",
        "gen_n_3",
        "gen_best_of_2",
        "gen_num_return_sequences_5",
    ],
)
def test_is_deterministic(request_type, gen_kwargs, expected):
    assert is_deterministic(request_type, gen_kwargs) is expected
