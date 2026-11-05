"""Tests for the model-free parts of gen/generate.py."""

import json

import pytest
import yaml

from gen.generate import (
    batch_seed, chunks, count_new_tokens, generation_kwargs, left_pad, question_order, read_done,
)

MAX = 128


@pytest.mark.parametrize("setting, expected", [
    ({"do_sample": False},
     {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0, "max_new_tokens": MAX}),
    ({"do_sample": False, "num_beams": 5},
     {"do_sample": False, "num_beams": 5, "repetition_penalty": 1.0, "max_new_tokens": MAX}),
    ({"do_sample": True, "temperature": 1.2},
     {"do_sample": True, "num_beams": 1, "repetition_penalty": 1.0, "max_new_tokens": MAX,
      "temperature": 1.2, "top_k": 0, "top_p": 1.0}),
    ({"do_sample": True, "temperature": 1.0, "top_p": 0.9},
     {"do_sample": True, "num_beams": 1, "repetition_penalty": 1.0, "max_new_tokens": MAX,
      "temperature": 1.0, "top_k": 0, "top_p": 0.9}),
])
def test_generation_kwargs(setting, expected):
    assert generation_kwargs(setting, MAX) == expected


def test_generation_kwargs_rejects_bad_settings():
    with pytest.raises(ValueError):
        generation_kwargs({"do_sample": True, "temprature": 0.7}, MAX)  # typo
    with pytest.raises(ValueError):
        generation_kwargs({"temperature": 0.7}, MAX)                    # no do_sample


def test_every_config_setting_is_valid():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    for name, setting in cfg["decoding"].items():
        kw = generation_kwargs(setting, cfg["max_new_tokens"])
        assert kw["repetition_penalty"] == 1.0, name
        if kw["do_sample"]:
            assert kw["top_k"] == 0, name


def test_question_order_is_fixed():
    ids = [f"A/2009/page_{i}.pdf-1" for i in range(20)]
    a = question_order(ids, seed=354)
    assert sorted(a) == sorted(ids)
    assert a == question_order(list(reversed(ids)), seed=354)
    assert a != question_order(ids, seed=1)


def test_batch_seed():
    assert batch_seed(354, "temp12", "A/2009/page_1.pdf-1") == batch_seed(354, "temp12", "A/2009/page_1.pdf-1")
    assert batch_seed(354, "temp12", "A/2009/page_1.pdf-1") != batch_seed(354, "temp07", "A/2009/page_1.pdf-1")


def test_left_pad():
    ids, mask = left_pad([[5, 6, 7], [8]], pad_id=0)
    assert ids == [[5, 6, 7], [0, 0, 8]]
    assert mask == [[1, 1, 1], [0, 0, 1]]


def test_count_new_tokens():
    stop = {99, 0}
    assert count_new_tokens([4, 5, 99, 0, 0], stop) == (2, False)   # stopped at EOS
    assert count_new_tokens([4, 5, 6], stop) == (3, True)           # ran to the limit
    assert count_new_tokens([99], stop) == (0, False)               # empty answer


def test_chunks():
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_read_done_drops_half_written_line(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n" + '{"id": "c", "ans')
    assert read_done(path) == {"a", "b"}
    assert path.read_text().splitlines() == [json.dumps({"id": "a"}), json.dumps({"id": "b"})]
    assert read_done(tmp_path / "missing.jsonl") == set()
