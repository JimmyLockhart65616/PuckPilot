"""The wire format: JSON a browser will actually parse."""

from __future__ import annotations

import json

import numpy as np
import pytest

from puckpilot.web import wire


def test_a_nan_becomes_null_not_a_bare_nan():
    """`json.dumps` writes NaN; `JSON.parse` rejects it, freezing the page."""
    text = wire.dumps({"vorp": float("nan"), "adp": float("inf"), "p": -float("inf")})
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"vorp": None, "adp": None, "p": None}


def test_nested_non_finite_values_are_found():
    payload = {"board": [{"vorp": 1.5}, {"vorp": float("nan")}], "gaps": {"x": [float("inf")]}}
    assert json.loads(wire.dumps(payload)) == {
        "board": [{"vorp": 1.5}, {"vorp": None}],
        "gaps": {"x": [None]},
    }


def test_numpy_scalars_serialize_instead_of_raising():
    """A stray np.int64 used to raise inside the console's push loop."""
    payload = {
        "made": np.int64(12),
        "p": np.float32(0.25),
        "flag": np.bool_(True),
        "bad": np.float64("nan"),
    }
    assert json.loads(wire.dumps(payload)) == {"made": 12, "p": 0.25, "flag": True, "bad": None}


def test_ordinary_values_pass_through_unchanged():
    payload = {"s": "Tim Stützle", "n": 3, "b": False, "none": None, "l": [1, "a"], "t": (1, 2)}
    assert json.loads(wire.dumps(payload)) == {**payload, "t": [1, 2]}


def test_loads_refuses_what_a_browser_would_refuse():
    with pytest.raises(ValueError):
        wire.loads('{"vorp": NaN}')
    with pytest.raises(ValueError):
        wire.loads('{"vorp": Infinity}')
    assert wire.loads('{"vorp": null}') == {"vorp": None}
