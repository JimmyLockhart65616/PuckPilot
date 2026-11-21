"""JSON exactly as a browser will parse it.

`json.dumps` writes a non-finite float as a bare `NaN` or `Infinity`. Python
reads that back happily; `JSON.parse` does not, so one NaN anywhere in a
snapshot makes the page's poll throw and the whole view freezes on "server
unreachable" - locally, and on the relay, which would faithfully store and
re-serve it. A value we do not have is `null` on the wire, which the page
already renders as absent.

numpy scalars are the other way this goes wrong: `json.dumps` refuses them
outright, and on the push path that exception used to escape into the draft
console's main loop.

Standard library only: the relay imports this, and the relay image has no numpy.
"""

from __future__ import annotations

import json
import math


def clean(obj):
    """A copy that `json.dumps(..., allow_nan=False)` accepts."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    item = getattr(obj, "item", None)
    if callable(item):
        # numpy scalar (np.float32, np.int64, np.bool_): unwrap, then clean
        return clean(item())
    return obj


def dumps(obj) -> str:
    return json.dumps(clean(obj), allow_nan=False)


def loads(text: str):
    """Parse the way a browser would: a bare NaN/Infinity is an error, not a float."""

    def refuse(token: str):
        raise ValueError(f"non-standard JSON constant {token!r}")

    return json.loads(text, parse_constant=refuse)
